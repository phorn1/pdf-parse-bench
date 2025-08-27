import os
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from dataclasses import dataclass, asdict
from tqdm import tqdm
from dotenv import load_dotenv

from openai import OpenAI
from pydantic import BaseModel, field_validator
import Levenshtein

load_dotenv()


# ========== PYDANTIC MODELS ==========

class FormulaEvaluationResponse(BaseModel):
    explanation: str
    is_correct: bool | None
    score: float | None
    errors: list[str]
    

# ========== DATA CLASSES ==========

@dataclass
class TextSimilarityResult:
    normalized_levenshtein_similarity: float
    ground_truth_text: str
    extracted_text: str
    text_number: int | None = None


@dataclass
class FormulaEvaluationResult:
    explanation: str
    is_correct: bool | None
    score: float | None
    errors: list[str]
    ground_truth_formula: str
    extracted_formula: str
    formula_number: int | None = None


@dataclass
class FormulaStatistics:
    total_formulas: int
    correct_formulas: int
    accuracy_percentage: float
    average_score: float
    average_inline_score: float
    average_display_score: float


@dataclass
class TextStatistics:
    total_texts: int
    average_levenshtein_similarity: float


@dataclass
class SummaryStatistics:
    formula_statistics: FormulaStatistics
    text_statistics: TextStatistics


# ========== DECORATORS ==========

def retry_formula_evaluation(max_retries: int = 10):
    def decorator(func) -> object:
        @wraps(func)
        def wrapper(client: OpenAI, model: str, gt_formula: str, extracted_formula: str) -> FormulaEvaluationResult:
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(client, model, gt_formula, extracted_formula)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    continue
            
            # If all retries failed, return error result
            return FormulaEvaluationResult(
                explanation=f"Failed to evaluate after {max_retries} attempts",
                is_correct=None,
                score=None,
                errors=[f"Evaluation failed: {last_error}"],
                ground_truth_formula=gt_formula,
                extracted_formula=extracted_formula,
                formula_number=None
            )
        return wrapper
    return decorator


# ========== EVALUATION FUNCTIONS ==========

def evaluate_text_similarity(gt_text: str, extracted_text: str) -> TextSimilarityResult:
    """
    Evaluate text similarity using normalized Levenshtein distance.

    Args:
        gt_text: Ground truth text
        extracted_text: Extracted text

    Returns:
        TextSimilarityResult with similarity metrics
    """
    lev_distance = Levenshtein.distance(gt_text, extracted_text)
    max_len = max(len(gt_text), len(extracted_text))
    normalized_lev_similarity = 1 - (lev_distance / max_len) if max_len > 0 else 1.0

    return TextSimilarityResult(
        normalized_levenshtein_similarity=round(normalized_lev_similarity, 4),
        ground_truth_text=gt_text,
        extracted_text=extracted_text,
        text_number=None
    )


@retry_formula_evaluation()
def evaluate_formula_pair(
        client: OpenAI,
        model: str,
        gt_formula: str,
        extracted_formula: str,
) -> FormulaEvaluationResult:
    """
    Evaluate formula correctness using LLM.

    Args:
        client: OpenAI client instance
        model: The name of the model to use
        gt_formula: Ground truth formula content
        extracted_formula: Extracted formula content
        
    Returns:
        FormulaEvaluationResult with evaluation metrics
    """
    prompt = _get_formula_evaluation_prompt(gt_formula, extracted_formula)

    response = client.responses.parse(
        model=model,
        input=prompt,
        text_format=FormulaEvaluationResponse
    )

    # Extract the parsed data from the structured response
    parsed_data = response.output_parsed

    return FormulaEvaluationResult(
        explanation=parsed_data.explanation,
        is_correct=parsed_data.is_correct,
        score=parsed_data.score,
        errors=parsed_data.errors,
        ground_truth_formula=gt_formula,
        extracted_formula=extracted_formula,
        formula_number=None
    )


# ========== HELPER FUNCTIONS ==========

def _get_formula_evaluation_prompt(gt_formula: str, extracted_formula: str) -> str:
    return f"""
You are a mathematical formula evaluator. Your task is to determine if the extracted formula correctly represents the ground truth formula, focusing on both semantic meaning AND proper mathematical notation.

Ground Truth Formula:
{gt_formula}

Extracted Formula:
{extracted_formula}

Evaluate the extracted formula using the following criteria:
1. Correctness: Are the mathematical symbols, variables, and operations accurately preserved?
2. Completeness: Are all parts of the formula present without omissions?
3. Semantic equivalence: Does the extracted formula convey the same mathematical meaning?

First, provide a thorough explanation of your evaluation. Then assign a score and determine correctness.
In case there is no  extracted formula, assign a score of 0.

Provide your evaluation STRICTLY in JSON format, starting with {{ and ending with }}:
{{
    "explanation": "Detailed explanation of your evaluation",
    "is_correct": true/false,
    "score": (0-10 scale, where 10 is perfect match),
    "errors": ["error1", "error2", ...]
}}
"""


def _evaluate_formulas(
    formula_pairs: list[tuple[str, str]], 
    model: str,
) -> list[FormulaEvaluationResult]:
    """Evaluate all formula pairs concurrently using ThreadPoolExecutor."""
    if os.getenv("LLM_PROXY_URL"):
        client = OpenAI(base_url=os.getenv("LLM_PROXY_URL"),
                        api_key=os.getenv("LLM_PROXY_API_KEY"))
    else:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    results = []
    
    with ThreadPoolExecutor() as executor:
        future_to_index = {
            executor.submit(evaluate_formula_pair, client, model, gt_formula, extracted_formula): i
            for i, (gt_formula, extracted_formula) in enumerate(formula_pairs)
        }
        
        for future in tqdm(as_completed(future_to_index), total=len(formula_pairs), desc="Evaluating Formulas"):
            formula_index = future_to_index[future]
            try:
                result = future.result()
                result.formula_number = formula_index
                results.append(result)
            except Exception as exc:
                print(f"Formula {formula_index} generated an exception: {exc}")
    
    return sorted(results, key=lambda x: x.formula_number or 0)


def _evaluate_texts(text_pairs: list[tuple[str, str]]) -> list[TextSimilarityResult]:
    """Evaluate all text pairs sequentially."""
    results = []
    for i, (gt_text, extracted_text) in enumerate(tqdm(text_pairs, desc="Evaluating Texts")):
        try:
            result = evaluate_text_similarity(gt_text, extracted_text)
            result.text_number = i
            results.append(result)
        except Exception as exc:
            print(f"Text {i} generated an exception: {exc}")
    
    return results


# ========== STATISTICS FUNCTIONS ==========

def calculate_statistics(
    results: list[FormulaEvaluationResult], 
    text_results: list[TextSimilarityResult],
    gt_formulas: list[dict[str, str]],
    extracted_formulas: list[dict[str, str]]
) -> SummaryStatistics:
    """
    Calculate summary statistics from evaluation results.

    Args:
        results: List of formula evaluation results
        text_results: List of text evaluation results

    Returns:
        SummaryStatistics with aggregated metrics
    """
    # ========== FORMULA STATISTICS ==========
    valid_formula_scores = [result.score for result in results if result.score is not None]
    total_formula_score = sum(valid_formula_scores) if valid_formula_scores else 0
    average_formula_score = total_formula_score / len(valid_formula_scores) if valid_formula_scores else 0

    correct_formula_count = sum(1 for result in results if result.is_correct is True)
    total_formulas_evaluated = sum(1 for result in results if result.is_correct is not None)
    formula_accuracy_percentage = (
        correct_formula_count / total_formulas_evaluated * 100) if total_formulas_evaluated else 0
    
    # ========== SEPARATE INLINE/DISPLAY STATISTICS ==========
    inline_scores = []
    display_scores = []
    
    for i, result in enumerate(results):
        if result.score is not None and i < len(gt_formulas):
            if gt_formulas[i]['type'] == 'inline-formula':
                inline_scores.append(result.score)
            elif gt_formulas[i]['type'] == 'display-formula':
                display_scores.append(result.score)
    
    average_inline_score = sum(inline_scores) / len(inline_scores) if inline_scores else 0
    average_display_score = sum(display_scores) / len(display_scores) if display_scores else 0

    # ========== TEXT STATISTICS ==========
    text_similarities = [result.normalized_levenshtein_similarity for result in text_results]
    average_text_similarity = sum(text_similarities) / len(text_similarities) if text_similarities else 0

    # ========== BUILD RESULT OBJECTS ==========
    formula_stats = FormulaStatistics(
        total_formulas=len(results),
        correct_formulas=correct_formula_count,
        accuracy_percentage=formula_accuracy_percentage,
        average_score=average_formula_score,
        average_inline_score=average_inline_score,
        average_display_score=average_display_score
    )
    
    text_stats = TextStatistics(
        total_texts=len(text_results),
        average_levenshtein_similarity=average_text_similarity
    )
    
    return SummaryStatistics(
        formula_statistics=formula_stats,
        text_statistics=text_stats
    )


# ========== MAIN EVALUATION FUNCTION ==========

def run_evaluation(
        llm_judge_model: str,
        gt_json_path: Path,
        parsed_json_path: Path,
        result_stats_path: Path,
        result_formula_evals_path: Path,
        result_text_evals_path: Path,
        use_hso_llm_proxy: bool = True,
) -> None:
    """
    Evaluate the parsed formulas and texts against ground truth.

    Args:
        llm_judge_model: The name of the model to use
        gt_json_path: Path to the ground truth json file
        parsed_json_path: Path to the parsed json file containing formulas and texts to evaluate
        result_stats_path: Path to write summary statistics JSON file
        result_formula_evals_path: Path to write formula evaluation results JSON file
        result_text_evals_path: Path to write text evaluation results JSON file

    Raises:
        FileNotFoundError: If input files don't exist
        ValueError: If no formulas found in either file or if formula count mismatch
        json.JSONDecodeError: If files contain invalid JSON
    """

    # ========== LOAD DATA ==========
    # Load ground truth data
    with open(gt_json_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    gt_formulas = [item for item in gt_data if item.get('type') in ['inline-formula', 'display-formula']]
    gt_texts = [item['data'] for item in gt_data if item.get('type') == 'text']

    # Load extracted data
    with open(parsed_json_path, 'r', encoding='utf-8') as f:
        parsed_data = json.load(f)
    extracted_formulas = [item for item in parsed_data if item.get('type') in ['inline-formula', 'display-formula']]
    extracted_texts = [item['data'] for item in parsed_data if item.get('type') == 'text']

    # ========== VALIDATE DATA COUNTS ==========
    if len(gt_formulas) != len(extracted_formulas):
        raise ValueError(
            f"Formula count mismatch: gt_formulas={len(gt_formulas)}, "
            f"extracted_formulas={len(extracted_formulas)}"
        )

    if len(gt_texts) != len(extracted_texts):
        raise ValueError(
            f"Text count mismatch: gt_texts={len(gt_texts)}, "
            f"extracted_texts={len(extracted_texts)}"
        )

    # ========== RUN EVALUATIONS ==========
    # Evaluate formulas
    print(f"Starting formula evaluation for {len(gt_formulas)} formulas...")
    formula_pairs = list(zip([f['data'] for f in gt_formulas], [f['data'] for f in extracted_formulas]))
    formula_results = _evaluate_formulas(formula_pairs, llm_judge_model)
    
    # Evaluate texts
    print(f"Starting text evaluation for {len(gt_texts)} texts...")
    text_pairs = list(zip(gt_texts, extracted_texts))
    text_results = _evaluate_texts(text_pairs)

    # ========== CALCULATE STATISTICS ==========
    summary_stats = calculate_statistics(formula_results, text_results, gt_formulas, extracted_formulas)

    # ========== WRITE RESULTS ==========
    with open(result_stats_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(summary_stats), f, indent=2, ensure_ascii=False)
    
    with open(result_formula_evals_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(result) for result in formula_results], f, indent=2, ensure_ascii=False)
    
    with open(result_text_evals_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(result) for result in text_results], f, indent=2, ensure_ascii=False)
