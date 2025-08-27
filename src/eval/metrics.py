import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from functools import wraps
from pathlib import Path

import Levenshtein
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

load_dotenv()


# ========== DATA MODELS ==========

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
    judge_model: str
    formula_number: int | None = None


@dataclass
class FormulaStatistics:
    judge_model: str
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
    formula_statistics: dict[str, FormulaStatistics]
    text_statistics: TextStatistics


class FormulaEvaluationResponse(BaseModel):
    """Structured response model for LLM formula evaluation."""
    explanation: str
    is_correct: bool | None
    score: float | None
    errors: list[str]


# ========== CORE EVALUATION FUNCTIONS ==========

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
                judge_model=model,
                formula_number=None
            )
        return wrapper
    return decorator

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
        judge_model=model,
        formula_number=None
    )




def _get_formula_evaluation_prompt(gt_formula: str, extracted_formula: str) -> str:
    """Generate evaluation prompt for formula comparison."""
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
    models: list[str],
) -> dict[str, list[FormulaEvaluationResult]]:
    """Evaluate all formula pairs concurrently using ThreadPoolExecutor for multiple models."""
    if os.getenv("LLM_PROXY_URL"):
        client = OpenAI(base_url=os.getenv("LLM_PROXY_URL"),
                        api_key=os.getenv("LLM_PROXY_API_KEY"))
    else:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    all_results = {model: [] for model in models}
    
    with ThreadPoolExecutor() as executor:
        future_to_info = {}
        
        # Submit tasks for all model-formula combinations
        for model in models:
            for i, (gt_formula, extracted_formula) in enumerate(formula_pairs):
                future = executor.submit(evaluate_formula_pair, client, model, gt_formula, extracted_formula)
                future_to_info[future] = (model, i)
        
        total_tasks = len(models) * len(formula_pairs)
        for future in tqdm(as_completed(future_to_info), total=total_tasks, desc="Evaluating Formulas"):
            model, formula_index = future_to_info[future]
            try:
                result = future.result()
                result.formula_number = formula_index
                all_results[model].append(result)
            except Exception as exc:
                print(f"Formula {formula_index} with model {model} generated an exception: {exc}")
    
    # Sort results by formula number for each model
    for model in models:
        all_results[model] = sorted(all_results[model], key=lambda x: x.formula_number or 0)
    
    return all_results


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


# ========== STATISTICS CALCULATION ==========

def calculate_statistics(
    results: dict[str, list[FormulaEvaluationResult]], 
    text_results: list[TextSimilarityResult],
    gt_formulas: list[dict[str, str]],
) -> SummaryStatistics:
    """
    Calculate summary statistics from evaluation results.

    Args:
        results: Dictionary mapping model names to lists of formula evaluation results
        text_results: List of text evaluation results
        gt_formulas: List of ground truth formulas with metadata

    Returns:
        SummaryStatistics with aggregated metrics per model
    """
    # ========== FORMULA STATISTICS PER MODEL ==========
    formula_statistics = {}
    
    for model_name, model_results in results.items():
        valid_formula_scores = [result.score for result in model_results if result.score is not None]
        total_formula_score = sum(valid_formula_scores) if valid_formula_scores else 0
        average_formula_score = total_formula_score / len(valid_formula_scores) if valid_formula_scores else 0

        correct_formula_count = sum(1 for result in model_results if result.is_correct is True)
        total_formulas_evaluated = sum(1 for result in model_results if result.is_correct is not None)
        formula_accuracy_percentage = (
            correct_formula_count / total_formulas_evaluated * 100) if total_formulas_evaluated else 0
        
        # ========== SEPARATE INLINE/DISPLAY STATISTICS ==========
        inline_scores = []
        display_scores = []
        
        for i, result in enumerate(model_results):
            if result.score is not None and i < len(gt_formulas):
                if gt_formulas[i]['type'] == 'inline-formula':
                    inline_scores.append(result.score)
                elif gt_formulas[i]['type'] == 'display-formula':
                    display_scores.append(result.score)
        
        average_inline_score = sum(inline_scores) / len(inline_scores) if inline_scores else 0
        average_display_score = sum(display_scores) / len(display_scores) if display_scores else 0

        # ========== BUILD RESULT OBJECTS ==========
        formula_statistics[model_name] = FormulaStatistics(
            judge_model=model_name,
            total_formulas=len(model_results),
            correct_formulas=correct_formula_count,
            accuracy_percentage=formula_accuracy_percentage,
            average_score=average_formula_score,
            average_inline_score=average_inline_score,
            average_display_score=average_display_score
        )
    
    # ========== TEXT STATISTICS ==========
    text_similarities = [result.normalized_levenshtein_similarity for result in text_results]
    average_text_similarity = sum(text_similarities) / len(text_similarities) if text_similarities else 0
    
    text_stats = TextStatistics(
        total_texts=len(text_results),
        average_levenshtein_similarity=average_text_similarity
    )
    
    return SummaryStatistics(
        formula_statistics=formula_statistics,
        text_statistics=text_stats
    )


# ========== MAIN EVALUATION PIPELINE ==========

def run_evaluation(
        llm_judge_models: list[str],
        gt_json_path: Path,
        parsed_json_path: Path,
        result_stats_path: Path,
        result_formula_evals_path: Path,
        result_text_evals_path: Path,
) -> None:
    """
    Complete evaluation pipeline: load data, validate, evaluate, calculate stats, and save results.

    Args:
        llm_judge_models: List of model names for formula evaluation
        gt_json_path: Ground truth JSON file path
        parsed_json_path: Parsed results JSON file path
        result_stats_path: Output path for summary statistics
        result_formula_evals_path: Output path for detailed formula evaluations
        result_text_evals_path: Output path for detailed text evaluations

    Raises:
        FileNotFoundError: If input files don't exist
        ValueError: If data counts don't match between ground truth and extracted
        json.JSONDecodeError: If files contain invalid JSON
    """
    # ========== LOAD DATA ==========
    with open(gt_json_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    with open(parsed_json_path, 'r', encoding='utf-8') as f:
        parsed_data = json.load(f)
    
    gt_formulas = [item for item in gt_data if item.get('type') in ['inline-formula', 'display-formula']]
    gt_texts = [item['data'] for item in gt_data if item.get('type') == 'text']
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
    # Evaluate formulas with multiple models
    formula_pairs = list(zip([f['data'] for f in gt_formulas], [f['data'] for f in extracted_formulas]))
    formula_results = _evaluate_formulas(formula_pairs, llm_judge_models)
    
    # Evaluate texts
    text_pairs = list(zip(gt_texts, extracted_texts))
    text_results = _evaluate_texts(text_pairs)

    # ========== CALCULATE STATISTICS ==========
    summary_stats = calculate_statistics(formula_results, text_results, gt_formulas)

    # ========== WRITE RESULTS ==========
    with open(result_stats_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(summary_stats), f, indent=2, ensure_ascii=False)
    
    # Write formula results with all models
    formula_results_flattened = []
    for model_name, model_results in formula_results.items():
        formula_results_flattened.extend([asdict(result) for result in model_results])
    
    with open(result_formula_evals_path, 'w', encoding='utf-8') as f:
        json.dump(formula_results_flattened, f, indent=2, ensure_ascii=False)
    
    with open(result_text_evals_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(result) for result in text_results], f, indent=2, ensure_ascii=False)
