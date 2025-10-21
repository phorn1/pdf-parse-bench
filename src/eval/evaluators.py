import base64
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from pathlib import Path
from typing import Literal

import Levenshtein
import requests
from dotenv import load_dotenv
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from openai import OpenAI
from google.genai import types
from google import genai
from mistralai import Mistral
from pydantic import BaseModel, Field
from tqdm import tqdm

load_dotenv()


# ========== CONSTANTS ==========

SUPPORTED_MODELS = {
    "gemini-2.5-flash": "gemini",
    "mistral-medium-2508": "mistral", 
    "gpt-5-nano": "openai",
    "gpt-5-mini": "openai",
    "gpt-5": "openai"
}

MODEL_MAX_WORKERS = {
    "gemini-2.5-flash": 8,
    "mistral-medium-2508": 2,
    "gpt-5-nano": 16,
    "gpt-5-mini": 12,
    "gpt-5": 8
}

FORMULA_EVALUATION_PROMPT_TEMPLATE = """
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



# ========== DATA MODELS ==========

class LLMJudgeEval(BaseModel):
    type: str = "llm_judge"
    judge_model: str
    explanation: str
    is_correct: bool
    score: int
    errors: list[str] = Field(default_factory=list)


class CDMEval(BaseModel):
    type: str = "cdm"
    score: float
    image_name: str | None = None


class FormulaEvaluationSummary(BaseModel):
    formula_number: int
    ground_truth_formula: str
    extracted_formula: str
    formula_type: Literal['inline-formula', 'display-formula']
    llm_evals: list[LLMJudgeEval] = Field(default_factory=list)
    cdm_eval: CDMEval | None = None
    bleu_score: float | None = None
    levenshtein_similarity: float | None = None
    
    @property
    def llm_evals_by_model(self) -> dict[str, LLMJudgeEval]:
        """Get LLM evaluations as dict by judge model for easier UI access."""
        return {eval.judge_model: eval for eval in self.llm_evals}


class CDMStatistics(BaseModel):
    total_formulas: int
    average_score: float
    average_inline_score: float
    average_display_score: float


class LLMJudgeStatistics(BaseModel):
    judge_model: str
    correct_formulas: int
    accuracy_percentage: float
    average_score: float
    average_inline_score: float
    average_display_score: float


class FormulaStatistics(BaseModel):
    total_formulas: int
    llm_judge: list[LLMJudgeStatistics]
    cdm: CDMStatistics | None


class SummaryStatistics(BaseModel):
    formula_statistics: FormulaStatistics

    @property
    def llm_judge_stats_by_model(self) -> dict[str, LLMJudgeStatistics]:
        """Get LLM judge statistics as dict by judge model for easier access."""
        return {stat.judge_model: stat for stat in self.formula_statistics.llm_judge}


# ========== EVALUATION CLASSES ==========

class CDMEvaluator:
    """Evaluates formulas using CDM service."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        
        # Check CDM service URL availability at initialization
        cdm_service_url = os.getenv("CDM_SERVICE_URL")
        if not cdm_service_url:
            raise ValueError("CDM_SERVICE_URL environment variable is required for CDM evaluation.\n"
                             "Note: CDM evaluation is an experimental feature that requires a separate local service installation. "
                             "This component is not part of the core benchmarking suite and does not work out-of-the-box.")
        self.cdm_service_url = cdm_service_url
    
    def evaluate_batch(self, pairs: list[tuple[str, str]]) -> list[CDMEval]:
        results = []
        for i, (gt_formula, extracted_formula) in enumerate(tqdm(pairs, desc="Evaluating CDM")):
            results.append(self._evaluate_single(gt_formula, extracted_formula, i))
        return results
    
    def _evaluate_single(self, gt_formula: str, extracted_formula: str, index: int) -> CDMEval:
        # Call CDM service to evaluate formula similarity and get visualization
        response = requests.post(self.cdm_service_url, json={
            'gt_formula': gt_formula,
            'pred_formula': extracted_formula,
            'case_id': f"formula_{index}"
        })
        response.raise_for_status()

        result = response.json()
        
        # Save visualization if available
        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_name = f"formula_{index:03}.png"
        image_path = self.output_dir / image_name
        
        # Check if visualization_base64 exists and is not None
        visualization_b64 = result.get('visualization_base64')
        if visualization_b64:
            with open(image_path, 'wb') as f:
                f.write(base64.b64decode(visualization_b64))
        else:
            image_name = None

        return CDMEval(score=result['cdm_f1'], image_name=image_name)


class NaiveEvaluator:
    """Evaluates formulas using naive text similarity metrics (BLEU and Levenshtein)."""
    
    @staticmethod
    def _clean_formula(formula: str) -> str:
        """Clean LaTeX formula by removing $$ delimiters and normalizing whitespace."""
        cleaned = re.sub(r'\$+', '', formula)
        cleaned = ' '.join(cleaned.split())
        return cleaned.strip()
    
    @staticmethod
    def _tokenize_formula(formula: str) -> list[str]:
        """Tokenize formula into meaningful parts for BLEU calculation."""
        tokens = re.findall(r'\\[a-zA-Z]+|[a-zA-Z0-9]+|[{}()\[\]|_^=+\-*/\\,.<>]|\'', formula)
        return [token for token in tokens if token.strip()]
    
    def evaluate_batch(self, pairs: list[tuple[str, str]]) -> list[tuple[float, float]]:
        """
        Calculate BLEU and Levenshtein similarity for formula pairs.
        Returns list of (bleu_score, levenshtein_similarity) tuples.
        """
        results = []
        for gt_formula, extracted_formula in tqdm(pairs, desc="Evaluating naive metrics"):
            # Clean formulas
            gt_clean = self._clean_formula(gt_formula)
            ext_clean = self._clean_formula(extracted_formula)
            
            # Calculate BLEU score
            try:
                tokens_gt = self._tokenize_formula(gt_clean)
                tokens_ext = self._tokenize_formula(ext_clean)
                smoothing_function = SmoothingFunction().method1
                bleu_score = sentence_bleu([tokens_gt], tokens_ext, smoothing_function=smoothing_function)
                bleu_score = round(bleu_score, 4)
            except:
                bleu_score = 0.0
            
            # Calculate Levenshtein similarity
            distance = Levenshtein.distance(gt_clean, ext_clean)
            max_length = max(len(gt_clean), len(ext_clean))
            
            if max_length == 0:
                levenshtein_similarity = 1.0
            else:
                similarity = 1 - (distance / max_length)
                levenshtein_similarity = round(similarity, 4)
            
            results.append((bleu_score, levenshtein_similarity))
        
        return results


class LLMEvaluator:
    """Evaluates formulas using LLM judges."""
    
    class FormulaResponse(BaseModel):
        explanation: str
        is_correct: bool
        score: int
        errors: list[str]
    

    @staticmethod
    def _retry_on_failure(max_retries: int = 10):
        def decorator(func):
            @wraps(func)
            def wrapper(model: str, gt_formula: str, extracted_formula: str) -> LLMJudgeEval:
                last_error = None
                for attempt in range(max_retries):
                    try:
                        return func(model, gt_formula, extracted_formula)
                    except Exception as e:
                        last_error = e
                        if attempt < max_retries - 1:
                            print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                
                # If we reach here, all attempts failed
                raise last_error
            return wrapper
        return decorator
    
    
    @staticmethod
    @_retry_on_failure()
    def evaluate_openai(model: str, gt_formula: str, extracted_formula: str) -> LLMJudgeEval:
        
        if os.getenv("LLM_PROXY_URL"):
            openai_client = OpenAI(
                base_url=os.getenv("LLM_PROXY_URL"),
                api_key=os.getenv("LLM_PROXY_API_KEY")
            )
        else:
            openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        prompt = FORMULA_EVALUATION_PROMPT_TEMPLATE.format(gt_formula=gt_formula, extracted_formula=extracted_formula)
        response = openai_client.responses.parse(
            model=model, input=prompt, text_format=LLMEvaluator.FormulaResponse
        )
        
        data = response.output_parsed
        return LLMJudgeEval(
            judge_model=model,
            explanation=data.explanation,
            is_correct=data.is_correct,
            score=data.score,
            errors=data.errors
        )
    
    @staticmethod
    @_retry_on_failure()
    def evaluate_gemini(model: str, gt_formula: str, extracted_formula: str) -> LLMJudgeEval:
        
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY required for Gemini models")
        gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        prompt = FORMULA_EVALUATION_PROMPT_TEMPLATE.format(gt_formula=gt_formula, extracted_formula=extracted_formula)
        response = gemini_client.models.generate_content(
            model=model,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=LLMEvaluator.FormulaResponse
            )
        )
        
        parsed_data = json.loads(response.text)
        data = LLMEvaluator.FormulaResponse(**parsed_data)
        return LLMJudgeEval(
            judge_model=model,
            explanation=data.explanation,
            is_correct=data.is_correct,
            score=data.score,
            errors=data.errors
        )
    
    @staticmethod
    @_retry_on_failure()
    def evaluate_mistral(model: str, gt_formula: str, extracted_formula: str) -> LLMJudgeEval:
        
        if not os.getenv("MISTRAL_API_KEY"):
            raise ValueError("MISTRAL_API_KEY required for Mistral models")
        mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        
        # Escape the formulas for safe JSON inclusion
        gt_formula_escaped = json.dumps(gt_formula)[1:-1]  # Remove outer quotes
        extracted_formula_escaped = json.dumps(extracted_formula)[1:-1]  # Remove outer quotes
        
        prompt = FORMULA_EVALUATION_PROMPT_TEMPLATE.format(
            gt_formula=gt_formula_escaped, 
            extracted_formula=extracted_formula_escaped
        )
        
        messages = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
        
        chat_response = mistral_client.chat.parse(
            model=model,
            messages=messages,
            response_format=LLMEvaluator.FormulaResponse,
            temperature=0
        )
        
        data = chat_response.choices[0].message.parsed
        return LLMJudgeEval(
            judge_model=model,
            explanation=data.explanation,
            is_correct=data.is_correct,
            score=data.score,
            errors=data.errors
        )
    


# ========== STATISTICS CALCULATOR ==========

class LLMStatisticsCalculator:
    """Calculates LLM evaluation statistics."""
    
    @staticmethod
    def calculate(summaries: list[FormulaEvaluationSummary]) -> list[LLMJudgeStatistics]:
        # Group by model
        model_evals = {}
        for summary in summaries:
            for llm_eval in summary.llm_evals:
                model = llm_eval.judge_model
                if model not in model_evals:
                    model_evals[model] = []
                model_evals[model].append((llm_eval, summary))
        
        stats = []
        for model, evals in model_evals.items():
            scores = [e.score for e, _ in evals]
            correct = sum(1 for e, _ in evals if e.is_correct is True)
            total = len(evals)
            
            # Calculate inline/display scores
            inline_scores = [e.score for e, s in evals if s.formula_type == 'inline-formula']
            display_scores = [e.score for e, s in evals if s.formula_type == 'display-formula']
            
            stats.append(LLMJudgeStatistics(
                judge_model=model,
                correct_formulas=correct,
                accuracy_percentage=correct / total * 100 if total else 0,
                average_score=sum(scores) / len(scores) if scores else 0,
                average_inline_score=sum(inline_scores) / len(inline_scores) if inline_scores else 0,
                average_display_score=sum(display_scores) / len(display_scores) if display_scores else 0
            ))
        
        return stats


class CDMStatisticsCalculator:
    """Calculates CDM evaluation statistics."""
    
    @staticmethod
    def calculate(summaries: list[FormulaEvaluationSummary]) -> CDMStatistics | None:
        cdm_evals = [(summary.cdm_eval, summary) for summary in summaries if summary.cdm_eval is not None]
        
        if not cdm_evals:
            return None
        
        scores = [e.score for e, _ in cdm_evals]
        
        # Calculate inline/display scores
        inline_scores = [e.score for e, s in cdm_evals if s.formula_type == 'inline-formula']
        display_scores = [e.score for e, s in cdm_evals if s.formula_type == 'display-formula']
        
        return CDMStatistics(
            total_formulas=len(cdm_evals),
            average_score=sum(scores) / len(scores) if scores else 0,
            average_inline_score=sum(inline_scores) / len(inline_scores) if inline_scores else 0,
            average_display_score=sum(display_scores) / len(display_scores) if display_scores else 0
        )


class StatisticsCalculator:
    """Main statistics calculator that coordinates all evaluation types."""

    @staticmethod
    def calculate(formula_summaries: list[FormulaEvaluationSummary]) -> SummaryStatistics:
        llm_judge_stats = LLMStatisticsCalculator.calculate(formula_summaries)
        cdm_stats = CDMStatisticsCalculator.calculate(formula_summaries)

        return SummaryStatistics(
            formula_statistics=FormulaStatistics(
                total_formulas=len(formula_summaries),
                llm_judge=llm_judge_stats,
                cdm=cdm_stats
            )
        )
    


# ========== FILE I/O HELPERS ==========

def load_formula_summaries(file_path: Path) -> list[FormulaEvaluationSummary]:
    """Load formula summaries from JSON file, return empty list if file doesn't exist."""
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return [FormulaEvaluationSummary(**item) for item in data]
    return []

def save_formula_summaries(file_path: Path, summaries: list[FormulaEvaluationSummary]) -> None:
    """Save formula summaries to JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump([s.model_dump() for s in summaries], f, indent=2, ensure_ascii=False)

def save_statistics(file_path: Path, stats: SummaryStatistics) -> None:
    """Save statistics to JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(stats.model_dump(), f, indent=2, ensure_ascii=False)


# ========== MAIN EVALUATION PIPELINE ==========

def run_evaluation(
    llm_judge_models: list[str],
    enable_cdm: bool,
    extracted_formulas_path: Path,
    result_stats_path: Path,
    result_formula_evals_path: Path,
    cdm_output_dir: Path,
) -> None:
    """
    Complete evaluation pipeline with incremental saving.

    Args:
        llm_judge_models: Models for formula evaluation
        enable_cdm: Whether to enable CDM scoring
        extracted_formulas_path: Path to JSON with paired formulas (gt_formula, parsed_formula)
        result_stats_path: Output path for statistics
        result_formula_evals_path: Output path for formula evaluations
        cdm_output_dir: CDM visualization output directory
    """
    # ========== LOAD AND PREPARE DATA ==========
    with open(extracted_formulas_path, 'r', encoding='utf-8') as f:
        formula_pairs_data = json.load(f)

    # Load existing results or initialize new ones
    formula_summaries = load_formula_summaries(result_formula_evals_path) or [
        FormulaEvaluationSummary(
            formula_number=i,
            ground_truth_formula=pair['gt_formula'],
            extracted_formula=pair['parsed_formula'],
            formula_type='display-formula'  # Type info no longer available, defaulting
        )
        for i, pair in enumerate(formula_pairs_data)
    ]

    # ========== LLM FORMULA EVALUATIONS ==========
    existing_models = {
        eval_result.judge_model
        for summary in formula_summaries
        for eval_result in summary.llm_evals
    }

    for model in [m for m in llm_judge_models if m not in existing_models]:
        # Validate model is supported
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model}")

        client_type = SUPPORTED_MODELS[model]

        if client_type == "gemini":
            evaluate_func = LLMEvaluator.evaluate_gemini
        elif client_type == "mistral":
            evaluate_func = LLMEvaluator.evaluate_mistral
        else:  # openai
            evaluate_func = LLMEvaluator.evaluate_openai

        # Parallel evaluation with results tracking
        with ThreadPoolExecutor(max_workers=MODEL_MAX_WORKERS[model]) as executor:
            future_to_index = {
                executor.submit(evaluate_func, model, pair['gt_formula'], pair['parsed_formula']): i
                for i, pair in enumerate(formula_pairs_data)
            }

            # Collect and sort results
            results = [(future.result(), future_to_index[future])
                      for future in tqdm(as_completed(future_to_index),
                                       total=len(future_to_index),
                                       desc=f"Evaluating with {model}")]
            results.sort(key=lambda x: x[1])

        # Apply results and save incrementally
        for (result, index) in results:
            formula_summaries[index].llm_evals.append(result)

        save_formula_summaries(result_formula_evals_path, formula_summaries)
        save_statistics(result_stats_path, StatisticsCalculator.calculate(formula_summaries))

    # ========== NAIVE FORMULA SIMILARITY EVALUATION ==========
    if any(summary.bleu_score is None or summary.levenshtein_similarity is None for summary in formula_summaries):
        naive_evaluator = NaiveEvaluator()
        formula_pairs = [(pair['gt_formula'], pair['parsed_formula']) for pair in formula_pairs_data]
        naive_results = naive_evaluator.evaluate_batch(formula_pairs)

        for i, (bleu_score, levenshtein_similarity) in enumerate(naive_results):
            formula_summaries[i].bleu_score = bleu_score
            formula_summaries[i].levenshtein_similarity = levenshtein_similarity

        save_formula_summaries(result_formula_evals_path, formula_summaries)

    # ========== CDM EVALUATION ==========
    if enable_cdm:
        cdm_evaluator = CDMEvaluator(cdm_output_dir)
        formula_pairs = [(pair['gt_formula'], pair['parsed_formula']) for pair in formula_pairs_data]
        cdm_results = cdm_evaluator.evaluate_batch(formula_pairs)

        for i, cdm_result in enumerate(cdm_results):
            formula_summaries[i].cdm_eval = cdm_result

        save_formula_summaries(result_formula_evals_path, formula_summaries)

    # ========== FINALIZE ==========
    save_statistics(result_stats_path, StatisticsCalculator.calculate(formula_summaries))
