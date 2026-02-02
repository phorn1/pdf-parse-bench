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

Respond with ONLY a single integer from 0 to 10, where 10 is a perfect match. No other text.
"""

TABLE_EVALUATION_PROMPT_TEMPLATE = """
You are a table evaluator. Your task is to determine if the extracted table correctly represents the ground truth table, focusing on both content accuracy AND structural preservation.

Ground Truth Table (LaTeX):
{gt_table}

Extracted Table:
{extracted_table}

Evaluate the extracted table using the following criteria:
1. Content accuracy: Are all cell values, headers, and data correctly preserved?
2. Structure preservation: Is the row/column structure maintained (even if format differs)?
3. Completeness: Are all rows and columns present without omissions or additions?

Note: The extracted table may be in a different format (markdown, HTML, list, plain text) than the ground truth LaTeX. Focus on semantic equivalence, not syntactic matching.

Respond with ONLY a single integer from 0 to 10, where 10 is a perfect match. No other text.
"""



# ========== DATA MODELS ==========

class LLMJudgeEval(BaseModel):
    type: str = "llm_judge"
    judge_model: str
    score: int


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
    average_score: float
    average_inline_score: float
    average_display_score: float


class FormulaStatistics(BaseModel):
    total_formulas: int
    llm_judge: list[LLMJudgeStatistics]
    cdm: CDMStatistics | None


class TableEvaluationSummary(BaseModel):
    table_number: int
    ground_truth_table: str
    extracted_table: str
    complexity: Literal['simple', 'moderate', 'complex']
    llm_evals: list[LLMJudgeEval] = Field(default_factory=list)

    @property
    def llm_evals_by_model(self) -> dict[str, LLMJudgeEval]:
        """Get LLM evaluations as dict by judge model for easier UI access."""
        return {eval.judge_model: eval for eval in self.llm_evals}


class TableLLMJudgeStatistics(BaseModel):
    judge_model: str
    average_score: float
    average_simple_score: float
    average_moderate_score: float
    average_complex_score: float


class TableStatistics(BaseModel):
    total_tables: int
    llm_judge: list[TableLLMJudgeStatistics]


class SummaryStatistics(BaseModel):
    formula_statistics: FormulaStatistics
    table_statistics: TableStatistics

    @property
    def llm_judge_stats_by_model(self) -> dict[str, LLMJudgeStatistics]:
        """Get LLM judge statistics as dict by judge model for easier access."""
        return {stat.judge_model: stat for stat in self.formula_statistics.llm_judge}

    @property
    def table_llm_judge_stats_by_model(self) -> dict[str, TableLLMJudgeStatistics]:
        """Get table LLM judge statistics as dict by judge model for easier access."""
        return {stat.judge_model: stat for stat in self.table_statistics.llm_judge}


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
    """Evaluates formulas and tables using LLM judges."""

    # Shared client instances (initialized lazily)
    _openai_client = None
    _gemini_client = None
    _mistral_client = None

    @classmethod
    def _get_openai_client(cls):
        """Get or create the shared OpenAI client."""
        if cls._openai_client is None:
            if os.getenv("LLM_PROXY_URL"):
                cls._openai_client = OpenAI(
                    base_url=os.getenv("LLM_PROXY_URL"),
                    api_key=os.getenv("LLM_PROXY_API_KEY")
                )
            else:
                cls._openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return cls._openai_client

    @classmethod
    def _get_gemini_client(cls):
        """Get or create the shared Gemini client."""
        if cls._gemini_client is None:
            if not os.getenv("GEMINI_API_KEY"):
                raise ValueError("GEMINI_API_KEY required for Gemini models")
            cls._gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        return cls._gemini_client

    @classmethod
    def _get_mistral_client(cls):
        """Get or create the shared Mistral client."""
        if cls._mistral_client is None:
            if not os.getenv("MISTRAL_API_KEY"):
                raise ValueError("MISTRAL_API_KEY required for Mistral models")
            cls._mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        return cls._mistral_client

    @staticmethod
    def _retry_on_failure(max_retries: int = 10):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs) -> LLMJudgeEval:
                last_error = None
                for attempt in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_error = e
                        if attempt < max_retries - 1:
                            print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                raise last_error
            return wrapper
        return decorator

    # ========== CORE EVALUATION METHODS ==========

    @staticmethod
    def _parse_score(text: str) -> int:
        """Parse integer score from LLM response text."""
        # Extract first integer found in the response
        match = re.search(r'\b(\d+)\b', text.strip())
        if match:
            score = int(match.group(1))
            return max(0, min(10, score))  # Clamp to 0-10
        raise ValueError(f"Could not parse score from response: {text}")

    @staticmethod
    def _evaluate(model: str, prompt: str) -> LLMJudgeEval:
        """Evaluate using the appropriate client based on model."""
        client_type = SUPPORTED_MODELS[model]
        if client_type == "openai":
            client = LLMEvaluator._get_openai_client()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            text = response.choices[0].message.content
        elif client_type == "gemini":
            client = LLMEvaluator._get_gemini_client()
            response = client.models.generate_content(
                model=model,
                contents=[prompt]
            )
            text = response.text
        else:  # mistral
            client = LLMEvaluator._get_mistral_client()
            chat_response = client.chat.complete(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            text = chat_response.choices[0].message.content
        score = LLMEvaluator._parse_score(text)
        return LLMJudgeEval(judge_model=model, score=score)

    @staticmethod
    @_retry_on_failure()
    def evaluate_formula(model: str, gt_formula: str, extracted_formula: str) -> LLMJudgeEval:
        """Evaluate a formula pair."""
        if SUPPORTED_MODELS[model] == "mistral":
            gt_formula = json.dumps(gt_formula)[1:-1]
            extracted_formula = json.dumps(extracted_formula)[1:-1]
        prompt = FORMULA_EVALUATION_PROMPT_TEMPLATE.format(
            gt_formula=gt_formula, extracted_formula=extracted_formula
        )
        return LLMEvaluator._evaluate(model, prompt)

    @staticmethod
    @_retry_on_failure()
    def evaluate_table(model: str, gt_table: str, extracted_table: str) -> LLMJudgeEval:
        """Evaluate a table pair."""
        if SUPPORTED_MODELS[model] == "mistral":
            gt_table = json.dumps(gt_table)[1:-1]
            extracted_table = json.dumps(extracted_table)[1:-1]
        prompt = TABLE_EVALUATION_PROMPT_TEMPLATE.format(
            gt_table=gt_table, extracted_table=extracted_table
        )
        return LLMEvaluator._evaluate(model, prompt)


# ========== STATISTICS FUNCTIONS ==========

def calculate_llm_stats(summaries: list[FormulaEvaluationSummary]) -> list[LLMJudgeStatistics]:
    """Calculate LLM evaluation statistics for formulas."""
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

        # Calculate inline/display scores
        inline_scores = [e.score for e, s in evals if s.formula_type == 'inline-formula']
        display_scores = [e.score for e, s in evals if s.formula_type == 'display-formula']

        stats.append(LLMJudgeStatistics(
            judge_model=model,
            average_score=sum(scores) / len(scores) if scores else 0,
            average_inline_score=sum(inline_scores) / len(inline_scores) if inline_scores else 0,
            average_display_score=sum(display_scores) / len(display_scores) if display_scores else 0
        ))

    return stats


def calculate_cdm_stats(summaries: list[FormulaEvaluationSummary]) -> CDMStatistics | None:
    """Calculate CDM evaluation statistics."""
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


def calculate_table_llm_stats(summaries: list[TableEvaluationSummary]) -> list[TableLLMJudgeStatistics]:
    """Calculate table LLM evaluation statistics."""
    if not summaries:
        return []

    # Group by model
    model_evals: dict[str, list[tuple[LLMJudgeEval, TableEvaluationSummary]]] = {}
    for summary in summaries:
        for llm_eval in summary.llm_evals:
            model = llm_eval.judge_model
            if model not in model_evals:
                model_evals[model] = []
            model_evals[model].append((llm_eval, summary))

    stats = []
    for model, evals in model_evals.items():
        scores = [e.score for e, _ in evals]

        # Calculate scores by complexity
        simple_scores = [e.score for e, s in evals if s.complexity == 'simple']
        moderate_scores = [e.score for e, s in evals if s.complexity == 'moderate']
        complex_scores = [e.score for e, s in evals if s.complexity == 'complex']

        stats.append(TableLLMJudgeStatistics(
            judge_model=model,
            average_score=sum(scores) / len(scores) if scores else 0,
            average_simple_score=sum(simple_scores) / len(simple_scores) if simple_scores else 0,
            average_moderate_score=sum(moderate_scores) / len(moderate_scores) if moderate_scores else 0,
            average_complex_score=sum(complex_scores) / len(complex_scores) if complex_scores else 0
        ))

    return stats


def calculate_statistics(
    formula_summaries: list[FormulaEvaluationSummary],
    table_summaries: list[TableEvaluationSummary]
) -> SummaryStatistics:
    """Calculate all evaluation statistics."""
    return SummaryStatistics(
        formula_statistics=FormulaStatistics(
            total_formulas=len(formula_summaries),
            llm_judge=calculate_llm_stats(formula_summaries),
            cdm=calculate_cdm_stats(formula_summaries)
        ),
        table_statistics=TableStatistics(
            total_tables=len(table_summaries),
            llm_judge=calculate_table_llm_stats(table_summaries)
        )
    )
    


# ========== FILE I/O HELPERS ==========

def save_summaries(file_path: Path, summaries: list[BaseModel]) -> None:
    """Save summaries to JSON file."""
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
    llm_judge_models: str | list[str] = "gpt-5-mini",
    enable_cdm: bool = False,
    skip_existing: bool = True,
    extracted_formulas_path: Path = None,
    extracted_tables_path: Path = None,
    result_stats_path: Path = None,
    result_formula_evals_path: Path = None,
    result_table_evals_path: Path = None,
    cdm_output_dir: Path = None,
) -> None:
    """
    Complete evaluation pipeline with incremental saving.

    Args:
        llm_judge_models: Model(s) for formula/table evaluation
        enable_cdm: Whether to enable CDM scoring for formulas
        skip_existing: If True, skip models that already have results. If False, re-evaluate and overwrite existing results
        extracted_formulas_path: Path to JSON with paired formulas (gt_formula, parsed_formula)
        extracted_tables_path: Path to JSON with paired tables (gt_table, parsed_table)
        result_stats_path: Output path for statistics
        result_formula_evals_path: Output path for formula evaluations
        result_table_evals_path: Output path for table evaluations
        cdm_output_dir: CDM visualization output directory
    """
    # Normalize to list
    if isinstance(llm_judge_models, str):
        llm_judge_models = [llm_judge_models]

    # ========== LOAD AND PREPARE FORMULA DATA ==========
    with open(extracted_formulas_path, 'r', encoding='utf-8') as f:
        formula_pairs_data = json.load(f)

    # Load existing results or initialize new ones
    formula_summaries = []
    if result_formula_evals_path and result_formula_evals_path.exists():
        with open(result_formula_evals_path, 'r', encoding='utf-8') as f:
            formula_summaries = [FormulaEvaluationSummary(**item) for item in json.load(f)]

    if not formula_summaries:
        formula_summaries = [
            FormulaEvaluationSummary(
                formula_number=i,
                ground_truth_formula=pair['gt_formula'],
                extracted_formula=pair['parsed_formula'],
                formula_type='display-formula' if pair['gt_formula'].startswith('$$') else 'inline-formula'
            )
            for i, pair in enumerate(formula_pairs_data)
        ]

    # ========== LOAD AND PREPARE TABLE DATA ==========
    table_pairs_data = []
    if extracted_tables_path and extracted_tables_path.exists():
        with open(extracted_tables_path, 'r', encoding='utf-8') as f:
            table_pairs_data = json.load(f)

    # Load existing results or initialize new ones
    table_summaries = []
    if result_table_evals_path and result_table_evals_path.exists():
        with open(result_table_evals_path, 'r', encoding='utf-8') as f:
            table_summaries = [TableEvaluationSummary(**item) for item in json.load(f)]

    if not table_summaries:
        table_summaries = [
            TableEvaluationSummary(
                table_number=i,
                ground_truth_table=pair['gt_table'],
                extracted_table=pair['parsed_table'],
                complexity=pair['complexity']
            )
            for i, pair in enumerate(table_pairs_data)
        ]

    # ========== DETERMINE MODELS TO EVALUATE ==========
    if skip_existing:
        # Skip models that already have results
        existing_formula_models = {
            eval_result.judge_model
            for summary in formula_summaries
            for eval_result in summary.llm_evals
        }
        existing_table_models = {
            eval_result.judge_model
            for summary in table_summaries
            for eval_result in summary.llm_evals
        }
        formula_models_to_evaluate = [m for m in llm_judge_models if m not in existing_formula_models]
        table_models_to_evaluate = [m for m in llm_judge_models if m not in existing_table_models]
    else:
        # Reprocess: remove old results for requested models and re-evaluate
        formula_models_to_evaluate = llm_judge_models
        table_models_to_evaluate = llm_judge_models
        models_to_reprocess = set(llm_judge_models)

        # Remove existing evaluations for models being reprocessed
        for summary in formula_summaries:
            summary.llm_evals = [
                eval_result for eval_result in summary.llm_evals
                if eval_result.judge_model not in models_to_reprocess
            ]
        for summary in table_summaries:
            summary.llm_evals = [
                eval_result for eval_result in summary.llm_evals
                if eval_result.judge_model not in models_to_reprocess
            ]

    # ========== LLM FORMULA EVALUATIONS ==========
    for model in formula_models_to_evaluate:
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model}")

        # Parallel evaluation with results tracking
        with ThreadPoolExecutor(max_workers=MODEL_MAX_WORKERS[model]) as executor:
            future_to_index = {
                executor.submit(LLMEvaluator.evaluate_formula, model, pair['gt_formula'], pair['parsed_formula']): i
                for i, pair in enumerate(formula_pairs_data)
            }

            # Collect and sort results
            results = [(future.result(), future_to_index[future])
                      for future in tqdm(as_completed(future_to_index),
                                       total=len(future_to_index),
                                       desc=f"Evaluating formulas with {model}")]
            results.sort(key=lambda x: x[1])

        # Apply results and save incrementally
        for (result, index) in results:
            formula_summaries[index].llm_evals.append(result)

        save_summaries(result_formula_evals_path, formula_summaries)
        save_statistics(result_stats_path, calculate_statistics(formula_summaries, table_summaries))

    # ========== LLM TABLE EVALUATIONS ==========
    for model in table_models_to_evaluate:
        if not table_pairs_data:
            break

        if model not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model}")

        # Parallel evaluation with results tracking
        with ThreadPoolExecutor(max_workers=MODEL_MAX_WORKERS[model]) as executor:
            future_to_index = {
                executor.submit(LLMEvaluator.evaluate_table, model, pair['gt_table'], pair['parsed_table']): i
                for i, pair in enumerate(table_pairs_data)
            }

            # Collect and sort results
            results = [(future.result(), future_to_index[future])
                      for future in tqdm(as_completed(future_to_index),
                                       total=len(future_to_index),
                                       desc=f"Evaluating tables with {model}")]
            results.sort(key=lambda x: x[1])

        # Apply results and save incrementally
        for (result, index) in results:
            table_summaries[index].llm_evals.append(result)

        save_summaries(result_table_evals_path, table_summaries)
        save_statistics(result_stats_path, calculate_statistics(formula_summaries, table_summaries))

    # ========== NAIVE FORMULA SIMILARITY EVALUATION ==========
    if any(summary.bleu_score is None or summary.levenshtein_similarity is None for summary in formula_summaries):
        naive_evaluator = NaiveEvaluator()
        formula_pairs = [(pair['gt_formula'], pair['parsed_formula']) for pair in formula_pairs_data]
        naive_results = naive_evaluator.evaluate_batch(formula_pairs)

        for i, (bleu_score, levenshtein_similarity) in enumerate(naive_results):
            formula_summaries[i].bleu_score = bleu_score
            formula_summaries[i].levenshtein_similarity = levenshtein_similarity

        save_summaries(result_formula_evals_path, formula_summaries)

    # ========== CDM EVALUATION ==========
    if enable_cdm:
        cdm_evaluator = CDMEvaluator(cdm_output_dir)
        formula_pairs = [(pair['gt_formula'], pair['parsed_formula']) for pair in formula_pairs_data]
        cdm_results = cdm_evaluator.evaluate_batch(formula_pairs)

        for i, cdm_result in enumerate(cdm_results):
            formula_summaries[i].cdm_eval = cdm_result

        save_summaries(result_formula_evals_path, formula_summaries)

    # ========== FINALIZE ==========
    save_statistics(result_stats_path, calculate_statistics(formula_summaries, table_summaries))
