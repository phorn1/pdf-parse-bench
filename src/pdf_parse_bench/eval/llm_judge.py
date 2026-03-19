import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Literal, TypeVar

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

T = TypeVar("T", bound=BaseModel)

load_dotenv()


# ========== DATA MODELS ==========

class LLMScore(BaseModel):
    judge_model: str
    score: int
    errors: list[str] = []


class FormulaResult(BaseModel):
    index: int
    gt_formula: str
    extracted_formula: str
    formula_type: Literal['inline-formula', 'display-formula']
    llm_scores: list[LLMScore] = []

    @property
    def llm_scores_by_model(self) -> dict[str, LLMScore]:
        """Get LLM scores as dict by judge model for easier UI access."""
        return {s.judge_model: s for s in self.llm_scores}


class TableResult(BaseModel):
    index: int
    gt_table: str
    extracted_table: str
    complexity: Literal['simple', 'moderate', 'complex']
    llm_scores: list[LLMScore] = []

    @property
    def llm_scores_by_model(self) -> dict[str, LLMScore]:
        """Get LLM scores as dict by judge model for easier UI access."""
        return {s.judge_model: s for s in self.llm_scores}


# ========== FILE I/O ==========

def save_results(file_path: Path, results: list[BaseModel]) -> None:
    """Save results to JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump([r.model_dump() for r in results], f, indent=2, ensure_ascii=False)


# ========== CONSTANTS ==========

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
You are a strict table evaluator. Your task is to determine if the extracted table correctly represents the ground truth table, focusing on content accuracy, structural preservation, and information completeness. The extracted table was parsed from the rendered table. Disregard LaTeX-specific elements in the ground truth (e.g., comments, styling commands, font formatting) that have no effect on content or structure.

Ground Truth Table (LaTeX):
{gt_table}

Extracted Table:
{extracted_table}

Evaluate the extracted table using the following criteria:
1. Content accuracy: Are all cell values, headers, and data correctly preserved?
2. Structure preservation: Are all rows and columns present, and can each value be unambiguously mapped to its row/column headers? Broken or ambiguous associations count as errors.

Note: Different output formats (markdown, HTML, plain text) are acceptable as long as no information is lost. Apply this key test: Could a reader who sees ONLY the extracted table — without access to the ground truth — unambiguously reconstruct every cell-to-header mapping and all content of the original table? If not, consider the parsing as failed and assign a low score.

First, enumerate all errors and ambiguities found. Then assign a score from 0 to 10, where 10 is a perfect match.
"""


class TableEvaluation(BaseModel):
    errors: list[str]
    score: int


# ========== EVALUATION CLASSES ==========

class LLMEvaluator:
    """Evaluates formulas and tables using LLM judges via OpenRouter."""

    _client = None

    @classmethod
    def _get_client(cls) -> OpenAI:
        """Get or create the shared OpenRouter client."""
        if cls._client is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable is required for LLM evaluation.")
            cls._client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        return cls._client

    @staticmethod
    def _retry_on_failure(max_retries: int = 10):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs) -> LLMScore:
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
        match = re.search(r'\b(\d+)\b', text.strip())
        if match:
            score = int(match.group(1))
            return max(0, min(10, score))
        raise ValueError(f"Could not parse score from response: {text}")

    @staticmethod
    def _evaluate(model: str, prompt: str) -> LLMScore:
        """Evaluate using OpenRouter (unstructured, returns score only)."""
        client = LLMEvaluator._get_client()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        score = LLMEvaluator._parse_score(response.choices[0].message.content)
        return LLMScore(judge_model=model, score=score)

    @staticmethod
    def _evaluate_structured(model: str, prompt: str, response_model: type[T]) -> T:
        """Evaluate using OpenRouter with structured JSON output."""
        client = LLMEvaluator._get_client()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": response_model.model_json_schema(),
                },
            },
        )
        return response_model.model_validate_json(response.choices[0].message.content)

    @staticmethod
    @_retry_on_failure()
    def evaluate_formula(model: str, gt_formula: str, extracted_formula: str) -> LLMScore:
        """Evaluate a formula pair."""
        prompt = FORMULA_EVALUATION_PROMPT_TEMPLATE.format(
            gt_formula=gt_formula, extracted_formula=extracted_formula
        )
        return LLMEvaluator._evaluate(model, prompt)

    @staticmethod
    @_retry_on_failure()
    def evaluate_table(model: str, gt_table: str, extracted_table: str) -> LLMScore:
        """Evaluate a table pair using structured output with chain-of-thought."""
        prompt = TABLE_EVALUATION_PROMPT_TEMPLATE.format(
            gt_table=gt_table, extracted_table=extracted_table
        )
        result = LLMEvaluator._evaluate_structured(model, prompt, TableEvaluation)
        return LLMScore(judge_model=model, score=max(0, min(10, result.score)), errors=result.errors)


# ========== BATCH EVALUATION ==========

@dataclass
class EvalPaths:
    """Paths for a single PDF's evaluation files."""
    formulas_path: Path
    tables_path: Path


def run_batch_evaluation(
    llm_judge_models: str | list[str],
    jobs: list[EvalPaths],
    skip_existing: bool = True,
    max_workers: int = 8,
) -> None:
    """
    Evaluate multiple PDFs in a single batched pass.

    Batches all LLM calls across PDFs into shared thread pools for efficiency.
    """
    if isinstance(llm_judge_models, str):
        llm_judge_models = [llm_judge_models]

    # Load all results, grouped by source file
    formula_groups: list[tuple[Path, list[FormulaResult]]] = []
    table_groups: list[tuple[Path, list[TableResult]]] = []

    for job in jobs:
        if job.formulas_path.exists():
            with open(job.formulas_path, 'r', encoding='utf-8') as f:
                formula_groups.append((job.formulas_path, [FormulaResult(**item) for item in json.load(f)]))
        if job.tables_path.exists():
            with open(job.tables_path, 'r', encoding='utf-8') as f:
                table_groups.append((job.tables_path, [TableResult(**item) for item in json.load(f)]))

    # Remove old scores if reprocessing
    if not skip_existing:
        models_to_reprocess = set(llm_judge_models)
        for _, results in formula_groups:
            for r in results:
                r.llm_scores = [s for s in r.llm_scores if s.judge_model not in models_to_reprocess]
        for _, results in table_groups:
            for r in results:
                r.llm_scores = [s for s in r.llm_scores if s.judge_model not in models_to_reprocess]

    # ========== BATCHED LLM EVALUATION ==========
    llm_eval_configs = [
        (formula_groups, lambda m, r: LLMEvaluator.evaluate_formula(m, r.gt_formula, r.extracted_formula), "formulas"),
        (table_groups, lambda m, r: LLMEvaluator.evaluate_table(m, r.gt_table, r.extracted_table), "tables"),
    ]

    for groups, eval_fn, type_name in llm_eval_configs:
        for model in llm_judge_models:
            # Collect all (group_idx, result_idx) pairs that need evaluation
            tasks = [
                (gi, ri)
                for gi, (_, results) in enumerate(groups)
                for ri, r in enumerate(results)
                if not any(s.judge_model == model for s in r.llm_scores)
            ]
            if not tasks:
                continue

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {
                    executor.submit(eval_fn, model, groups[gi][1][ri]): (gi, ri)
                    for gi, ri in tasks
                }
                for future in tqdm(as_completed(future_to_task), total=len(future_to_task),
                                   desc=f"Evaluating {type_name} with {model}"):
                    gi, ri = future_to_task[future]
                    groups[gi][1][ri].llm_scores.append(future.result())

            # Save all affected groups after each model
            for save_path, results in groups:
                save_results(save_path, results)
