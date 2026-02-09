import base64
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

import Levenshtein
import requests
from dotenv import load_dotenv
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from openai import OpenAI
from tqdm import tqdm

from .results import LLMScore, CDMScore, FormulaResult, TableResult, save_results

load_dotenv()


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
You are a table evaluator. Your task is to determine if the extracted table correctly represents the ground truth table, focusing on content accuracy, structural preservation, and information completeness. The extracted table was parsed from the rendered table, so LaTeX-specific elements that do not affect structure or content (e.g., comments, styling commands, font formatting) should be disregarded.

Ground Truth Table (LaTeX):
{gt_table}

Extracted Table:
{extracted_table}

Evaluate the extracted table using the following criteria:
1. Content accuracy: Are all cell values, headers, and data correctly preserved?
2. Structure preservation: Are all rows and columns present, and can each value be unambiguously mapped to its row/column headers? Broken associations count as information loss.

Note: The extracted table may be in a different format (markdown, HTML, list, plain text) than the ground truth LaTeX. This is expected and acceptable. Focus on semantic equivalence.

Respond with ONLY a single integer from 0 to 10, where 10 is a perfect match. No other text.
"""


# ========== EVALUATION CLASSES ==========

class FormulaCDMEvaluator:
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

    def evaluate_batch(self, pairs: list[tuple[str, str]]) -> list[CDMScore]:
        results = []
        for i, (gt_formula, extracted_formula) in enumerate(tqdm(pairs, desc="Evaluating CDM")):
            results.append(self._evaluate_single(gt_formula, extracted_formula, i))
        return results

    def _evaluate_single(self, gt_formula: str, extracted_formula: str, index: int) -> CDMScore:
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

        return CDMScore(score=result['cdm_f1'], image_name=image_name)


class FormulaNaiveEvaluator:
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
        """Evaluate using OpenRouter."""
        client = LLMEvaluator._get_client()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        score = LLMEvaluator._parse_score(response.choices[0].message.content)
        return LLMScore(judge_model=model, score=score)

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
        """Evaluate a table pair."""
        prompt = TABLE_EVALUATION_PROMPT_TEMPLATE.format(
            gt_table=gt_table, extracted_table=extracted_table
        )
        return LLMEvaluator._evaluate(model, prompt)


# ========== BATCH EVALUATION ==========

@dataclass
class EvalPaths:
    """Paths for a single PDF's evaluation files."""
    formulas_path: Path
    tables_path: Path
    cdm_output_dir: Path


def run_batch_evaluation(
    llm_judge_models: str | list[str],
    jobs: list[EvalPaths],
    enable_cdm: bool = False,
    skip_existing: bool = True,
    max_workers: int = 8,
) -> None:
    """
    Evaluate multiple PDFs in a single batched pass.

    Batches all LLM calls across PDFs into shared thread pools for efficiency.
    Naive and CDM evaluations are run per-PDF afterwards.
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

    # ========== NAIVE FORMULA SIMILARITY (per group) ==========
    for save_path, results in formula_groups:
        if results and any(r.bleu_score is None or r.levenshtein_similarity is None for r in results):
            naive_evaluator = FormulaNaiveEvaluator()
            pairs = [(r.gt_formula, r.extracted_formula) for r in results]
            naive_results = naive_evaluator.evaluate_batch(pairs)

            for i, (bleu_score, levenshtein_similarity) in enumerate(naive_results):
                results[i].bleu_score = bleu_score
                results[i].levenshtein_similarity = levenshtein_similarity

            save_results(save_path, results)

    # ========== CDM EVALUATION (per group) ==========
    if enable_cdm:
        cdm_dirs = {job.formulas_path: job.cdm_output_dir for job in jobs}
        for save_path, results in formula_groups:
            if results and save_path in cdm_dirs:
                cdm_evaluator = FormulaCDMEvaluator(cdm_dirs[save_path])
                pairs = [(r.gt_formula, r.extracted_formula) for r in results]
                cdm_results = cdm_evaluator.evaluate_batch(pairs)

                for i, cdm_result in enumerate(cdm_results):
                    results[i].cdm_score = cdm_result

                save_results(save_path, results)