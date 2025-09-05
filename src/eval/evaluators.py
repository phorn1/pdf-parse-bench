import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from pathlib import Path
from typing import Literal

import Levenshtein
import requests
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

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

class TextSimilarityResult(BaseModel):
    normalized_levenshtein_similarity: float
    ground_truth_text: str
    extracted_text: str
    text_number: int


class LLMJudgeEval(BaseModel):
    type: str = "llm_judge"
    judge_model: str
    explanation: str
    is_correct: bool
    score: float
    errors: list[str] = Field(default_factory=list)


class CDMEval(BaseModel):
    type: str = "cdm"
    score: float
    visualization_path: str


class FormulaEvaluationSummary(BaseModel):
    formula_number: int
    ground_truth_formula: str
    extracted_formula: str
    formula_type: Literal['inline-formula', 'display-formula']
    llm_evals: list[LLMJudgeEval] = Field(default_factory=list)
    cdm_eval: CDMEval | None = None
    
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


class TextStatistics(BaseModel):
    total_texts: int
    average_levenshtein_similarity: float


class SummaryStatistics(BaseModel):
    formula_statistics: FormulaStatistics
    text_statistics: TextStatistics
    
    @property
    def llm_judge_stats_by_model(self) -> dict[str, LLMJudgeStatistics]:
        """Get LLM judge statistics as dict by judge model for easier access."""
        return {stat.judge_model: stat for stat in self.formula_statistics.llm_judge}


# ========== EVALUATION CLASSES ==========

class TextEvaluator:
    """Evaluates text similarity using Levenshtein distance."""
    
    def evaluate_batch(self, pairs: list[tuple[str, str]]) -> list[TextSimilarityResult]:
        results = []
        for i, (gt_text, extracted_text) in enumerate(tqdm(pairs, desc="Evaluating texts")):
            lev_distance = Levenshtein.distance(gt_text, extracted_text)
            max_len = max(len(gt_text), len(extracted_text))
            similarity = 1 - (lev_distance / max_len) if max_len > 0 else 1.0
            
            results.append(TextSimilarityResult(
                normalized_levenshtein_similarity=round(similarity, 4),
                ground_truth_text=gt_text,
                extracted_text=extracted_text,
                text_number=i
            ))
        return results


class CDMEvaluator:
    """Evaluates formulas using CDM service."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        
        # Check CDM service URL availability at initialization
        cdm_service_url = os.getenv("CDM_SERVICE_URL")
        if not cdm_service_url:
            raise ValueError("CDM_SERVICE_URL environment variable is required for CDM evaluation. "
                             "Note: CDM evaluation is an experimental feature that requires a separate local service installation. "
                             "This component is not part of the core benchmarking suite and does not work out-of-the-box.")
        self.cdm_service_url = cdm_service_url
    
    def evaluate_batch(self, pairs: list[tuple[str, str]]) -> list[CDMEval]:
        results = []
        for i, (gt_formula, extracted_formula) in enumerate(tqdm(pairs, desc="Evaluating CDM")):
            results.append(self._evaluate_single(gt_formula, extracted_formula, f"formula_{i}"))
        return results
    
    def _evaluate_single(self, gt_formula: str, extracted_formula: str, case_id: str) -> CDMEval:
        # Call CDM service to evaluate formula similarity and get visualization
        response = requests.post(self.cdm_service_url, json={
            'gt_formula': gt_formula,
            'pred_formula': extracted_formula,
            'case_id': case_id
        })
        response.raise_for_status()

        result = response.json()
        
        # Save visualization
        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_path = self.output_dir / f"cdm_visualization_{case_id}.png"
        
        with open(image_path, 'wb') as f:
            f.write(base64.b64decode(result['visualization_base64']))
        visualization_path = str(image_path.resolve())

        return CDMEval(score=result['cdm_f1'], visualization_path=visualization_path)


class LLMEvaluator:
    """Evaluates formulas using LLM judges."""
    
    def __init__(self, models: list[str]):
        self.models = models
        self.openai_client, self.gemini_client, self.mistral_client = self._init_clients()
    
    def _init_clients(self):
        openai_client = None
        gemini_client = None
        mistral_client = None
        
        openai_models = [m for m in self.models if not m.startswith("gemini-") and not m.startswith("mistral-")]
        gemini_models = [m for m in self.models if m.startswith("gemini-")]
        mistral_models = [m for m in self.models if m.startswith("mistral-")]
        
        if openai_models:
            if os.getenv("LLM_PROXY_URL"):
                openai_client = OpenAI(
                    base_url=os.getenv("LLM_PROXY_URL"),
                    api_key=os.getenv("LLM_PROXY_API_KEY")
                )
            else:
                openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        if gemini_models:
            from google import genai
            if not os.getenv("GEMINI_API_KEY"):
                raise ValueError("GEMINI_API_KEY required for Gemini models")
            gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        if mistral_models:
            try:
                from mistralai import Mistral
            except ImportError:
                raise ImportError("mistralai package required for Mistral models. Install with: pip install mistralai")
            if not os.getenv("MISTRAL_API_KEY"):
                raise ValueError("MISTRAL_API_KEY required for Mistral models")
            mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        
        return openai_client, gemini_client, mistral_client
    
    def evaluate_batch(self, gt_formulas: list[dict], extracted_formulas: list[dict]) -> list[FormulaEvaluationSummary]:
        all_results = {model: [] for model in self.models}
        
        with ThreadPoolExecutor() as executor:
            futures = {}
            
            for model in self.models:
                for i, (gt, extracted) in enumerate(zip(gt_formulas, extracted_formulas)):
                    gt_text = gt['data']
                    extracted_text = extracted['data']
                    if model.startswith("gemini-"):
                        future = executor.submit(self._evaluate_gemini, model, gt_text, extracted_text)
                    elif model.startswith("mistral-"):
                        future = executor.submit(self._evaluate_mistral, model, gt_text, extracted_text)
                    else:
                        future = executor.submit(self._evaluate_openai, model, gt_text, extracted_text)
                    futures[future] = (model, i, gt, extracted)
            
            total = len(self.models) * len(gt_formulas)
            for future in tqdm(as_completed(futures), total=total, desc="Evaluating formulas"):
                model, idx, gt, extracted = futures[future]
                result = future.result()
                all_results[model].append((result, idx))
        
        # Sort results by index
        for model in self.models:
            all_results[model].sort(key=lambda x: x[1])
        
        # Create summaries with typed llm_evals and formula_type
        summaries = []
        for i, (gt, extracted) in enumerate(zip(gt_formulas, extracted_formulas)):
            llm_evals = []
            for model in self.models:
                model_results = [r for r, idx in all_results[model] if idx == i]
                if model_results:
                    llm_evals.append(model_results[0])
            
            summaries.append(FormulaEvaluationSummary(
                formula_number=i,
                ground_truth_formula=gt['data'],
                extracted_formula=extracted['data'],
                formula_type=gt['type'],
                llm_evals=llm_evals
            ))
        
        return summaries
    
    @staticmethod
    def _retry_on_failure(max_retries: int = 10):
        def decorator(func):
            @wraps(func)
            def wrapper(self, model: str, gt_formula: str, extracted_formula: str) -> LLMJudgeEval:
                last_error = None
                for attempt in range(max_retries):
                    try:
                        return func(self, model, gt_formula, extracted_formula)
                    except Exception as e:
                        last_error = e
                        if attempt < max_retries - 1:
                            print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                
                # If we reach here, all attempts failed
                raise last_error
            return wrapper
        return decorator
    
    @_retry_on_failure()
    def _evaluate_openai(self, model: str, gt_formula: str, extracted_formula: str) -> LLMJudgeEval:
        class FormulaResponse(BaseModel):
            explanation: str
            is_correct: bool
            score: float
            errors: list[str]
        
        prompt = FORMULA_EVALUATION_PROMPT_TEMPLATE.format(gt_formula=gt_formula, extracted_formula=extracted_formula)
        response = self.openai_client.responses.parse(
            model=model, input=prompt, text_format=FormulaResponse
        )
        
        data = response.output_parsed
        return LLMJudgeEval(
            judge_model=model,
            explanation=data.explanation,
            is_correct=data.is_correct,
            score=data.score,
            errors=data.errors
        )
    
    @_retry_on_failure()
    def _evaluate_gemini(self, model: str, gt_formula: str, extracted_formula: str) -> LLMJudgeEval:
        from google.genai import types
        from pydantic import BaseModel
        
        class FormulaResponse(BaseModel):
            explanation: str
            is_correct: bool
            score: float
            errors: list[str]
        
        prompt = FORMULA_EVALUATION_PROMPT_TEMPLATE.format(gt_formula=gt_formula, extracted_formula=extracted_formula)
        response = self.gemini_client.models.generate_content(
            model=model,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FormulaResponse
            )
        )
        
        data = json.loads(response.text)
        return LLMJudgeEval(
            judge_model=model,
            explanation=data.get("explanation", ""),
            is_correct=data["is_correct"],
            score=data.get("score"),
            errors=data.get("errors", [])
        )
    
    @_retry_on_failure()
    def _evaluate_mistral(self, model: str, gt_formula: str, extracted_formula: str) -> LLMJudgeEval:
        from pydantic import BaseModel
        
        class FormulaResponse(BaseModel):
            explanation: str
            is_correct: bool
            score: float
            errors: list[str]
        
        prompt = FORMULA_EVALUATION_PROMPT_TEMPLATE.format(gt_formula=gt_formula, extracted_formula=extracted_formula)
        
        messages = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
        
        chat_response = self.mistral_client.chat.parse(
            model=model,
            messages=messages,
            response_format=FormulaResponse,
            temperature=0
        )
        
        # Access the parsed Pydantic object directly
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


class TextStatisticsCalculator:
    """Calculates text similarity statistics."""
    
    @staticmethod
    def calculate(text_results: list[TextSimilarityResult]) -> TextStatistics:
        similarities = [r.normalized_levenshtein_similarity for r in text_results]
        return TextStatistics(
            total_texts=len(text_results),
            average_levenshtein_similarity=sum(similarities) / len(similarities) if similarities else 0
        )


class StatisticsCalculator:
    """Main statistics calculator that coordinates all evaluation types."""
    
    @staticmethod
    def calculate(
        formula_summaries: list[FormulaEvaluationSummary],
        text_results: list[TextSimilarityResult]
    ) -> SummaryStatistics:
        
        llm_judge_stats = LLMStatisticsCalculator.calculate(formula_summaries)
        cdm_stats = CDMStatisticsCalculator.calculate(formula_summaries)
        text_stats = TextStatisticsCalculator.calculate(text_results)
        
        formula_stats = FormulaStatistics(
            total_formulas=len(formula_summaries),
            llm_judge=llm_judge_stats,
            cdm=cdm_stats
        )
        
        return SummaryStatistics(
            formula_statistics=formula_stats,
            text_statistics=text_stats
        )
    


# ========== MAIN EVALUATION PIPELINE ==========

def run_evaluation(
    llm_judge_models: list[str],
    enable_cdm: bool,
    gt_json_path: Path,
    parsed_json_path: Path,
    result_stats_path: Path,
    result_formula_evals_path: Path,
    result_text_evals_path: Path,
    cdm_output_dir: Path,
) -> None:
    """
    Complete evaluation pipeline.
    
    Args:
        llm_judge_models: Models for formula evaluation
        enable_cdm: Whether to enable CDM scoring
        gt_json_path: Ground truth JSON path
        parsed_json_path: Parsed results JSON path
        result_stats_path: Output path for statistics
        result_formula_evals_path: Output path for formula evaluations
        result_text_evals_path: Output path for text evaluations
        cdm_output_dir: CDM visualization output directory
    """
    # ========== LOAD DATA ==========
    with open(gt_json_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    with open(parsed_json_path, 'r', encoding='utf-8') as f:
        parsed_data = json.load(f)
    
    # Extract formulas and texts
    gt_formulas = [item for item in gt_data if item.get('type') in ['inline-formula', 'display-formula']]
    gt_texts = [item['data'] for item in gt_data if item.get('type') == 'text']
    extracted_formulas = [item for item in parsed_data if item.get('type') in ['inline-formula', 'display-formula']]
    extracted_texts = [item['data'] for item in parsed_data if item.get('type') == 'text']

    # ========== RUN EVALUATIONS ==========

    # Formula evaluation with LLMs
    llm_evaluator = LLMEvaluator(llm_judge_models)
    formula_summaries = llm_evaluator.evaluate_batch(gt_formulas, extracted_formulas)
    
    # CDM evaluation if enabled
    if enable_cdm:
        cdm_evaluator = CDMEvaluator(cdm_output_dir)
        formula_pairs = [(gt['data'], e['data']) for gt, e in zip(gt_formulas, extracted_formulas, strict=True)]
        cdm_results = cdm_evaluator.evaluate_batch(formula_pairs)
        
        # Add CDM results to summaries
        for i, cdm_result in enumerate(cdm_results):
            if i < len(formula_summaries):
                formula_summaries[i].cdm_eval = cdm_result
    
    # Text evaluation
    text_evaluator = TextEvaluator()
    text_pairs = list(zip(gt_texts, extracted_texts, strict=True))
    text_results = text_evaluator.evaluate_batch(text_pairs)

    # ========== CALCULATE STATISTICS ==========
    stats = StatisticsCalculator.calculate(formula_summaries, text_results)

    # ========== WRITE RESULTS ==========
    with open(result_stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats.model_dump(), f, indent=2, ensure_ascii=False)
    
    with open(result_formula_evals_path, 'w', encoding='utf-8') as f:
        json.dump([s.model_dump() for s in formula_summaries], f, indent=2, ensure_ascii=False)
    
    with open(result_text_evals_path, 'w', encoding='utf-8') as f:
        json.dump([r.model_dump() for r in text_results], f, indent=2, ensure_ascii=False)
