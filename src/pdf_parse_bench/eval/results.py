import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


# ========== DATA MODELS ==========

class LLMScore(BaseModel):
    judge_model: str
    score: int


class CDMScore(BaseModel):
    score: float
    image_name: str | None = None


class FormulaResult(BaseModel):
    index: int
    gt_formula: str
    extracted_formula: str
    formula_type: Literal['inline-formula', 'display-formula']
    llm_scores: list[LLMScore] = []
    cdm_score: CDMScore | None = None
    bleu_score: float | None = None
    levenshtein_similarity: float | None = None

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