"""Content models and random content generators."""

import logging
from abc import ABC, abstractmethod
from typing import Callable

import duckdb
from faker import Faker
from pydantic import BaseModel, Field

# Suppress verbose Faker locale loading logs
logging.getLogger('faker').setLevel(logging.WARNING)
logging.getLogger('faker.factory').setLevel(logging.WARNING)


# ========== CONTENT BLOCKS ==========

class ContentBlock(BaseModel, ABC):
    """Base class for all content blocks."""

    @abstractmethod
    def to_latex(self) -> str:
        """Convert this block to LaTeX format."""
        pass

    @abstractmethod
    def to_ground_truth(self) -> dict[str, str] | list[dict[str, str]]:
        """Convert this block to ground truth format."""
        pass


class ParagraphBlock(ContentBlock):
    """Text paragraph content block."""
    text: str

    def to_latex(self) -> str:
        return self.text + "\n"

    def to_ground_truth(self) -> dict[str, str]:
        return {"type": "text", "data": self.text}


class FormulaBlock(ContentBlock):
    """Mathematical formula content block."""
    latex_formula: str

    def to_latex(self) -> str:
        return f"$${self.latex_formula}$$\n"

    def to_ground_truth(self) -> dict[str, str]:
        return {"type": "display-formula", "data": f"$${self.latex_formula}$$"}


class MixedTextBlock(ContentBlock):
    """Mixed text block with inline formulas between text segments."""
    text_segments: list[str]
    inline_formulas: list[str]

    def to_latex(self) -> str:
        # First text, then alternating: formula as separator, next text
        result = self.text_segments[0]
        for formula, text in zip(self.inline_formulas, self.text_segments[1:]):
            result += f" \\mbox{{${formula}$}} " + text
        return result + "\n"

    def to_ground_truth(self) -> list[dict[str, str]]:
        # First text, then alternating: formula, text
        result = [{"type": "text", "data": self.text_segments[0]}]
        for formula, text in zip(self.inline_formulas, self.text_segments[1:]):
            result.append({"type": "inline-formula", "data": f"${formula}$"})
            result.append({"type": "text", "data": text})
        return result


class TableBlock(ContentBlock):
    """Table content block with pre-computed dimensions."""
    latex_table: str
    width_pt: float
    height_pt: float
    complexity: str

    def to_latex(self) -> str:
        return f"\\addvspace{{10pt}}\n{{\\centering\n\\adjustbox{{max width=\\columnwidth}}{{{self.latex_table}}}\n\\par}}\n\\addvspace{{10pt}}\n"

    def to_ground_truth(self) -> dict[str, str]:
        return {"type": "table", "data": self.latex_table, "complexity": self.complexity}


class PageContent(BaseModel):
    """Content structure for a single page."""
    content_blocks: list[ContentBlock] = Field(default_factory=list)


    def to_latex(self) -> str:
        """Convert all content blocks to LaTeX format."""
        return "\n".join(block.to_latex() for block in self.content_blocks)

    def to_ground_truth(self) -> list[dict[str, str]]:
        """Convert all content blocks to flattened ground truth format."""
        gt_data = []
        for block in self.content_blocks:
            block_gt = block.to_ground_truth()
            if isinstance(block_gt, list):
                # MixedTextBlock returns a list - extend to flatten
                gt_data.extend(block_gt)
            else:
                # Other blocks return single dict - append
                gt_data.append(block_gt)
        return gt_data


# ========== RANDOM CONTENT GENERATORS ==========

def create_text_generator(language: str = "en_US", seed: int | None = None) -> Callable[[int], str]:
    """Create a text generator function using Faker.

    Args:
        language: Language locale (e.g., 'en_US', 'de_DE', 'fr_FR', 'es_ES', etc.)
        seed: Random seed for reproducible text generation

    Returns:
        A function that generates text of specified max length.

    Usage:
        generate = create_text_generator()
        text = generate(150)  # Generate text up to 150 chars
    """
    fake = Faker(locale=language)
    if seed is not None:
        fake.seed_instance(seed)

    def generate(max_chars: int) -> str:
        return fake.text(max_nb_chars=max_chars).replace('\n', ' ')

    return generate


def load_formulas() -> list[str]:
    """
    Load formulas from Hugging Face dataset.
    Uses DuckDB with HTTP range requests to efficiently fetch only the 'formula' column
    without downloading the entire 751MB dataset (only ~35MB of text data is transferred).

    Returns:
        list[str]: List of all LaTeX formulas from the dataset
    """
    # Parquet file URL for the dataset
    parquet_url = (
        "https://huggingface.co/datasets/piushorn/wikipedia-latex-formulas-319k/"
        "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
    )

    # Use DuckDB to fetch only the 'formula' column via HTTP range requests
    # This leverages Parquet's columnar format to download only needed data (~35MB vs 751MB)
    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false")
    result = con.execute(f"SELECT formula FROM read_parquet('{parquet_url}')").fetchall()
    con.close()

    # Extract formulas from query result
    return [formula for (formula,) in result]


def load_tables() -> dict[str, list[TableBlock]]:
    """Load all LaTeX tables from the tables jsonl files, grouped by complexity."""
    import json
    from pathlib import Path
    base_path = Path(__file__).parents[3]
    tables_files = {
        "simple": base_path / "tables_simple.jsonl",
        "moderate": base_path / "tables_moderate.jsonl",
        "complex": base_path / "tables_complex.jsonl",
    }
    tables = {"simple": [], "moderate": [], "complex": []}
    for complexity, tables_file in tables_files.items():
        with open(tables_file, encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                tables[complexity].append(TableBlock(
                    latex_table=record["tabular"],
                    width_pt=record["width_pt"],
                    height_pt=record["height_pt"],
                    complexity=complexity,
                ))
    return tables