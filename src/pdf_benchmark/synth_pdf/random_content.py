"""Content generators for synthetic PDF content."""
import logging
import random
from typing import Generator

from faker import Faker
import duckdb

# Suppress verbose Faker locale loading logs
logging.getLogger('faker').setLevel(logging.WARNING)
logging.getLogger('faker.factory').setLevel(logging.WARNING)



def generate_text_paragraphs(language: str = "en_US", default_max_chars: int = 345, seed: int | None = None) -> Generator[str, int, None]:
    """Generate random text paragraphs using Faker with dynamic max_chars.

    Args:
        language: Language locale (e.g., 'en_US', 'de_DE', 'fr_FR', 'es_ES', etc.)
        default_max_chars: Default maximum number of characters for text generation
        seed: Random seed for reproducible text generation

    Usage:
        gen = generate_text_paragraphs()
        text = next(gen)  # Uses default_max_chars
        text = gen.send(150)  # Uses 150 as max_chars
    """
    fake = Faker(locale=language)
    if seed is not None:
        fake.seed_instance(seed)

    max_chars = default_max_chars
    while True:
        text = fake.text(max_nb_chars=max_chars).replace('\n', ' ')
        # yield returns None when next() is called, or the sent value when send() is used
        max_chars = (yield text) or default_max_chars


def load_formulas_from_dataset() -> list[str]:
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


def load_formula_generator(seed: int | None = None, formulas: list[str] | None = None) -> Generator[str, None, None]:
    """
    Create a generator that yields random formulas.

    Args:
        seed (int | None): Random seed for reproducible formula selection
        formulas (list[str] | None): Pre-loaded formula list. If None, formulas will be downloaded.

    Yields:
        str: Individual LaTeX formulas from the dataset (randomly selected)
    """
    # Load formulas if not provided
    if formulas is None:
        formulas = load_formulas_from_dataset()

    rng = random.Random(seed)
    # Infinite generator that randomly samples formulas without copying the list
    while True:
        yield rng.choice(formulas)


            
