"""Content generators for synthetic PDF content."""
import logging
import random
from typing import Callable

from faker import Faker
import duckdb

# Suppress verbose Faker locale loading logs
logging.getLogger('faker').setLevel(logging.WARNING)
logging.getLogger('faker.factory').setLevel(logging.WARNING)



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


def create_formula_generator(seed: int | None = None, formulas: list[str] | None = None) -> Callable[[], str]:
    """
    Create a formula generator function.

    Args:
        seed: Random seed for reproducible formula selection
        formulas: Pre-loaded formula list. If None, formulas will be downloaded.

    Returns:
        A function that returns a random formula on each call.

    Usage:
        get_formula = create_formula_generator()
        formula = get_formula()  # Get random formula
    """
    # Load formulas if not provided
    if formulas is None:
        formulas = load_formulas_from_dataset()

    rng = random.Random(seed)

    def generate() -> str:
        return rng.choice(formulas)

    return generate


            
