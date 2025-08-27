"""Content generators for synthetic PDF content."""
import json
import random
from pathlib import Path
from typing import Generator

from faker import Faker



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


def load_formula_generator(input_json_path: Path, seed: int | None = None) -> Generator[str, None, None]:
    """
    Load formulas from a JSON file as a generator

    Args:
        input_json_path (Path): Path to JSON file containing formulas
        seed (int | None): Random seed for reproducible formula order

    Yields:
        str: Individual formulas from the JSON file
    """
    with input_json_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    # Flatten formulas from all URLs
    all_formulas = []
    for url, formulas_list in data.items():
        all_formulas.extend(formulas_list)

    # Use seed for reproducible shuffling
    if seed is not None:
        random.seed(seed)
    random.shuffle(all_formulas)
    
    for formula in all_formulas:
        yield formula
