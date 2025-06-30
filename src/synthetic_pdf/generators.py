"""Content generators for synthetic PDF content."""
import json
import random
from pathlib import Path

from faker import Faker


def generate_text_paragraphs(seed=None):
    """Generate random text paragraphs using Faker."""
    fake = Faker()
    if seed is not None:
        fake.seed_instance(seed)
    while True:
        text_paragraph = fake.text(max_nb_chars=345).replace('\n', ' ')
        yield text_paragraph


def load_formula_generator(input_json_path: Path, seed=None):
    """
    Load formulas from a JSON file as a generator

    Args:
        input_json_path (Path): Path to JSON file containing formulas
        seed (int): Random seed for reproducible results

    Yields:
        str: Individual formulas from the JSON file
    """
    with input_json_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    # Flatten formulas from all URLs
    all_formulas = []
    for url, formulas_list in data.items():
        all_formulas.extend(formulas_list)

    random.seed(seed)
    random.shuffle(all_formulas)
    
    for formula in all_formulas:
        yield formula