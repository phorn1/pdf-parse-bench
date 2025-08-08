"""Content generators for synthetic PDF content."""
import json
import random
from pathlib import Path
from typing import Generator

from faker import Faker
from .style_config import StyleConfig



def generate_text_paragraphs(seed: int | None = None) -> Generator[str, None, None]:
    """Generate random text paragraphs using Faker."""
    fake = Faker()
    if seed is not None:
        fake.seed_instance(seed)
    while True:
        text_paragraph = fake.text(max_nb_chars=345).replace('\n', ' ')
        yield text_paragraph


def load_formula_generator(input_json_path: Path, seed: int | None =None) -> Generator[str, None, None]:
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


# def _calculate_available_content_width(style: StyleConfig) -> float:
#     """Calculate available content width in mm based on page layout."""
#     page_width_mm = 210  # A4 width
#     margin_left_mm = float(style.margin_left.replace('mm', ''))
#     margin_right_mm = float(style.margin_right.replace('mm', ''))
#     content_width_mm = page_width_mm - margin_left_mm - margin_right_mm
#
#     # Account for column layout
#     if style.column_count > 1:
#         column_gap_mm = float(style.column_gap.replace('mm', ''))
#         content_width_mm = (content_width_mm - (style.column_count - 1) * column_gap_mm) / style.column_count
#
#     return content_width_mm


