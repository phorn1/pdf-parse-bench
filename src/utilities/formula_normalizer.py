import re
import logging
from typing import Any

logger = logging.getLogger(__name__)


def normalize_gathered_display_formula(formula_data: str) -> str:
    """
    Normalize display formula by removing unmatched gathered environments
    and ensuring proper $$ delimiters.
    
    Args:
        formula_data: Raw LaTeX formula string
        
    Returns:
        Normalized LaTeX formula string with proper delimiters
    """
    # Remove unmatched gathered environments
    begin_matches = list(re.finditer(r'\\begin\{gathered\}', formula_data))
    end_matches = list(re.finditer(r'\\end\{gathered\}', formula_data))
    
    if len(begin_matches) != len(end_matches):
        formula_data = re.sub(r'\\(?:begin|end)\{gathered\}', '', formula_data)
        logger.debug(f"Removed unmatched gathered environment: {len(begin_matches)} begin + {len(end_matches)} end")
    else:
        return formula_data
    
    # Ensure proper $$ delimiters for the repaired formula
    stripped = formula_data.strip()
    
    if not stripped.startswith('$$'):
        stripped = f"$${stripped}"
    if not stripped.endswith('$$'):
        stripped = f"{stripped}$$"
    
    return stripped


def normalize_segments_formulas(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Normalize formula data in formula segments.
    
    Args:
        segments: List of segment dictionaries
        
    Returns:
        List of segments with normalized formula data
    """
    return [
        {**segment, 'data': normalize_gathered_display_formula(segment['data'])}
        if segment.get('type') == 'display-formula'
        else segment
        for segment in segments
    ]