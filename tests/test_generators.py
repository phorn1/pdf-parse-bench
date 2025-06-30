"""Tests for generators to verify seed reproducibility."""
import json
import tempfile
from pathlib import Path

from src.synthetic_pdf.generators import generate_text_paragraphs, load_formula_generator


def test_text_paragraphs_reproducibility():
    """Test that generate_text_paragraphs produces identical results with same seed."""
    seed = 42
    
    # Generate first batch
    gen1 = generate_text_paragraphs(seed=seed)
    batch1 = [next(gen1) for _ in range(5)]
    
    # Generate second batch with same seed
    gen2 = generate_text_paragraphs(seed=seed)
    batch2 = [next(gen2) for _ in range(5)]
    
    # Should be identical
    assert batch1 == batch2, "Text paragraphs should be identical with same seed"


def test_text_paragraphs_randomness():
    """Test that generate_text_paragraphs produces different results without seed."""
    gen1 = generate_text_paragraphs()
    batch1 = [next(gen1) for _ in range(5)]
    
    gen2 = generate_text_paragraphs()
    batch2 = [next(gen2) for _ in range(5)]
    
    # Should be different (very unlikely to be identical by chance)
    assert batch1 != batch2, "Text paragraphs should be different without seed"


def test_formula_generator_reproducibility():
    """Test that load_formula_generator produces identical results with same seed."""
    # Create temporary JSON file with test formulas
    test_data = {
        "url1": ["formula1", "formula2", "formula3"],
        "url2": ["formula4", "formula5", "formula6"]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        temp_path = Path(f.name)
    
    try:
        seed = 42
        
        # Generate first batch
        gen1 = load_formula_generator(temp_path, seed=seed)
        batch1 = [next(gen1) for _ in range(6)]
        
        # Generate second batch with same seed
        gen2 = load_formula_generator(temp_path, seed=seed)
        batch2 = [next(gen2) for _ in range(6)]
        
        # Should be identical
        assert batch1 == batch2, "Formulas should be identical with same seed"
        
    finally:
        temp_path.unlink()  # Clean up temp file


def test_formula_generator_randomness():
    """Test that load_formula_generator produces different results with different seeds."""
    # Create temporary JSON file with test formulas
    test_data = {
        "url1": ["formula1", "formula2", "formula3"],
        "url2": ["formula4", "formula5", "formula6"]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        temp_path = Path(f.name)
    
    try:
        # Generate with different seeds
        gen1 = load_formula_generator(temp_path, seed=42)
        batch1 = [next(gen1) for _ in range(6)]
        
        gen2 = load_formula_generator(temp_path, seed=123)
        batch2 = [next(gen2) for _ in range(6)]
        
        # Should be different (same formulas but different order)
        assert batch1 != batch2, "Formulas should be in different order with different seeds"
        
        # But should contain same elements
        assert set(batch1) == set(batch2), "Should contain same formulas"
        
    finally:
        temp_path.unlink()  # Clean up temp file