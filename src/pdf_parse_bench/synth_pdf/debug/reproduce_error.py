#!/usr/bin/env python3
"""Script to reproduce LaTeX generation errors from debug files."""

import json
import sys
import tempfile
from pathlib import Path

from pdf_parse_bench.synth_pdf import LaTeXConfig, LaTeXSinglePagePDFGenerator


def reproduce_error_from_debug_file(debug_file: Path) -> None:
    """Reproduce an error from a saved debug configuration file."""
    
    if not debug_file.exists():
        print(f"Debug file not found: {debug_file}")
        sys.exit(1)
    
    # Load error configuration
    with open(debug_file, 'r', encoding='utf-8') as f:
        error_info = json.load(f)
    
    config_data = error_info['config']
    seed = config_data.get('seed')
    
    print(f"Reproducing error with seed: {seed}")
    print(f"Original error: {error_info['error_type']}: {error_info['error_message']}")
    print("=" * 60)
    
    # Recreate the configuration
    config = LaTeXConfig.random(seed=seed)
    
    # Path to formulas file (adjust if needed)
    formulas_file = Path("data/formulas.json")
    if not formulas_file.exists():
        print(f"Formulas file not found: {formulas_file}")
        sys.exit(1)
    
    # Create temporary output files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        latex_path = temp_path / "test.tex"
        pdf_path = temp_path / "test.pdf" 
        gt_path = temp_path / "gt.json"
        
        # Try to reproduce the error
        try:
            generator = LaTeXSinglePagePDFGenerator(config)
            generator.generate_single_page_pdf(latex_path, pdf_path, gt_path)
            print("✓ Generation succeeded - error might be fixed!")
            
        except Exception as e:
            print(f"✗ Error reproduced: {type(e).__name__}: {e}")
            print("\nYou can now debug this specific case.")
            
            # Save the generated LaTeX for inspection
            if latex_path.exists():
                debug_latex = Path("debug") / f"reproduced_seed_{seed}.tex"
                debug_latex.parent.mkdir(exist_ok=True)
                latex_path.rename(debug_latex)
                print(f"LaTeX file saved to: {debug_latex}")


def main():
    """Main entry point."""
    debug_file = Path("debug") / "failed_config_seed_-333568025366769423.json"
    reproduce_error_from_debug_file(debug_file)


if __name__ == "__main__":
    main()