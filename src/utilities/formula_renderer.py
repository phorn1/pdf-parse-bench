import os
import tempfile
import subprocess
from pathlib import Path
import logging
from pylatexenc.latexencode import UnicodeToLatexEncoder


# ========== EXCEPTIONS ==========

class LatexRenderError(Exception):
    """Exception raised when LaTeX rendering fails."""
    pass


# ========== FORMULA RENDERING ==========

class FormulaRenderer:
    """Handles LaTeX to PNG conversion."""
    
    LATEX_TEMPLATE = """
\\documentclass[preview, border=5pt]{{standalone}}
\\usepackage{{amsmath,amssymb,amsfonts}}
\\usepackage[version=4]{{mhchem}}
\\usepackage{{varwidth}}
\\begin{{document}}
\\begin{{varwidth}}{{25cm}}
{formula}
\\end{{varwidth}}
\\end{{document}}
"""
    
    def preprocess_unicode(self, text: str) -> str:
        """Convert Unicode mathematical symbols to LaTeX commands using pylatexenc."""
        try:
            has_dollars = '$' in text
            rules = 'unicode-xml' if has_dollars else 'defaults'
            
            encoder = UnicodeToLatexEncoder(
                conversion_rules=[rules],
                non_ascii_only=True,
                replacement_latex_protection='braces',
            )
            
            converted = encoder.unicode_to_latex(text)
            return converted
        except Exception as e:
            raise LatexRenderError(f"Unicode conversion failed: {e}")

    def render(self, latex_formula: str) -> str:
        """Convert LaTeX formula to PNG using pdflatex + magick."""
        preprocessed_formula = self.preprocess_unicode(latex_formula)
        latex_doc = self.LATEX_TEMPLATE.format(formula=preprocessed_formula)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_png:
            output_path = tmp_png.name
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tex_file = os.path.join(tmpdir, 'formula.tex')
            pdf_file = os.path.join(tmpdir, 'formula.pdf')
            
            with open(tex_file, 'w') as f:
                f.write(latex_doc)
            
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', '-output-directory', tmpdir, tex_file], 
                capture_output=True, text=True, encoding='utf-8', errors='replace'
            )
            
            if result.returncode != 0:
                logging.debug(f"LaTeX Warning: {result.stdout}\n{result.stderr}")

            if not os.path.exists(pdf_file):
                raise LatexRenderError(f"PDF not generated for formula: {latex_formula}")
            
            try:
                subprocess.run(
                    ['magick', '-density', '800', pdf_file, '-quality', '100', output_path],
                    check=True, capture_output=True
                )
            except subprocess.CalledProcessError as e:
                raise LatexRenderError(f"ImageMagick conversion failed: {e}")
            
            return output_path
    
    def create_error_image(self, error_msg: str) -> str:
        """Create a simple error image with text."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_png:
            error_png_path = tmp_png.name
        
        short_error = error_msg[:50] + "..." if len(error_msg) > 50 else error_msg

        subprocess.run([
            'magick', '-size', '600x300', 'xc:white',
            '-pointsize', '14', '-fill', 'red',
            '-gravity', 'center', '-annotate', '+0-20', 'LaTeX Render Error',
            '-pointsize', '10', '-annotate', '+0+20', short_error,
            error_png_path
        ], check=True, capture_output=True)
        return error_png_path
    
    def render_formulas_in_segments(self, segments: list[dict], rendered_formulas_dir: Path):
        """Render all formulas in segments and update them with rendered_png filenames."""

        rendered_formulas_dir.mkdir(parents=True, exist_ok=True)
        
        formula_counter = 0
        for segment in segments:
            if segment["type"] in ["inline-formula", "display-formula"]:
                try:
                    # Render formula to PNG
                    temp_png_path = self.render(segment["data"])
                    
                    # Create final filename and path
                    formula_filename = f"formula_{formula_counter:03d}.png"
                    final_png_path = rendered_formulas_dir / formula_filename
                    
                    # Move temp file to final location
                    os.rename(temp_png_path, final_png_path)
                    
                    # Add rendered_png field to segment
                    segment["rendered_png"] = formula_filename
                    
                except LatexRenderError as e:
                    logging.warning(f"Failed to render formula {formula_counter}: {e}")
                    # Create error image
                    error_png_path = self.create_error_image(str(e))
                    formula_filename = f"formula_{formula_counter:03d}_error.png"
                    final_png_path = rendered_formulas_dir / formula_filename
                    os.rename(error_png_path, final_png_path)
                    segment["rendered_png"] = formula_filename
                except Exception as e:
                    logging.warning(f"Failed to render formula {formula_counter}: {e}")
                    # Create error image
                    try:
                        error_png_path = self.create_error_image(str(e))
                        formula_filename = f"formula_{formula_counter:03d}_error.png"
                        final_png_path = rendered_formulas_dir / formula_filename
                        os.rename(error_png_path, final_png_path)
                        segment["rendered_png"] = formula_filename
                    except Exception as render_error:
                        logging.error(f"Failed to create error image for formula {formula_counter}: {render_error}")
                
                formula_counter += 1