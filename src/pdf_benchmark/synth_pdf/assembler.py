"""LaTeX content generation utilities."""

import random
import tempfile
import re
from typing import Generator
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from .style_config import LaTeXConfig
from .compiler import LaTeXCompiler


# ========== CONTENT BLOCK TYPES ==========

@dataclass
class ContentBlock(ABC):
    """Base class for all content blocks."""
    
    @staticmethod
    def _fix_formula_for_latex(formula: str) -> str:
        """Fix formula for LaTeX compilation (e.g., Spanish babel conflicts)."""
        return formula.replace("\\%", "\\text{\\%}")
    
    @abstractmethod
    def to_latex(self) -> str:
        """Convert this block to LaTeX format."""
        pass
    
    @abstractmethod
    def to_ground_truth(self) -> dict[str, str] | list[dict[str, str]]:
        """Convert this block to ground truth format."""
        pass


@dataclass
class ParagraphBlock(ContentBlock):
    """Text paragraph content block."""
    text: str
    
    def to_latex(self) -> str:
        return self.text + "\n"
    
    def to_ground_truth(self) -> dict[str, str]:
        return {"type": "text", "data": self.text}


@dataclass
class FormulaBlock(ContentBlock):
    """Mathematical formula content block."""
    latex_formula: str
    
    def to_latex(self) -> str:
        formula = self._fix_formula_for_latex(self.latex_formula)
        return f"$${formula}$$\n"

    def to_ground_truth(self) -> dict[str, str]:
        return {"type": "display-formula", "data": f"$${self.latex_formula}$$"}


@dataclass
class MixedTextBlock(ContentBlock):
    """Mixed text block with inline formulas between text segments."""
    text_segments: list[str]
    inline_formulas: list[str]
    
    def to_latex(self) -> str:
        result = []
        
        # Interleave text segments and inline formulas
        for i, text in enumerate(self.text_segments):
            result.append(text)
            # Add inline formula after each text segment except the last
            if i < len(self.inline_formulas):
                formula = self._fix_formula_for_latex(self.inline_formulas[i])
                result.append(f" \\mbox{{${formula}$}} ")
        
        return "".join(result) + "\n"
    
    def to_ground_truth(self) -> list[dict[str, str]]:
        ground_truth_entries = []
        
        for i, text in enumerate(self.text_segments):
            ground_truth_entries.append({"type": "text", "data": text})
            if i < len(self.inline_formulas):
                ground_truth_entries.append({
                    "type": "inline-formula", 
                    "data": f"${self.inline_formulas[i]}$"
                })
        
        return ground_truth_entries


# ========== PAGE CONTENT ==========

@dataclass
class PageContent:
    """Content structure for a single page."""
    content_blocks: list[ContentBlock] = field(default_factory=list)
    
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


class LaTeXDocumentTemplate:
    """Handles LaTeX document template generation (preamble, packages, etc.)."""
    
    def __init__(self, config: LaTeXConfig):
        self.config = config
    
    def build_document_template(self, page_content: PageContent) -> str:
        """Build complete LaTeX document with given page content."""
        latex_code = ""

        # Document class and options
        latex_code += self._build_documentclass() + "\n"
        
        # Packages
        latex_code += self._build_packages_section() + "\n"
        
        # Geometry settings  
        latex_code += self._build_geometry() + "\n"
        
        # Typography settings
        latex_code += self._build_typography_section() + "\n"
        
        # Line spacing commands (must come in preamble)
        # Handle memoir class differently as it has its own spacing commands
        if self.config.document_class.value == "memoir":
            if self.config.typography.line_spacing == "onehalf":
                latex_code += "\\OnehalfSpacing\n"
            elif self.config.typography.line_spacing == "double":
                latex_code += "\\DoubleSpacing\n"
        else:
            if self.config.typography.line_spacing == "onehalf":
                latex_code += "\\onehalfspacing\n"
            elif self.config.typography.line_spacing == "double":
                latex_code += "\\doublespacing\n"
        
        # Header/footer setup
        if self.config.use_fancy_headers:
            latex_code += self._build_fancy_headers_section() + "\n"
        
        # Document begin
        latex_code += "\n\\begin{document}\n"
        
        # Document content
        latex_code += page_content.to_latex()
        
        # Document end
        latex_code += "\\end{document}"
        
        return latex_code
    
    def _build_documentclass(self) -> str:
        """Build document class line."""
        options = []
        
        # Font size
        options.append(self.config.typography.font_size)
        
        # Paper size
        options.append(self.config.page_geometry.value)
        
        # Two column
        if self.config.two_column:
            options.append("twocolumn")
        
        options_str = ",".join(options)
        return f"\\documentclass[{options_str}]{{{self.config.document_class.value}}}"
    
    def _build_packages_section(self) -> str:
        """Build required packages section."""
        packages = [
            "\\usepackage[utf8]{inputenc}",
            "\\usepackage[T1]{fontenc}",
        ]
        
        # Handle Spanish babel with proper options to avoid active character conflicts
        if self.config.language.babel_name == "spanish":
            packages.append("\\usepackage[spanish,es-noshorthands]{babel}")
        else:
            packages.append(f"\\usepackage[{self.config.language.babel_name}]{{babel}}")
        
        packages.extend([
            "\\usepackage{amsmath}",
            "\\usepackage{geometry}"
        ])
        
        # Don't load setspace with memoir class - it has its own spacing commands
        if self.config.document_class.value != "memoir":
            packages.append("\\usepackage{setspace}")
        
        # Font packages
        font_packages = self.config.font_family.packages
        packages.extend(font_packages)
        
        # Add amsfonts and amssymb only if not using mathdesign (which conflicts with these)
        uses_mathdesign = any("mathdesign" in pkg for pkg in font_packages)
        if not uses_mathdesign:
            packages.extend(["\\usepackage{amsfonts}", "\\usepackage{amssymb}"])
        
        if self.config.use_fancy_headers:
            packages.append("\\usepackage{fancyhdr}")

        packages.append("\\usepackage[version=4]{mhchem}")
        packages.append("\\usepackage{xcolor}")

        if self.config.two_column:
            packages.append("\\usepackage{multicol}")
        
        # Add backward compatibility for old font commands
        packages.append("\\DeclareOldFontCommand{\\rm}{\\normalfont\\rmfamily}{\\mathrm}")
        packages.append("\\DeclareOldFontCommand{\\bf}{\\normalfont\\bfseries}{\\mathbf}")
        packages.append("\\DeclareOldFontCommand{\\it}{\\normalfont\\itshape}{\\mathit}")
        packages.append("\\DeclareOldFontCommand{\\tt}{\\normalfont\\ttfamily}{\\mathtt}")
        packages.append("\\DeclareOldFontCommand{\\sf}{\\normalfont\\sffamily}{\\mathsf}")
        packages.append("\\DeclareOldFontCommand{\\sc}{\\normalfont\\scshape}{\\mathsc}")

        return "\n".join(packages)
    
    def _build_geometry(self) -> str:
        """Build geometry package configuration."""
        options = [self.config.page_geometry.value] + self.config.margins.to_latex_options()
        return f"\\geometry{{{','.join(options)}}}"
    
    def _build_typography_section(self) -> str:
        """Build typography settings section."""
        settings = []
        
        # Paragraph settings
        settings.append(f"\\setlength{{\\parindent}}{{{self.config.typography.paragraph_indent}}}")
        settings.append(f"\\setlength{{\\parskip}}{{{self.config.typography.paragraph_skip}}}")
        
        # Column separation for two-column layout
        if self.config.two_column:
            settings.append(f"\\setlength{{\\columnsep}}{{{self.config.column_sep}}}")
        
        return "\n".join(settings)
    
    def _build_fancy_headers_section(self) -> str:
        """Build fancy header configuration section without page numbers."""
        headers = [
            "\\pagestyle{fancy}",
            "\\fancyhf{}",
            "\\fancyhead[L]{\\leftmark}",
            "\\renewcommand{\\headrulewidth}{0.4pt}"
        ]
        return "\n".join(headers)


# ========== PAGE FITTING VALIDATOR ==========

class PageFittingValidator:
    """Handles page fitting validation."""
    
    def __init__(self, template: LaTeXDocumentTemplate):
        self.template = template
    
    def check_fits_one_page(self, page_content: PageContent) -> bool:
        """Check if page content fits on one page by compiling LaTeX."""
        latex_content = self.template.build_document_template(page_content)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            tex_file = temp_path / "test.tex"

            tex_file.write_text(latex_content, encoding='utf-8')
            
            temp_pdf = temp_path / "test.pdf"
            LaTeXCompiler.compile_latex(tex_file, output_pdf_path=temp_pdf, timeout=30)

            log_file = temp_path / "test.log"
            page_count = LaTeXCompiler.get_page_count_from_log(log_file)
            return page_count == 1
    
    def check_block_fits_bounds(self, block: ContentBlock) -> bool:
        """Check if a single content block fits within page bounds."""
        single_block_content = PageContent(content_blocks=[block])
        latex_content = self.template.build_document_template(single_block_content)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            tex_file = temp_path / "test_block.tex"

            tex_file.write_text(latex_content, encoding='utf-8')

            LaTeXCompiler.compile_latex(tex_file, output_pdf_path=None, timeout=30)
            log_path = tex_file.with_suffix('.log')
            return LaTeXCompiler.check_bounds_from_log(log_path)

    def check_inline_formula_height(self, formula: str, max_height_pt: float = 10.0) -> bool:
        """Check if formula is flat enough for inline use.

        Args:
            formula: LaTeX formula string (without $ delimiters)
            max_height_pt: Maximum allowed height in points

        Returns:
            True if formula height is within bounds, False otherwise
        """
        # Create minimal test document that measures formula height
        test_latex = r"""
\documentclass{article}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage[version=4]{mhchem}
\begin{document}
\setbox0=\hbox{$""" + formula + r"""$}
\typeout{FORMULA_HEIGHT_PT:\the\ht0}
\end{document}
"""

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                tex_file = temp_path / "height_test.tex"

                tex_file.write_text(test_latex, encoding='utf-8')

                LaTeXCompiler.compile_latex(tex_file, output_pdf_path=None)

                # Extract height from log file
                log_path = tex_file.with_suffix('.log')
                log_content = log_path.read_text(encoding='utf-8', errors='ignore')

                # Search for our typeout message
                height_match = re.search(r'FORMULA_HEIGHT_PT:([\d.]+)pt', log_content)

                if height_match:
                    height_pt = float(height_match.group(1))
                    return height_pt <= max_height_pt
                else:
                    raise Exception
        except Exception:
            print("wtf")
            raise Exception


# ========== CONTENT GENERATOR ==========

class LaTeXContentGenerator:
    """Handles content generation using page fitting validation."""
    
    def __init__(self, config: LaTeXConfig,
                 text_generator: Generator[str, int, None],
                 formula_generator: Generator[str, None, None]):
        self.config = config
        self.template = LaTeXDocumentTemplate(config)
        self.validator = PageFittingValidator(self.template)
        self.text_generator = text_generator
        self.formula_generator = formula_generator
        
        # Prime the text generator so we can use send() later
        next(self.text_generator)
    
    def generate_page_content(self) -> PageContent:
        """Generate page content that fills exactly one page."""

        page_content = PageContent()

        # Add blocks iteratively until page is full
        while True:
            # Choose block type first, then generate (lazy evaluation)
            block_generator = random.choice([
                self._generate_paragraph,
                self._generate_formula,
                self._generate_mixed_text,
            ])
            block = block_generator()

            # Test if adding this block would exceed one page
            page_content.content_blocks.append(block)
            if not self.validator.check_fits_one_page(page_content):
                # Adding this block would exceed one page, remove it
                page_content.content_blocks.pop()
                break

        return page_content
    
    def _generate_paragraph(self) -> ParagraphBlock:
        """Generate a text paragraph with random length."""
        paragraph_length = random.randint(
            self.config.content.paragraph_min_chars,
            self.config.content.paragraph_max_chars
        )
        content = self.text_generator.send(paragraph_length)
        return ParagraphBlock(text=content)
    
    def _generate_formula(self) -> FormulaBlock:
        """Generate a mathematical formula that fits within bounds."""
        while True:
            formula = next(self.formula_generator)
            block = FormulaBlock(latex_formula=formula)

            # Check if formula fits bounds by testing compilation
            if self.validator.check_block_fits_bounds(block):
                return block
            # If not, skip this formula and try the next one

    def _get_inline_formula(self) -> str:
        """Get a formula suitable for inline use (not too tall).

        Returns:
            Formula string that is flat enough for inline use
        """
        while True:
            formula = next(self.formula_generator)

            # Check if formula height is suitable for inline use
            if self.validator.check_inline_formula_height(formula):
                return formula
            # If not, skip this formula and try the next one

    def _generate_mixed_text(self) -> MixedTextBlock:
        """Generate a mixed text block with inline formulas that fits within bounds."""
        while True:
            # Generate random number of text segments based on config
            num_segments = random.randint(2, self.config.content.mixed_segments_max_count)

            text_segments = []
            inline_formulas = []

            for i in range(num_segments):
                # Generate text segment with variable length
                segment_length = random.randint(
                    self.config.content.mixed_segment_min_chars,
                    self.config.content.mixed_segment_max_chars
                )

                # Use generator.send() to specify exact length for this segment
                segment_text = self.text_generator.send(segment_length)
                text_segments.append(segment_text)

                # Add inline formula between segments (except for the last one)
                if i < num_segments - 1:
                    # Use height-validated formula for inline use
                    inline_formulas.append(self._get_inline_formula())

            block = MixedTextBlock(text_segments=text_segments, inline_formulas=inline_formulas)

            # Check if mixed text block fits bounds by testing compilation
            if self.validator.check_block_fits_bounds(block):
                return block
            # If not, skip this block and try again
    
