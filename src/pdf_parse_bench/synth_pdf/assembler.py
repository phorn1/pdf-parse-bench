"""LaTeX content generation utilities."""

import random
import tempfile
import re
from typing import Callable
from pathlib import Path
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from .style_config import LaTeXConfig
from .compiler import LaTeXCompiler


# ========== DATA MODELS ==========

class ContentBlock(BaseModel, ABC):
    """Base class for all content blocks."""

    @abstractmethod
    def to_latex(self) -> str:
        """Convert this block to LaTeX format."""
        pass

    @abstractmethod
    def to_ground_truth(self) -> dict[str, str] | list[dict[str, str]]:
        """Convert this block to ground truth format."""
        pass


class ParagraphBlock(ContentBlock):
    """Text paragraph content block."""
    text: str

    def to_latex(self) -> str:
        return self.text + "\n"

    def to_ground_truth(self) -> dict[str, str]:
        return {"type": "text", "data": self.text}


class FormulaBlock(ContentBlock):
    """Mathematical formula content block."""
    latex_formula: str

    def to_latex(self) -> str:
        return f"$${self.latex_formula}$$\n"

    def to_ground_truth(self) -> dict[str, str]:
        return {"type": "display-formula", "data": f"$${self.latex_formula}$$"}


class MixedTextBlock(ContentBlock):
    """Mixed text block with inline formulas between text segments."""
    text_segments: list[str]
    inline_formulas: list[str]

    def to_latex(self) -> str:
        # First text, then alternating: formula as separator, next text
        result = self.text_segments[0]
        for formula, text in zip(self.inline_formulas, self.text_segments[1:]):
            result += f" \\mbox{{${formula}$}} " + text
        return result + "\n"

    def to_ground_truth(self) -> list[dict[str, str]]:
        # First text, then alternating: formula, text
        result = [{"type": "text", "data": self.text_segments[0]}]
        for formula, text in zip(self.inline_formulas, self.text_segments[1:]):
            result.append({"type": "inline-formula", "data": f"${formula}$"})
            result.append({"type": "text", "data": text})
        return result


class PageContent(BaseModel):
    """Content structure for a single page."""
    content_blocks: list[ContentBlock] = Field(default_factory=list)

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


# ========== DOCUMENT ASSEMBLY ==========

class LaTeXDocument:
    """LaTeX document with preamble and content rendering."""

    def __init__(self, config: LaTeXConfig):
        self.config = config

    @property
    def documentclass_line(self) -> str:
        """Document class declaration with options."""
        options = [self.config.typography.font_size, "a4paper"]
        if self.config.two_column:
            options.append("twocolumn")
        return f"\\documentclass[{','.join(options)}]{{{self.config.document_class.value}}}"

    @property
    def packages(self) -> list[str]:
        """All required package declarations."""
        pkgs = [
            "\\usepackage[utf8]{inputenc}",
            "\\usepackage[T1]{fontenc}",
            self.config.language.babel_package,
            "\\usepackage{amsmath}",
            "\\usepackage{geometry}",
            "\\usepackage{setspace}",
            *self.config.font_family.packages,
            "\\usepackage[version=4]{mhchem}",
            "\\usepackage{xcolor}",
        ]

        if not self.config.font_family.conflicts_with_amsfonts:
            pkgs.extend(["\\usepackage{amsfonts}", "\\usepackage{amssymb}"])

        if self.config.two_column:
            pkgs.append("\\usepackage{multicol}")

        return pkgs

    @property
    def preamble_settings(self) -> list[str]:
        """All preamble settings (geometry, typography, font commands)."""
        cfg = self.config
        settings = [
            f"\\geometry{{a4paper,{','.join(cfg.margins.to_latex_options())}}}",
            f"\\setlength{{\\parindent}}{{{cfg.typography.paragraph_indent}}}",
            f"\\setlength{{\\parskip}}{{{cfg.typography.paragraph_skip}}}",
        ]

        if cfg.two_column:
            settings.append(f"\\setlength{{\\columnsep}}{{{cfg.column_sep}}}")

        if cfg.typography.line_spacing.command:
            settings.append(cfg.typography.line_spacing.command)

        # Backward compatibility for old font commands
        settings.extend([
            "\\DeclareOldFontCommand{\\rm}{\\normalfont\\rmfamily}{\\mathrm}",
            "\\DeclareOldFontCommand{\\bf}{\\normalfont\\bfseries}{\\mathbf}",
            "\\DeclareOldFontCommand{\\it}{\\normalfont\\itshape}{\\mathit}",
            "\\DeclareOldFontCommand{\\tt}{\\normalfont\\ttfamily}{\\mathtt}",
            "\\DeclareOldFontCommand{\\sf}{\\normalfont\\sffamily}{\\mathsf}",
            "\\DeclareOldFontCommand{\\sc}{\\normalfont\\scshape}{\\mathsc}",
        ])

        return settings

    def assemble(self, page_content: PageContent) -> str:
        """Assemble complete LaTeX document from page content."""
        sections = [
            self.documentclass_line,
            "\n".join(self.packages),
            "\n".join(self.preamble_settings),
            "",
            "\\begin{document}",
            page_content.to_latex(),
            "\\end{document}",
        ]
        return "\n".join(sections)


# ========== VALIDATION ==========

class PageFittingValidator:
    """Validates content fits within page bounds."""

    def __init__(self, document: LaTeXDocument):
        self.document = document

    def check_fits_one_page(self, page_content: PageContent) -> bool:
        """Check if page content fits on one page."""
        latex_content = self.document.assemble(page_content)

        with tempfile.TemporaryDirectory() as temp_dir:
            tex_file = Path(temp_dir) / "test.tex"
            tex_file.write_text(latex_content, encoding='utf-8')
            LaTeXCompiler.compile_latex(tex_file, output_pdf_path=tex_file.with_suffix('.pdf'), timeout=30)

            # Pattern: "Output written on test.pdf (1 page, ...)"
            log_content = tex_file.with_suffix('.log').read_text(encoding='utf-8', errors='ignore')
            if match := re.search(r'Output written on.*?\((\d+) page', log_content):
                return int(match.group(1)) == 1

            raise RuntimeError(f"Could not extract page count from LaTeX log: {log_content}")

    def check_block_fits_bounds(self, block: ContentBlock) -> bool:
        """Check if a single content block fits within page bounds."""
        latex_content = self.document.assemble(PageContent(content_blocks=[block]))

        with tempfile.TemporaryDirectory() as temp_dir:
            tex_file = Path(temp_dir) / "test.tex"
            tex_file.write_text(latex_content, encoding='utf-8')
            LaTeXCompiler.compile_latex(tex_file, output_pdf_path=None, timeout=30)

            log_content = tex_file.with_suffix('.log').read_text(encoding='utf-8', errors='ignore')

            # Content exceeds box dimensions
            if "Overfull \\hbox" in log_content or "Overfull \\vbox" in log_content:
                return False

            # Severe spacing issues (badness >= 5000 indicates poor line breaking)
            if (match := re.search(r"Underfull \\hbox.*badness (\d+)", log_content)) \
                    and int(match.group(1)) >= 5000:
                return False

            return True

    def check_inline_formula_height(self, formula: str, max_height_pt: float = 10.0) -> bool:
        """Check if formula is flat enough for inline use."""
        test_latex = f"""{self.document.documentclass_line}
{"\n".join(self.document.packages)}
\\begin{{document}}
\\setbox0=\\hbox{{${formula}$}}
\\typeout{{FORMULA_HEIGHT_PT:\\the\\ht0}}
\\end{{document}}
"""

        with tempfile.TemporaryDirectory() as temp_dir:
            tex_file = Path(temp_dir) / "test.tex"
            tex_file.write_text(test_latex, encoding='utf-8')
            LaTeXCompiler.compile_latex(tex_file, output_pdf_path=None, timeout=30)

            # Pattern: "FORMULA_HEIGHT_PT:6.94444pt"
            log_content = tex_file.with_suffix('.log').read_text(encoding='utf-8', errors='ignore')
            if match := re.search(r'FORMULA_HEIGHT_PT:([\d.]+)pt', log_content):
                return float(match.group(1)) <= max_height_pt

            raise RuntimeError(f"Could not extract formula height from LaTeX log: {log_content}")


# ========== CONTENT GENERATION ==========

class LaTeXContentGenerator:
    """Generates random content that fits within page bounds."""

    def __init__(self, config: LaTeXConfig,
                 text_generator: Callable[[int], str],
                 formula_generator: Callable[[], str]):
        self.config = config
        self.document = LaTeXDocument(config)
        self.validator = PageFittingValidator(self.document)
        self.text_generator = text_generator
        self.formula_generator = formula_generator
        self.rng = random.Random(config.seed)

    def generate_page(self) -> PageContent:
        """Generate random page content that fills exactly one page."""
        page_content = PageContent()

        # Add blocks iteratively until page is full
        while True:
            block = self.rng.choice([
                self._generate_paragraph,
                self._generate_formula,
                self._generate_mixed_text,
            ])()

            # Test if adding this block would exceed one page
            page_content.content_blocks.append(block)
            if not self.validator.check_fits_one_page(page_content):
                # Adding this block would exceed one page, remove it
                page_content.content_blocks.pop()
                break

        return page_content
    
    def _generate_paragraph(self) -> ParagraphBlock:
        """Generate a text paragraph with random length."""
        paragraph_length = self.rng.randint(
            self.config.content.paragraph_min_chars,
            self.config.content.paragraph_max_chars
        )
        content = self.text_generator(paragraph_length)
        return ParagraphBlock(text=content)
    
    def _generate_formula(self) -> FormulaBlock:
        """Generate a display formula that fits within bounds."""
        while True:
            formula = self.formula_generator()
            block = FormulaBlock(latex_formula=formula)
            if self.validator.check_block_fits_bounds(block):
                return block

    def _get_inline_formula(self) -> str:
        """Generate a formula suitable for inline use (not too tall)."""
        while True:
            formula = self.formula_generator()
            if self.validator.check_inline_formula_height(formula):
                return formula

    def _generate_mixed_text(self) -> MixedTextBlock:
        """Generate a mixed text block with inline formulas that fits within bounds."""
        while True:
            # Generate random number of text segments based on config
            num_segments = self.rng.randint(2, self.config.content.mixed_segments_max_count)

            text_segments = []
            inline_formulas = []

            for i in range(num_segments):
                # Generate text segment with variable length
                segment_length = self.rng.randint(
                    self.config.content.mixed_segment_min_chars,
                    self.config.content.mixed_segment_max_chars
                )
                segment_text = self.text_generator(segment_length)
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
    
