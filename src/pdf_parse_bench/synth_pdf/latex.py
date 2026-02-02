"""LaTeX document assembly, compilation, and validation."""

import random
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, NamedTuple

from .latex_config import LaTeXConfig
from .content import ContentBlock, ParagraphBlock, FormulaBlock, MixedTextBlock, TableBlock, PageContent


# ========== COMPILER ==========

def compile_latex(tex_path: Path, output_pdf_path: Path | None, timeout: int = 30) -> None:
    """Compile LaTeX file with optional PDF output."""
    cmd = ["pdflatex", "-halt-on-error"]
    if output_pdf_path is None:
        cmd.append("-draftmode")
    cmd.append(tex_path.name)

    result = subprocess.run(
        cmd,
        cwd=tex_path.parent,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        timeout=timeout
    )

    if result.returncode != 0:
        error_msg = f"PDFlatex failed with exit code {result.returncode}"
        if result.stderr:
            error_msg += f"\nSTDERR: {result.stderr}"
        if result.stdout:
            error_msg += f"\nSTDOUT: {result.stdout}"
        raise RuntimeError(error_msg)

    if output_pdf_path is not None:
        tex_path.with_suffix('.pdf').rename(output_pdf_path)


# ========== DOCUMENT ASSEMBLY ==========

class LaTeXDocument:
    """LaTeX document with preamble and content rendering."""

    class PageFitResult(NamedTuple):
        fits: bool
        remaining_space_pt: float | None

    def __init__(self, config: LaTeXConfig):
        self._config = config

    @property
    def documentclass_line(self) -> str:
        """Document class declaration with options."""
        options = [self._config.typography.font_size, "a4paper"]
        if self._config.two_column:
            options.append("twocolumn")
        return f"\\documentclass[{','.join(options)}]{{{self._config.document_class.value}}}"

    @property
    def packages(self) -> list[str]:
        """All required package declarations."""
        pkgs = [
            "\\usepackage[utf8]{inputenc}",
            "\\usepackage[T1]{fontenc}",
            self._config.language.babel_package,
            "\\usepackage{amsmath}",
            "\\usepackage{geometry}",
            "\\usepackage{setspace}",
            *self._config.font_family.packages,
            "\\usepackage[version=4]{mhchem}",
            "\\usepackage[table]{xcolor}",
            # Table packages
            "\\usepackage{booktabs,multirow,makecell,graphicx,array}",
            "\\usepackage{colortbl}",
            "\\usepackage{adjustbox,caption,diagbox}",
        ]

        if not self._config.font_family.conflicts_with_amsfonts:
            pkgs.extend(["\\usepackage{amsfonts}", "\\usepackage{amssymb}"])

        if self._config.two_column:
            pkgs.append("\\usepackage{multicol}")

        return pkgs

    @property
    def preamble_settings(self) -> list[str]:
        """All preamble settings (geometry, typography, font commands)."""
        cfg = self._config
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

        # Macro to output raw LaTeX register values (calculations done in Python)
        settings.extend([
            "\\makeatletter",
            "\\newcommand{\\dumpregisters}{%",
            "  \\par%",
            "  \\typeout{TEXTHEIGHT_PT:\\strip@pt\\textheight}%",
            "  \\typeout{PAGETOTAL_PT:\\strip@pt\\pagetotal}%",
            "  \\if@twocolumn",
            "    \\if@firstcolumn\\typeout{COLUMN:first}\\else\\typeout{COLUMN:second}\\fi",
            "  \\else",
            "    \\typeout{COLUMN:single}%",
            "  \\fi",
            "}",
            "\\makeatother",
        ])

        return settings

    def assemble_latex(self, page_content: PageContent) -> str:
        """Assemble complete LaTeX document from page content."""
        sections = [
            self.documentclass_line,
            "\n".join(self.packages),
            "\n".join(self.preamble_settings),
            "",
            "\\begin{document}",
            page_content.to_latex(),
            "\\dumpregisters",
            "\\end{document}",
        ]
        return "\n".join(sections)

    # ========== VALIDATION ==========

    def compile_and_measure(self, page_content: PageContent) -> PageFitResult:
        """Compile document and measure page usage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tex_file = Path(temp_dir) / "test.tex"
            tex_file.write_text(self.assemble_latex(page_content), encoding='utf-8')
            compile_latex(tex_file, output_pdf_path=tex_file.with_suffix('.pdf'), timeout=30)
            log = tex_file.with_suffix('.log').read_text(encoding='utf-8', errors='ignore')

        # Extract page count
        if not (match := re.search(r'Output written on.*?\((\d+) page', log)):
            raise RuntimeError("Could not extract page count from LaTeX log")
        if int(match.group(1)) != 1:
            return self.PageFitResult(fits=False, remaining_space_pt=None)

        # Parse registers and calculate remaining space
        textheight = float(re.search(r'TEXTHEIGHT_PT:([\d.]+)', log).group(1))
        pagetotal = float(re.search(r'PAGETOTAL_PT:([\d.]+)', log).group(1))
        column = re.search(r'COLUMN:(\w+)', log).group(1)

        remaining = textheight - pagetotal
        if column == "first":
            remaining = 2 * textheight - pagetotal
        elif column == "second":
            remaining = textheight - pagetotal

        return self.PageFitResult(fits=True, remaining_space_pt=remaining)

    def check_block_fits_bounds(self, block: ContentBlock) -> bool:
        """Check if a single content block fits within page bounds."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tex_file = Path(temp_dir) / "test.tex"
            tex_file.write_text(self.assemble_latex(PageContent(content_blocks=[block])), encoding='utf-8')
            compile_latex(tex_file, output_pdf_path=None, timeout=30)
            log = tex_file.with_suffix('.log').read_text(encoding='utf-8', errors='ignore')

        # Content exceeds box dimensions
        if "Overfull \\hbox" in log or "Overfull \\vbox" in log:
            return False
        # Severe spacing issues (badness >= 5000 indicates poor line breaking)
        if (match := re.search(r"Underfull \\hbox.*badness (\d+)", log)) and int(match.group(1)) >= 5000:
            return False
        return True

    def check_inline_formula_height(self, formula: str, max_height_pt: float = 10.0) -> bool:
        """Check if formula is flat enough for inline use."""
        test_latex = f"""{self.documentclass_line}
{"\n".join(self.packages)}
{"\n".join(self.preamble_settings)}
\\begin{{document}}
\\setbox0=\\hbox{{${formula}$}}
\\typeout{{FORMULA_HEIGHT_PT:\\the\\ht0}}
\\end{{document}}
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            tex_file = Path(temp_dir) / "test.tex"
            tex_file.write_text(test_latex, encoding='utf-8')
            compile_latex(tex_file, output_pdf_path=None, timeout=30)
            log = tex_file.with_suffix('.log').read_text(encoding='utf-8', errors='ignore')

        # Pattern: "FORMULA_HEIGHT_PT:6.94444pt"
        if match := re.search(r'FORMULA_HEIGHT_PT:([\d.]+)pt', log):
            return float(match.group(1)) <= max_height_pt
        raise RuntimeError("Could not extract formula height from LaTeX log")


# ========== PAGE BUILDING ==========

class PageBuilder:
    """Generates random content that fits within page bounds."""

    def __init__(self, latex_config: LaTeXConfig,
                 text_generator: Callable[[int], str],
                 formulas: list[str],
                 tables: dict[str, list[TableBlock]]):
        self._latex_config = latex_config
        self._document = LaTeXDocument(latex_config)
        self._text_generator = text_generator
        self._formulas = formulas
        self._tables = tables
        self._rng = random.Random(latex_config.seed)
        self._column_width_pt = self._calculate_column_width()
        self._remaining_space_pt: float | None = None

    def _calculate_column_width(self) -> float:
        """Calculate the width of a single column in pt."""
        # A4 width in pt
        a4_width_pt = 595.276

        # Parse margin values (format: "45pt", "55pt", etc.)
        def parse_pt(value: str) -> float:
            return float(value.replace("pt", ""))

        left = parse_pt(self._latex_config.margins.left)
        right = parse_pt(self._latex_config.margins.right)
        text_width = a4_width_pt - left - right

        if self._latex_config.two_column:
            columnsep = parse_pt(self._latex_config.column_sep)
            return (text_width - columnsep) / 2
        return text_width

    def assemble_latex(self, page_content: PageContent) -> str:
        """Assemble complete LaTeX document from page content."""
        return self._document.assemble_latex(page_content)

    def generate_page(self) -> PageContent:
        """Generate random page content that fills exactly one page."""
        page_content = PageContent()
        self._remaining_space_pt = None
        generators: list[Callable[[], ContentBlock | None]] = [self._generate_paragraph]
        if self._latex_config.include_formulas:
            generators.extend([self._generate_formula, self._generate_mixed_text])
        if self._latex_config.include_tables:
            generators.append(self._generate_table)

        while True:
            # Retry if _generate_table returns None (not enough space)
            block = None
            while block is None:
                block = self._rng.choice(generators)()

            page_content.content_blocks.append(block)
            result = self._document.compile_and_measure(page_content)

            if not result.fits:
                page_content.content_blocks.pop()
                break

            self._remaining_space_pt = result.remaining_space_pt

        return page_content

    def _generate_table(self) -> TableBlock | None:
        """Generate a table that fits, or None if conditions don't allow."""
        # Skip tables if less than 100pt remaining (prefer text/formulas for small gaps)
        if self._remaining_space_pt is not None and self._remaining_space_pt < 100:
            return None
        return self._select_fitting_table()

    def _select_fitting_table(self) -> TableBlock | None:
        """Select a table that fits in the available space, or None if no table fits."""
        complexity = self._rng.choice(list(self._tables.keys()))
        candidates = self._tables[complexity].copy()
        self._rng.shuffle(candidates)

        for table in candidates:
            # Skip tables that are too tall for remaining space (+ 20pt for addvspace above/below)
            if self._remaining_space_pt is not None and table.height_pt + 20 > self._remaining_space_pt:
                continue

            # Skip tables wider than 130% of column width (adjustbox can shrink to fit)
            if table.width_pt > 1.3 * self._column_width_pt:
                continue

            # Verify it compiles without overflow
            if self._document.check_block_fits_bounds(table):
                return table

        return None

    def _generate_paragraph(self) -> ParagraphBlock:
        """Generate a text paragraph with random length."""
        paragraph_length = self._rng.randint(
            self._latex_config.content.paragraph_min_chars,
            self._latex_config.content.paragraph_max_chars
        )
        content = self._text_generator(paragraph_length)
        return ParagraphBlock(text=content)

    def _generate_formula(self) -> FormulaBlock:
        """Generate a display formula that fits within bounds."""
        while True:
            formula = self._rng.choice(self._formulas)
            block = FormulaBlock(latex_formula=formula)
            if self._document.check_block_fits_bounds(block):
                return block

    def _generate_inline_formula(self) -> str:
        """Generate a formula suitable for inline use (not too tall)."""
        while True:
            formula = self._rng.choice(self._formulas)
            if self._document.check_inline_formula_height(formula):
                return formula

    def _generate_mixed_text(self) -> MixedTextBlock:
        """Generate a mixed text block with inline formulas that fits within bounds."""
        while True:
            # Generate random number of text segments based on config
            num_segments = self._rng.randint(2, self._latex_config.content.mixed_segments_max_count)

            text_segments = []
            inline_formulas = []

            for i in range(num_segments):
                # Generate text segment with variable length
                segment_length = self._rng.randint(
                    self._latex_config.content.mixed_segment_min_chars,
                    self._latex_config.content.mixed_segment_max_chars
                )
                segment_text = self._text_generator(segment_length)
                text_segments.append(segment_text)

                # Add inline formula between segments (except for the last one)
                if i < num_segments - 1:
                    inline_formulas.append(self._generate_inline_formula())

            block = MixedTextBlock(text_segments=text_segments, inline_formulas=inline_formulas)

            # Check if mixed text block fits bounds by testing compilation
            if self._document.check_block_fits_bounds(block):
                return block
            # If not, skip this block and try again
