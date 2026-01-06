"""LaTeX compilation utilities."""

import subprocess
from pathlib import Path


class LaTeXCompiler:
    """Handles LaTeX compilation with consistent configuration."""

    @staticmethod
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