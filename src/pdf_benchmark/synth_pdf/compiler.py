"""LaTeX compilation utilities."""

import subprocess
import re
from pathlib import Path


class LaTeXCompiler:
    """Handles LaTeX compilation with consistent configuration."""
    
    @staticmethod
    def compile_latex(tex_path: Path, output_pdf_path: Path | None, timeout: int = 30) -> None:
        """Compile LaTeX file with optional PDF output.
        
        Args:
            tex_path: Path to the .tex file to compile
            output_pdf_path: If provided, copy PDF to this location. If None, run in draft mode.
            timeout: Compilation timeout in seconds
        """
        work_dir = tex_path.parent
        tex_name = tex_path.name

        cmd = [
            "pdflatex",
            "-halt-on-error",
            tex_name
        ]
        
        # Use draft mode if no output PDF path is specified
        if output_pdf_path is None:
            cmd.insert(-1, "-draftmode")

        result = subprocess.run(
            cmd,
            cwd=work_dir,
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

        # If output path is specified, copy the generated PDF
        if output_pdf_path is not None:
            generated_pdf_path = work_dir / tex_path.with_suffix('.pdf').name
            
            # Copy to desired output location
            generated_pdf_path.rename(output_pdf_path)
    
    @staticmethod
    def get_page_count_from_log(log_path: Path) -> int:
        """Extract page count from LaTeX log file."""
        log_content = log_path.read_text(encoding='utf-8', errors='ignore')
        
        # Search pattern for page count
        page_match = re.search(r'Output written on.*?\((\d+) page', log_content)
        if page_match:
            return int(page_match.group(1))

        raise RuntimeError(f"Could not extract page count from LaTeX log: {log_content}")


    @staticmethod
    def check_bounds_from_log(log_path: Path) -> bool:
        """Check if content fits within page bounds by analyzing log warnings."""
        if not log_path.exists():
            return False
            
        log_content = log_path.read_text(encoding='utf-8', errors='ignore')
        
        # Check for overfull hbox/vbox warnings indicating out-of-bounds content
        if "Overfull \\hbox" in log_content or "Overfull \\vbox" in log_content:
            return False
        
        # Check for severe badness warnings that indicate formatting issues
        badness_match = re.search(r"Underfull \\hbox.*badness (\d+)", log_content)
        if badness_match and int(badness_match.group(1)) >= 5000:
            return False
        
        return True