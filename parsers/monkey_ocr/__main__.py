"""CLI entry point for MonkeyOCR parser benchmark."""

import tempfile
import zipfile
from pathlib import Path

import requests

from pdf_parse_bench.pipeline import run_cli
from pdf_parse_bench.utilities import PDFParser


class MonkeyOCRParser(PDFParser):
    """PDF parser using MonkeyOCR via FastAPI."""

    def __init__(self):
        """Initialize MonkeyOCR parser with FastAPI client."""
        super().__init__()
        self.api_url = "http://localhost:7861"

    def _check_health(self) -> bool:
        """Check if MonkeyOCR API is healthy and ready."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get("status") == "healthy" and data.get("model_loaded", False)
            return False
        except requests.RequestException:
            return False

    @classmethod
    def parser_name(cls) -> str:
        """Return parser name identifier."""
        return "monkey_ocr"

    def parse(self, pdf_path: Path, output_path: Path) -> str:
        """Parse single-page PDF to markdown using MonkeyOCR."""
        # ========== CHECK API HEALTH ==========
        if not self._check_health():
            raise RuntimeError(
                f"MonkeyOCR API is not available at {self.api_url}. "
                "Please ensure the Docker container is running with 'docker compose up monkeyocr-api'"
            )

        # ========== SEND PDF TO API ==========
        with open(pdf_path, "rb") as f:
            files = {"file": (pdf_path.name, f, "application/pdf")}
            response = requests.post(
                f"{self.api_url}/parse",
                files=files,
                timeout=300,  # 5 minutes timeout for processing
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"MonkeyOCR API request failed with status {response.status_code}: {response.text}"
            )

        # ========== EXTRACT MARKDOWN FROM RESPONSE ==========
        result = response.json()

        if not result.get("success", False):
            raise RuntimeError(
                f"MonkeyOCR parsing failed: {result.get('message', 'Unknown error')}"
            )

        # The API returns a download_url for a ZIP file containing all results
        download_url = result.get("download_url")
        if not download_url:
            raise RuntimeError("No download_url in API response")

        # Download the ZIP file
        zip_url = f"{self.api_url}{download_url}"
        zip_response = requests.get(zip_url, timeout=30)

        if zip_response.status_code != 200:
            raise RuntimeError(
                f"Failed to download ZIP file: {zip_response.status_code}"
            )

        # Extract markdown from ZIP
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as zip_file:
            zip_file.write(zip_response.content)
            zip_path = Path(zip_file.name)

        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                # Find the markdown file in the ZIP
                markdown_files = [f for f in zip_ref.namelist() if f.endswith(".md")]

                if not markdown_files:
                    raise RuntimeError("No markdown file found in ZIP archive")

                # Read the first markdown file
                with zip_ref.open(markdown_files[0]) as md_file:
                    markdown = md_file.read().decode("utf-8")
        finally:
            zip_path.unlink()

        self._write_output(markdown, output_path)
        return markdown


if __name__ == "__main__":
    run_cli(parser=MonkeyOCRParser())
