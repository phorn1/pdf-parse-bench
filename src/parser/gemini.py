import os
from pathlib import Path
from dotenv import load_dotenv
from .core import PDFParser, parser_registry, PDF_TO_MARKDOWN_PROMPT

# Load environment variables
load_dotenv()


class BaseGeminiParser(PDFParser):
    def __init__(self, model: str):
        super().__init__()
        self.model = model
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

    def parse(self, pdf_path: Path, output_path: Path) -> str:
        from google import genai
        from google.genai import types
        
        client = genai.Client(api_key=self.api_key)
        
        with open(pdf_path, "rb") as file:
            pdf_bytes = file.read()

        response = client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=pdf_bytes, mime_type='application/pdf'),
                PDF_TO_MARKDOWN_PROMPT
            ]
        )
        
        self._write_output(response.text, output_path)
        return response.text


@parser_registry()
class GeminiProParser(BaseGeminiParser):
    def __init__(self):
        super().__init__(model="gemini-2.5-pro")

    @classmethod
    def parser_name(cls) -> str:
        return "gemini-2.5-pro"


@parser_registry()
class GeminiFlashParser(BaseGeminiParser):
    def __init__(self):
        super().__init__(model="gemini-2.5-flash")

    @classmethod
    def parser_name(cls) -> str:
        return "gemini-2.5-flash"
