import os
from pathlib import Path
from dotenv import load_dotenv
from .core import PDFParser, parser_registry, PDF_TO_MARKDOWN_PROMPT

# Load environment variables
load_dotenv()


class BaseOpenAIParser(PDFParser):
    def __init__(self, model: str):
        super().__init__()
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

    def parse(self, pdf_path: Path, output_path: Path) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)

        file = client.files.create(
            file=open(pdf_path, "rb"),
            purpose="user_data"
        )

        response = client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "file_id": file.id,
                        },
                        {
                            "type": "input_text",
                            "text": f"{PDF_TO_MARKDOWN_PROMPT}",
                        },
                    ]
                }
            ]
        )

        self._write_output(response.output_text, output_path)
        return response.output_text


@parser_registry()
class OpenAIGPT4oParser(BaseOpenAIParser):
    def __init__(self):
        super().__init__(model="gpt-4o")

    @classmethod
    def parser_name(cls) -> str:
        return "openai_gpt-4o"


@parser_registry()
class OpenAIGPT4oMiniParser(BaseOpenAIParser):
    def __init__(self):
        super().__init__(model="gpt-4o-mini")

    @classmethod
    def parser_name(cls) -> str:
        return "openai_gpt-4o-mini"