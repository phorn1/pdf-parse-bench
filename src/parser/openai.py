import os
import base64
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

        with open(pdf_path, "rb") as f:
            data = f.read()

        base64_string = base64.b64encode(data).decode("utf-8")

        response = client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": pdf_path.name,
                            "file_data": f"data:application/pdf;base64,{base64_string}",
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
        return "gpt-4o"


@parser_registry()
class OpenAIGPT4oMiniParser(BaseOpenAIParser):
    def __init__(self):
        super().__init__(model="gpt-4o-mini")

    @classmethod
    def parser_name(cls) -> str:
        return "gpt-4o-mini"


@parser_registry()
class OpenAIGPT5Parser(BaseOpenAIParser):
    def __init__(self):
        super().__init__(model="gpt-5")

    @classmethod
    def parser_name(cls) -> str:
        return "gpt-5"


@parser_registry()
class OpenAIGPT5MiniParser(BaseOpenAIParser):
    def __init__(self):
        super().__init__(model="gpt-5-mini")

    @classmethod
    def parser_name(cls) -> str:
        return "gpt-5-mini"


@parser_registry()
class OpenAIGPT5NanoParser(BaseOpenAIParser):
    def __init__(self):
        super().__init__(model="gpt-5-nano")

    @classmethod
    def parser_name(cls) -> str:
        return "gpt-5-nano"