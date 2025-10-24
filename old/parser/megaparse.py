import os
from pathlib import Path
from dotenv import load_dotenv

from .core import PDFParser, parser_registry

# Load environment variables
load_dotenv()


@parser_registry()
class MegaParseParser(PDFParser):
    """PDF parser using MegaParse Vision."""
    
    def __init__(self):
        super().__init__()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
    
    @classmethod
    def parser_name(cls) -> str:
        return "megaparse"

    def parse(self, pdf_path: Path, output_path: Path) -> str:
        """
        Parse PDF to markdown using MegaParse.
        
        Args:
            pdf_path: Path to input PDF file
            output_path: Path for output markdown file
            
        Returns:
            str: Generated markdown content
        """
        
        from megaparse.parser.megaparse_vision import MegaParseVision
        from langchain_openai import ChatOpenAI
        
        # For now, MegaParse uses hardcoded OpenAI settings
        model = ChatOpenAI(model='gpt-4o', api_key=self.openai_api_key)  # type: ignore
        parser = MegaParseVision(model=model)
        response = parser.convert(str(pdf_path))
        text_blocks = response.content
        combined_text = "\n\n".join(block.text for block in text_blocks)
        
        self._write_output(combined_text, output_path)
        return combined_text


