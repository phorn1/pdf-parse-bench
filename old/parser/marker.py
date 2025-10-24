import os
from pathlib import Path
from dotenv import load_dotenv

from .core import PDFParser, parser_registry

# Load environment variables
load_dotenv()


class BaseMarkerParser(PDFParser):
    """Base class for Marker parsers with different LLM services."""
    
    def __init__(self, service: str, api_key_env: str):
        super().__init__()
        self.marker_service = service
        self.api_key = os.getenv(api_key_env)
        
        if not self.api_key:
            raise ValueError(f"{api_key_env} environment variable is required")
    
    @property
    def marker_config_class(self) -> str:
        return {
            "gemini": "marker.services.gemini.GoogleGeminiService",
            "openai": "marker.services.openai.OpenAIService"
        }[self.marker_service]
    
    def parse(self, pdf_path: Path, output_path: Path) -> str:
        """
        Parse PDF to markdown using Marker.
        
        Args:
            pdf_path: Path to input PDF file
            output_path: Path for output markdown file
            
        Returns:
            str: Generated markdown content
            
        Raises:
            Exception: For errors during conversion
        """
        # Configure Marker settings
        marker_config = {
            "output_format": "markdown",
            "use_llm": True,
            "llm_service": self.marker_config_class,
            "output_dir": str(output_path.parent) if output_path.is_file() else str(output_path),
            "redo_inline_math": True,
            f"{self.marker_service}_api_key": self.api_key,
        }

        # Import marker modules only when needed
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.config.parser import ConfigParser

        # Initialize Marker components
        config_parser = ConfigParser(marker_config)
        config_dict = config_parser.generate_config_dict()
        artifact_dict = create_model_dict()

        # Initialize PDF converter
        converter = PdfConverter(
            artifact_dict=artifact_dict,
            config=config_dict,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service()
        )

        try:
            # Perform conversion
            rendered = converter(str(pdf_path))

            # Write output
            self._write_output(rendered.markdown, output_path)
            
            return rendered.markdown

        except Exception as e:
            raise Exception(f"Error during PDF conversion: {e}")


@parser_registry()
class MarkerGeminiParser(BaseMarkerParser):
    """Marker parser using Gemini LLM service."""
    
    def __init__(self):
        super().__init__(service="gemini", api_key_env="GEMINI_API_KEY")
    
    @classmethod
    def parser_name(cls) -> str:
        return "marker-gemini"


@parser_registry()
class MarkerOpenAIParser(BaseMarkerParser):
    """Marker parser using OpenAI LLM service."""
    
    def __init__(self):
        super().__init__(service="openai", api_key_env="OPENAI_API_KEY")
    
    @classmethod
    def parser_name(cls) -> str:
        return "marker-openai"
