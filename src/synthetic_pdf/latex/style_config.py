"""LaTeX configuration system for automated PDF generation."""

import random
from enum import Enum
from dataclasses import dataclass
from pydantic import BaseModel


class DocumentClass(str, Enum):
    """Available LaTeX document classes."""
    ARTICLE = "article"
    REPORT = "report" 
    BOOK = "book"
    MEMOIR = "memoir"
    SCRARTCL = "scrartcl"
    SCRREPRT = "scrreprt"
    SCRBOOK = "scrbook"


class FontFamily(Enum):
    """Available font families with their LaTeX packages."""
    TIMES = (["\\usepackage{times}", "\\usepackage{txfonts}"])
    PALATINO = (["\\usepackage{palatino}", "\\usepackage{euler}"])
    LIBERTINE = (["\\usepackage{libertine}", "\\usepackage{libertinust1math}"])
    CHARTER = (["\\usepackage{charter}", "\\usepackage[charter]{mathdesign}"])
    LMODERN = (["\\usepackage{lmodern}"])
    KPFONTS = (["\\usepackage{kpfonts}"])
    
    @property
    def packages(self) -> list[str]:
        """Get required LaTeX packages for this font family."""
        return self.value


class Language(Enum):
    """Available document languages."""
    ENGLISH = ("english", "en_US")
    GERMAN = ("german", "de_DE")
    SPANISH = ("spanish", "es_ES")
    FRENCH = ("french", "fr_FR")
    
    def __init__(self, babel_name: str, locale_code: str):
        self.babel_name = babel_name
        self.locale_code = locale_code


class PageGeometry(str, Enum):
    """Page size options."""
    A4 = "a4paper"
    # LETTER = "letterpaper"
    # A5 = "a5paper"
    # B5 = "b5paper"


@dataclass
class MarginSettings:
    """Margin configuration."""
    top: str
    bottom: str
    left: str
    right: str
    
    @classmethod
    def random(cls) -> 'MarginSettings':
        """Generate random margin settings."""
        margins = ["1.5cm", "2cm", "2.5cm", "3cm"]
        return cls(
            top=random.choice(margins),
            bottom=random.choice(margins), 
            left=random.choice(margins),
            right=random.choice(margins)
        )
    
    def to_latex_options(self) -> list[str]:
        """Convert margins to LaTeX geometry options."""
        return [
            f"top={self.top}",
            f"bottom={self.bottom}",
            f"left={self.left}",
            f"right={self.right}"
        ]


@dataclass
class TypographySettings:
    """Typography configuration."""
    font_size: str = "11pt"
    line_spacing: str = "single"  # single, onehalf, double
    paragraph_indent: str = "1.5em"
    paragraph_skip: str = "0pt"
    
    @classmethod
    def random(cls) -> 'TypographySettings':
        """Generate random typography settings."""
        font_sizes = ["10pt", "11pt", "12pt"]
        line_spacings = ["single", "onehalf", "double"]
        indents = ["0pt", "1em", "1.5em", "2em"]
        skips = ["0pt", "0.5em", "1em"]
        
        return cls(
            font_size=random.choice(font_sizes),
            line_spacing=random.choice(line_spacings),
            paragraph_indent=random.choice(indents),
            paragraph_skip=random.choice(skips)
        )


@dataclass
class ContentSettings:
    """Content generation configuration."""
    # Mixed text block settings
    mixed_segment_min_chars: int = 50
    mixed_segment_max_chars: int = 90
    mixed_segments_min_count: int = 2
    mixed_segments_max_count: int = 5
    # Paragraph block settings
    paragraph_min_chars: int = 120
    paragraph_max_chars: int = 200
    


class LaTeXConfig(BaseModel):
    """Complete LaTeX document configuration."""
    
    # Document structure
    document_class: DocumentClass
    font_family: FontFamily
    page_geometry: PageGeometry
    language: Language
    margins: MarginSettings
    typography: TypographySettings
    content: ContentSettings
    
    # Layout options
    two_column: bool = False
    column_sep: str = "1cm"
    
    # Content features
    include_headers: bool = True
    use_fancy_headers: bool = False
    
    # Reproducibility
    seed: int | None = None
    
    
    @classmethod
    def random(cls, seed: int | None = None) -> 'LaTeXConfig':
        """Generate random LaTeX configuration."""
        if seed is not None:
            random.seed(seed)
            
        return cls(
            document_class=random.choice(list(DocumentClass)),
            font_family=random.choice(list(FontFamily)),
            page_geometry=random.choice(list(PageGeometry)),
            language=random.choice(list(Language)),
            margins=MarginSettings.random(),
            typography=TypographySettings.random(),
            content=ContentSettings(),
            two_column=random.choice([True, False]),
            column_sep=random.choice(["0.8cm", "1cm", "1.2cm"]),
            include_headers=random.choice([True, False]),
            use_fancy_headers=random.choice([True, False]),
            seed=seed
        )
    
    
    