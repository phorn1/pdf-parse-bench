"""LaTeX configuration system for automated PDF generation."""

import random
from enum import Enum
from dataclasses import dataclass
from pydantic import BaseModel


class DocumentClass(Enum):
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


@dataclass
class MarginSettings:
    """Margin configuration."""
    top: str
    bottom: str
    left: str
    right: str

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


@dataclass
class ContentSettings:
    """Content generation configuration."""
    # Mixed text block settings
    mixed_segment_min_chars: int = 50
    mixed_segment_max_chars: int = 90
    mixed_segments_max_count: int = 5
    # Paragraph block settings
    paragraph_min_chars: int = 120
    paragraph_max_chars: int = 200
    


class LaTeXConfig(BaseModel):
    """Complete LaTeX document configuration."""
    
    # Document structure
    document_class: DocumentClass
    font_family: FontFamily
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
        rng = random.Random(seed)

        margins = ["1.5cm", "2cm", "2.5cm", "3cm"]
        font_sizes = ["10pt", "11pt", "12pt"]
        line_spacings = ["single", "onehalf", "double"]
        indents = ["0pt", "1em", "1.5em", "2em"]
        skips = ["0pt", "0.5em", "1em"]

        return cls(
            document_class=rng.choice(list(DocumentClass)),
            font_family=rng.choice(list(FontFamily)),
            language=rng.choice(list(Language)),
            margins=MarginSettings(
                top=rng.choice(margins),
                bottom=rng.choice(margins),
                left=rng.choice(margins),
                right=rng.choice(margins),
            ),
            typography=TypographySettings(
                font_size=rng.choice(font_sizes),
                line_spacing=rng.choice(line_spacings),
                paragraph_indent=rng.choice(indents),
                paragraph_skip=rng.choice(skips),
            ),
            content=ContentSettings(),
            two_column=rng.choice([True, False]),
            column_sep=rng.choice(["0.8cm", "1cm", "1.2cm"]),
            include_headers=rng.choice([True, False]),
            use_fancy_headers=rng.choice([True, False]),
            seed=seed,
        )
