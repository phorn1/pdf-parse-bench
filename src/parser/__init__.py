from .core import PDFParser, ParserRegistry

# Auto-discover and import all parser modules
import importlib
import pkgutil
from pathlib import Path

# Import all parser modules dynamically to trigger decorator registration
_current_dir = Path(__file__).parent
for finder, module_name, ispkg in pkgutil.iter_modules([str(_current_dir)]):
    if not module_name.startswith('_') and not module_name == 'core':
        importlib.import_module(f'.{module_name}', package=__name__)

__all__ = [
    'PDFParser',
    'ParserRegistry', 
]