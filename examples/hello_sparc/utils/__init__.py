"""
Utilities package for Hello SPARC application.

This package contains helper modules that demonstrate modularity
and separation of concerns according to SPARC methodology.
"""

from hello_sparc.utils.formatter import format_greeting, format_colored_text
from hello_sparc.utils.logger import get_logger, log_greeting

# For convenience, expose key functions directly
__all__ = [
    'format_greeting',
    'format_colored_text',
    'get_logger',
    'log_greeting'
]