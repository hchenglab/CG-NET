#!/usr/bin/env python
"""CG-NET Command Line Interface Package.

This package provides a comprehensive CLI for CG-NET training and prediction,
including parameter management, configuration templates, and validation utilities.
"""

from .main import main, parse_args
from .utils import CLIUtils
from .templates import ConfigTemplateManager
from .validators import ConfigValidator

__all__ = [
    'main',
    'parse_args', 
    'CLIUtils',
    'ConfigTemplateManager',
    'ConfigValidator'
] 