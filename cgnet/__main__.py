#!/usr/bin/env python
"""
Entry point for running cgnet as a module with python -m cgnet.
"""

from .cli.main import main

if __name__ == "__main__":
    exit(main()) 