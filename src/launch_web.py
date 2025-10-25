#!/usr/bin/env python3
"""
Launch script for the Cryptography RAG Chat Web Interface
"""
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from web_ui import main

if __name__ == "__main__":
    main()