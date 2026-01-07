# src/web_ui_v2.py
"""
Cryptography Component Composer Web Interface

This module can be imported or run directly.
For better experience, use launch_composer_ui.py from the project root.
"""
import sys
import os

# Handle both direct execution and module import
if __name__ == "__main__":
    # When run directly, add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.web_ui_components import build_ui
else:
    # When imported as a module
    from .web_ui_components import build_ui

def main():
    """Launch the Component Composer interface."""
    print("🚀 Starting Cryptography Component Composer...")
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7861, show_api=False, inbrowser=True)

if __name__ == "__main__":
    main()
