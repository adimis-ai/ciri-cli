#!/usr/bin/env python3
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Optional, Any
from contextlib import contextmanager


import yaml
from dotenv import load_dotenv

from .agent import Ciri
from .controllers import CiriController
from .serializers import (
    CiriSerializer,
    CiriJSONEncoder,
    serialize_ciri_state,
    serialize_any_message,
    serialize_state_snapshot,
    serialize_interrupt,
    serialize_resume_command,
)

load_dotenv()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO

    # Clear any existing handlers to avoid conflicts
    logging.root.handlers.clear()

    # Configure logging to use stderr exclusively
    logging.basicConfig(
        level=level,
        format="%(levelname)-8s [%(name)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
        force=True,  # Override any existing configuration
    )


def parse_json_arg(value: str) -> Any:
    """Parse a JSON string argument."""
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {e}")


def parse_list_arg(value: str) -> list:
    """Parse a comma-separated list or JSON array."""
    if not value:
        return []
    try:
        # Try parsing as JSON first
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
        raise argparse.ArgumentTypeError("Value must be a JSON array")
    except json.JSONDecodeError:
        # Fallback to comma-separated values
        return [item.strip() for item in value.split(",") if item.strip()]


def main():
    """Main entry point for Ciri CLI."""
    parser = argparse.ArgumentParser(description="Ciri AI Copilot")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        default=True,
        help="Launch interactive TUI (default)",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.tui:
        from .tui import CiriApp

        app = CiriApp()
        app.run()
    else:
        print(
            "CLI mode not implemented. Use --tui to launch the interactive interface."
        )


if __name__ == "__main__":
    main()
