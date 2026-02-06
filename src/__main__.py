#!/usr/bin/env python3
"""
Ciri Copilot CLI

Usage:
    ciri [command] [options]

Commands:
    compile    Compile the Ciri agent with database and cache configuration
    stream     Stream responses from the Ciri agent
    invoke     Invoke the Ciri agent and get a single response
    history    Get conversation history for a thread

Examples:
    ciri compile --debug
    ciri stream --input '{"messages": [{"role": "user", "content": "Hello"}]}'
    ciri invoke --input '{"messages": [{"role": "user", "content": "Hello"}]}' --thread-id my-thread
    ciri history --thread-id my-thread --limit 10
"""

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

# Load environment variables before importing modules that may depend on them
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


def parse_ciri_spec(spec_input: str) -> dict:
    """
    Parse Ciri specification from JSON string or file path.

    Args:
        spec_input: Either a JSON string or a file path to JSON/YAML file

    Returns:
        Parsed specification as a dictionary

    Raises:
        argparse.ArgumentTypeError: If parsing fails
    """
    # Quick check if it looks like JSON (starts with '{' or '[')
    # or if it's reasonably short enough to be a filename
    if (
        len(spec_input) < 260
        and not spec_input.strip().startswith("{")
        and not spec_input.strip().startswith("[")
    ):
        # Check if it's a file path
        path = Path(spec_input)
        if path.exists() and path.is_file():
            try:
                with open(path, "r") as f:
                    # Determine format by extension
                    if path.suffix.lower() in [".yaml", ".yml"]:
                        return yaml.safe_load(f)
                    elif path.suffix.lower() == ".json":
                        return json.load(f)
                    else:
                        # Try to parse as YAML first, then JSON
                        content = f.read()
                        try:
                            return yaml.safe_load(content)
                        except yaml.YAMLError:
                            return json.loads(content)
            except Exception as e:
                raise argparse.ArgumentTypeError(f"Failed to parse spec file: {e}")

    # Try to parse as JSON string
    try:
        return json.loads(spec_input)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON spec: {e}")


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="ciri",
        description="Ciri Copilot CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose/debug logging",
    )

    parser.add_argument(
        "--sqlite-url",
        type=str,
        help="SQLite URL for storage (default: from SQLITE_URL env var)",
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Compile command
    compile_parser = subparsers.add_parser(
        "compile", help="Compile the Ciri agent with database and cache configuration"
    )
    compile_parser.add_argument(
        "--spec",
        type=str,
        required=True,
        help="Ciri specification as JSON string or path to JSON/YAML file",
    )
    compile_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for the agent",
    )

    # Stream command
    stream_parser = subparsers.add_parser(
        "stream", help="Stream responses from the Ciri agent"
    )
    stream_parser.add_argument(
        "--input",
        type=parse_json_arg,
        required=True,
        help="Input message state or command (JSON string)",
    )
    stream_parser.add_argument(
        "--thread-id",
        type=str,
        help="Thread ID for conversation persistence",
    )
    stream_parser.add_argument(
        "--config",
        type=parse_json_arg,
        help="Runnable configuration (JSON string)",
    )
    stream_parser.add_argument(
        "--context",
        type=parse_json_arg,
        help="Context dictionary (JSON string)",
    )
    stream_parser.add_argument(
        "--no-subgraphs",
        action="store_true",
        help="Disable subgraph streaming",
    )
    stream_parser.add_argument(
        "--spec",
        type=str,
        required=True,
        help="Ciri specification as JSON string or path to JSON/YAML file (required for auto-compilation)",
    )

    # Invoke command
    invoke_parser = subparsers.add_parser(
        "invoke", help="Invoke the Ciri agent and get a single response"
    )
    invoke_parser.add_argument(
        "--input",
        type=parse_json_arg,
        required=True,
        help="Input message state or command (JSON string)",
    )
    invoke_parser.add_argument(
        "--thread-id",
        type=str,
        help="Thread ID for conversation persistence",
    )
    invoke_parser.add_argument(
        "--config",
        type=parse_json_arg,
        help="Runnable configuration (JSON string)",
    )
    invoke_parser.add_argument(
        "--context",
        type=parse_json_arg,
        help="Context dictionary (JSON string)",
    )
    invoke_parser.add_argument(
        "--spec",
        type=str,
        required=True,
        help="Ciri specification as JSON string or path to JSON/YAML file (required for auto-compilation)",
    )

    # History command
    history_parser = subparsers.add_parser(
        "history", help="Get conversation history for a thread"
    )
    history_parser.add_argument(
        "--thread-id",
        type=str,
        required=True,
        help="Thread ID to retrieve history for",
    )
    history_parser.add_argument(
        "--filter",
        type=parse_json_arg,
        help="Filter criteria (JSON string)",
    )
    history_parser.add_argument(
        "--before",
        type=parse_json_arg,
        help="Get history before this config (JSON string)",
    )
    history_parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of history entries to retrieve",
    )
    history_parser.add_argument(
        "--spec",
        type=str,
        required=True,
        help="Ciri specification as JSON string or path to JSON/YAML file (required for auto-compilation)",
    )

    return parser


def handle_compile(args: argparse.Namespace, controller: CiriController) -> int:
    """Handle the compile command."""
    try:
        # Parse the Ciri specification
        spec = parse_ciri_spec(args.spec)
        logging.debug(f"Parsed Ciri spec: {spec}")

        # # Install embedding model
        # try:
        #     install_embedding_model()
        # except Exception as e:
        #     logging.error(f"Failed to install embedding model: {e}")

        # Create Ciri instance from spec
        # Assuming Ciri can be instantiated from a dict spec
        # If Ciri has a from_dict or similar method, use that instead
        ciri = Ciri(**spec) if isinstance(spec, dict) else Ciri(spec)

        # Compile with the Ciri instance
        controller.compile(ciri=ciri, debug=args.debug)
        print("✓ Ciri agent compiled successfully")
        return 0
    except Exception as e:
        logging.error(f"Failed to compile: {e}")
        return 1


@contextmanager
def suppress_logging():
    """Context manager to temporarily suppress all logging output."""
    # Save current handlers
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    original_level = root_logger.level

    # Remove all handlers temporarily
    for handler in original_handlers:
        root_logger.removeHandler(handler)

    # Set root logger to CRITICAL+1 to suppress all messages
    root_logger.setLevel(logging.CRITICAL + 1)

    # Also suppress module-specific loggers that might exist
    original_levels = {}
    for name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(name)
        if logger.level != logging.NOTSET and logger.hasHandlers():
            original_levels[name] = logger.level
            logger.setLevel(logging.CRITICAL + 1)
            # Also remove their handlers
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

    try:
        yield
    finally:
        # Restore original handlers
        for handler in original_handlers:
            root_logger.addHandler(handler)

        # Restore root logger level
        root_logger.setLevel(original_level)

        # Restore original levels and handlers
        for name, level in original_levels.items():
            logger = logging.getLogger(name)
            logger.setLevel(level)
            # Handlers will be restored by their original owners


def auto_compile_from_spec(
    args: argparse.Namespace, controller: CiriController, suppress_output: bool = False
) -> None:
    """Auto-compile the Ciri agent from spec argument.

    This is a helper used by stream, invoke, and history commands to ensure
    the agent is compiled before execution.

    Args:
        args: CLI arguments containing the spec
        controller: The Ciri controller instance
        suppress_output: If True, temporarily suppress logging output during compilation
    """
    if not hasattr(args, "spec") or not args.spec:
        raise ValueError("--spec argument is required for this command")

    spec = parse_ciri_spec(args.spec)

    if suppress_output:
        with suppress_logging():
            ciri = Ciri(**spec) if isinstance(spec, dict) else Ciri(spec)
            controller.compile(ciri=ciri, debug=False)
    else:
        logging.debug(f"Auto-compiling Ciri spec")
        ciri = Ciri(**spec) if isinstance(spec, dict) else Ciri(spec)
        controller.compile(ciri=ciri, debug=False)


def handle_stream(args: argparse.Namespace, controller: CiriController) -> int:
    """Handle the stream command."""
    try:
        # Auto-compile the agent before streaming
        auto_compile_from_spec(args, controller)

        # Build config if thread_id is provided
        config = args.config
        if args.thread_id:
            if config is None:
                config = {}
            if "configurable" not in config:
                config["configurable"] = {}
            config["configurable"]["thread_id"] = args.thread_id

        # Stream responses - flush immediately for real-time output
        for data in controller.stream(
            config=config,
            input=args.input,
            context=args.context,
            subgraphs=not args.no_subgraphs,
        ):
            try:
                json_output = json.dumps(data, cls=CiriJSONEncoder)
                print(json_output, flush=True)
            except (TypeError, ValueError) as json_error:
                # If serialization fails, output an error event
                error_data = {
                    "type": "error",
                    "message": f"JSON serialization error: {str(json_error)}",
                }
                print(json.dumps(error_data), flush=True)

        return 0
    except Exception as e:
        logging.error(f"Failed to stream: {e}")
        return 1


def handle_invoke(args: argparse.Namespace, controller: CiriController) -> int:
    """Handle the invoke command."""
    try:
        # Auto-compile the agent before invoking (suppress logging for clean JSON output)
        auto_compile_from_spec(args, controller, suppress_output=True)

        # Build config if thread_id is provided
        config = args.config
        if args.thread_id:
            if config is None:
                config = {}
            if "configurable" not in config:
                config["configurable"] = {}
            config["configurable"]["thread_id"] = args.thread_id

        # Invoke and get response
        result = controller.invoke(
            input=args.input,
            config=config,
            context=args.context,
        )
        print(json.dumps(result, indent=2, cls=CiriJSONEncoder))

        return 0
    except Exception as e:
        logging.error(f"Failed to invoke: {e}")
        return 1


def handle_history(args: argparse.Namespace, controller: CiriController) -> int:
    """Handle the history command."""
    try:
        # Auto-compile the agent before fetching history (suppress logging for clean JSON output)
        auto_compile_from_spec(args, controller, suppress_output=True)

        for snapshot in controller.history(
            thread_id=args.thread_id,
            filter=args.filter,
            before=args.before,
            limit=args.limit,
        ):
            # Explicitly serialize StateSnapshot to dict before JSON encoding
            # (StateSnapshot is a NamedTuple which json.dumps serializes as an array)
            serialized = serialize_state_snapshot(snapshot)
            print(json.dumps(serialized, cls=CiriJSONEncoder), flush=True)

        return 0
    except Exception as e:
        logging.error(f"Failed to get history: {e}")
        return 1


def main() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Show help if no command provided
    if not args.command:
        parser.print_help()
        return 0

    # Initialize controller (Ciri will be created in compile command)
    try:
        with CiriController(
            sqlite_url=args.sqlite_url,
        ) as controller:
            # Route to appropriate command handler
            if args.command == "compile":
                return handle_compile(args, controller)
            elif args.command == "stream":
                return handle_stream(args, controller)
            elif args.command == "invoke":
                return handle_invoke(args, controller)
            elif args.command == "history":
                return handle_history(args, controller)
            else:
                parser.print_help()
                return 1

    except Exception as e:
        logging.error(f"Failed to initialize: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
