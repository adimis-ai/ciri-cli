#!/usr/bin/env python3
"""
Build script for compiling ciri-copilot CLI to a standalone binary.

Usage:
    python build.py [--target TARGET]

Options:
    --target TARGET    Target platform (auto-detected if not specified)
                       Examples: x86_64-unknown-linux-gnu, x86_64-pc-windows-msvc, aarch64-apple-darwin
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def get_target_triple() -> str:
    """Get the Rust-style target triple for the current platform."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    # Map machine architectures
    arch_map = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "aarch64": "aarch64",
        "arm64": "aarch64",
        "i686": "i686",
        "i386": "i686",
    }
    arch = arch_map.get(machine, machine)

    # Map OS to target triple
    if system == "linux":
        return f"{arch}-unknown-linux-gnu"
    elif system == "darwin":
        return f"{arch}-apple-darwin"
    elif system == "windows":
        return f"{arch}-pc-windows-msvc"
    else:
        raise ValueError(f"Unsupported platform: {system}")


def get_binary_name(target: str) -> str:
    """Get the output binary name based on target platform."""
    base_name = "ciri"
    if "windows" in target:
        return f"{base_name}-{target}.exe"
    return f"{base_name}-{target}"


def build(target: str | None = None) -> Path:
    """Build the ciri CLI binary using PyInstaller."""
    src_dir = Path(__file__).parent
    src_code_dir = src_dir / "src"
    dist_dir = src_dir / "dist"
    build_dir = src_dir / "build"

    # Get target triple
    if target is None:
        target = get_target_triple()

    print(f"Building ciri-copilot for target: {target}")

    # Clean previous builds
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    if build_dir.exists():
        shutil.rmtree(build_dir)

    # Determine binary name (Tauri expects: name-target[.exe])
    binary_name = get_binary_name(target)

    # Hidden imports for the langchain/langgraph ecosystem and dependencies
    hidden_imports = [
        # Core langchain modules
        "langchain",
        "langchain_core",
        "langchain_core.tools",
        "langchain_core.messages",
        "langchain_core.runnables",
        "langchain_openai",
        "langchain_community",
        "langchain_community.tools",
        "langchain_unstructured",
        "langchain_mcp_adapters",
        "langchain_mcp_adapters.client",
        # Langgraph
        "langgraph",
        "langgraph.types",
        "langgraph.store",
        "langgraph.store.base",
        "langgraph.store.sqlite",
        "langgraph.checkpoint.sqlite",
        "sqlite_vec",
        "langgraph.cache",
        "langgraph.cache.base",
        "langgraph_swarm",
        # Deep agents
        "deepagents",
        "crawl4ai",
        # CLI and UI
        "typer",
        "rich",
        "anyio",
        "httpx",
        "tenacity",
        "orjson",
        # Other dependencies
        "pydantic",
        "pydantic._internal",
        "pydantic._internal._model_construction",
        "pydantic._internal._generate_schema",
        "pydantic._internal._schema_generation_shared",
        "pydantic.fields",
        "pydantic_core",
        "dotenv",
        "python_dotenv",
        "yaml",
        "openai",
        "duckduckgo_search",
        "ddgs",
        "sqlalchemy",
        "structlog",
        "opentelemetry",
        "packaging",
        "platformdirs",
        # Local src modules
        "src",
        "src.db",
        "src.subagents",
        "src.serializers",
        "src.utils",
        "src.toolkit",
        "src.middlewares",
        "src.skills",
        "src.prompts",
    ]

    # PyInstaller command
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",
        "--name",
        binary_name.replace(".exe", "").replace(
            f"-{target}", ""
        ),  # Base name without target
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(build_dir),
        "--specpath",
        str(build_dir),
        # Add all source modules
        "--paths",
        str(src_code_dir),
        # Collect all submodules from key packages
        "--collect-all",
        "langchain",
        "--collect-all",
        "langchain_core",
        "--collect-all",
        "langchain_openai",
        "--collect-all",
        "langgraph",
        "--collect-all",
        "pydantic",
        "--collect-all",
        "pydantic_core",
        "--collect-all",
        "crawl4ai",
        "--collect-all",
        "playwright",
    ]

    # Add all hidden imports
    for hidden_import in hidden_imports:
        cmd.extend(["--hidden-import", hidden_import])

    # Add the main entry point
    cmd.append(str(src_code_dir / "__main__.py"))

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=src_dir)

    if result.returncode != 0:
        print("PyInstaller build failed!")
        sys.exit(1)

    # Rename the output to include target triple for Tauri sidecar
    output_binary = dist_dir / ("ciri.exe" if "windows" in target else "ciri")
    final_binary = dist_dir / binary_name

    if output_binary.exists() and output_binary != final_binary:
        output_binary.rename(final_binary)

    # Copy to src-tauri/binaries for Tauri sidecar
    tauri_binaries_dir = src_dir.parent / "src-tauri" / "binaries"
    tauri_binaries_dir.mkdir(parents=True, exist_ok=True)

    dest_binary = tauri_binaries_dir / binary_name
    shutil.copy2(final_binary, dest_binary)

    print(f"Build complete: {final_binary}")
    print(f"Copied to: {dest_binary}")

    return final_binary


def main():
    parser = argparse.ArgumentParser(description="Build ciri-copilot CLI binary")
    parser.add_argument(
        "--target",
        type=str,
        help="Target platform triple (auto-detected if not specified)",
    )
    args = parser.parse_args()

    build(target=args.target)


if __name__ == "__main__":
    main()
