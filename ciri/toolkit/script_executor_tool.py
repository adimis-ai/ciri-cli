import os
import sys
import uuid
import shutil
import logging
import subprocess
from pathlib import Path
from typing import Optional, List, Literal, Annotated, Callable

from pydantic import BaseModel, Field
from langgraph.types import interrupt
from langchain_core.tools import StructuredTool, InjectedToolCallId

logger = logging.getLogger(__name__)


class ScriptExecutorInput(BaseModel):
    script_content: str = Field(
        description="The full script source code to execute.",
    )
    language: Literal["python", "javascript"] = Field(
        description="The scripting language: 'python' or 'javascript'.",
    )
    dependencies: Optional[List[str]] = Field(
        default=None,
        description=(
            "Packages to install before running the script. "
            "For Python: pip package names (e.g. ['playwright', 'requests']). "
            "For JavaScript: npm package names (e.g. ['puppeteer', 'axios'])."
        ),
    )
    working_dir: Optional[str] = Field(
        default=None,
        description=(
            "Directory where the script executes. Defaults to a temp directory. "
            "Use this when the script needs access to specific local files."
        ),
    )
    output_dir: Optional[str] = Field(
        default=None,
        description=(
            "Directory where the script should save output files (screenshots, "
            "HTML, data files, etc.). If not set, defaults to working_dir or "
            "the temp execution directory."
        ),
    )
    timeout: int = Field(
        default=120,
        description="Maximum execution time in seconds. Default is 120.",
    )
    cleanup: bool = Field(
        default=True,
        description=(
            "If True, the temporary execution environment (venv/node_modules) "
            "is removed after execution. Output files in output_dir are preserved."
        ),
    )


def _run_streaming(
    cmd: list,
    callback: Optional[Callable[[str], None]] = None,
    timeout: int = 120,
    cwd: str = None,
    env: dict = None,
) -> subprocess.CompletedProcess:
    """Run a subprocess, streaming output line-by-line via callback."""
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=cwd,
        env=env,
    )
    output_lines = []
    try:
        for line in process.stdout:
            output_lines.append(line)
            if callback:
                callback(line.rstrip("\n"))
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        raise

    stdout = "".join(output_lines)
    return subprocess.CompletedProcess(
        args=cmd, returncode=process.returncode, stdout=stdout, stderr=""
    )


def build_script_executor_tool(
    name: str = "execute_sandboxed_script",
    description: str = (
        "Execute standalone Python or JavaScript scripts in an isolated environment. "
        "The tool handles dependency installation (pip/npm), creates a temp workspace, "
        "and performs cleanup by default. Requires human-in-the-loop approval. "
        "\n\nWHEN TO USE:\n"
        "- Complex data processing or transformation that is inefficient via multiple tool calls.\n"
        "- Web automation tasks (e.g., using Playwright, Puppeteer) requiring custom logic.\n"
        "- Using specialized libraries (e.g. pandas, beautifulsoup4) not available as built-in tools.\n"
        "- Generating artifacts like CSVs, images, or JSON files that need to be preserved.\n"
        "\n\nWHEN NOT TO USE:\n"
        "- Simple file reads/writes, directory listings, or basic shell commands.\n"
        "- When a more direct tool (e.g., web search, specialized API tool) is available.\n"
        "- Tasks that can be accomplished with standard library code without external dependencies "
        "if those dependencies are large or slow to install.\n"
        "\n\nHOW TO USE:\n"
        "- List all required packages in the 'dependencies' argument.\n"
        "- Use the 'CIRI_OUTPUT_DIR' environment variable in your script to save persistent output.\n"
        "- To use scripts from skills: Copy them from the skill directory to the necessary structure inside a temp folder, then pass that folder as 'working_dir' so they are available in the sandbox.\n"
        "- Default 'cleanup=True' removes the environment but preserves contents of 'output_dir'."
    ),
    output_callback: Optional[Callable[[str], None]] = None,
) -> StructuredTool:
    """Build a script executor tool with HITL approval via interrupt."""

    def run_script(
        script_content: str,
        language: Literal["python", "javascript"],
        dependencies: Optional[List[str]] = None,
        working_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        timeout: int = 120,
        cleanup: bool = True,
        tool_call_id: Annotated[Optional[str], InjectedToolCallId] = None,
    ) -> str:
        # --- HITL approval via interrupt ---
        approval = interrupt(
            {
                "type": "script_execution",
                "script_content": script_content,
                "language": language,
                "dependencies": dependencies or [],
                "working_dir": working_dir,
                "output_dir": output_dir,
                "timeout": timeout,
                "cleanup": cleanup,
            }
        )

        # Handle rejection
        if isinstance(approval, dict):
            if approval.get("status") == "rejected":
                return f"Script execution rejected by user: {approval.get('reason', 'No reason given')}"
            if approval.get("status") == "edited":
                script_content = approval.get("script_content", script_content)
        elif isinstance(approval, str) and approval.lower() in ("rejected", "reject"):
            return "Script execution rejected by user."

        # --- Setup execution environment ---
        exec_id = uuid.uuid4().hex[:12]
        temp_base = Path("/tmp/ciri_scripts") / exec_id
        temp_base.mkdir(parents=True, exist_ok=True)

        effective_working_dir = Path(working_dir) if working_dir else temp_base
        effective_working_dir.mkdir(parents=True, exist_ok=True)

        effective_output_dir = Path(output_dir) if output_dir else effective_working_dir
        effective_output_dir.mkdir(parents=True, exist_ok=True)

        result_parts = []

        try:
            if language == "python":
                result_parts = _execute_python(
                    script_content=script_content,
                    dependencies=dependencies,
                    temp_base=temp_base,
                    working_dir=effective_working_dir,
                    output_dir=effective_output_dir,
                    timeout=timeout,
                    output_callback=output_callback,
                )
            elif language == "javascript":
                result_parts = _execute_javascript(
                    script_content=script_content,
                    dependencies=dependencies,
                    temp_base=temp_base,
                    working_dir=effective_working_dir,
                    output_dir=effective_output_dir,
                    timeout=timeout,
                    output_callback=output_callback,
                )
        except subprocess.TimeoutExpired:
            result_parts.append(f"ERROR: Script timed out after {timeout} seconds.")
        except Exception as e:
            logger.error(f"Script execution error: {e}")
            result_parts.append(f"ERROR: {e}")
        finally:
            if cleanup:
                cleanup_status = _cleanup(temp_base, effective_output_dir)
                result_parts.append(cleanup_status)

        return "\n".join(result_parts)

    return StructuredTool.from_function(
        func=run_script,
        name=name,
        description=description,
        args_schema=ScriptExecutorInput,
    )


def _execute_python(
    script_content: str,
    dependencies: Optional[List[str]],
    temp_base: Path,
    working_dir: Path,
    output_dir: Path,
    timeout: int,
    output_callback: Optional[Callable[[str], None]] = None,
) -> List[str]:
    """Execute a Python script in an isolated venv."""
    results = []
    venv_dir = temp_base / ".venv"

    # Create venv (fast, no streaming needed)
    logger.info(f"Creating Python venv at {venv_dir}")
    proc = subprocess.run(
        [sys.executable, "-m", "venv", str(venv_dir)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        results.append(f"ERROR creating venv: {proc.stderr}")
        return results
    results.append("Virtual environment created.")

    # Determine python/pip paths
    if sys.platform == "win32":
        python_bin = venv_dir / "Scripts" / "python.exe"
        pip_bin = venv_dir / "Scripts" / "pip.exe"
    else:
        python_bin = venv_dir / "bin" / "python"
        pip_bin = venv_dir / "bin" / "pip"

    # Install dependencies
    if dependencies:
        logger.info(f"Installing Python deps: {dependencies}")
        proc = _run_streaming(
            [str(pip_bin), "install", *dependencies],
            callback=output_callback,
            timeout=300,
            cwd=str(working_dir),
        )
        if proc.returncode != 0:
            results.append(f"ERROR installing dependencies:\n{proc.stdout}")
            return results
        results.append(f"Installed: {', '.join(dependencies)}")

        # Special handling: if playwright is a dependency, install browsers
        if any("playwright" in dep.lower() for dep in dependencies):
            logger.info("Installing Playwright browsers (chromium)")
            proc = _run_streaming(
                [str(python_bin), "-m", "playwright", "install", "chromium"],
                callback=output_callback,
                timeout=300,
            )
            if proc.returncode != 0:
                results.append(
                    f"WARNING: Playwright browser install failed: {proc.stdout}"
                )
            else:
                results.append("Playwright chromium browser installed.")

    # Write script
    script_path = temp_base / "script.py"
    script_path.write_text(script_content, encoding="utf-8")

    # Execute
    env = os.environ.copy()
    env["CIRI_OUTPUT_DIR"] = str(output_dir)
    logger.info(f"Executing Python script: {script_path}")
    proc = _run_streaming(
        [str(python_bin), str(script_path)],
        callback=output_callback,
        timeout=timeout,
        cwd=str(working_dir),
        env=env,
    )

    results.append(f"Exit code: {proc.returncode}")
    if proc.stdout.strip():
        results.append(f"STDOUT:\n{proc.stdout.strip()}")

    # List output files
    output_files = _list_output_files(output_dir, temp_base)
    if output_files:
        results.append(f"Output files:\n" + "\n".join(f"  - {f}" for f in output_files))

    return results


def _execute_javascript(
    script_content: str,
    dependencies: Optional[List[str]],
    temp_base: Path,
    working_dir: Path,
    output_dir: Path,
    timeout: int,
    output_callback: Optional[Callable[[str], None]] = None,
) -> List[str]:
    """Execute a JavaScript script with npm dependencies."""
    results = []

    # Find node and npm
    node_bin = shutil.which("node")
    npm_bin = shutil.which("npm")
    if not node_bin:
        results.append("ERROR: Node.js not found on PATH. Please install Node.js.")
        return results
    if not npm_bin:
        results.append("ERROR: npm not found on PATH. Please install Node.js/npm.")
        return results

    # Initialize npm project in temp_base (fast, no streaming needed)
    proc = subprocess.run(
        [npm_bin, "init", "-y"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(temp_base),
    )
    if proc.returncode != 0:
        results.append(f"ERROR initializing npm project: {proc.stderr}")
        return results

    # Install dependencies
    if dependencies:
        logger.info(f"Installing npm deps: {dependencies}")
        proc = _run_streaming(
            [npm_bin, "install", *dependencies],
            callback=output_callback,
            timeout=300,
            cwd=str(temp_base),
        )
        if proc.returncode != 0:
            results.append(f"ERROR installing dependencies:\n{proc.stdout}")
            return results
        results.append(f"Installed: {', '.join(dependencies)}")

    # Write script
    script_path = temp_base / "script.js"
    script_path.write_text(script_content, encoding="utf-8")

    # Execute
    env = os.environ.copy()
    env["CIRI_OUTPUT_DIR"] = str(output_dir)
    # Add node_modules/.bin to PATH for installed CLI tools
    node_modules_bin = temp_base / "node_modules" / ".bin"
    env["PATH"] = str(node_modules_bin) + os.pathsep + env.get("PATH", "")
    # Set NODE_PATH so require() finds locally installed packages
    env["NODE_PATH"] = str(temp_base / "node_modules")

    logger.info(f"Executing JavaScript script: {script_path}")
    proc = _run_streaming(
        [node_bin, str(script_path)],
        callback=output_callback,
        timeout=timeout,
        cwd=str(working_dir),
        env=env,
    )

    results.append(f"Exit code: {proc.returncode}")
    if proc.stdout.strip():
        results.append(f"STDOUT:\n{proc.stdout.strip()}")

    # List output files
    output_files = _list_output_files(output_dir, temp_base)
    if output_files:
        results.append(f"Output files:\n" + "\n".join(f"  - {f}" for f in output_files))

    return results


def _list_output_files(output_dir: Path, temp_base: Path) -> List[str]:
    """List files in output_dir that aren't part of the temp environment."""
    if not output_dir.exists():
        return []

    output_files = []
    for f in output_dir.rglob("*"):
        if f.is_file():
            # Skip venv/node_modules internals
            rel = str(f.relative_to(output_dir))
            if rel.startswith(".venv") or rel.startswith("node_modules"):
                continue
            # Skip the script itself if output_dir == temp_base
            if f.name in (
                "script.py",
                "script.js",
                "package.json",
                "package-lock.json",
            ):
                continue
            output_files.append(str(f))

    return output_files


def _cleanup(temp_base: Path, output_dir: Path) -> str:
    """Remove the temp execution environment, preserving output files."""
    try:
        # If output_dir is inside temp_base, move output files out first
        if output_dir != temp_base and str(output_dir).startswith(str(temp_base)):
            # Output dir is a subdirectory of temp — nothing to move, it'll be deleted
            pass

        if temp_base.exists():
            shutil.rmtree(temp_base, ignore_errors=True)
        return "Cleanup: temporary environment removed."
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        return f"Cleanup warning: {e}"
