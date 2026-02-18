import os
import tomllib
import json
import logging
import asyncio
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Union


from langchain.agents.middleware import AgentMiddleware
from langchain_mcp_adapters.client import MultiServerMCPClient
from ..utils import get_default_filesystem_root, get_core_harness_dir

logger = logging.getLogger(__name__)


class ToolkitInjectionMiddleware(AgentMiddleware):
    """
    AgentMiddleware that discovers and injects tools from local and nested MCP toolkits.

    This middleware scans for toolkits in `.ciri/toolkits` and recursively in the project root
    (excluding other `.ciri` directories). It manages the lifecycle of these toolkit servers,
    restarting them if their version changes, and uses UV to sync dependencies.
    """

    # Class-level registry to track "running" toolkits and their versions across instances
    _toolkit_versions: Dict[str, str] = {}
    _active_clients: Dict[str, MultiServerMCPClient] = {}

    def __init__(self, scan_root: Optional[Union[str, Path]] = None):
        super().__init__()
        self.root = Path(scan_root) if scan_root else get_default_filesystem_root()
        self.tools = []

        # Track state to avoid unnecessary refreshes
        self._last_toolkit_state = set()

    async def refresh(self, force: bool = False):
        """Discovers and loads tools, awaiting the fetch process."""
        await self._async_refresh_tools(force=force)

    def _refresh_tools(self, force: bool = False):
        """Discover and load tools if toolkits have changed (sync wrapper)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, we can't easily wait for it synchronously
                # but we can schedule it.
                asyncio.ensure_future(self._async_refresh_tools(force=force))
            else:
                loop.run_until_complete(self._async_refresh_tools(force=force))
        except Exception as e:
            logger.error(f"Error refreshing toolkits (sync): {e}")

    async def _async_refresh_tools(self, force: bool = False):
        """Discover and load tools if toolkits have changed (async)."""
        try:
            # 1. Discover toolkit directories
            toolkit_dirs = self._discover_toolkits(self.root)

            # 2. Check current state (paths + versions) to see if update is needed
            current_state = set()
            for tk_dir in toolkit_dirs:
                try:
                    # We read version to detect updates
                    if (tk_dir / "pyproject.toml").exists():
                        with open(tk_dir / "pyproject.toml", "rb") as f:
                            data = tomllib.load(f)
                            version = data.get("project", {}).get("version", "0.1.0")
                    else:
                        with open(tk_dir / "package.json", "r") as f:
                            data = json.load(f)
                            version = data.get("version", "0.1.0")
                    current_state.add((str(tk_dir.resolve()), version))
                except Exception:
                    current_state.add((str(tk_dir.resolve()), "unknown"))

            if not force and current_state == self._last_toolkit_state:
                logger.debug("Toolkits unchanged, skipping refresh.")
                return

            logger.info("Toolkits changed or updated. Refreshing tools...")

            # 3. Manage servers (sync dependencies and handle restarts on version change)
            self._sync_and_manage_servers(toolkit_dirs)

            # 4. Initialize tools using MultiServerMCPClient (await it)
            await self._init_mcp_tools_async(toolkit_dirs)

            # Update state only after successful init
            self._last_toolkit_state = current_state

        except Exception as e:
            logger.error(f"Error refreshing toolkits (async): {e}")

    def _discover_toolkits(self, root: Path) -> List[Path]:
        """Discover toolkits: core harness first, then project .ciri/toolkits, then recursive scan.

        Ordering:
        1. Core harness: get_core_harness_dir() / "toolkits" / <each-toolkit-dir>
        2. Project harness: all <root>/**/.ciri/toolkits/<each-toolkit-dir>
        3. Recursive project scan: toolkits found directly in the project tree

        De-duplication: by resolved absolute path using an ordered dict with
        first-insertion-wins semantics — core harness toolkits are never displaced.
        """
        # Use an ordered dict to preserve insertion order while de-duplicating
        # (first occurrence wins — core harness entries cannot be overridden)
        unique_toolkits: dict = {}

        # 1. Core harness toolkits
        try:
            core_toolkits_dir = get_core_harness_dir() / "toolkits"
            if core_toolkits_dir.is_dir():
                for tk_dir in sorted(core_toolkits_dir.iterdir()):
                    if tk_dir.is_dir() and (
                        (tk_dir / "pyproject.toml").exists()
                        or (tk_dir / "package.json").exists()
                    ):
                        if self._is_mcp_toolkit(tk_dir):
                            resolved = str(tk_dir.resolve())
                            if resolved not in unique_toolkits:
                                unique_toolkits[resolved] = tk_dir.resolve()
        except Exception as e:
            logger.error(f"Error scanning core harness toolkits: {e}")

        # 2. Project harness: all .ciri/toolkits directories
        try:
            for ciri_dir in root.rglob(".ciri"):
                # Ensure we are not inside another .ciri folder
                if ciri_dir.is_dir() and not any(
                    p.name == ".ciri" for p in ciri_dir.parents if p != root
                ):
                    toolkits_dir = ciri_dir / "toolkits"
                    if toolkits_dir.is_dir():
                        for tk_dir in toolkits_dir.iterdir():
                            if tk_dir.is_dir() and (
                                (tk_dir / "pyproject.toml").exists()
                                or (tk_dir / "package.json").exists()
                            ):
                                if self._is_mcp_toolkit(tk_dir):
                                    resolved = str(tk_dir.resolve())
                                    if resolved not in unique_toolkits:
                                        unique_toolkits[resolved] = tk_dir.resolve()
        except Exception as e:
            logger.error(f"Error scanning for toolkits in .ciri directories: {e}")

        # 3. Recursive scan of project tree (excluding .ciri directories)
        try:
            for item in root.iterdir():
                if item.is_dir() and item.name != ".ciri":
                    for tk in self._recursive_toolkit_discovery(item):
                        resolved = str(tk.resolve())
                        if resolved not in unique_toolkits:
                            unique_toolkits[resolved] = tk
        except Exception as e:
            logger.error(f"Error during recursive toolkit discovery: {e}")

        return list(unique_toolkits.values())

    def _recursive_toolkit_discovery(self, path: Path) -> List[Path]:
        """Recursively find toolkits, skipping .ciri directories."""
        toolkits = []
        try:
            # Check if this directory is a toolkit
            if (path / "pyproject.toml").exists() or (path / "package.json").exists():
                if self._is_mcp_toolkit(path):
                    toolkits.append(path.resolve())

            # Recurse into subdirectories
            for item in path.iterdir():
                if item.is_dir() and item.name != ".ciri":
                    toolkits.extend(self._recursive_toolkit_discovery(item))
        except (PermissionError, Exception):
            pass
        return toolkits

    def _is_mcp_toolkit(self, path: Path) -> bool:
        """Check if a project is an MCP toolkit (Python or TypeScript)."""
        # Check Python (FastMCP)
        if (path / "pyproject.toml").exists():
            try:
                with open(path / "pyproject.toml", "rb") as f:
                    data = tomllib.load(f)
                    deps = data.get("project", {}).get("dependencies", [])
                    if any("fastmcp" in dep.lower() for dep in deps):
                        # Verify structure
                        if (path / "src" / "main.py").exists():
                            return True
            except Exception:
                pass

        # Check TypeScript (MCP SDK)
        if (path / "package.json").exists():
            try:
                with open(path / "package.json", "r") as f:
                    data = json.load(f)
                    deps = data.get("dependencies", {})
                    if "@modelcontextprotocol/sdk" in deps:
                        # Verify structure or blindly accept?
                        # Let's check for src/index.ts or dist/index.js as common patterns
                        # But user might configure main differently.
                        # Just checking dependency is safest for now,
                        # assuming build process handles the rest.
                        return True
            except Exception:
                pass

        return False

    def _sync_and_manage_servers(self, toolkit_dirs: List[Path]):
        """Ensure dependencies are synced and track version changes for restarts."""
        for tk_dir in toolkit_dirs:
            tk_path = str(tk_dir)
            try:
                version = "0.0.0"
                is_python = (tk_dir / "pyproject.toml").exists()

                if is_python:
                    with open(tk_dir / "pyproject.toml", "rb") as f:
                        data = tomllib.load(f)
                        version = data.get("project", {}).get("version", "0.1.0")
                else:
                    # TypeScript
                    with open(tk_dir / "package.json", "r") as f:
                        data = json.load(f)
                        version = data.get("version", "0.1.0")

                # Check if we need to sync/restart
                if (
                    tk_path not in self._toolkit_versions
                    or self._toolkit_versions[tk_path] != version
                ):
                    logger.info(
                        f"Toolkit {tk_dir.name} version changed or new (v{version}). Syncing..."
                    )

                    if is_python:
                        # Install & sync dependencies
                        subprocess.run(
                            ["uv", "sync"], cwd=tk_dir, check=True, capture_output=True
                        )
                    else:
                        # TypeScript: npm install && npm run build
                        # We use 'npm ci' if package-lock.json exists for clean install, else 'npm install'
                        cmd = (
                            ["npm", "ci"]
                            if (tk_dir / "package-lock.json").exists()
                            else ["npm", "install"]
                        )
                        subprocess.run(cmd, cwd=tk_dir, check=True, capture_output=True)

                        # Build if script exists
                        # Check package.json for build script
                        with open(tk_dir / "package.json", "r") as f:
                            pkg_data = json.load(f)
                            if "build" in pkg_data.get("scripts", {}):
                                subprocess.run(
                                    ["npm", "run", "build"],
                                    cwd=tk_dir,
                                    check=True,
                                    capture_output=True,
                                )

                    # Update version tracking
                    self._toolkit_versions[tk_path] = version
            except Exception as e:
                logger.error(f"Failed to manage toolkit server {tk_dir}: {e}")

    def _init_mcp_tools(self, toolkit_dirs: List[Path]):
        """Sync wrapper for _init_mcp_tools_async."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self._init_mcp_tools_async(toolkit_dirs))
            else:
                loop.run_until_complete(self._init_mcp_tools_async(toolkit_dirs))
        except Exception as e:
            logger.error(f"Failed to initialize MCP tools (sync): {e}")

    async def _init_mcp_tools_async(self, toolkit_dirs: List[Path]):
        """Initialize MultiServerMCPClient and fetch tools from discovered toolkits."""
        if not toolkit_dirs:
            return

        # Sanitize environment to avoid uv warnings and other issues
        safe_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("VIRTUAL_ENV", "PYTHONPATH")
        }

        connections = {}
        for tk_dir in toolkit_dirs:
            if (tk_dir / "pyproject.toml").exists() and (
                tk_dir / "src" / "main.py"
            ).exists():
                # Python / FastMCP
                connections[tk_dir.name] = {
                    "command": "uv",
                    "args": ["run", "src/main.py"],
                    "cwd": str(tk_dir),
                    "transport": "stdio",
                    "env": {
                        "FASTMCP_LOG_LEVEL": "WARNING",
                        "FASTMCP_BANNER": "0",
                        **safe_env,
                    },
                }
            elif (tk_dir / "package.json").exists():
                # TypeScript / Node
                entry_point = "dist/index.js"  # default
                try:
                    with open(tk_dir / "package.json", "r") as f:
                        data = json.load(f)
                        if "main" in data:
                            entry_point = data["main"]
                except Exception:
                    pass

                connections[tk_dir.name] = {
                    "command": "node",
                    "args": [entry_point],
                    "cwd": str(tk_dir),
                    "transport": "stdio",
                    "env": {
                        "FASTMCP_LOG_LEVEL": "WARNING",
                        "FASTMCP_BANNER": "0",
                        **safe_env,
                    },
                }

        if connections:
            await self._async_fetch_tools(connections)

    async def _async_fetch_tools(self, connections: Dict[str, Any]):
        """Async helper to connect and fetch tools."""
        try:
            client = MultiServerMCPClient(connections=connections)
            # Fetch tools from all connected servers
            all_tools = await client.get_tools()

            # Ensure unique tool names in self.tools
            unique_tools = {}
            for t in all_tools:
                if t.name not in unique_tools:
                    unique_tools[t.name] = t
                else:
                    logger.warning(f"Duplicate tool name '{t.name}' found. Skipping.")

            self.tools = list(unique_tools.values())
            logger.info(
                f"ToolkitInjector: Successfully injected {len(self.tools)} tools"
            )
        except Exception as e:
            logger.error(f"Error fetching tools from MultiServerMCPClient: {e}")

    async def awrap_model_call(self, request, handler):
        """Inject toolkit tools into the model call request (async)."""
        self._refresh_tools()
        self._inject_tools(request)
        return await handler(request)

    def wrap_model_call(self, request, handler):
        """Inject toolkit tools into the model call request (sync)."""
        self._refresh_tools()
        self._inject_tools(request)
        return handler(request)

    def _inject_tools(self, request: Any):
        """Helper to inject discovered tools into the request."""
        if self.tools:
            request_tool_names = {t.name for t in request.tools}
            for tool in self.tools:
                if tool.name not in request_tool_names:
                    request.tools.append(tool)
                    request_tool_names.add(tool.name)
