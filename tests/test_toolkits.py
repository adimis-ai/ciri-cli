import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock, call
from src.middlewares.toolkits import ToolkitInjectionMiddleware


@pytest.fixture
def project_root(tmp_path):
    """Create a mock project structure with toolkits."""
    ciri_dir = tmp_path / ".ciri"
    toolkits_dir = ciri_dir / "toolkits"

    # Python Toolkit
    py_tk_dir = toolkits_dir / "test_toolkit_py"
    py_src_dir = py_tk_dir / "src"
    py_tk_dir.mkdir(parents=True)
    py_src_dir.mkdir(parents=True)

    (py_tk_dir / "pyproject.toml").write_text("""
[project]
name = "test_toolkit_py"
version = "0.1.0"
dependencies = ["fastmcp"]
""")
    (py_src_dir / "main.py").write_text("print('hello')")

    # TypeScript Toolkit
    ts_tk_dir = toolkits_dir / "test_toolkit_ts"
    ts_tk_dir.mkdir(parents=True)

    (ts_tk_dir / "package.json").write_text(
        json.dumps(
            {
                "name": "test_toolkit_ts",
                "version": "0.1.0",
                "dependencies": {"@modelcontextprotocol/sdk": "^0.6.0"},
                "scripts": {"build": "tsc"},
                "main": "dist/index.js",
            }
        )
    )

    return tmp_path


class MockTool:
    def __init__(self, name):
        self.name = name


class MockRequest:
    def __init__(self, tools=None):
        self.tools = tools or []


@pytest.mark.asyncio
async def test_toolkit_discovery_and_injection(project_root):
    # Mock subprocess.run to avoid actual uv sync / npm install
    # Mock MultiServerMCPClient to avoid actual MCP server startup
    with (
        patch("subprocess.run") as mock_run,
        patch("src.middlewares.toolkits.MultiServerMCPClient") as mock_client_class,
    ):

        # Setup mock client
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(
            return_value=[MockTool("echo"), MockTool("add")]
        )
        mock_client_class.return_value = mock_client

        # Initialize middleware
        middleware = ToolkitInjectionMiddleware(scan_root=project_root)

        # Verify discovery of BOTH toolkits
        assert len(middleware._toolkit_versions) == 2

        py_tk_path = project_root / ".ciri" / "toolkits" / "test_toolkit_py"
        ts_tk_path = project_root / ".ciri" / "toolkits" / "test_toolkit_ts"

        assert str(py_tk_path) in middleware._toolkit_versions
        assert str(ts_tk_path) in middleware._toolkit_versions

        # Verify sync/install calls
        # We expect:
        # 1. uv sync (for python)
        # 2. npm install (for ts)
        # 3. npm run build (for ts)

        assert mock_run.call_count >= 3

        calls = mock_run.call_args_list
        commands = [c[0][0] for c in calls]

        uv_sync_called = False
        npm_install_called = False
        npm_build_called = False

        for cmd in commands:
            if cmd == ["uv", "sync"]:
                uv_sync_called = True
            if cmd == ["npm", "install"]:
                npm_install_called = True
            if cmd == ["npm", "run", "build"]:
                npm_build_called = True

        assert uv_sync_called, "uv sync should be called"
        assert npm_install_called, "npm install should be called"
        assert npm_build_called, "npm run build should be called"

        # Verify tools are fetched
        # Give some time for the background task to complete
        for _ in range(10):
            if middleware.tools:
                break
            await asyncio.sleep(0.1)

        assert len(middleware.tools) == 2
        assert {t.name for t in middleware.tools} == {"echo", "add"}

        # Test injection
        request = MockRequest(tools=[MockTool("existing_tool")])
        middleware._inject_tools(request)

        assert len(request.tools) == 3
        assert {t.name for t in request.tools} == {"existing_tool", "echo", "add"}

        # Verify connection args passed to client
        _, kwargs = mock_client_class.call_args
        connections = kwargs.get("connections", {})

        assert "test_toolkit_py" in connections
        assert "test_toolkit_ts" in connections

        assert connections["test_toolkit_py"]["command"] == "uv"
        assert connections["test_toolkit_ts"]["command"] == "node"
        assert connections["test_toolkit_ts"]["args"] == ["dist/index.js"]


@pytest.mark.asyncio
async def test_toolkit_version_change(project_root):
    with (
        patch("subprocess.run") as mock_run,
        patch("src.middlewares.toolkits.MultiServerMCPClient") as mock_client_class,
    ):

        # Setup mock client
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(return_value=[])
        mock_client_class.return_value = mock_client

        # Initial run
        middleware = ToolkitInjectionMiddleware(scan_root=project_root)

        # Reset mock calls (but versions are tracked in class var, so we need to be careful of leakage if we don't clear it or if it persists)
        # The class variable _toolkit_versions persists across instances.
        mock_run.reset_mock()

        # Update version in pyproject.toml
        py_tk_path = project_root / ".ciri" / "toolkits" / "test_toolkit_py"
        (py_tk_path / "pyproject.toml").write_text("""
[project]
name = "test_toolkit_py"
version = "0.2.0"
dependencies = ["fastmcp"]
""")

        # Second run
        middleware2 = ToolkitInjectionMiddleware(scan_root=project_root)

        # Verify sync was called again for the updated toolkit
        # We should only see calls for the updated toolkit.
        # However, since test execution order mock_run might be fresh?
        # If test_toolkit_ts is still there and version unchanged, it should NOT trigger sync.

        assert mock_run.call_count == 1
        assert mock_run.call_args[0][0] == ["uv", "sync"]
        assert middleware2._toolkit_versions[str(py_tk_path.resolve())] == "0.2.0"
