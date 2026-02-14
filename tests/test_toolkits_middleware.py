import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY
from src.middlewares.tookits import ToolkitInjectorMiddleware


@pytest.fixture
def project_root(tmp_path):
    """Create a mock project structure with toolkits."""
    ciri_dir = tmp_path / ".ciri"
    toolkits_dir = ciri_dir / "toolkits"
    toolkits_dir.mkdir(parents=True)

    # Create a toolkit
    tk1 = toolkits_dir / "tk1"
    tk1.mkdir()
    (tk1 / "pyproject.toml").write_text(
        '["project"]\nname="tk1"\nversion="0.1.0"\ndependencies=["fastmcp"]'
    )
    (tk1 / "src").mkdir()
    (tk1 / "src" / "main.py").touch()

    return tmp_path


@pytest.mark.asyncio
async def test_toolkits_dynamic_scanning(project_root):
    # Mock subprocess.run to avoid actual uv sync
    # Mock MultiServerMCPClient to avoid actual connections
    with (
        patch("subprocess.run") as mock_run,
        patch("src.middlewares.tookits.MultiServerMCPClient") as mock_client_cls,
    ):

        # Setup mock client instance
        mock_client = mock_client_cls.return_value
        mock_tool = MagicMock()
        mock_tool.name = "tool1"
        mock_client.get_tools = MagicMock(
            side_effect=lambda: asyncio.sleep(0, result=[mock_tool])
        )

        # Initialize middleware
        middleware = ToolkitInjectorMiddleware(scan_root=project_root)

        # Give the background task (ensure_future) a moment to run
        await asyncio.sleep(0.1)

        # Verify initial init
        mock_run.assert_called()  # Should have synced
        mock_client_cls.assert_called()  # Should have initialized client
        assert len(middleware.tools) == 1
        assert middleware.tools[0].name == "tool1"

        # Reset mocks to verify subsequent calls
        mock_run.reset_mock()
        mock_client_cls.reset_mock()

        # Trigger refresh with NO changes via wrap call
        class MockRequest:
            tools = []

        request = MockRequest()

        async def async_handler(r):
            return r

        await middleware.awrap_model_call(request, async_handler)

        # Should NOT have called sync or init client again (optimization)
        mock_run.assert_not_called()
        mock_client_cls.assert_not_called()

        # Add a NEW toolkit
        tk2 = project_root / ".ciri" / "toolkits" / "tk2"
        tk2.mkdir()
        (tk2 / "pyproject.toml").write_text(
            '["project"]\nname="tk2"\nversion="0.1.0"\ndependencies=["fastmcp"]'
        )
        (tk2 / "src").mkdir()
        (tk2 / "src" / "main.py").touch()

        # Update mock client to return tools for both
        mock_tool2 = MagicMock()
        mock_tool2.name = "tool2"
        mock_client = mock_client_cls.return_value  # New instance will be created
        mock_client.get_tools = MagicMock(
            side_effect=lambda: asyncio.sleep(0, result=[mock_tool, mock_tool2])
        )

        # Trigger refresh via wrap call
        await middleware.awrap_model_call(request, async_handler)

        # Allow background task to run
        await asyncio.sleep(0.1)

        # Should HAVE called sync (for new toolkit) and init client
        mock_run.assert_called()
        mock_client_cls.assert_called()
        assert len(middleware.tools) == 2

        # Reset again
        mock_run.reset_mock()
        mock_client_cls.reset_mock()

        # Update VERSION of existing toolkit
        tk1 = project_root / ".ciri" / "toolkits" / "tk1"
        (tk1 / "pyproject.toml").write_text(
            '["project"]\nname="tk1"\nversion="0.2.0"\ndependencies=["fastmcp"]'
        )

        # Trigger refresh
        await middleware.awrap_model_call(request, async_handler)

        # Allow background task to run
        await asyncio.sleep(0.1)

        # Should HAVE called sync (version changed) and init client
        mock_run.assert_called()
        mock_client_cls.assert_called()
