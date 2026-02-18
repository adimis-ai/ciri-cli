import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch
from ciri.middlewares.subagents import SubAgentMiddleware


@pytest.fixture
def project_root(tmp_path):
    """Create a mock project structure with subagents."""
    ciri_dir = tmp_path / ".ciri"
    subagents_dir = ciri_dir / "subagents"
    subagents_dir.mkdir(parents=True)

    # Create a subagent file
    agent1 = subagents_dir / "agent1.yaml"
    agent1.write_text(
        "name: agent1\ndescription: test agent\nsystem_prompt: prompt\nmodel: model"
    )

    return tmp_path


@pytest.mark.asyncio
async def test_subagents_dynamic_scanning(project_root):
    # Mock the base class __init__ to avoid LangChain validation
    with patch(
        "src.middlewares.subagents.BaseSubAgentMiddleware.__init__", return_value=None
    ) as mock_base_init:
        # Initialize middleware
        mock_model = "test-model"
        middleware = SubAgentMiddleware(
            scan_root=project_root,
            default_model=mock_model,
            default_tools=[],
            default_middleware=[],
            default_interrupt_on=[],
            subagents=[],
            system_prompt="sys",
            general_purpose_agent="gen",
            task_description="task",
        )

        # Verify base init was called
        mock_base_init.assert_called_once()

        # Manually set attributes that base init would have set if we need them
        # But our test checks middleware.subagents which IS set by our init BEFORE calling super

    # Verify initial discovery
    # Note: Our init sets self.subagents before calling super, so it should be populated
    # middleware.subagents likely contains dicts based on previous errors
    discovered = []
    for s in middleware.subagents:
        if isinstance(s, dict):
            discovered.append(s["name"])
        else:
            discovered.append(s.name)
    discovered = sorted(discovered)
    assert "agent1" in discovered

    # Add a new subagent file
    agent2 = project_root / ".ciri" / "subagents" / "agent2.yaml"
    agent2.write_text(
        "name: agent2\ndescription: test agent 2\nsystem_prompt: prompt 2\nmodel: model"
    )

    # Trigger scan via wrap call
    # We need to mock _wrap_model_call_common because it calls super().wrap_model_call/awrap_model_call
    # which might fail if base init didn't run properly.
    # However, we only care about _refresh_subagents being called.

    # We can mock the super().awrap_model_call using patch.object maybe?
    # Or just mock the entire awrap_model_call of the BASE class.

    # We need AsyncMock for async method
    from unittest.mock import AsyncMock

    with patch(
        "src.middlewares.subagents.BaseSubAgentMiddleware.awrap_model_call",
        new_callable=AsyncMock,
    ) as mock_base_awrap:

        class MockRequest:
            tools = []

        request = MockRequest()

        async def async_handler(r):
            return r

        await middleware.awrap_model_call(request, async_handler)

    # Verify new subagent is discovered
    discovered_new = []
    for s in middleware.subagents:
        if isinstance(s, dict):
            discovered_new.append(s["name"])
        else:
            discovered_new.append(s.name)
    discovered_new = sorted(discovered_new)
    assert "agent1" in discovered_new
    assert "agent2" in discovered_new
