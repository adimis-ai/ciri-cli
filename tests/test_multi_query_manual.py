import asyncio
import json
from unittest.mock import MagicMock, patch
from ciri.toolkit.human_follow_up_tool import follow_up_with_human
from ciri.__main__ import _handle_follow_up, _render_follow_up_interrupt


async def test_multi_query_tool():
    print("Testing multi-query tool...")

    # Mock interrupt to return responses
    with patch(
        "src.toolkit.human_follow_up_tool.interrupt",
        return_value=["Answer 1", "Answer 2"],
    ) as mock_interrupt:
        queries = [
            {"question": "What is your name?", "options": ["Alice", "Bob"]},
            {"question": "How old are you?"},
        ]

        result = follow_up_with_human.invoke(
            {
                "args": {"queries": queries},
                "name": "follow_up_with_human",
                "type": "tool_call",
                "id": "test_id",
            }
        )

        # Verify interrupt was called with correct data
        mock_interrupt.assert_called_once_with(
            {"type": "human_follow_up", "queries": queries}
        )

        # Verify ToolMessage content
        msg = result.update["messages"][0]
        print(f"ToolMessage content:\n{msg.content}")
        assert "Question: What is your name?" in msg.content
        assert "Response: Answer 1" in msg.content
        assert "Question: How old are you?" in msg.content
        assert "Response: Answer 2" in msg.content
        assert msg.tool_call_id == "test_id"


async def test_cli_rendering():
    print("\nTesting CLI rendering...")
    val = {"queries": [{"question": "Q1", "options": ["O1", "O2"]}, {"question": "Q2"}]}

    # Just call it and see if it crashes
    with patch("src.__main__.console.print") as mock_print:
        _render_follow_up_interrupt(val)

        # Verify it printed something for both queries
        print(f"Print calls: {len(mock_print.call_args_list)}")
        assert any("Q1" in str(args) for args, kwargs in mock_print.call_args_list)
        assert any("Q2" in str(args) for args, kwargs in mock_print.call_args_list)


async def test_cli_handling():
    print("\nTesting CLI handling...")
    val = {"queries": [{"question": "Q1", "options": ["O1", "O2"]}, {"question": "Q2"}]}

    controller = MagicMock()
    config = {}
    completer = MagicMock()

    # Mock pt_prompt to return answers
    with (
        patch("src.__main__.pt_prompt", side_effect=["1", "Answer 2"]) as mock_prompt,
        patch("src.__main__.run_graph") as mock_run_graph,
    ):

        await _handle_follow_up(controller, val, config, completer)

        # Verify run_graph was called with correct responses
        # Question 1 option 1 is "O1"
        mock_run_graph.assert_called_once()
        command = mock_run_graph.call_args[0][1]
        print(f"Resume value: {command.resume}")
        assert command.resume == ["O1", "Answer 2"]


if __name__ == "__main__":
    asyncio.run(test_multi_query_tool())
    asyncio.run(test_cli_rendering())
    asyncio.run(test_cli_handling())
    print("\nAll tests passed!")
