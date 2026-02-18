import os
import shutil
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from ciri.toolkit.script_executor_tool import build_script_executor_tool


class TestScriptExecutor(unittest.TestCase):
    def setUp(self):
        self.tool = build_script_executor_tool()
        self.test_dir = Path("/tmp/ciri_scripts_test")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    @patch("src.toolkit.script_executor_tool.interrupt")
    def test_python_execution(self, mock_interrupt):
        # Mock approval
        mock_interrupt.return_value = {"status": "approved"}

        script = "print('Hello from Python'); import os; print(f'Output dir: {os.environ.get(\"CIRI_OUTPUT_DIR\")}')"
        # Tool uses InjectedToolCallId, so we must invoke it with a full ToolCall if we use .invoke()
        # or just call the function directly if we have access to it.
        # Here we use the .run() method or pass a dict that looks like a ToolCall
        result = self.tool.invoke(
            {
                "type": "tool_call",
                "name": "execute_sandboxed_script",
                "args": {
                    "script_content": script,
                    "language": "python",
                    "cleanup": True,
                },
                "id": "test_call_id",
            }
        )

        print("\nPython Result:\n", result.content)
        self.assertIn("Hello from Python", result.content)
        self.assertIn("Output dir:", result.content)
        self.assertIn("Virtual environment created.", result.content)
        self.assertIn("Cleanup: temporary environment removed.", result.content)

    @patch("src.toolkit.script_executor_tool.interrupt")
    def test_javascript_execution(self, mock_interrupt):
        # Mock approval
        mock_interrupt.return_value = {"status": "approved"}

        script = "console.log('Hello from JS'); console.log('Output dir: ' + process.env.CIRI_OUTPUT_DIR);"
        result = self.tool.invoke(
            {
                "type": "tool_call",
                "name": "execute_sandboxed_script",
                "args": {
                    "script_content": script,
                    "language": "javascript",
                    "cleanup": True,
                },
                "id": "test_call_id",
            }
        )

        print("\nJS Result:\n", result.content)
        self.assertIn("Hello from JS", result.content)
        self.assertIn("Output dir:", result.content)
        self.assertIn("Cleanup: temporary environment removed.", result.content)

    @patch("src.toolkit.script_executor_tool.interrupt")
    def test_python_dependencies(self, mock_interrupt):
        # Mock approval
        mock_interrupt.return_value = {"status": "approved"}

        script = "import requests; print('Requests imported successfully')"
        result = self.tool.invoke(
            {
                "type": "tool_call",
                "name": "execute_sandboxed_script",
                "args": {
                    "script_content": script,
                    "language": "python",
                    "dependencies": ["requests"],
                    "cleanup": True,
                },
                "id": "test_call_id",
            }
        )

        print("\nPython Deps Result:\n", result.content)
        self.assertIn("Requests imported successfully", result.content)
        self.assertIn("Installed: requests", result.content)

    @patch("src.toolkit.script_executor_tool.interrupt")
    def test_cleanup_off(self, mock_interrupt):
        mock_interrupt.return_value = {"status": "approved"}

        script = "print('No cleanup')"

        result = self.tool.invoke(
            {
                "type": "tool_call",
                "name": "execute_sandboxed_script",
                "args": {
                    "script_content": script,
                    "language": "python",
                    "cleanup": False,
                },
                "id": "test_call_id",
            }
        )

        self.assertNotIn("Cleanup: temporary environment removed.", result.content)
        # Note: actually verifying the folder exists is tricky without capturing the ID
        # but the lack of cleanup message is a good indicator.


if __name__ == "__main__":
    unittest.main()
