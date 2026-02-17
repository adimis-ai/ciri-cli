import subprocess
import asyncio
from typing import Optional, Callable
from deepagents.backends import FilesystemBackend
from deepagents.backends.sandbox import SandboxBackendProtocol, ExecuteResponse

from .utils import get_default_filesystem_root


class CiriBackend(SandboxBackendProtocol, FilesystemBackend):
    def __init__(
        self,
        root_dir=None,
        virtual_mode=False,
        max_file_size_mb=10,
        output_callback: Optional[Callable[[str], None]] = None,
    ):
        if not root_dir:
            root_dir = get_default_filesystem_root()
        super().__init__(root_dir, virtual_mode, max_file_size_mb)
        self.output_callback = output_callback
        self._last_streamed = False

    @property
    def id(self) -> str:
        return "local"

    def execute(self, command: str) -> ExecuteResponse:
        self._last_streamed = False
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=self.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            output_lines = []
            for line in process.stdout:
                output_lines.append(line)
                if self.output_callback:
                    self._last_streamed = True
                    self.output_callback(line.rstrip("\n"))
            process.wait()
            return ExecuteResponse(
                output="".join(output_lines),
                exit_code=process.returncode,
                truncated=False,
            )
        except Exception as e:
            return ExecuteResponse(
                output=f"Error executing command: {e}", exit_code=1, truncated=False
            )

    async def aexecute(self, command: str) -> ExecuteResponse:
        self._last_streamed = False
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=self.cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            output_lines = []
            async for raw_line in process.stdout:
                line = raw_line.decode()
                output_lines.append(line)
                if self.output_callback:
                    self._last_streamed = True
                    self.output_callback(line.rstrip("\n"))
            await process.wait()
            return ExecuteResponse(
                output="".join(output_lines),
                exit_code=process.returncode,
                truncated=False,
            )
        except Exception as e:
            return ExecuteResponse(
                output=f"Error executing command: {e}", exit_code=1, truncated=False
            )
