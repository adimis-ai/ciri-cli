
import subprocess
import asyncio
from deepagents.backends import FilesystemBackend
from deepagents.backends.sandbox import SandboxBackendProtocol, ExecuteResponse

from .utils import get_default_filesystem_root



class CiriBackend(SandboxBackendProtocol, FilesystemBackend):
    def __init__(self, root_dir=None, virtual_mode=False, max_file_size_mb=10):
        if not root_dir:
            root_dir = get_default_filesystem_root()
        super().__init__(root_dir, virtual_mode, max_file_size_mb)

    @property
    def id(self) -> str:
        return "local"

    def execute(self, command: str) -> ExecuteResponse:
        try:
            # We run the command in the root_dir
            process = subprocess.run(
                command,
                shell=True,
                cwd=self.cwd,
                capture_output=True,
                text=True,
                check=False
            )
            return ExecuteResponse(
                output=process.stdout + process.stderr,
                exit_code=process.returncode,
                truncated=False
            )
        except Exception as e:
            return ExecuteResponse(
                output=f"Error executing command: {e}",
                exit_code=1,
                truncated=False
            )

    async def aexecute(self, command: str) -> ExecuteResponse:
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=self.cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            output = stdout.decode() + stderr.decode()
            return ExecuteResponse(
                output=output,
                exit_code=process.returncode,
                truncated=False
            )
        except Exception as e:
            return ExecuteResponse(
                output=f"Error executing command: {e}",
                exit_code=1,
                truncated=False
            )
