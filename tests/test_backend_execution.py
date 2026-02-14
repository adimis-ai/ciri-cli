
import pytest
import asyncio
from src.backend import CiriBackend

@pytest.mark.asyncio
async def test_backend_execution_sync():
    backend = CiriBackend()
    # Test simple echo command
    result = backend.execute("echo 'hello world'")
    assert result.exit_code == 0
    assert "hello world" in result.output.strip()

@pytest.mark.asyncio
async def test_backend_execution_async():
    backend = CiriBackend()
    # Test simple echo command asynchronously
    result = await backend.aexecute("echo 'hello async'")
    assert result.exit_code == 0
    assert "hello async" in result.output.strip()

@pytest.mark.asyncio
async def test_backend_execution_error():
    backend = CiriBackend()
    # Test invalid command
    result = backend.execute("nonexistentcommand")
    assert result.exit_code != 0
    assert "not found" in result.output.lower() or "error" in result.output.lower()
