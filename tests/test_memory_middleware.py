import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.middlewares.memory import MemoryMiddleware


@pytest.fixture
def project_root(tmp_path):
    """Create a mock project structure with memory files."""
    ciri_dir1 = tmp_path / ".ciri"
    memory_dir1 = ciri_dir1 / "memory"
    memory_dir1.mkdir(parents=True)
    (memory_dir1 / "mem1.md").write_text("memory 1")
    (memory_dir1 / "mem2.md").write_text("memory 2")

    # Nested project with its own .ciri
    sub_project = tmp_path / "sub_project"
    ciri_dir2 = sub_project / ".ciri"
    memory_dir2 = ciri_dir2 / "memory"
    memory_dir2.mkdir(parents=True)
    (memory_dir2 / "mem3.md").write_text("memory 3")

    # Deeply nested .ciri that should be ignored if inside another .ciri (per discovery logic)
    # But wait, the logic says: not any(p.name == ".ciri" for p in ciri_dir.parents if p != root)
    # This means if root is tmp_path, then sub_project/.ciri is NOT inside another .ciri relative to root.
    # But sub_project/.ciri/.nested/.ciri WOULD be ignored.

    return tmp_path


class MockRequest:
    def __init__(self, memory_sources=None):
        self.memory_sources = memory_sources if memory_sources is not None else []


@pytest.mark.asyncio
async def test_memory_discovery(project_root):
    # Initialize middleware
    mock_backend = MagicMock()
    middleware = MemoryMiddleware(backend=mock_backend, scan_root=project_root)

    # Verify discovery
    discovered = sorted([Path(s).name for s in middleware.sources])
    assert discovered == ["mem1.md", "mem2.md", "mem3.md"]


@pytest.mark.asyncio
async def test_memory_manual_sources(project_root):
    mock_backend = MagicMock()
    manual_file = project_root / "manual.md"
    manual_file.write_text("manual")

    middleware = MemoryMiddleware(
        backend=mock_backend, sources=[str(manual_file)], scan_root=project_root
    )

    # Should have manual + discovered
    assert len(middleware.sources) == 4
    names = {Path(s).name for s in middleware.sources}
    assert "manual.md" in names
    assert "mem1.md" in names


@pytest.mark.asyncio
async def test_memory_dynamic_scanning(project_root):
    # Initialize middleware
    mock_backend = MagicMock()
    middleware = MemoryMiddleware(backend=mock_backend, scan_root=project_root)

    # Verify initial discovery
    discovered = sorted([Path(s).name for s in middleware.sources])
    assert discovered == ["mem1.md", "mem2.md", "mem3.md"]

    # Add a new memory file
    new_memory = project_root / ".ciri" / "memory" / "mem_new.md"
    new_memory.write_text("new memory")

    # Trigger scan via wrap call
    class MockRequest:
        pass

    request = MockRequest()

    async def async_handler(r):
        return r

    await middleware.awrap_model_call(request, async_handler)

    # Verify new file is discovered
    discovered_new = sorted([Path(s).name for s in middleware.sources])
    assert discovered_new == ["mem1.md", "mem2.md", "mem3.md", "mem_new.md"]
