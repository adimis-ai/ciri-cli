import logging
from pathlib import Path
from typing import List, Optional, Union, Any
from deepagents.middleware import MemoryMiddleware as BaseMemoryMiddleware
from ..utils import get_default_filesystem_root

logger = logging.getLogger(__name__)


class MemoryMiddleware(BaseMemoryMiddleware):
    """
    Extended MemoryMiddleware that automatically scans for memory files in .ciri/memory folders.

    This middleware looks for all `.ciri/memory/*.md` files within the project root
    and adds them as memory sources.
    """

    def __init__(
        self,
        *,
        backend,
        sources: Optional[List[str]] = None,
        scan_root: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the memory middleware with automatic source discovery.

        Args:
            backend: The storage backend for memory.
            sources: Explicit list of memory source paths.
            scan_root: Root directory to scan for .ciri/memory folders.
                      Defaults to the project root.
        """
        if sources is None:
            sources = []

        # 1. Store configuration for dynamic scanning
        self.root = Path(scan_root) if scan_root else get_default_filesystem_root()
        self.explicit_sources = sources
        logger.debug(f"Scanning for memory in: {self.root}")

        # 2. Initial scan
        self._refresh_sources()
        super().__init__(backend=backend, sources=self.sources)

    def _refresh_sources(self):
        """Discover memory sources and update self.sources."""
        # Discover auto sources
        auto_sources = self._discover_memory_sources(self.root)

        # Merge sources: Explicit sources first, then discovered ones (de-duplicated)
        final_sources = list(self.explicit_sources)
        seen = {str(Path(s).resolve()) for s in self.explicit_sources}

        for s in auto_sources:
            resolved_s = str(Path(s).resolve())
            if resolved_s not in seen:
                final_sources.append(s)
                seen.add(resolved_s)

        self.sources = final_sources

    def _discover_memory_sources(self, root: Path) -> List[str]:
        """Recursively find all .ciri/memory/*.md files."""
        discovered = []
        try:
            # Recursive search for all .ciri folders
            for ciri_dir in root.rglob(".ciri"):
                # Ensure we are not inside another .ciri folder
                if ciri_dir.is_dir() and not any(
                    p.name == ".ciri" for p in ciri_dir.parents if p != root
                ):
                    memory_dir = ciri_dir / "memory"
                    if memory_dir.is_dir():
                        for md_file in memory_dir.glob("*.md"):
                            discovered.append(str(md_file.resolve()))
        except Exception as e:
            logger.error(f"Error while scanning for memory files: {e}")

        return sorted(discovered)

    async def awrap_model_call(self, request, handler):
        self._refresh_sources()
        return await handler(request)

    def wrap_model_call(self, request, handler):
        self._refresh_sources()
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        self._refresh_sources()
        return await handler(request)

    def wrap_tool_call(self, request, handler):
        self._refresh_sources()
        return handler(request)
