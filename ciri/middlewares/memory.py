import logging
from pathlib import Path
from typing import List, Optional, Union, Any
from deepagents.middleware import MemoryMiddleware as BaseMemoryMiddleware
from ..utils import get_default_filesystem_root, get_core_harness_dir

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
        """Find all memory .md files: core harness first, then project .ciri/memory dirs.

        Ordering:
        1. Core harness: get_core_harness_dir() / "memory" / "*.md"
        2. Project harness: all <root>/**/.ciri/memory/*.md

        Both sets of files are loaded additively — memory is not de-duplicated
        by name since different files contain different context.
        De-duplication is by resolved path only (handled in _refresh_sources()).
        """
        discovered = []

        # 1. Core harness memory files (global / cross-project context)
        try:
            core_memory_dir = get_core_harness_dir() / "memory"
            if core_memory_dir.is_dir():
                for md_file in sorted(core_memory_dir.glob("*.md")):
                    discovered.append(str(md_file.resolve()))
        except Exception as e:
            logger.error(f"Error accessing core harness memory directory: {e}")

        # 2. Project harness memory files (workspace-specific context)
        try:
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

        return discovered  # NOT sorted — core harness files come first

    async def awrap_model_call(self, request, handler):
        self._refresh_sources()
        return await handler(request)

    def wrap_model_call(self, request, handler):
        self._refresh_sources()
        return handler(request)
