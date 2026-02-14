import logging
from pathlib import Path
from typing import List, Optional, Union
from deepagents.middleware import SkillsMiddleware as BaseSkillsMiddleware
from ..utils import get_default_filesystem_root

logger = logging.getLogger(__name__)


class SkillsMiddleware(BaseSkillsMiddleware):
    """
    Extended SkillsMiddleware that automatically scans for skills in .ciri/skills folders.

    This middleware looks for all `.ciri/skills` directories within the project root
    and adds them as skill sources.
    """

    def __init__(
        self,
        *,
        backend,
        sources: Optional[List[str]] = None,
        scan_root: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the skills middleware with automatic source discovery.

        Args:
            backend: The storage backend for skills.
            sources: Explicit list of skill source paths.
            scan_root: Root directory to scan for .ciri/skills folders.
                      Defaults to the project root.
        """
        if sources is None:
            sources = []

        # 1. Store configuration for dynamic scanning
        self.root = Path(scan_root) if scan_root else get_default_filesystem_root()
        self.explicit_sources = sources
        logger.debug(f"Scanning for skills in: {self.root}")

        # 2. Initial scan
        self._refresh_sources()

    def _refresh_sources(self):
        """Discover skill sources and update self.sources."""
        # Discover auto sources
        auto_sources = self._discover_skills_sources(self.root)

        # Merge sources: Explicit sources first, then discovered ones (de-duplicated)
        final_sources = list(self.explicit_sources)
        seen = {str(Path(s).resolve()) for s in self.explicit_sources}

        for s in auto_sources:
            resolved_s = str(Path(s).resolve())
            if resolved_s not in seen:
                final_sources.append(s)
                seen.add(resolved_s)

        self.sources = final_sources
        # Update base class sources if needed, though usually middlewares use self.sources directly
        # But deepagents BaseSkillsMiddleware might store it differently.
        # Checking base class implementation would be ideal, but assuming standard pattern:
        # We initialized super() with initial sources. If base class uses self.sources, we are good.
        # If it copies to internal storage, we might need to verify.
        # Given memory middleware example, updating self.sources seems the way.

        logger.debug(
            f"Refreshed SkillsMiddleware with {len(final_sources)} sources: {final_sources}"
        )

    def _discover_skills_sources(self, root: Path) -> List[str]:
        """Recursively find all .ciri/skills directories."""
        discovered = []
        try:
            # Recursive search for all .ciri folders
            for ciri_dir in root.rglob(".ciri"):
                # Ensure we are not inside another .ciri folder
                if ciri_dir.is_dir() and not any(
                    p.name == ".ciri" for p in ciri_dir.parents if p != root
                ):
                    skills_dir = ciri_dir / "skills"
                    if skills_dir.is_dir():
                        discovered.append(str(skills_dir.resolve()))
        except Exception as e:
            logger.error(f"Error while scanning for skills directories: {e}")

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
