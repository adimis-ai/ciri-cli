import shutil
import logging
import os
from pathlib import Path
from typing import List, Optional, Union
from deepagents.middleware import SkillsMiddleware as BaseSkillsMiddleware
from ..utils import get_default_filesystem_root, get_core_harness_dir

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

        # 2. Bootstrap default skills before scanning
        self._bootstrap_default_skills()

        logger.debug(f"Scanning for skills in: {self.root}")

        # 3. Initial scan
        self._refresh_sources()
        super().__init__(backend=backend, sources=self.sources)

    def _bootstrap_default_skills(self):
        """Copy default skills from src/skills/ into the core harness skills directory.

        Target is get_core_harness_dir() / "skills" (the OS-level persistent
        directory) so that default skills are available globally, not just in
        the current project's .ciri/skills/.
        The copy is skipped for skills that already exist (first-run-only bootstrap).
        """
        default_skills_source = Path(__file__).parent.parent / "skills"

        # Only proceed if source skills exist
        if not default_skills_source.exists():
            logger.warning(
                f"Default skills source not found at {default_skills_source}"
            )
            return

        # Target: OS-level core harness, not the project-local .ciri/
        target_skills_dir = get_core_harness_dir() / "skills"
        # get_core_harness_dir() already ensures the directory exists

        try:
            for item in default_skills_source.iterdir():
                if item.is_dir():
                    skill_name = item.name
                    target_skill_path = target_skills_dir / skill_name

                    # Copy only if it doesn't exist in target
                    if not target_skill_path.exists():
                        logger.info(f"Bootstrapping default skill: {skill_name}")
                        try:
                            shutil.copytree(str(item), str(target_skill_path))
                        except Exception as e:
                            logger.error(f"Failed to copy skill {skill_name}: {e}")
                    else:
                        logger.debug(
                            f"Skill {skill_name} already exists in core harness, skipping."
                        )

        except Exception as e:
            logger.error(f"Error bootstrapping default skills: {e}")

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

        logger.debug(
            f"Refreshed SkillsMiddleware with {len(final_sources)} sources: {final_sources}"
        )

    def _discover_skills_sources(self, root: Path) -> List[str]:
        """Find all skills directories: core harness first, then project .ciri/skills dirs.

        Ordering:
        1. Core harness: get_core_harness_dir() / "skills"  (highest priority)
        2. Project harness: all <root>/**/.ciri/skills directories

        De-duplication is handled in _refresh_sources() by resolved path.
        Order is intentional — NOT sorted, so core harness always comes first.
        """
        discovered = []

        # 1. Core harness skills directory (highest priority / first loaded)
        try:
            core_skills_dir = get_core_harness_dir() / "skills"
            if core_skills_dir.is_dir():
                discovered.append(str(core_skills_dir.resolve()))
        except Exception as e:
            logger.error(f"Error accessing core harness skills directory: {e}")

        # 2. Project harness: scan for all .ciri/skills directories under root
        try:
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

        return discovered  # NOT sorted — order matters (core harness first)

    async def awrap_model_call(self, request, handler):
        self._refresh_sources()
        return await handler(request)

    def wrap_model_call(self, request, handler):
        self._refresh_sources()
        return handler(request)
