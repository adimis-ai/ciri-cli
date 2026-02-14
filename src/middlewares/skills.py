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
        scan_root: Optional[Union[str, Path]] = None
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
            
        # 1. Determine scan root
        root = Path(scan_root) if scan_root else get_default_filesystem_root()
        logger.debug(f"Scanning for skills in: {root}")
        
        # 2. Discover auto sources
        auto_sources = self._discover_skills_sources(root)
        
        # 3. Merge sources: Explicit sources first, then discovered ones (de-duplicated)
        final_sources = list(sources)
        seen = {str(Path(s).resolve()) for s in sources}
        
        for s in auto_sources:
            resolved_s = str(Path(s).resolve())
            if resolved_s not in seen:
                final_sources.append(s)
                seen.add(resolved_s)
        
        logger.info(f"Initialized SkillsMiddleware with {len(final_sources)} sources: {final_sources}")
        super().__init__(backend=backend, sources=final_sources)

    def _discover_skills_sources(self, root: Path) -> List[str]:
        """Recursively find all .ciri/skills directories."""
        discovered = []
        try:
            # Recursive search for all .ciri folders
            for ciri_dir in root.rglob(".ciri"):
                if ciri_dir.is_dir():
                    skills_dir = ciri_dir / "skills"
                    if skills_dir.is_dir():
                        discovered.append(str(skills_dir.resolve()))
        except Exception as e:
            logger.error(f"Error while scanning for skills directories: {e}")
            
        return sorted(discovered)