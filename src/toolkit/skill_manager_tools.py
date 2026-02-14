# Skill Manager Tools: LangChain tools for managing skills in .ciri/skills directories

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
import yaml
import json
from datetime import datetime

from langchain_core.tools import tool
from deepagents.backends import FilesystemBackend

from ..utils import get_default_filesystem_root

logger = logging.getLogger(__name__)


class SkillMetadata(BaseModel):
    """Metadata for a skill following Agent Skills specification."""
    
    name: str = Field(..., description="Skill identifier (max 64 chars, lowercase alphanumeric and hyphens)")
    description: str = Field(..., description="Description of what the skill does (max 1024 chars)")
    license: Optional[str] = Field(None, description="License name or reference")
    compatibility: Optional[str] = Field(None, description="Environment requirements")
    metadata: Dict[str, str] = Field(default_factory=dict, description="Additional metadata key-value pairs")
    allowed_tools: List[str] = Field(default_factory=list, description="Pre-approved tools list")

    model_config = ConfigDict(extra="allow")  # Allow additional metadata fields


class SkillInput(BaseModel):
    """Input schema for creating or updating a skill."""
    
    name: str = Field(..., description="Name/identifier of the skill", min_length=1, max_length=64)
    description: str = Field(..., description="Description of what the skill does", min_length=1, max_length=1024)
    content: str = Field(..., description="Main markdown content of the skill instructions")
    license: Optional[str] = Field(None, description="License name or reference")
    compatibility: Optional[str] = Field(None, description="Environment requirements")
    metadata: Dict[str, str] = Field(default_factory=dict, description="Additional metadata key-value pairs")
    allowed_tools: List[str] = Field(default_factory=list, description="Pre-approved tools list")
    target_directory: Optional[str] = Field(None, description="Specific .ciri/skills directory to use (if multiple exist)")

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        """Validate skill name per Agent Skills specification."""
        # Max 64 characters, lowercase alphanumeric and hyphens only
        # Cannot start or end with hyphen, no consecutive hyphens
        if not re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", v):
            raise ValueError("Skill name must be lowercase alphanumeric with single hyphens between segments")
        return v


class SkillIdentifier(BaseModel):
    """Input schema for identifying a skill."""
    
    name: str = Field(..., description="Name/identifier of the skill to target")
    target_directory: Optional[str] = Field(None, description="Specific .ciri/skills directory to search in")


class SkillInfo(BaseModel):
    """Detailed information about a skill."""
    
    name: str
    description: str
    content: str
    frontmatter: SkillMetadata
    file_path: str
    directory_path: str
    size_bytes: int
    last_modified: str


class SkillSummary(BaseModel):
    """Summary information about a skill."""
    
    name: str
    description: str
    directory_path: str
    license: Optional[str]
    compatibility: Optional[str]
    metadata: Dict[str, str]
    last_modified: str


class SkillManager:
    """Helper class to manage skills in the filesystem."""
    
    def __init__(self, backend: FilesystemBackend):
        self.backend = backend
        self.root = get_default_filesystem_root()
    
    def _discover_skill_directories(self) -> List[Path]:
        """Discover all .ciri/skills directories."""
        skills_dirs = []
        try:
            # Recursive search for all .ciri folders
            for ciri_dir in self.root.rglob(".ciri"):
                # Ensure we are not inside another .ciri folder
                if ciri_dir.is_dir() and not any(
                    p.name == ".ciri" for p in ciri_dir.parents if p != self.root
                ):
                    skills_dir = ciri_dir / "skills"
                    if skills_dir.is_dir():
                        skills_dirs.append(skills_dir)
        except Exception as e:
            logger.error(f"Error discovering skill directories: {e}")
        
        return sorted(skills_dirs)
    
    def _find_skill_path(self, name: str, target_directory: Optional[str] = None) -> Optional[Path]:
        """Find the path to a specific skill."""
        skill_dirs = self._discover_skill_directories()
        
        if target_directory:
            # Search in specific directory first
            target_path = Path(target_directory).resolve()
            if target_path in skill_dirs:
                skill_dirs = [target_path] + [d for d in skill_dirs if d != target_path]
        
        for skills_dir in skill_dirs:
            skill_path = skills_dir / name
            if skill_path.is_dir():
                return skill_path
        
        return None
    
    def _get_skill_content_path(self, skill_dir: Path) -> Path:
        """Get the main content file path for a skill."""
        return skill_dir / "SKILL.md"
    
    def _parse_skill_frontmatter(self, content: str) -> tuple[SkillMetadata, str]:
        """Parse YAML frontmatter from SKILL.md content.
        
        Returns:
            Tuple of (SkillMetadata, markdown_content_without_frontmatter)
        """
        # Match YAML frontmatter between --- delimiters
        frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n"
        match = re.match(frontmatter_pattern, content, re.DOTALL)
        
        if not match:
            raise ValueError("SKILL.md file must have YAML frontmatter delimited by ---")
        
        frontmatter_str = match.group(1)
        markdown_content = content[match.end():]
        
        try:
            frontmatter_data = yaml.safe_load(frontmatter_str) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML frontmatter: {e}")
        
        if not isinstance(frontmatter_data, dict):
            raise ValueError("YAML frontmatter must be a dictionary")
        
        # Validate required fields
        name = frontmatter_data.get("name")
        description = frontmatter_data.get("description")
        
        if not name or not description:
            raise ValueError("YAML frontmatter must include 'name' and 'description' fields")
        
        metadata = SkillMetadata(
            name=str(name),
            description=str(description),
            license=frontmatter_data.get("license"),
            compatibility=frontmatter_data.get("compatibility"),
            metadata=frontmatter_data.get("metadata", {}),
            allowed_tools=frontmatter_data.get("allowed_tools", [])
        )
        
        return metadata, markdown_content
    
    def _create_skill_content(self, skill_input: SkillInput) -> str:
        """Create SKILL.md content with YAML frontmatter."""
        frontmatter = {
            "name": skill_input.name,
            "description": skill_input.description
        }
        
        if skill_input.license:
            frontmatter["license"] = skill_input.license
        if skill_input.compatibility:
            frontmatter["compatibility"] = skill_input.compatibility
        if skill_input.metadata:
            frontmatter["metadata"] = skill_input.metadata
        if skill_input.allowed_tools:
            frontmatter["allowed_tools"] = skill_input.allowed_tools
        
        # Create YAML frontmatter
        yaml_content = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
        
        # Combine frontmatter with content
        full_content = f"---\n{yaml_content}---\n\n{skill_input.content}"
        
        return full_content
    
    def _load_skill_from_file(self, skill_dir: Path) -> tuple[SkillMetadata, str]:
        """Load skill metadata and content from SKILL.md file.
        
        Returns:
            Tuple of (SkillMetadata, markdown_content)
        """
        content_path = self._get_skill_content_path(skill_dir)
        
        if not content_path.exists():
            raise FileNotFoundError(f"SKILL.md not found: {content_path}")
        
        with open(content_path, 'r', encoding='utf-8') as f:
            full_content = f.read()
        
        return self._parse_skill_frontmatter(full_content)
    
    def _parse_description_from_content(self, content: str) -> str:
        """Extract description from skill content if not provided separately."""
        lines = content.strip().split('\n')
        for line in lines:
            if line.strip().startswith('#'):
                # Extract title/description from first header
                return line.strip().lstrip('# ').strip()
        
        # Fall back to first non-empty line
        for line in lines:
            if line.strip():
                return line.strip()[:100] + ("..." if len(line.strip()) > 100 else "")
        
        return "No description available"
    
    def upsert_skill(self, skill_input: SkillInput) -> Dict[str, Any]:
        """Create or update a skill."""
        try:
            skill_dirs = self._discover_skill_directories()
            
            if not skill_dirs:
                # Create a default .ciri/skills directory
                default_skills_dir = self.root / ".ciri" / "skills"
                default_skills_dir.mkdir(parents=True, exist_ok=True)
                skill_dirs = [default_skills_dir]
            
            # Determine target directory
            if skill_input.target_directory:
                target_path = Path(skill_input.target_directory).resolve()
                if target_path in skill_dirs:
                    target_skills_dir = target_path
                else:
                    raise ValueError(f"Specified target directory not found: {skill_input.target_directory}")
            else:
                # Use first available skills directory
                target_skills_dir = skill_dirs[0]
            
            # Set up skill directory
            skill_dir = target_skills_dir / skill_input.name
            content_path = self._get_skill_content_path(skill_dir)
            is_update = content_path.exists()
            
            skill_dir.mkdir(parents=True, exist_ok=True)
            
            # Create full content with YAML frontmatter
            full_content = self._create_skill_content(skill_input)
            
            # Save skill file
            with open(content_path, 'w', encoding='utf-8') as f:
                f.write(full_content)
            
            return {
                "success": True,
                "message": f"Skill '{skill_input.name}' {'updated' if is_update else 'created'} successfully",
                "skill_path": str(skill_dir),
                "content_path": str(content_path)
            }
        
        except Exception as e:
            logger.error(f"Error upserting skill '{skill_input.name}': {e}")
            return {
                "success": False,
                "message": f"Failed to upsert skill: {str(e)}"
            }
    
    def delete_skill(self, identifier: SkillIdentifier) -> Dict[str, Any]:
        """Delete a skill."""
        try:
            skill_path = self._find_skill_path(identifier.name, identifier.target_directory)
            
            if not skill_path:
                return {
                    "success": False,
                    "message": f"Skill '{identifier.name}' not found"
                }
            
            # Remove the entire skill directory
            import shutil
            shutil.rmtree(skill_path)
            
            return {
                "success": True,
                "message": f"Skill '{identifier.name}' deleted successfully",
                "deleted_path": str(skill_path)
            }
        
        except Exception as e:
            logger.error(f"Error deleting skill '{identifier.name}': {e}")
            return {
                "success": False,
                "message": f"Failed to delete skill: {str(e)}"
            }
    
    def list_skills(self) -> Dict[str, Any]:
        """List all available skills."""
        try:
            skills = []
            skill_dirs = self._discover_skill_directories()
            
            for skills_dir in skill_dirs:
                if not skills_dir.exists():
                    continue
                
                for skill_dir in skills_dir.iterdir():
                    if not skill_dir.is_dir():
                        continue
                    
                    content_path = self._get_skill_content_path(skill_dir)
                    if not content_path.exists():
                        continue
                    
                    try:
                        # Load skill metadata and content
                        frontmatter, _ = self._load_skill_from_file(skill_dir)
                        
                        skill_summary = SkillSummary(
                            name=frontmatter.name,
                            description=frontmatter.description,
                            directory_path=str(skills_dir),
                            license=frontmatter.license,
                            compatibility=frontmatter.compatibility,
                            metadata=frontmatter.metadata,
                            last_modified=datetime.fromtimestamp(
                                content_path.stat().st_mtime
                            ).isoformat()
                        )
                        
                        skills.append(skill_summary.model_dump())
                    
                    except Exception as e:
                        logger.warning(f"Error processing skill {skill_dir.name}: {e}")
                        continue
            
            return {
                "success": True,
                "skills": skills,
                "count": len(skills),
                "skills_directories": [str(d) for d in skill_dirs]
            }
        
        except Exception as e:
            logger.error(f"Error listing skills: {e}")
            return {
                "success": False,
                "message": f"Failed to list skills: {str(e)}",
                "skills": []
            }
    
    def get_skill_info(self, identifier: SkillIdentifier) -> Dict[str, Any]:
        """Get detailed information about a specific skill."""
        try:
            skill_path = self._find_skill_path(identifier.name, identifier.target_directory)
            
            if not skill_path:
                return {
                    "success": False,
                    "message": f"Skill '{identifier.name}' not found"
                }
            
            content_path = self._get_skill_content_path(skill_path)
            
            if not content_path.exists():
                return {
                    "success": False,
                    "message": f"Skill content file not found: {content_path}"
                }
            
            # Load skill metadata and content
            frontmatter, content = self._load_skill_from_file(skill_path)
            
            # Get file stats
            stat = content_path.stat()
            
            skill_info = SkillInfo(
                name=frontmatter.name,
                description=frontmatter.description,
                content=content,
                frontmatter=frontmatter,
                file_path=str(content_path),
                directory_path=str(skill_path.parent),
                size_bytes=stat.st_size,
                last_modified=datetime.fromtimestamp(stat.st_mtime).isoformat()
            )
            
            return {
                "success": True,
                "skill": skill_info.model_dump()
            }
        
        except Exception as e:
            logger.error(f"Error getting skill info for '{identifier.name}': {e}")
            return {
                "success": False,
                "message": f"Failed to get skill info: {str(e)}"
            }


# Global skill manager instance (will be initialized with backend)
_skill_manager: Optional[SkillManager] = None


def initialize_skill_manager(backend: FilesystemBackend):
    """Initialize the global skill manager with a backend."""
    global _skill_manager
    _skill_manager = SkillManager(backend)


def get_skill_manager() -> SkillManager:
    """Get the initialized skill manager."""
    if _skill_manager is None:
        raise RuntimeError("SkillManager not initialized. Call initialize_skill_manager first.")
    return _skill_manager


# LangChain Tools
@tool
def upsert_skill(skill_input: SkillInput) -> Dict[str, Any]:
    """
    Create or update a skill in the .ciri/skills directory.
    
    Takes structured input with skill name, description, content, and metadata.
    Validates the input and ensures it adheres to the expected schema for skills.
    
    Args:
        skill_input: SkillInput object containing all skill information
        
    Returns:
        Dict with success status, message, and file paths
    """
    manager = get_skill_manager()
    return manager.upsert_skill(skill_input)


@tool
def delete_skill(identifier: SkillIdentifier) -> Dict[str, Any]:
    """
    Delete a skill from the .ciri/skills directory.
    
    Takes the name or identifier of the skill to be deleted as input.
    Validates that the skill exists before deletion and handles errors gracefully.
    
    Args:
        identifier: SkillIdentifier object with skill name and optional target directory
        
    Returns:
        Dict with success status and message
    """
    manager = get_skill_manager()
    return manager.delete_skill(identifier)


@tool
def list_skills() -> Dict[str, Any]:
    """
    List all existing skills in the .ciri/skills directories.
    
    Returns a structured list of skills with names, descriptions, tags, versions,
    and other relevant information to help users understand available skills.
    
    Returns:
        Dict with success status, list of skills, count, and skills directories
    """
    manager = get_skill_manager()
    return manager.list_skills()


@tool
def get_skill_info(identifier: SkillIdentifier) -> Dict[str, Any]:
    """
    Retrieve detailed information about a specific skill.
    
    Takes the name or identifier of the skill and returns comprehensive information
    including functionality, content, metadata, and usage details.
    
    Args:
        identifier: SkillIdentifier object with skill name and optional target directory
        
    Returns:
        Dict with success status and detailed skill information
    """
    manager = get_skill_manager()
    return manager.get_skill_info(identifier)


# Export the tools for use in skill_builder_agent
SKILL_MANAGEMENT_TOOLS = [
    upsert_skill,
    delete_skill,
    list_skills,
    get_skill_info,
]