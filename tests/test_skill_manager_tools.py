import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from deepagents.backends import FilesystemBackend

from src.toolkit.skill_manager_tools import (
    SkillInput,
    SkillIdentifier,
    SkillMetadata,
    initialize_skill_manager,
    get_skill_manager,
    upsert_skill,
    delete_skill,
    list_skills,
    get_skill_info,
)


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = Path(tmp_dir)
        
        # Create a .ciri/skills directory structure
        ciri_dir = workspace / ".ciri"
        skills_dir = ciri_dir / "skills"
        skills_dir.mkdir(parents=True)
        
        # Mock the get_default_filesystem_root to return our temp workspace
        import src.toolkit.skill_manager_tools
        original_get_root = src.toolkit.skill_manager_tools.get_default_filesystem_root
        src.toolkit.skill_manager_tools.get_default_filesystem_root = lambda: workspace
        
        yield workspace
        
        # Restore original function
        src.toolkit.skill_manager_tools.get_default_filesystem_root = original_get_root


@pytest.fixture
def mock_backend():
    """Create a mock FilesystemBackend."""
    return MagicMock(spec=FilesystemBackend)


def test_skill_manager_initialization(temp_workspace, mock_backend):
    """Test that skill manager initializes correctly."""
    initialize_skill_manager(mock_backend)
    manager = get_skill_manager()
    
    assert manager is not None
    assert manager.backend == mock_backend


def test_upsert_skill_creates_new_skill(temp_workspace, mock_backend):
    """Test creating a new skill."""
    initialize_skill_manager(mock_backend)
    
    skill_input = SkillInput(
        name="test-skill",
        description="A test skill for demonstration purposes",
        content="# Test Skill\n\n## When to Use\n\n- User asks for a test demonstration\n\n## Instructions\n\nThis is a test skill for demonstration.",
        license="MIT",
        metadata={"author": "test_user", "version": "1.0.0"}
    )
    
    result = upsert_skill.func(skill_input)
    
    assert result["success"] is True
    assert "test-skill" in result["message"]
    assert "created" in result["message"]
    
    # Verify files were created
    skill_dir = temp_workspace / ".ciri" / "skills" / "test-skill"
    assert skill_dir.exists()
    assert (skill_dir / "SKILL.md").exists()
    # No separate metadata.yaml file should exist
    assert not (skill_dir / "metadata.yaml").exists()
    
    # Verify content has YAML frontmatter
    with open(skill_dir / "SKILL.md", 'r') as f:
        content = f.read()
    assert content.startswith("---\n")
    assert "name: test-skill" in content
    assert "description: A test skill for demonstration purposes" in content
    assert "This is a test skill" in content


def test_list_skills_empty_directory(temp_workspace, mock_backend):
    """Test listing skills when directory is empty."""
    initialize_skill_manager(mock_backend)
    
    result = list_skills.func()
    
    assert result["success"] is True
    assert result["count"] == 0
    assert len(result["skills"]) == 0


def test_list_skills_with_existing_skills(temp_workspace, mock_backend):
    """Test listing skills with existing skills."""
    initialize_skill_manager(mock_backend)
    
    # Create a test skill first
    skill_input = SkillInput(
        name="test-skill-1",
        description="First test skill for demonstration",
        content="# Test Skill 1\n\n## When to Use\n\n- When testing\n\n## Instructions\n\nThis is the first test skill.",
        metadata={"category": "test"}
    )
    upsert_skill.func(skill_input)
    
    skill_input2 = SkillInput(
        name="test-skill-2", 
        description="Second test skill for advanced testing",
        content="# Test Skill 2\n\n## When to Use\n\n- When advanced testing is needed\n\n## Instructions\n\nThis is the second test skill.",
        license="Apache-2.0",
        metadata={"category": "test", "level": "advanced"}
    )
    upsert_skill.func(skill_input2)
    
    result = list_skills.func()
    
    assert result["success"] is True
    assert result["count"] == 2
    assert len(result["skills"]) == 2
    
    skill_names = [skill["name"] for skill in result["skills"]]
    assert "test-skill-1" in skill_names
    assert "test-skill-2" in skill_names


def test_get_skill_info_existing_skill(temp_workspace, mock_backend):
    """Test getting info for an existing skill."""
    initialize_skill_manager(mock_backend)
    
    # Create a test skill first
    skill_input = SkillInput(
        name="info-test-skill",
        description="Skill for testing get_skill_info functionality",
        content="# Info Test Skill\n\n## When to Use\n\n- When testing get_skill_info\n\n## Instructions\n\nThis skill is for testing get_skill_info.",
        license="BSD-3-Clause",
        compatibility="Python 3.8+",
        metadata={"author": "info_tester", "version": "1.5.0"}
    )
    upsert_skill.func(skill_input)
    
    identifier = SkillIdentifier(name="info-test-skill")
    result = get_skill_info.func(identifier)
    
    assert result["success"] is True
    assert result["skill"]["name"] == "info-test-skill"
    assert "This skill is for testing" in result["skill"]["content"]
    assert result["skill"]["frontmatter"]["license"] == "BSD-3-Clause"
    assert result["skill"]["frontmatter"]["compatibility"] == "Python 3.8+"
    assert result["skill"]["frontmatter"]["metadata"]["author"] == "info_tester"
    assert result["skill"]["frontmatter"]["metadata"]["version"] == "1.5.0"


def test_get_skill_info_nonexistent_skill(temp_workspace, mock_backend):
    """Test getting info for a nonexistent skill."""
    initialize_skill_manager(mock_backend)
    
    identifier = SkillIdentifier(name="nonexistent-skill")
    result = get_skill_info.func(identifier)
    
    assert result["success"] is False
    assert "not found" in result["message"]


def test_delete_skill_existing_skill(temp_workspace, mock_backend):
    """Test deleting an existing skill."""
    initialize_skill_manager(mock_backend)
    
    # Create a test skill first
    skill_input = SkillInput(
        name="delete-test-skill",
        description="Skill created for testing deletion functionality", 
        content="# Delete Test Skill\n\n## When to Use\n\n- When testing deletion\n\n## Instructions\n\nThis skill will be deleted.",
    )
    upsert_skill.func(skill_input)
    
    # Verify skill exists
    skill_dir = temp_workspace / ".ciri" / "skills" / "delete-test-skill"
    assert skill_dir.exists()
    
    # Delete the skill
    identifier = SkillIdentifier(name="delete-test-skill")
    result = delete_skill.func(identifier)
    
    assert result["success"] is True
    assert "deleted successfully" in result["message"]
    
    # Verify skill is gone
    assert not skill_dir.exists()


def test_delete_skill_nonexistent_skill(temp_workspace, mock_backend):
    """Test deleting a nonexistent skill."""
    initialize_skill_manager(mock_backend)
    
    identifier = SkillIdentifier(name="nonexistent-skill")
    result = delete_skill.func(identifier)
    
    assert result["success"] is False
    assert "not found" in result["message"]


def test_skill_input_validation():
    """Test SkillInput validation."""
    # Valid input - Agent Skills spec format
    skill_input = SkillInput(
        name="valid-skill-name",
        description="A valid skill description",
        content="# Valid Skill\n\n## When to Use\n\n- When validation is needed\n\n## Instructions\n\nContent here."
    )
    assert skill_input.name == "valid-skill-name"
    
    # Invalid name with underscores (not allowed in Agent Skills spec)
    with pytest.raises(ValueError, match="lowercase alphanumeric"):
        SkillInput(
            name="invalid_skill_name",
            description="Description",
            content="Content"
        )
    
    # Invalid name with uppercase
    with pytest.raises(ValueError, match="lowercase alphanumeric"):
        SkillInput(
            name="Invalid-Skill-Name",
            description="Description",
            content="Content"
        )
    
    # Empty name
    with pytest.raises(ValueError):
        SkillInput(
            name="",
            description="Description", 
            content="Content"
        )


def test_upsert_skill_updates_existing(temp_workspace, mock_backend):
    """Test updating an existing skill."""
    initialize_skill_manager(mock_backend)
    
    # Create initial skill
    skill_input = SkillInput(
        name="update-test-skill",
        description="Initial skill description",
        content="# Initial Content\n\n## When to Use\n\n- Initial use case\n\n## Instructions\n\nInitial version.",
        metadata={"version": "1.0.0"}
    )
    result1 = upsert_skill.func(skill_input)
    assert "created" in result1["message"]
    
    # Update the skill
    updated_skill_input = SkillInput(
        name="update-test-skill",
        description="Updated skill description with more details",
        content="# Updated Content\n\n## When to Use\n\n- Updated use case\n- Additional scenarios\n\n## Instructions\n\nThis is the updated version.",
        license="MIT",
        metadata={"version": "2.0.0", "status": "updated"}
    )
    result2 = upsert_skill.func(updated_skill_input)
    assert "updated" in result2["message"]
    
    # Verify update
    identifier = SkillIdentifier(name="update-test-skill")
    info_result = get_skill_info.func(identifier)
    
    assert info_result["success"] is True
    assert "updated version" in info_result["skill"]["content"]
    assert info_result["skill"]["frontmatter"]["license"] == "MIT"
    assert info_result["skill"]["frontmatter"]["metadata"]["version"] == "2.0.0"
    assert info_result["skill"]["frontmatter"]["metadata"]["status"] == "updated"