import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock
from src.middlewares.skills import SkillsMiddleware


@pytest.fixture
def project_root(tmp_path):
    """Create a mock project structure with skills folders."""
    ciri_dir = tmp_path / ".ciri"
    skills_dir = ciri_dir / "skills"
    skills_dir.mkdir(parents=True)

    # Create a skill folder
    skill1 = skills_dir / "skill1"
    skill1.mkdir()
    (skill1 / "skill.py").write_text("def run(): pass")

    return tmp_path


@pytest.mark.asyncio
async def test_skills_dynamic_scanning(project_root):
    # Initialize middleware
    mock_backend = MagicMock()
    middleware = SkillsMiddleware(backend=mock_backend, scan_root=project_root)

    # Verify initial discovery
    # Note: SkillsMiddleware discovers the PATH to the skills directory, not individual skills
    discovered = sorted([Path(s).name for s in middleware.sources])
    assert "skills" in discovered

    # Add a new skill folder in a DIFFERENT .ciri location
    # because if we add to existing project_root/.ciri/skills, the path is already in sources
    sub_project = project_root / "sub_project"
    sub_project.mkdir()
    new_skills_dir = sub_project / ".ciri" / "skills"
    new_skills_dir.mkdir(parents=True)

    # Trigger scan via wrap call
    class MockRequest:
        pass

    request = MockRequest()

    async def async_handler(r):
        return r

    await middleware.awrap_model_call(request, async_handler)

    # Verify new skills directory is discovered (it should appear twice as "skills", so check count or resolve)
    discovered_paths = sorted([str(Path(s).resolve()) for s in middleware.sources])
    assert str((project_root / ".ciri" / "skills").resolve()) in discovered_paths
    assert str(new_skills_dir.resolve()) in discovered_paths
