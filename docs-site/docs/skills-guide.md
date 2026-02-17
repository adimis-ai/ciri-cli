# Skills Guide

Skills are the primary extension mechanism for user-facing features.

Where to put skills:
- Built-in skills: src/skills/
- User-installed: .ciri/skills/

Skill structure (recommended):
- skill.json — metadata (name, version, description)
- module.py — implementation module exposing the expected hooks
- README.md — short usage & examples

Creating a skill:
1. Create a new folder under .ciri/skills/my-skill
2. Add a skill.json with minimal metadata
3. Implement an entry function and handlers
4. Run CIRI and call /sync to discover the skill

Testing skills:
- Write unit tests under tests/ that import and exercise your skill code
- Use pytest and mock LLM responses where appropriate
