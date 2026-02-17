# Skills, Toolkits and Subagents

CIRI is extended with three primary extension types:

- Skills: packaged user-facing features (autocomplete, file ops, content generation)
  - Located in src/skills/ for built-in skills, and in .ciri/skills/ for installed skills
  - Each skill should include metadata and an entrypoint for registration
- Toolkits: adapter packages that expose external APIs or local OS features
  - Placed under src/toolkit/ or .ciri/toolkits/
- Subagents: isolated agents for background processing or long-running tasks
  - Placed under src/subagents/ or .ciri/subagents/

Development notes:
- The repository includes many example skills under .ciri/skills/
- To add a new skill, create a directory with a metadata file (skill.json) and a python module that exposes the skill API. See the onboarding skill in .ciri/skills/project-onboarding-ciri for examples.
- Use /sync from inside the CLI to let CIRI discover and self-train on new skills.
