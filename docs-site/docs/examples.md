# Examples & Tutorials

This page contains hands-on examples and short tutorials to get you productive quickly.

1) First run: inspect threads and run /sync
- Start CIRI:
  ciri
- Create a new thread:
  /new-thread
- Trigger workspace sync to detect local skills:
  /sync

2) Create a simple skill (example)
- Directory: .ciri/skills/hello-world/
- Files:
  - skill.json (metadata)
  - hello.py (implementation)
- Minimal skill.json:

```json
{
  "name": "hello-world",
  "version": "0.1.0",
  "description": "A minimal example skill that responds with a greeting."
}
```

- Example implementation pattern (hello.py):

```python
def handle(args):
    return "Hello from skill!"
```

- After creating the folder, run /sync in the CLI to discover and register the skill.

3) Adding a model provider (example)
- Add OPENROUTER_API_KEY to your .env
- Update .ciri/settings.json if you want a persistent default model

4) Running a Playwright-based toolkit example
- Ensure browsers are installed:
  playwright install
- Start the toolkit from within a skill or toolkit adapter that uses Playwright APIs.

See tutorials for more guided workflows.
