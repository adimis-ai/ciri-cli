# FAQ

Q: What models does CIRI support?
A: CIRI integrates with OpenRouter and other compatible model providers via LangChain adapters. See the runtime providers section for configuration.

Q: Where do I add a new skill?
A: Add it to .ciri/skills/ for user-installed skills or src/skills/ for built-in ones. Run /sync to detect.

Q: How do I build a single binary?
A: Use python build.py --onefile to run the PyInstaller-based builder. Builds are platform specific.

Q: How do I contribute?
A: Follow CONTRIBUTING.md and the docs site guide.
