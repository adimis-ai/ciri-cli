# CLI - Advanced Usage and Integration

This page documents deeper integration points and automation patterns.

- Non-interactive use: you can script interactions by invoking the main entrypoint and piping commands, but note the CLI is optimized for interactive usage.
- Integrating with Tauri desktop app: build the CLI binary and include it in the Tauri bundle under src-tauri/binaries.
- Programmatic usage: import src.copilot and create a Copilot instance in custom scripts (advanced; internal APIs may change).

Example: headless sync and export

  python -c "from ciri.__main__ import run_headless_sync; run_headless_sync()"

(Only for advanced users — check source for helper functions and breakage risk.)
