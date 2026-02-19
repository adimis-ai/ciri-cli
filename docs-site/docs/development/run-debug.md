# Run & Debug

Run the CLI:

- Installed globally or in venv:
  ciri

- Or from source:
  python -m ciri.__main__

Common runtime flags and tips:
- To get help: ciri --help
- Use the built-in commands at the prompt, e.g., /new-thread, /sync, /change-model
- To increase logging: export CIRI_LOG_LEVEL=DEBUG

Debugging with breakpoints:
- Use python -m debugpy --listen 5678 -m ciri.__main__ and attach your IDE
- For interactive prompt issues, use logging and inspect the log file (CIRI writes logs to app data dir)
