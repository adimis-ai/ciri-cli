# Build & Packaging

CIRI supports distribution as a wheel (recommended) and a single-file binary via PyInstaller (optional).

Build wheel (hatchling):

1. Ensure build backend is hatchling (see pyproject.toml)
2. Build:

   python -m build --wheel
   # or using hatch directly
   python -m hatchling.build

The wheel result will be in dist/.

PyInstaller single-file binary (via build.py):

1. PyInstaller is included in dev deps. To create a binary run:

   python build.py --onefile

2. build.py will create the binary and copy it to the src-tauri/binaries/ folder for Tauri packaging.

Notes:
- PyInstaller builds are platform-specific. Build on the target OS or use CI cross-build.
- For reproducible builds use a pinned environment (pip-tools or lock file).
