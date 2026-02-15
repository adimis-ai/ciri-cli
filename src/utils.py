import os
import sys
import json
import shutil
import asyncio
import logging
import platform
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple, Any
from dotenv import load_dotenv, set_key

# ---------------------------------------------------------------------------
# Settings Persistence
# ---------------------------------------------------------------------------

def get_settings_path() -> Path:
    """Return the path to the project-local settings.json file."""
    return get_default_filesystem_root() / ".ciri" / "settings.json"


def load_settings() -> dict:
    """Load settings from the project-local settings.json file."""
    path = get_settings_path()
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load settings from {path}: {e}")
    return {}


def save_settings(settings: dict):
    """Save settings to the project-local settings.json file."""
    path = get_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4)
    except OSError as e:
        logger.error(f"Failed to save settings to {path}: {e}")

try:
    import pathspec
except ImportError:
    pathspec = None

logger = logging.getLogger(__name__)


def get_app_data_dir() -> Path:
    """Determine OS-specific application data directory for Ciri."""
    system = platform.system()
    user_home = Path.home()

    if system == "Windows":
        root = user_home / "AppData" / "Local" / "Ciri"
    elif system == "Darwin":
        root = user_home / "Library" / "Application Support" / "Ciri"
    else:  # Linux and others
        root = user_home / ".local" / "share" / "ciri"

    root.mkdir(parents=True, exist_ok=True)
    return root


def load_all_dotenv():
    """Load .env from current directory and global app data directory."""
    # Load from current directory
    load_dotenv()
    # Load from global app data directory
    global_env = get_app_data_dir() / ".env"
    if global_env.exists():
        load_dotenv(dotenv_path=global_env, override=False)


# def initialize_embeddings():
#     """
#     Initialize Hugging Face embeddings using configuration from Django settings.
#     """
#     embedding_model = os.getenv("EMBEDDING_MODEL", "thenlper/gte-large")
#     device = os.getenv("EMBEDDING_DEVICE", "cpu")
#     normalize = os.getenv("EMBEDDING_NORMALIZE", "True") == "True"
#     model_kwargs = os.getenv("EMBEDDING_MODEL_KWARGS", {})
#     encode_kwargs = os.getenv("EMBEDDING_ENCODE_KWARGS", {})

#     return HuggingFaceEmbeddings(
#         model_name=embedding_model,
#         model_kwargs={
#             "device": device,
#             **model_kwargs,
#         },
#         encode_kwargs={
#             "normalize_embeddings": normalize,
#             **encode_kwargs,
#         },
#     )


# def install_embedding_model():
#     """
#     Pre-install/download the default embedding model on startup.
#     This prevents API calls from being blocked by initial download.
#     """
#     import logging

#     logger = logging.getLogger(__name__)

#     logger.info("Pre-installing embedding model")
#     try:
#         initialize_embeddings()
#         logger.info("Successfully installed embedding model")
#     except Exception as e:
#         logger.error(f"Failed to install embedding model: {e}")


def find_windows_bash() -> tuple[str, ...] | None:
    """Find a POSIX-compatible bash on Windows (Git Bash or WSL).

    The ShellToolMiddleware session uses POSIX features (printf, $?) so we
    need a real bash, not cmd.exe or PowerShell.  Returns None on non-Windows
    (letting the upstream default of /bin/bash take effect).
    """
    if sys.platform != "win32":
        return None

    # 1. Check if bash is directly on PATH (e.g. Git Bash added to PATH)
    bash_on_path = shutil.which("bash")
    if bash_on_path:
        return (bash_on_path,)

    # 2. Check common Git Bash install locations
    for candidate in (
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
    ):
        if os.path.isfile(candidate):
            return (candidate,)

    # 3. Try WSL bash as last resort
    wsl_path = shutil.which("wsl")
    if wsl_path:
        return (wsl_path, "bash")

    return None


def get_default_filesystem_root() -> Path:
    """Return the current working directory as the default filesystem root."""
    return Path(os.getcwd()).resolve()


# ---------------------------------------------------------------------------
# Browser profile detection & copy
# ---------------------------------------------------------------------------

# Directories to skip when copying a browser profile (large/unnecessary)
_PROFILE_COPY_IGNORE = {
    "Service Worker",
    "GrShaderCache",
    "ShaderCache",
    "BrowserMetrics",
    "Crashpad",
    "Code Cache",
    "Cache",
    "GPUCache",
    "blob_storage",
    "component_crx_cache",
    "optimization_guide_model_store",
    "safe_browsing",
    "DawnGraphiteCache",
    "DawnWebGPUCache",
    "Extensions",
    "Extension State",
    "Extension Scripts",
    "Extension Rules",
    "GraphiteDawnCache",
    "File System",
    "GCM Store",
    "Platform Notifications",
    "Site Characteristics Database",
    "Sync Data",
    "Sync Extension Settings",
    "Local Extension Settings",
    "shared_proto_db",
    "Safe Browsing Network",
}


def is_wsl() -> bool:
    """Check if running inside WSL."""
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def has_display() -> bool:
    """Check whether a graphical display is available.

    Returns True on Windows and macOS (always have a desktop), and on Linux
    only when an X11 or Wayland session is active. WSL2 without WSLg will
    return False.
    """
    system = platform.system()
    if system in ("Windows", "Darwin"):
        return True
    # Linux — check for X11 / Wayland environment variables
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def get_chrome_channel() -> Optional[str]:
    """Detect the installed Chrome/Edge browser and return the Playwright
    ``channel`` name (``"chrome"``, ``"msedge"``, or ``None``).
    """
    system = platform.system()

    if system == "Windows":
        candidates = [
            (r"C:\Program Files\Google\Chrome\Application\chrome.exe", "chrome"),
            (r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe", "chrome"),
            (r"C:\Program Files\Microsoft\Edge\Application\msedge.exe", "msedge"),
        ]
        for exe, channel in candidates:
            if Path(exe).exists():
                return channel

    elif system == "Darwin":
        if Path("/Applications/Google Chrome.app").exists():
            return "chrome"
        if Path("/Applications/Microsoft Edge.app").exists():
            return "msedge"

    else:  # Linux (including WSL)
        for cmd, channel in [
            ("google-chrome-stable", "chrome"),
            ("google-chrome", "chrome"),
            ("microsoft-edge-stable", "msedge"),
            ("microsoft-edge", "msedge"),
        ]:
            if shutil.which(cmd):
                return channel

    return None


def _get_wsl_windows_user() -> Optional[str]:
    """Get the Windows username from inside WSL."""
    try:
        result = subprocess.run(
            ["cmd.exe", "/C", "echo", "%USERNAME%"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        username = result.stdout.strip()
        if username and username != "%USERNAME%":
            return username
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # Fallback: scan /mnt/c/Users for non-system directories
    users_dir = Path("/mnt/c/Users")
    if users_dir.is_dir():
        skip = {"Public", "Default", "Default User", "All Users"}
        for entry in users_dir.iterdir():
            if entry.is_dir() and entry.name not in skip:
                # Check if this user has Chrome data
                chrome_data = (
                    entry / "AppData" / "Local" / "Google" / "Chrome" / "User Data"
                )
                if chrome_data.is_dir():
                    return entry.name
    return None


def _get_browser_user_data_dirs() -> list[dict]:
    """Return a list of (browser_name, user_data_dir) for known browsers per OS."""
    system = platform.system()
    home = Path.home()
    results = []

    browser_paths: list[tuple[str, Path]] = []

    if system == "Windows":
        local = home / "AppData" / "Local"
        browser_paths = [
            ("chrome", local / "Google" / "Chrome" / "User Data"),
            ("edge", local / "Microsoft" / "Edge" / "User Data"),
            ("chromium", local / "Chromium" / "User Data"),
        ]
    elif system == "Darwin":
        app_support = home / "Library" / "Application Support"
        browser_paths = [
            ("chrome", app_support / "Google" / "Chrome"),
            ("edge", app_support / "Microsoft Edge"),
            ("chromium", app_support / "Chromium"),
        ]
    else:  # Linux
        config = home / ".config"
        browser_paths = [
            ("chrome", config / "google-chrome"),
            ("edge", config / "microsoft-edge"),
            ("chromium", config / "chromium"),
        ]

        # WSL: also check Windows-side Chrome profiles via /mnt/c/
        if is_wsl():
            win_user = _get_wsl_windows_user()
            if win_user:
                win_local = Path("/mnt/c/Users") / win_user / "AppData" / "Local"
                browser_paths.extend(
                    [
                        ("chrome", win_local / "Google" / "Chrome" / "User Data"),
                        ("edge", win_local / "Microsoft" / "Edge" / "User Data"),
                        ("chromium", win_local / "Chromium" / "User Data"),
                    ]
                )

    for browser_name, data_dir in browser_paths:
        if data_dir.is_dir():
            results.append({"browser": browser_name, "user_data_dir": data_dir})

    return results


def _read_profile_name(profile_dir: Path) -> str:
    """Read the human-readable profile name from a Chrome profile's Preferences file."""
    prefs_file = profile_dir / "Preferences"
    if not prefs_file.is_file():
        return profile_dir.name

    try:
        with open(prefs_file, "r", encoding="utf-8") as f:
            prefs = json.load(f)
        return prefs.get("profile", {}).get("name", profile_dir.name)
    except (json.JSONDecodeError, OSError):
        return profile_dir.name


def detect_browser_profiles() -> list[dict]:
    """
    Auto-detect Chrome/Edge/Chromium profiles across all OSes.

    Returns a list of dicts:
        {
            "browser": "chrome" | "edge" | "chromium",
            "user_data_dir": Path,        # parent dir (e.g. ~/.config/google-chrome)
            "profile_directory": str,      # subdirectory name (e.g. "Default", "Profile 1")
            "display_name": str,           # human-readable name from Preferences
        }
    """
    profiles = []

    for browser_info in _get_browser_user_data_dirs():
        user_data_dir = browser_info["user_data_dir"]
        browser = browser_info["browser"]

        # Check "Default" profile
        default_dir = user_data_dir / "Default"
        if default_dir.is_dir() and (default_dir / "Preferences").is_file():
            profiles.append(
                {
                    "browser": browser,
                    "user_data_dir": user_data_dir,
                    "profile_directory": "Default",
                    "display_name": _read_profile_name(default_dir),
                }
            )

        # Check "Profile N" directories
        for entry in sorted(user_data_dir.iterdir()):
            if (
                entry.is_dir()
                and entry.name.startswith("Profile ")
                and (entry / "Preferences").is_file()
            ):
                profiles.append(
                    {
                        "browser": browser,
                        "user_data_dir": user_data_dir,
                        "profile_directory": entry.name,
                        "display_name": _read_profile_name(entry),
                    }
                )

    return profiles


def copy_browser_profile(
    source_user_data_dir: Path,
    profile_directory: str,
) -> Path:
    """
    Copy a browser profile to a CIRI-managed directory to avoid Chrome v136+ CDP restrictions.

    Copies the specific profile subdirectory and essential root files (Local State).
    Skips large/unnecessary directories for speed.

    Returns the new user_data_dir path (parent containing the copied profile).
    """
    # Target: ~/.ciri/browser_profiles/<browser_dirname>_<profile_directory>/
    dir_name = f"{source_user_data_dir.name}_{profile_directory}".replace(" ", "_")
    target_dir = get_app_data_dir() / "browser_profiles" / dir_name
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy root-level "Local State" file (needed for profile registry)
    local_state = source_user_data_dir / "Local State"
    if local_state.is_file():
        try:
            shutil.copy2(str(local_state), str(target_dir / "Local State"))
        except (PermissionError, OSError):
            pass

    # 2. Copy the profile subdirectory, skipping heavy dirs and locked files
    source_profile = source_user_data_dir / profile_directory
    target_profile = target_dir / profile_directory

    def _ignore(directory: str, contents: list[str]) -> set[str]:
        ignored = {c for c in contents if c in _PROFILE_COPY_IGNORE}
        # Also skip LOCK files (Chrome holds them while running)
        ignored.update(c for c in contents if c == "LOCK")
        return ignored

    def _safe_copy(src: str, dst: str, **kwargs) -> None:
        """Copy a single file, silently skipping locked/permission-denied files."""
        try:
            shutil.copy2(src, dst)
        except (PermissionError, OSError):
            # File is locked by Chrome or inaccessible (common on Windows/WSL)
            pass

    if source_profile.is_dir():
        shutil.copytree(
            str(source_profile),
            str(target_profile),
            ignore=_ignore,
            dirs_exist_ok=True,
            copy_function=_safe_copy,
        )

    logger.info("Copied browser profile to %s", target_dir)
    return target_dir


def resolve_browser_profile(
    browser_name: Optional[str] = None,
    profile_directory: Optional[str] = None,
) -> Optional[dict]:
    """Find the best matching browser profile and copy it for safe use.

    Returns:
        A dict with ``user_data_dir`` (Path to the copied parent),
        ``profile_directory`` (str), and ``browser`` (str) — or ``None``.
    """
    profiles = detect_browser_profiles()
    if not profiles:
        return None

    # Narrow down to the requested browser / profile
    if browser_name:
        filtered = [p for p in profiles if p["browser"] == browser_name]
        if filtered:
            profiles = filtered

    if profile_directory:
        filtered = [p for p in profiles if p["profile_directory"] == profile_directory]
        if filtered:
            profiles = filtered

    if not profiles:
        return None

    selected = profiles[0]

    # Copy to a CIRI-managed directory to avoid Chrome lock conflicts
    copied_user_data_dir = copy_browser_profile(
        source_user_data_dir=selected["user_data_dir"],
        profile_directory=selected["profile_directory"],
    )

    return {
        "user_data_dir": copied_user_data_dir,
        "profile_directory": selected["profile_directory"],
        "browser": selected["browser"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Autocomplete Discovery Functions
# ──────────────────────────────────────────────────────────────────────────────


# Directories always excluded from file/folder autocomplete
_EXCLUDED_DIRS = {
    # Version control
    ".git",
    ".hg",
    ".svn",
    # Python
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    ".eggs",
    # Node.js
    "node_modules",
    ".npm",
    ".next",
    ".nuxt",
    # IDE/Editor
    ".idea",
    ".vscode",
    # Java/Maven/Gradle
    "target",
    ".gradle",
    ".m2",
    # PHP/Composer
    "vendor",
    # Frontend
    "bower_components",
    # Build outputs
    "out",
    "dist",
    "build",
    # Coverage
    "htmlcov",
    "coverage",
    # CIRI
    ".ciri",
}


def _load_gitignore_spec(directory: Path):
    """Load a single .gitignore file from a directory, returning a PathSpec or None."""
    if pathspec is None:
        return None
    gitignore_file = directory / ".gitignore"
    if not gitignore_file.is_file():
        return None
    try:
        with open(gitignore_file, "r", encoding="utf-8") as f:
            return pathspec.PathSpec.from_lines("gitwildmatch", f)
    except Exception:
        return None


def _is_ignored_by_specs(
    rel_path: str, is_dir: bool, specs: List[Tuple[Path, Any]], root: Path
) -> bool:
    """Check if a relative path is ignored by any gitignore spec.

    Args:
        rel_path: Path relative to root (e.g. "src/foo/bar.py")
        is_dir: Whether the path is a directory
        specs: List of (spec_directory, PathSpec) tuples
        root: The workspace root
    """
    # For directory matching, gitignore patterns like "build/" need trailing slash
    match_path = rel_path + "/" if is_dir else rel_path

    for spec_dir, spec in specs:
        try:
            # Get path relative to the spec's directory
            spec_rel = spec_dir.relative_to(root)
            spec_rel_str = str(spec_rel)

            if spec_rel_str == ".":
                # Root-level gitignore — match against full rel_path
                path_for_match = match_path
            else:
                # Nested gitignore — match against path relative to that gitignore's dir
                prefix = spec_rel_str + "/"
                if not rel_path.startswith(prefix):
                    continue
                path_for_match = match_path[len(prefix) :]

            if spec.match_file(path_for_match):
                return True
        except (ValueError, Exception):
            continue

    return False


def _walk_with_gitignore(
    root: Path,
    collect_files: bool = True,
    collect_dirs: bool = False,
    prefix: str = "",
) -> List[str]:
    """Walk directory tree respecting .gitignore and excluded dirs.

    Uses os.walk with directory pruning so we never descend into ignored dirs.
    Loads .gitignore specs incrementally as we encounter them.
    """
    results = []
    # Accumulate gitignore specs: list of (directory_path, PathSpec)
    gitignore_specs: List[Tuple[Path, Any]] = []

    # Load root .gitignore first
    root_spec = _load_gitignore_spec(root)
    if root_spec is not None:
        gitignore_specs.append((root, root_spec))

    for dirpath_str, dirnames, filenames in os.walk(root, topdown=True):
        dirpath = Path(dirpath_str)

        # Load .gitignore from current directory (if not root, already loaded)
        if dirpath != root:
            dir_spec = _load_gitignore_spec(dirpath)
            if dir_spec is not None:
                gitignore_specs.append((dirpath, dir_spec))

        # Prune directories: remove ignored dirs from dirnames in-place
        pruned = []
        for d in dirnames:
            # Always exclude hardcoded dirs
            if d in _EXCLUDED_DIRS:
                continue
            # Check against gitignore specs
            child_rel = str((dirpath / d).relative_to(root))
            if _is_ignored_by_specs(child_rel, True, gitignore_specs, root):
                continue
            pruned.append(d)
        dirnames[:] = sorted(pruned)

        # Collect directories
        if collect_dirs and dirpath != root:
            rel_dir = str(dirpath.relative_to(root))
            if prefix == "" or rel_dir.startswith(prefix):
                results.append(rel_dir)

        # Collect files
        if collect_files:
            for fname in filenames:
                # Skip common non-useful files
                if fname.endswith((".swp", ".swo", ".pyc", ".pyo")):
                    continue
                file_path = dirpath / fname
                rel_file = str(file_path.relative_to(root))
                if _is_ignored_by_specs(rel_file, False, gitignore_specs, root):
                    continue
                if prefix == "" or rel_file.startswith(prefix):
                    results.append(rel_file)

    return sorted(results)


def list_files_with_gitignore(root: Path, prefix: str = "") -> List[str]:
    """List all files respecting .gitignore, excluding common directories."""
    try:
        return _walk_with_gitignore(
            root, collect_files=True, collect_dirs=False, prefix=prefix
        )
    except Exception:
        return []


def list_folders_with_gitignore(root: Path, prefix: str = "") -> List[str]:
    """List all directories respecting .gitignore, excluding common directories."""
    try:
        return _walk_with_gitignore(
            root, collect_files=False, collect_dirs=True, prefix=prefix
        )
    except Exception:
        return []


def list_skills(root: Path, prefix: str = "") -> List[str]:
    """Discover all skill names from .ciri/skills directories."""
    skills = []

    try:
        for ciri_dir in root.rglob(".ciri"):
            if ciri_dir.is_dir():
                skills_dir = ciri_dir / "skills"
                if skills_dir.is_dir():
                    for item in skills_dir.iterdir():
                        if item.is_dir():
                            skill_name = item.name
                            if prefix == "" or skill_name.startswith(prefix):
                                skills.append(skill_name)
    except Exception:
        pass

    return sorted(set(skills))


def list_toolkits(root: Path, prefix: str = "") -> List[str]:
    """Discover all toolkit names from .ciri/toolkits directories."""
    toolkits = []

    try:
        for ciri_dir in root.rglob(".ciri"):
            if ciri_dir.is_dir():
                toolkits_dir = ciri_dir / "toolkits"
                if toolkits_dir.is_dir():
                    for item in toolkits_dir.iterdir():
                        if item.is_dir():
                            toolkit_name = item.name
                            if prefix == "" or toolkit_name.startswith(prefix):
                                toolkits.append(toolkit_name)
    except Exception:
        pass

    return sorted(set(toolkits))


def list_subagents(root: Path, prefix: str = "") -> List[str]:
    """Discover all subagent names from .ciri/subagents/*.{yaml,yml,json}."""
    subagents = []

    try:
        for ciri_dir in root.rglob(".ciri"):
            if ciri_dir.is_dir():
                subagents_dir = ciri_dir / "subagents"
                if subagents_dir.is_dir():
                    for ext in ["*.yaml", "*.yml", "*.json"]:
                        for config_file in subagents_dir.glob(ext):
                            subagent_name = config_file.stem
                            if prefix == "" or subagent_name.startswith(prefix):
                                subagents.append(subagent_name)
    except Exception:
        pass

    return sorted(set(subagents))
