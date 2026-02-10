import os
import sys
import json
import shutil
import asyncio
import logging
import platform
import subprocess
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv, set_key

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


def _is_wsl() -> bool:
    """Check if running inside WSL."""
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


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
        if _is_wsl():
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
