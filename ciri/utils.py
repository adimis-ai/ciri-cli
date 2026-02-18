import os
import sys
import json
import shutil
import asyncio
import logging
import platform
import subprocess
import socket
import time as _time
import urllib.request
import urllib.error
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

    def _default(obj):
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(
            f"Object of type {obj.__class__.__name__} is not JSON serializable"
        )

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4, default=_default)
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


def get_core_harness_dir() -> Path:
    """Return the OS-level core harness directory, shared across all projects.

    This is the persistent home for default skills, reusable toolkits, global
    subagent configs, and cross-project memory.  It lives inside
    get_app_data_dir() so it shares the same OS-level root as the database
    and .env file:
      Linux   : ~/.local/share/ciri/
      macOS   : ~/Library/Application Support/Ciri/
      Windows : ~/AppData/Local/Ciri/

    Subdirectories created on first call:
      skills/    — default skills (synced from src/skills/ at startup)
      toolkits/  — core MCP toolkit servers
      subagents/ — core subagent configs
      memory/    — global / cross-project memory files
    """
    root = get_app_data_dir()
    for subdir in ("skills", "toolkits", "subagents", "memory"):
        (root / subdir).mkdir(parents=True, exist_ok=True)
    return root


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


# ---------------------------------------------------------------------------
# CDP (Chrome DevTools Protocol) browser helpers
# ---------------------------------------------------------------------------

# UX flags added when launching Chrome for CDP control.
# NOTE: Do NOT add --disable-blink-features=AutomationControlled here;
# modern Chrome versions show a warning banner for that flag.
_CDP_LAUNCH_ARGS: list[str] = [
    "--disable-infobars",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-popup-blocking",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding",
]


def _get_browser_executable(browser_name: Optional[str] = None) -> Optional[str]:
    """Return the absolute path (or command name) of Chrome / Edge on this OS.

    If *browser_name* is ``None`` the function auto-detects the first
    installed Chromium-based browser.
    """
    system = platform.system()

    candidates: list[tuple[str, str]] = []

    if system == "Windows":
        candidates = [
            (r"C:\Program Files\Google\Chrome\Application\chrome.exe", "chrome"),
            (r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe", "chrome"),
            (r"C:\Program Files\Microsoft\Edge\Application\msedge.exe", "edge"),
        ]
    elif system == "Darwin":
        candidates = [
            ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "chrome"),
            ("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge", "edge"),
        ]
    else:  # Linux / WSL
        for cmd, bname in [
            ("google-chrome-stable", "chrome"),
            ("google-chrome", "chrome"),
            ("microsoft-edge-stable", "edge"),
            ("microsoft-edge", "edge"),
        ]:
            path = shutil.which(cmd)
            if path:
                candidates.append((path, bname))

    for exe, bname in candidates:
        if browser_name and bname != browser_name:
            continue
        if Path(exe).exists() or shutil.which(exe):
            return exe

    return None


def is_cdp_port_open(port: int = 9222) -> bool:
    """Return ``True`` if a CDP-enabled browser is listening on *port*.

    Uses an HTTP GET to ``/json/version`` — the standard CDP discovery
    endpoint.  Tries ``127.0.0.1`` first (IPv4) then ``localhost`` so
    we handle both Chrome's default binding and edge-cases where the
    browser listens only on IPv6.
    """
    for host in ("127.0.0.1", "localhost"):
        url = f"http://{host}:{port}/json/version"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError, ValueError):
            continue
    return False


def _kill_browser_processes(exe_path: str) -> bool:
    """Terminate running Chrome / Edge processes that match *exe_path*.

    On Windows this uses ``taskkill /IM <name>``.  On Linux/macOS it uses
    ``pkill -f``.  Returns ``True`` if any processes were found and a
    termination signal was sent.  Waits until all matching processes have
    actually exited (up to 15 s on Windows, 8 s elsewhere).
    """
    exe_basename = Path(
        exe_path
    ).name.lower()  # e.g. "chrome.exe", "google-chrome-stable"

    try:
        if sys.platform == "win32":
            # taskkill is reliable on Windows for closing all Chrome processes
            result = subprocess.run(
                ["taskkill", "/F", "/IM", exe_basename],
                capture_output=True,
                timeout=10,
            )
            killed = result.returncode == 0
        else:
            # On Linux/macOS, pkill by process name
            # Strip paths — match on the binary name
            result = subprocess.run(
                ["pkill", "-f", exe_basename],
                capture_output=True,
                timeout=10,
            )
            killed = result.returncode == 0

        if killed:
            logger.info("Terminated existing %s processes", exe_basename)
            # Poll until the processes are truly gone, rather than using a
            # fixed sleep.  On Windows, profile locks and TCP ports may not
            # be released until the kernel has fully reaped the process.
            max_wait = 15 if sys.platform == "win32" else 8
            for _ in range(max_wait * 2):  # check every 0.5 s
                if not _is_browser_running(exe_path):
                    break
                _time.sleep(0.5)
            else:
                logger.warning(
                    "%s processes still running after %ds — proceeding anyway",
                    exe_basename,
                    max_wait,
                )
            # Extra grace period for the OS to release file locks / ports
            _time.sleep(1)
        return killed

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        logger.warning("Could not kill browser processes: %s", exc)
        return False


def _is_browser_running(exe_path: str) -> bool:
    """Return ``True`` if a process matching *exe_path* is currently running."""
    exe_basename = Path(exe_path).name.lower()

    try:
        if sys.platform == "win32":
            result = subprocess.run(
                ["tasklist", "/FI", f"IMAGENAME eq {exe_basename}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return exe_basename in result.stdout.lower()
        else:
            result = subprocess.run(
                ["pgrep", "-f", exe_basename],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def launch_browser_with_cdp(
    port: int = 9222,
    browser_name: Optional[str] = None,
    user_data_dir: Optional[Path] = None,
    profile_directory: Optional[str] = None,
    *,
    timeout: float = 20.0,
) -> str:
    """Ensure Chrome/Edge is running with ``--remote-debugging-port`` and
    return the CDP HTTP endpoint (``http://localhost:<port>``).

    If the port is already open we assume the browser is already running with
    CDP and return the endpoint immediately.

    If the browser is running *without* the debug port, it is terminated and
    relaunched with the flag — this is necessary because Chrome's single-
    instance model causes a second launch to simply open a new window inside
    the existing (non-debug) process.

    Args:
        port: TCP port for Chrome DevTools Protocol.
        browser_name: ``"chrome"`` or ``"edge"`` (auto-detected if ``None``).
        user_data_dir: ``--user-data-dir`` value.  Uses the real profile
            directory so the user's cookies / sessions / extensions are
            present.
        profile_directory: ``--profile-directory`` value (e.g. ``"Default"``).
        timeout: Seconds to wait for the debugging port to open.

    Returns:
        The CDP endpoint URL, e.g. ``"http://localhost:9222"``.

    Raises:
        RuntimeError: If the browser could not be started or the port did not
            open within *timeout* seconds.
    """
    endpoint = f"http://localhost:{port}"

    # Already listening on the debug port? Return immediately.
    if is_cdp_port_open(port):
        logger.info("CDP port %d already open — reusing existing browser", port)
        return endpoint

    exe = _get_browser_executable(browser_name)
    if not exe:
        raise RuntimeError(
            "Could not find a Chrome or Edge installation.  "
            "Please install Google Chrome or Microsoft Edge."
        )

    # Chrome/Edge single-instance guard:  if Chrome is already running
    # *without* the debug port, a second launch with --remote-debugging-port
    # just asks the existing instance to open a new window (ignoring the
    # debug flag).  We must close the existing instance first.
    if _is_browser_running(exe):
        logger.info(
            "Browser is running without CDP — closing it to relaunch with "
            "--remote-debugging-port=%d",
            port,
        )
        _kill_browser_processes(exe)

    args: list[str] = [
        exe,
        f"--remote-debugging-port={port}",
        *_CDP_LAUNCH_ARGS,
    ]
    if user_data_dir:
        args.append(f"--user-data-dir={user_data_dir}")
    if profile_directory:
        args.append(f"--profile-directory={profile_directory}")

    logger.info("Launching browser for CDP: %s", " ".join(args))

    # Launch as a fully detached process so it survives CIRI exit.
    # On Windows, DETACHED_PROCESS makes pipes unreliable, so we use
    # DEVNULL for stderr there and skip the proc.poll() early-exit
    # check (poll() is unreliable for fully detached processes).
    kwargs: dict = {}
    is_win = sys.platform == "win32"
    if is_win:
        # CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
    else:
        kwargs["start_new_session"] = True

    proc = subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL if is_win else subprocess.PIPE,
        **kwargs,
    )

    # Wait for the debugging port to become reachable.
    # Windows often needs extra time after a kill/relaunch cycle.
    effective_timeout = timeout if not is_win else max(timeout, 30.0)
    deadline = _time.monotonic() + effective_timeout
    while _time.monotonic() < deadline:
        if is_cdp_port_open(port):
            logger.info("CDP endpoint ready at %s", endpoint)
            return endpoint
        # Check if the process died early (skip on Windows — poll() is
        # unreliable for DETACHED_PROCESS and may return None even after
        # the process has exited).
        if not is_win and proc.poll() is not None:
            stderr_out = ""
            if proc.stderr:
                try:
                    stderr_out = proc.stderr.read().decode(errors="replace")[:500]
                except Exception:
                    pass
            raise RuntimeError(
                f"Browser process exited immediately (code {proc.returncode}). "
                f"stderr: {stderr_out or '(empty)'}"
            )
        _time.sleep(0.5)

    raise RuntimeError(
        f"Browser was launched but CDP port {port} did not open within "
        f"{effective_timeout}s.  Check that no other process is blocking the port."
    )


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
    """Discover all skill names from core harness and .ciri/skills directories."""
    skills = []

    # 1. Core harness skills (OS-level, shared across all projects)
    try:
        core_skills_dir = get_core_harness_dir() / "skills"
        if core_skills_dir.is_dir():
            for item in core_skills_dir.iterdir():
                if item.is_dir():
                    skill_name = item.name
                    if prefix == "" or skill_name.startswith(prefix):
                        skills.append(skill_name)
    except Exception:
        pass

    # 2. Project harness skills (all .ciri/skills dirs under root)
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


def sync_default_skills():
    """Sync default skills from src/skills to the core harness skills directory.

    Target is get_core_harness_dir() / "skills" (the OS-level persistent
    directory) rather than the project-local <cwd>/.ciri/skills/.  This ensures
    default skills are available globally, accessible from any project on the
    user's machine.
    """
    try:
        target_skills_dir = get_core_harness_dir() / "skills"

        # Determine the source skills directory (src/skills relative to this file)
        source_skills_dir = Path(__file__).parent / "skills"

        if not source_skills_dir.is_dir():
            logger.warning(f"Source skills directory not found: {source_skills_dir}")
            return

        # target_skills_dir already created by get_core_harness_dir()
        logger.info(f"Syncing default skills from {source_skills_dir} to {target_skills_dir}")

        # Copy each skill directory from source to target
        for item in source_skills_dir.iterdir():
            if item.is_dir() and not item.name.startswith("__"):
                target_skill_path = target_skills_dir / item.name
                # Use shutil.copytree with dirs_exist_ok=True for upsert behavior
                shutil.copytree(item, target_skill_path, dirs_exist_ok=True)

    except Exception as e:
        logger.error(f"Failed to sync default skills: {e}")


def list_toolkits(root: Path, prefix: str = "") -> List[str]:
    """Discover all toolkit names from core harness and .ciri/toolkits directories."""
    toolkits = []

    # 1. Core harness toolkits
    try:
        core_toolkits_dir = get_core_harness_dir() / "toolkits"
        if core_toolkits_dir.is_dir():
            for item in core_toolkits_dir.iterdir():
                if item.is_dir():
                    toolkit_name = item.name
                    if prefix == "" or toolkit_name.startswith(prefix):
                        toolkits.append(toolkit_name)
    except Exception:
        pass

    # 2. Project harness toolkits (all .ciri/toolkits dirs under root)
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
    """Discover all subagent names from core harness and .ciri/subagents directories."""
    subagents = []

    # 1. Core harness subagents
    try:
        core_subagents_dir = get_core_harness_dir() / "subagents"
        if core_subagents_dir.is_dir():
            for ext in ["*.yaml", "*.yml", "*.json"]:
                for config_file in core_subagents_dir.glob(ext):
                    subagent_name = config_file.stem
                    if prefix == "" or subagent_name.startswith(prefix):
                        subagents.append(subagent_name)
    except Exception:
        pass

    # 2. Project harness subagents (all .ciri/subagents dirs under root)
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
