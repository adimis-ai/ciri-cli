import logging
import os
import platform
import shutil
from pathlib import Path
from typing import Optional

from deepagents import CompiledSubAgent
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchResults

from langgraph.errors import GraphInterrupt
from langchain.agents.middleware import (
    TodoListMiddleware,
    ToolRetryMiddleware,
    SummarizationMiddleware,
)

from ..toolkit.web_crawler_tool import (
    build_web_crawler_tool,
    BrowserConfig as CrawlerBrowserConfig,
)
from ..toolkit.human_follow_up_tool import follow_up_with_human
from ..utils import (
    _is_wsl,
    detect_browser_profiles,
    copy_browser_profile,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Anti-detection browser arguments
# ---------------------------------------------------------------------------

_STEALTH_ARGS: list[str] = [
    "--disable-blink-features=AutomationControlled",
    "--disable-infobars",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-popup-blocking",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding",
]


# ---------------------------------------------------------------------------
# Platform detection helpers
# ---------------------------------------------------------------------------


def _has_display() -> bool:
    """Check whether a graphical display is available.

    Returns True on Windows and macOS (always have a desktop), and on Linux
    only when an X11 or Wayland session is active.  WSL2 without WSLg will
    return False.
    """
    system = platform.system()
    if system in ("Windows", "Darwin"):
        return True
    # Linux — check for X11 / Wayland environment variables
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _get_chrome_channel() -> Optional[str]:
    """Detect the installed Chrome/Edge browser and return the Playwright
    ``channel`` name (``"chrome"``, ``"msedge"``, or ``None``).

    Playwright's ``channel`` param tells it to launch the *system-installed*
    browser instead of its own bundled Chromium — this inherits the real
    browser fingerprint and reduces bot-detection signals.
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


# ---------------------------------------------------------------------------
# Browser profile resolution
# ---------------------------------------------------------------------------


def _resolve_browser_profile(
    browser_name: Optional[str] = None,
    profile_directory: Optional[str] = None,
) -> Optional[dict]:
    """Find the best matching browser profile and copy it for safe use.

    Chrome v136+ locks its profile directory via CDP restrictions when the
    main browser is running.  ``copy_browser_profile`` copies the essential
    data to a CIRI-managed directory so we can open it without conflicts.

    On WSL2, this also picks up Windows-side profiles from ``/mnt/c/``.

    Returns:
        A dict with ``user_data_dir`` (Path to the copied parent),
        ``profile_directory`` (str), and ``browser`` (str) — or ``None``
        when no profile is found.
    """
    profiles = detect_browser_profiles()
    if not profiles:
        logger.info("No browser profiles detected on this system")
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
        logger.warning("No matching browser profile found")
        return None

    selected = profiles[0]
    logger.info(
        "Using browser profile: %s (%s / %s)",
        selected["display_name"],
        selected["browser"],
        selected["profile_directory"],
    )

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
# Playwright persistent-context adapters
#
# PlayWrightBrowserToolkit expects a Browser object (with a .contexts list).
# Playwright's launch_persistent_context() returns a BrowserContext directly.
# These lightweight adapters bridge the gap so the toolkit works unchanged.
# ---------------------------------------------------------------------------


class _AsyncBrowserContextAdapter:
    """Make an async ``BrowserContext`` quack like an async ``Browser``."""

    def __init__(self, context):  # noqa: ANN001
        self._context = context

    @property
    def contexts(self):  # noqa: ANN201
        return [self._context]

    async def close(self) -> None:
        await self._context.close()


class _SyncBrowserContextAdapter:
    """Make a sync ``BrowserContext`` quack like a sync ``Browser``."""

    def __init__(self, context):  # noqa: ANN001
        self._context = context

    @property
    def contexts(self):  # noqa: ANN201
        return [self._context]

    def close(self) -> None:
        self._context.close()


# ---------------------------------------------------------------------------
# Playwright browser launchers (persistent context)
# ---------------------------------------------------------------------------


async def _launch_persistent_async_browser(
    user_data_dir: Path,
    profile_directory: str = "Default",
    headless: bool = False,
    channel: Optional[str] = None,
) -> tuple:
    """Launch an async Playwright browser with a persistent context.

    Using ``launch_persistent_context`` keeps cookies, localStorage, and
    session data across runs — critical for staying logged-in on sites like
    LinkedIn and Twitter without triggering bot detection.

    Returns:
        ``(adapter, playwright_instance)`` so the caller can tear down later.
    """
    from playwright.async_api import async_playwright

    pw = await async_playwright().start()

    # Persistent context user_data_dir = the actual profile sub-folder
    profile_path = str(user_data_dir / profile_directory)

    launch_kwargs: dict = {
        "user_data_dir": profile_path,
        "headless": headless,
        "args": list(_STEALTH_ARGS),
        "ignore_https_errors": True,
        "viewport": {"width": 1920, "height": 1080},
    }
    if channel:
        launch_kwargs["channel"] = channel

    context = await pw.chromium.launch_persistent_context(**launch_kwargs)
    return _AsyncBrowserContextAdapter(context), pw


def _launch_persistent_sync_browser(
    user_data_dir: Path,
    profile_directory: str = "Default",
    headless: bool = False,
    channel: Optional[str] = None,
) -> tuple:
    """Sync variant of :func:`_launch_persistent_async_browser`."""
    from playwright.sync_api import sync_playwright

    pw = sync_playwright().start()
    profile_path = str(user_data_dir / profile_directory)

    launch_kwargs: dict = {
        "user_data_dir": profile_path,
        "headless": headless,
        "args": list(_STEALTH_ARGS),
        "ignore_https_errors": True,
        "viewport": {"width": 1920, "height": 1080},
    }
    if channel:
        launch_kwargs["channel"] = channel

    context = pw.chromium.launch_persistent_context(**launch_kwargs)
    return _SyncBrowserContextAdapter(context), pw


# ---------------------------------------------------------------------------
# Playwright tools builder
# ---------------------------------------------------------------------------


def _build_tools_from_browser(async_browser) -> list[BaseTool]:  # noqa: ANN001
    """Build Playwright tools from an async browser or adapter.

    Uses ``model_construct`` to bypass pydantic ``isinstance`` validation so
    that our ``_AsyncBrowserContextAdapter`` (or a real ``Browser``) can be
    used interchangeably.
    """
    from langchain_community.tools.playwright.click import ClickTool
    from langchain_community.tools.playwright.current_page import CurrentWebPageTool
    from langchain_community.tools.playwright.extract_hyperlinks import (
        ExtractHyperlinksTool,
    )
    from langchain_community.tools.playwright.extract_text import ExtractTextTool
    from langchain_community.tools.playwright.get_elements import GetElementsTool
    from langchain_community.tools.playwright.navigate import NavigateTool
    from langchain_community.tools.playwright.navigate_back import NavigateBackTool

    tool_classes = [
        ClickTool,
        NavigateTool,
        NavigateBackTool,
        ExtractTextTool,
        ExtractHyperlinksTool,
        GetElementsTool,
        CurrentWebPageTool,
    ]
    return [
        cls.model_construct(async_browser=async_browser, sync_browser=None)
        for cls in tool_classes
    ]


async def get_playwright_tools(
    profile_info: Optional[dict] = None,
    headless: Optional[bool] = None,
    channel: Optional[str] = None,
) -> list[BaseTool]:
    """Create Playwright browser tools, preferring a persistent context with
    the user's real browser profile for anti-detection.

    Falls back to a plain Chromium launch (with stealth args) when no profile
    is available or persistent-context launch fails.
    """
    if headless is None:
        headless = not _has_display()
        if headless:
            logger.info("No display server detected — running Playwright headless")

    if channel is None:
        channel = _get_chrome_channel()

    # --- attempt persistent-context launch with real profile ---
    if profile_info:
        try:
            adapter, _pw = await _launch_persistent_async_browser(
                user_data_dir=profile_info["user_data_dir"],
                profile_directory=profile_info["profile_directory"],
                headless=headless,
                channel=channel,
            )
            return _build_tools_from_browser(adapter)
        except Exception:
            logger.warning(
                "Failed to launch persistent browser context — "
                "falling back to default Chromium",
                exc_info=True,
            )

    # --- fallback: plain launch with stealth args ---
    from playwright.async_api import async_playwright

    pw = await async_playwright().start()
    browser = await pw.chromium.launch(
        headless=headless, args=list(_STEALTH_ARGS)
    )
    return _build_tools_from_browser(browser)


# ---------------------------------------------------------------------------
# crawl4ai BrowserConfig builder
# ---------------------------------------------------------------------------


def build_crawler_browser_config(
    profile_info: Optional[dict] = None,
    headless: Optional[bool] = None,
    channel: Optional[str] = None,
) -> CrawlerBrowserConfig:
    """Build a ``crawl4ai.BrowserConfig`` that uses the user's real browser
    profile for anti-detection.

    Key settings:
    - ``use_persistent_context`` + ``user_data_dir`` — reuse cookies/sessions
    - ``chrome_channel`` — launch the system-installed Chrome/Edge
    - ``enable_stealth`` — inject playwright-stealth patches
    - ``extra_args`` — disable automation-detection Blink features
    """
    if headless is None:
        headless = not _has_display()

    if channel is None:
        channel = _get_chrome_channel() or "chromium"

    kwargs: dict = {
        "browser_type": "chromium",
        "headless": headless,
        "chrome_channel": channel,
        "channel": channel,
        "extra_args": list(_STEALTH_ARGS),
        "enable_stealth": True,
        "ignore_https_errors": True,
        "viewport_width": 1920,
        "viewport_height": 1080,
    }

    if profile_info:
        profile_path = str(
            profile_info["user_data_dir"] / profile_info["profile_directory"]
        )
        kwargs["use_persistent_context"] = True
        kwargs["user_data_dir"] = profile_path

    return CrawlerBrowserConfig(**kwargs)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

WEB_RESEARCHER_SYSTEM_PROMPT = """\
You are CIRI Web Researcher — an elite, autonomous web research agent \
operating a *real* browser session loaded with the user's actual browser \
profile (cookies, sessions, saved logins). You think step-by-step, \
choose the optimal tool at every stage, and always verify your findings \
before reporting.

# ═══════════════════════════════════════════════════════════════════════
# 1. COMPLETE TOOL INVENTORY
# ═══════════════════════════════════════════════════════════════════════

You have access to **10 tools** across four categories.  Always use the \
exact names shown below when calling a tool.

## A. Playwright Interactive Browser Tools
These operate on a live browser page backed by the user's real profile.

| # | Tool name            | Input                              | What it does                                                         | When to use                                                                   |
|---|----------------------|------------------------------------|----------------------------------------------------------------------|-------------------------------------------------------------------------------|
| 1 | `navigate_browser`   | `url: str`                         | Navigates the browser to the given URL                               | Opening any page — always the first step before interacting                   |
| 2 | `click_element`      | `selector: str` (CSS)              | Clicks the element matching the CSS selector                         | Buttons, links, tabs, "Load More", accepting cookie banners, pagination       |
| 3 | `extract_text`       | *(none)*                           | Returns **all** visible text on the current page                     | Reading page content after navigation (articles, search results, profiles)    |
| 4 | `extract_hyperlinks` | `absolute_urls: bool` (default True) | Returns every hyperlink on the current page                          | Discovering sub-pages, outbound references, sitemaps                          |
| 5 | `get_elements`       | `selector: str`, `attributes: list[str]` (default `["innerText"]`) | Returns matching elements with requested attributes | Scraping structured data (tables, lists, cards), inspecting forms, reading specific sections |
| 6 | `current_webpage`    | *(none)*                           | Returns the URL of the page currently loaded in the browser          | Verifying redirects, confirming navigation, logging current state             |
| 7 | `previous_webpage`   | *(none)*                           | Navigates back to the previous page (browser back button)            | Returning to search results, undoing a misclick, backtracking                 |

## B. Web Crawler (crawl4ai)
| # | Tool name     | Key inputs                                          | What it does                                                                  |
|---|---------------|-----------------------------------------------------|-------------------------------------------------------------------------------|
| 8 | `web_crawler`  | `url`, `deep_crawl: bool`, `max_depth`, `max_pages`, `word_count_threshold`, `bypass_cache` | Extracts clean, LLM-ready **markdown** from a page.  With `deep_crawl=True` it performs BFS to discover and crawl linked pages across an entire site. |

## C. Web Search
| # | Tool name           | Input          | What it does                                   |
|---|---------------------|----------------|-------------------------------------------------|
| 9 | `simple_web_search` | `query: str`  | DuckDuckGo search — returns result snippets & URLs |

## D. Human Follow-Up
| #  | Tool name               | Input                                      | What it does                                                                |
|----|-------------------------|--------------------------------------------|------------------------------------------------------------------------------|
| 10 | `follow_up_with_human`  | `question: str`, `options: list[str]` (optional) | Pauses execution and asks the **user** a question; resumes with their answer |

# ═══════════════════════════════════════════════════════════════════════
# 2. THINKING PROCESS  (follow this for EVERY research task)
# ═══════════════════════════════════════════════════════════════════════

Before calling any tool, walk through these mental steps **in order**:

## Step 1 — PLAN
- Decompose the request into concrete sub-questions.
- Identify what *kind* of information each sub-question needs \
(factual lookup, data extraction, interactive browsing, live account data).
- Estimate which tools are needed and in what order.

## Step 2 — DISCOVER
- Use `simple_web_search` to find candidate URLs.
- Scan the result snippets — often you can answer simple factual \
questions directly from snippets without visiting a single page.

## Step 3 — EXTRACT
- For content-heavy pages (articles, docs, blogs, wikis): \
prefer `web_crawler` — it returns clean markdown with no JS noise.
- For pages behind login, heavy JS, or interactive widgets: \
use Playwright (`navigate_browser` → `extract_text` / `get_elements`).

## Step 4 — INTERACT (when necessary)
- If the page requires clicking (pagination, "Show more", tabs, \
cookie-consent dialogs): use `click_element`.
- If you need to fill a form (search bars on-site, filters): \
use `click_element` to focus + Playwright tools to type.
- If the site requires login and you have an existing session, \
just navigate — cookies handle it.
- If the site requires login and **no session exists**, call \
`follow_up_with_human` to ask the user for credentials or \
manual assistance (see Section 5).

## Step 5 — SYNTHESIZE
- Combine extracted information across sources.
- Resolve contradictions by preferring primary / authoritative sources.
- Note confidence level and gaps.

## Step 6 — VERIFY
- Cross-reference key claims across ≥ 2 independent sources.
- Check dates — prefer the most recent information.
- If only one source exists, say so explicitly.

# ═══════════════════════════════════════════════════════════════════════
# 3. TOOL-CALLING DECISION TREE
# ═══════════════════════════════════════════════════════════════════════

Use this flowchart to pick the right tool:

```
Need to FIND pages/URLs?
  └─► simple_web_search

Have a URL and need CONTENT?
  ├─ Static or content-heavy page? ──► web_crawler
  ├─ JS-heavy / SPA / needs interaction? ──► navigate_browser → extract_text
  └─ Unsure? ──► Try web_crawler first; if it returns poor/empty results,
                  fall back to navigate_browser → extract_text

Need to INTERACT with a page?
  ├─ Click a button/link ──► click_element (CSS selector)
  ├─ Get structured data (table, list) ──► get_elements (CSS selector + attributes)
  ├─ Find sub-pages ──► extract_hyperlinks
  ├─ Check where you are ──► current_webpage
  └─ Go back ──► previous_webpage

Need to CRAWL an entire site section?
  └─► web_crawler with deep_crawl=True, max_depth=2-3, max_pages=10-20

Blocked by CAPTCHA / login wall / paywall / ambiguity?
  └─► follow_up_with_human
```

# ═══════════════════════════════════════════════════════════════════════
# 4. COMPOUND TOOL SEQUENCES  (common patterns)
# ═══════════════════════════════════════════════════════════════════════

**Pattern A — General Research:**
1. `simple_web_search` → get candidate URLs
2. `web_crawler(url=best_url)` → extract markdown content
3. Repeat for 2-3 more sources → cross-reference

**Pattern B — Deep Site Exploration:**
1. `web_crawler(url=root, deep_crawl=True, max_depth=2, max_pages=10)`
2. Identify most relevant sub-pages from crawl output
3. `web_crawler(url=sub_page)` for each key sub-page

**Pattern C — Interactive / JS-Heavy Site:**
1. `navigate_browser(url=target)` → load the page
2. `extract_text` → read current content
3. `click_element(selector="...")` → interact (pagination, tabs, etc.)
4. `extract_text` → read updated content
5. Repeat 3-4 as needed

**Pattern D — Structured Data Extraction:**
1. `navigate_browser(url=target)`
2. `get_elements(selector="table tr", attributes=["innerText"])` → extract rows
3. Or `get_elements(selector=".card", attributes=["innerText", "href"])` → cards

**Pattern E — Link Discovery & Follow:**
1. `navigate_browser(url=target)`
2. `extract_hyperlinks(absolute_urls=True)` → get all links
3. Filter relevant links by pattern
4. `web_crawler(url=relevant_link)` for each → extract content

**Pattern F — Authenticated Research:**
1. `navigate_browser(url=target)` → page loads with user's cookies
2. `current_webpage` → verify you landed on the right page (not a login redirect)
3. If redirected to login → `follow_up_with_human(question="...")` to get creds
4. If logged in → proceed with `extract_text` / `get_elements`

**Pattern G — Fallback Chain (resilience):**
1. Try `web_crawler(url=target)` first
2. If it fails or returns empty → `navigate_browser(url=target)` + `extract_text`
3. If Playwright is also blocked → `simple_web_search` for cached/alternate versions
4. If still stuck → `follow_up_with_human` to ask user for help

# ═══════════════════════════════════════════════════════════════════════
# 5. HUMAN-IN-THE-LOOP — when and how to call `follow_up_with_human`
# ═══════════════════════════════════════════════════════════════════════

This tool **pauses your execution** and asks the user a question.  Use it \
sparingly but decisively in these scenarios:

| Scenario | Example question | Options (if applicable) |
|----------|-----------------|-------------------------|
| **CAPTCHA encountered** | "I've hit a CAPTCHA on {url}. Could you solve it in the browser or provide an alternative URL?" | ["I'll solve it", "Skip this source", "Try a different URL"] |
| **Login required, no session** | "The page {url} requires login and I don't have an active session. Could you provide credentials or log in manually?" | ["Here are my credentials", "I'll log in manually", "Skip this source"] |
| **Ambiguous research query** | "Your request could mean X or Y. Which direction should I focus on?" | ["X", "Y", "Both"] |
| **Paywall blocking content** | "This article is behind a paywall. Should I look for a free alternative or do you have access?" | ["Find free alternative", "I have access, I'll help", "Skip it"] |
| **2FA / MFA challenge** | "The site is asking for two-factor authentication. Could you complete this step?" | ["I'll complete it", "Skip this source"] |
| **Critical decision point** | "I found conflicting information from sources A and B. Which do you trust more, or should I dig deeper?" | ["Trust A", "Trust B", "Dig deeper"] |

**Rules for `follow_up_with_human`:**
- Always provide clear, actionable `options` when possible — this makes it \
easy for the user to respond quickly.
- Never ask more than one question at a time.
- After receiving the user's response, immediately act on it.
- Do NOT call this for things you can figure out yourself — exhaust your \
own tools first.

# ═══════════════════════════════════════════════════════════════════════
# 6. ANTI-DETECTION & STEALTH GUIDELINES
# ═══════════════════════════════════════════════════════════════════════

You are operating the user's **real browser** with their actual cookies, \
sessions, and browser fingerprint.  This is your biggest advantage — \
use it wisely:

- **Pace yourself**: Insert natural delays between actions on the same \
domain.  Never fire 10 rapid requests to one site.
- **Leverage existing sessions**: On platforms like LinkedIn, Twitter/X, \
GitHub, Google — the user is likely already logged in.  Just navigate \
and read.  Do NOT try to log in again.
- **Rotate extraction methods**: If `web_crawler` gets blocked on a \
domain, switch to Playwright (and vice versa).
- **Respect robots.txt in spirit**: Don't aggressively deep-crawl sites \
that clearly don't want it unless the user explicitly asks.
- **Handle cookie banners**: Many sites show consent dialogs.  Use \
`click_element` to accept/dismiss them before extracting content.
- **Check for redirects**: After `navigate_browser`, call \
`current_webpage` to ensure you weren't redirected to a login page, \
error page, or geo-blocked page.

# ═══════════════════════════════════════════════════════════════════════
# 7. FAILURE RECOVERY
# ═══════════════════════════════════════════════════════════════════════

When a tool call fails or returns useless results, follow this \
escalation chain:

1. **Retry once** with adjusted parameters (e.g., different CSS selector, \
`bypass_cache=True` for web_crawler).
2. **Switch tools**: web_crawler ↔ Playwright ↔ simple_web_search.
3. **Try alternative sources**: search for mirrors, cached versions, \
or related pages.
4. **Ask the human**: call `follow_up_with_human` with a clear \
explanation of what failed and what options exist.
5. **Report honestly**: if all avenues are exhausted, explain what you \
tried and why access was not possible.

# ═══════════════════════════════════════════════════════════════════════
# 8. OUTPUT STANDARDS
# ═══════════════════════════════════════════════════════════════════════

- **Structure**: Use headers, bullet points, and tables for clarity.
- **Source attribution**: Every factual claim must include the source URL.
- **Quoting**: Use blockquotes for direct passages from sources.
- **Confidence signaling**: Distinguish between well-sourced facts, \
single-source claims, and your own inferences.
- **Recency**: Always note the publication date when available; prefer \
the most recent information.
- **Completeness**: If you could not access certain sources or \
information, explain *why* and suggest alternatives.
- **Conciseness**: Provide comprehensive but focused answers.  Lead with \
the answer, then provide supporting evidence.
"""


# ---------------------------------------------------------------------------
# Agent builder
# ---------------------------------------------------------------------------


async def build_web_researcher_agent(
    model: BaseChatModel,
    browser_name: Optional[str] = None,
    profile_directory: Optional[str] = None,
    headless: Optional[bool] = None,
    crawler_browser_config: Optional[CrawlerBrowserConfig] = None,
) -> CompiledSubAgent:
    """Build a web-researcher sub-agent with real-browser anti-detection.

    The agent uses the user's actual installed browser (Chrome / Edge) with
    their existing profile (cookies, sessions, fingerprint) to minimise
    bot-detection on guarded platforms such as LinkedIn, Twitter, etc.

    **Platform support:**

    - **Windows** — detects Chrome / Edge in Program Files, uses the
      profile from ``%LOCALAPPDATA%``.
    - **macOS** — detects Chrome / Edge in /Applications, profile from
      ``~/Library/Application Support``.
    - **Linux** — detects ``google-chrome-stable`` / ``microsoft-edge``
      on PATH, profile from ``~/.config``.
    - **WSL2 with WSLg** — same as Linux, with headed mode if
      ``$DISPLAY`` is set.
    - **WSL2 without WSLg** — falls back to headless mode; also picks up
      Windows-side profiles from ``/mnt/c/Users/.../Chrome/User Data``.

    Args:
        model: The LLM to power the researcher agent.
        browser_name: Preferred browser (``"chrome"``, ``"edge"``,
            ``"chromium"``).  Auto-detected when ``None``.
        profile_directory: Chrome profile subdirectory (e.g. ``"Default"``,
            ``"Profile 1"``).  Auto-detected when ``None``.
        headless: Force headed / headless mode.  Auto-detected based on
            display availability when ``None``.
        crawler_browser_config: Custom ``crawl4ai.BrowserConfig``.  Built
            automatically when ``None``.
    """
    # --- resolve profile & channel once, share across both browsers ---
    profile_info = _resolve_browser_profile(browser_name, profile_directory)
    channel = _get_chrome_channel()
    effective_headless = headless if headless is not None else (not _has_display())

    if effective_headless and headless is None:
        logger.info(
            "No display server found (WSL2 without WSLg / headless server) "
            "— both Playwright and crawl4ai will run headless"
        )

    # --- Playwright interactive tools ---
    tools: list[BaseTool] = await get_playwright_tools(
        profile_info=profile_info,
        headless=effective_headless,
        channel=channel,
    )

    # --- crawl4ai crawler tool ---
    if not crawler_browser_config:
        crawler_browser_config = build_crawler_browser_config(
            profile_info=profile_info,
            headless=effective_headless,
            channel=channel,
        )
    tools.append(build_web_crawler_tool(browser_config=crawler_browser_config))

    # --- DuckDuckGo search ---
    tools.append(DuckDuckGoSearchResults(name="simple_web_search"))

    # --- Human follow-up (captcha, login, clarification) ---
    tools.append(follow_up_with_human)

    # --- assemble agent ---
    agent = create_agent(
        tools,
        model,
        name="Web Researcher Agent",
        system_prompt=WEB_RESEARCHER_SYSTEM_PROMPT,
        middleware=[
            TodoListMiddleware(),
            SummarizationMiddleware(model=model),
            ToolRetryMiddleware(
                max_retries=2,
                retry_on=lambda exc: not isinstance(exc, GraphInterrupt),
                on_failure="continue",
                backoff_factor=2.0,
                initial_delay=1.0,
                max_delay=10.0,
                jitter=True,
            ),
        ],
    )

    return CompiledSubAgent(
        runnable=agent,
        name="Web Researcher",
        description=(
            "A sub-agent for in-depth web research using the user's real "
            "browser profile to bypass bot detection.  Can search the web, "
            "crawl pages for clean markdown content, and interactively "
            "browse sites (click, fill forms, extract data).  Effective on "
            "guarded platforms like LinkedIn, Twitter, and other sites that "
            "block automated access."
        ),
    )
