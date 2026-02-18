import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Any, Type

from pydantic import BaseModel, Field, model_validator
from deepagents import CompiledSubAgent
from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langgraph.cache.memory import InMemoryCache
from langchain_core.language_models import BaseChatModel
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.base import BaseBrowserTool
from langchain_community.tools.playwright.utils import (
    aget_current_page,
    get_current_page,
)
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.errors import GraphInterrupt
from langchain.agents.middleware import (
    TodoListMiddleware,
    ToolRetryMiddleware,
)
from .._retry_helpers import graphinterrupt_aware_failure
from langchain.agents.middleware import (
    SummarizationMiddleware,
    HumanInTheLoopMiddleware,
)

if TYPE_CHECKING:
    from playwright.async_api import Browser as AsyncBrowser

from ..toolkit.web_crawler_tool import (
    build_web_crawler_tool,
    build_crawler_browser_config,
    BrowserConfig as CrawlerBrowserConfig,
)
from ..toolkit.human_follow_up_tool import follow_up_with_human

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Playwright tools — SPA-friendly overrides
# ---------------------------------------------------------------------------


class _NavigateToolInput(BaseModel):
    """Input for NavigateTool."""

    url: str = Field(..., description="url to navigate to")


class SPANavigateTool(BaseBrowserTool):
    """Navigate to a URL and wait for the page to fully load (network idle).

    This override uses ``wait_until="networkidle"`` so that JavaScript-heavy
    SPAs (Twitter/X, React apps, etc.) finish rendering before control is
    returned.
    """

    name: str = "navigate_browser"
    description: str = "Navigate a browser to the specified URL"
    args_schema: Type[BaseModel] = _NavigateToolInput

    def _run(
        self,
        url: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if self.sync_browser is None:
            raise ValueError(f"Synchronous browser not provided to {self.name}")
        page = get_current_page(self.sync_browser)
        response = page.goto(url, wait_until="networkidle")
        status = response.status if response else "unknown"
        return f"Navigating to {url} returned status code {status}"

    async def _arun(
        self,
        url: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        if self.async_browser is None:
            raise ValueError(f"Asynchronous browser not provided to {self.name}")
        page = await aget_current_page(self.async_browser)
        response = await page.goto(url, wait_until="networkidle")
        status = response.status if response else "unknown"
        return f"Navigating to {url} returned status code {status}"


class _ExtractTextToolInput(BaseModel):
    """Explicit no-args input for ExtractTextTool."""


class SPAExtractTextTool(BaseBrowserTool):
    """Extract visible text from the current page, stripping ``<noscript>`` tags.

    SPA sites like Twitter/X include ``<noscript>`` fallback content (e.g.
    "JavaScript is disabled") that pollutes extracted text.  This override
    removes those tags before extraction.
    """

    name: str = "extract_text"
    description: str = "Extract all the text on the current webpage"
    args_schema: Type[BaseModel] = _ExtractTextToolInput

    @model_validator(mode="before")
    @classmethod
    def _check_bs4(cls, values: dict) -> Any:
        try:
            from bs4 import BeautifulSoup  # noqa: F401
        except ImportError:
            raise ImportError(
                "The 'beautifulsoup4' package is required. "
                "Install it with 'pip install beautifulsoup4'."
            )
        return values

    def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        from bs4 import BeautifulSoup

        if self.sync_browser is None:
            raise ValueError(f"Synchronous browser not provided to {self.name}")
        page = get_current_page(self.sync_browser)
        html_content = page.content()
        soup = BeautifulSoup(html_content, "lxml")
        for tag in soup.find_all("noscript"):
            tag.decompose()
        return " ".join(text for text in soup.stripped_strings)

    async def _arun(
        self, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        from bs4 import BeautifulSoup

        if self.async_browser is None:
            raise ValueError(f"Asynchronous browser not provided to {self.name}")
        page = await aget_current_page(self.async_browser)
        html_content = await page.content()
        soup = BeautifulSoup(html_content, "lxml")
        for tag in soup.find_all("noscript"):
            tag.decompose()
        return " ".join(text for text in soup.stripped_strings)


# ---------------------------------------------------------------------------
# Playwright CDP connection
# ---------------------------------------------------------------------------


class PlaywrightCDPConnection:
    """Connect to an already-running browser via Chrome DevTools Protocol."""

    def __init__(self, cdp_endpoint: str):
        self.cdp_endpoint = cdp_endpoint
        self._pw = None
        self._browser = None

    async def get_async_browser(self) -> "AsyncBrowser":
        if self._browser is None:
            from playwright.async_api import async_playwright

            self._pw = await async_playwright().start()
            self._browser = await self._pw.chromium.connect_over_cdp(
                self.cdp_endpoint
            )
        return self._browser


class NoInput(BaseModel):
    """Schema for tools that take no input."""

    pass


class TakeScreenshotTool(BaseTool):
    """Tool for taking a screenshot of the current webpage."""

    name: str = "take_screenshot"
    description: str = (
        "Takes a screenshot of the current webpage and returns it as a base64-encoded PNG data URL. Only use this when you need visual information about the page."
    )
    args_schema: Type[BaseModel] = NoInput
    async_browser: Any = Field(default=None, exclude=True)

    def _run(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Use _arun instead")

    async def _arun(self, **kwargs: Any) -> str:
        """Capture a screenshot of the current page."""
        if not self.async_browser:
            return "Error: Browser not initialized"

        try:
            # Get the current active page
            contexts = self.async_browser.contexts
            if not contexts:
                return "Error: No active browser context"

            pages = contexts[0].pages
            if not pages:
                return "Error: No active page"

            page = pages[0]

            import base64

            screenshot_bytes = await page.screenshot(type="png", full_page=False)
            base64_image = base64.b64encode(screenshot_bytes).decode("utf-8")
            return f"data:image/png;base64,{base64_image}"
        except Exception as e:
            return f"Error taking screenshot: {e}"


async def get_playwright_tools(cdp_endpoint: str):
    """Get Playwright browser tools by connecting to the user's real browser
    via Chrome DevTools Protocol.

    Args:
        cdp_endpoint: The CDP endpoint URL (e.g. ``"http://localhost:9222"``).
    """
    logger.debug(
        "[get_playwright_tools] Connecting to real browser via CDP at %s",
        cdp_endpoint,
    )

    connection = PlaywrightCDPConnection(cdp_endpoint)
    browser = await connection.get_async_browser()

    adapter = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
    tools = adapter.get_tools()

    # Replace default navigate & extract_text with SPA-friendly versions
    # that wait for networkidle and strip <noscript> tags respectively.
    _replacements = {
        "navigate_browser": SPANavigateTool,
        "extract_text": SPAExtractTextTool,
    }
    tools = [
        _replacements[t.name].from_browser(async_browser=browser)
        if t.name in _replacements
        else t
        for t in tools
    ]

    # Add the screenshot tool
    tools.append(TakeScreenshotTool(async_browser=adapter.async_browser))

    logger.info(
        "Connected to real browser via CDP at %s — tools: %s",
        cdp_endpoint,
        [tool.name for tool in tools],
    )
    return tools


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

You have access to **11 tools** across four categories.  Always use the \
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
| 8 | `take_screenshot`    | *(none)*                           | Returns the current page as a base64-encoded PNG image               | When you need to "see" the page (charts, layout, maps, or visual data)         |

## B. Web Crawler (crawl4ai)
| # | Tool name     | Key inputs                                          | What it does                                                                  |
|---|---------------|-----------------------------------------------------|-------------------------------------------------------------------------------|
| 9 | `web_crawler`  | `url`, `deep_crawl: bool`, `max_depth`, `max_pages`, `word_count_threshold`, `bypass_cache` | Extracts clean, LLM-ready **markdown** from a page.  With `deep_crawl=True` it performs BFS to discover and crawl linked pages across an entire site. |

## C. Web Search
| # | Tool name           | Input          | What it does                                   |
|---|---------------------|----------------|-------------------------------------------------|
| 10 | `simple_web_search` | `query: str`  | DuckDuckGo search — returns result snippets & URLs |

## D. Human Follow-Up
| #  | Tool name               | Input                                      | What it does                                                                |
|----|-------------------------|--------------------------------------------|------------------------------------------------------------------------------|
| 11 | `follow_up_with_human`  | `queries: list[dict]` | Pauses execution and asks the **user** one or more questions; resumes with their answers. Each query in the list should be a dict with `question` and optional `options` (list of strings). |

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
  ├─ Take a visual snapshot ──► take_screenshot
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
- Always provide clear, actionable `options` when possible — this makes it easy for the user to respond quickly.
- You can batch multiple questions into a single tool call when you need several pieces of information at once.
- After receiving the user's responses, immediately act on them.
- Do NOT call this for things you can figure out yourself — exhaust your own tools first.

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
    cdp_endpoint: Optional[str] = None,
    crawler_browser_config: Optional[CrawlerBrowserConfig] = None,
    all_allowed: bool = False,
) -> CompiledSubAgent:
    """Build a web-researcher sub-agent that connects to the user's real
    browser via Chrome DevTools Protocol (CDP).

    The agent operates on the user's actual running browser — real tabs,
    real cookies, real sessions, real extensions.

    Args:
        model: The LLM to power the researcher agent.
        cdp_endpoint: The CDP HTTP endpoint (e.g. ``"http://localhost:9222"``).
            When ``None``, Playwright interactive tools and the CDP-based
            crawler are omitted; only DuckDuckGo search is available.
        crawler_browser_config: Custom ``crawl4ai.BrowserConfig``.  Built
            automatically when ``None``.
        all_allowed: If ``True``, all tool calls are auto-approved. If ``False``,
            certain tools require human approval via ``HumanInTheLoopMiddleware``.
    """
    tools: list[BaseTool] = []

    # --- Playwright interactive tools via CDP (requires a running browser) ---
    if cdp_endpoint:
        tools.extend(await get_playwright_tools(cdp_endpoint=cdp_endpoint))

    # --- crawl4ai crawler tool ---
    if cdp_endpoint or crawler_browser_config:
        if not crawler_browser_config:
            crawler_browser_config = build_crawler_browser_config(
                cdp_url=cdp_endpoint,
            )
        tools.append(build_web_crawler_tool(browser_config=crawler_browser_config))

    # --- DuckDuckGo search ---
    tools.append(DuckDuckGoSearchResults(name="simple_web_search"))

    # --- Human follow-up (captcha, login, clarification) ---
    tools.append(follow_up_with_human)

    # --- assemble agent ---
    middleware = [
        TodoListMiddleware(),
        SummarizationMiddleware(model=model),
        ToolRetryMiddleware(
            max_retries=2,
            retry_on=lambda exc: not isinstance(exc, GraphInterrupt),
            on_failure=graphinterrupt_aware_failure,
            backoff_factor=2.0,
            initial_delay=1.0,
            max_delay=10.0,
            jitter=True,
        ),
    ]

    if not all_allowed:
        middleware.append(
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "navigate_browser": True,
                    "click_element": True,
                    "web_crawler": True,
                    "simple_web_search": True,
                    "take_screenshot": True,
                }
            )
        )

    agent = create_agent(
        model=model,
        tools=tools,
        cache=InMemoryCache(),
        name="web_research_agent",
        system_prompt=WEB_RESEARCHER_SYSTEM_PROMPT,
        middleware=middleware,
    )

    return CompiledSubAgent(
        runnable=agent,
        name="web_research_agent",
        description=(
            "Web research agent using the user's real browser (cookies, sessions) "
            "via CDP. Can search, crawl pages to markdown, and interactively browse "
            "JS-heavy/authenticated sites. Invoke for any internet lookup, API doc "
            "research, or live data extraction."
        ),
    )
