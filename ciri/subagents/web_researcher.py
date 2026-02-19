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

from ..prompts import WEB_RESEARCHER_SYSTEM_PROMPT

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
            self._browser = await self._pw.chromium.connect_over_cdp(self.cdp_endpoint)
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
        (
            _replacements[t.name].from_browser(async_browser=browser)
            if t.name in _replacements
            else t
        )
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
