import logging
from typing import Optional
from pydantic import BaseModel, Field
from browser_use.llm.base import BaseChatModel
from langchain_core.tools import StructuredTool
from browser_use import Agent, Browser, AgentHistoryList, sandbox

logger = logging.getLogger(__name__)


class BrowserUseInput(BaseModel):
    task: str = Field(
        description=(
            "A natural language description of the web automation task. "
            "Examples: 'Log in to LinkedIn and search for AI companies', 'Go to amazon.com and find the price of the latest iPhone', "
            "'Navigate to the settings page and toggle the dark mode switch'."
        )
    )


def build_web_surfer_tool(
    llm: BaseChatModel,
    name: str = "web_surfer_tool",
    description: str = (
        "Advanced browser automation tool so that you can surf the web and can interact with websites like a human. "
        "Use this for complex scenarios requiring multi-step navigation, form filling, clicking buttons, handling authentication, "
        "or scraping content from JavaScript-heavy Single Page Applications (SPAs). "
        "Define tasks clearly in natural language."
    ),
    *,
    browser: Optional[Browser] = None,
) -> StructuredTool:
    """
    Builds a generic LangChain StructuredTool for browser automation using browser-use.
    Supports both sync and async execution.
    """
    # Use provided browser or create a default instance
    _browser = browser or Browser()

    @sandbox()
    async def run_browser_task(task: str, browser: Browser) -> AgentHistoryList:
        """Async implementation of the browser use tool."""
        try:
            agent = Agent(llm=llm, browser=browser, task=task)
            result = await agent.run()
            return result
        except Exception as e:
            logger.error(f"Error executing browser use async task: {e}")
            return f"Error: {str(e)}"

    @sandbox()
    def sync_browser_task(task: str, browser: Browser) -> AgentHistoryList:
        """Sync implementation of the browser use tool."""
        try:
            agent = Agent(llm=llm, browser=browser, task=task)
            result = agent.run_sync()
            return result
        except Exception as e:
            logger.error(f"Error executing browser use sync task: {e}")
            return f"Error: {str(e)}"

    # Wrapper functions that pass the browser instance
    async def async_wrapper(task: str) -> AgentHistoryList:
        return await run_browser_task(task, _browser)
    
    def sync_wrapper(task: str) -> AgentHistoryList:
        return sync_browser_task(task, _browser)

    return StructuredTool.from_function(
        func=sync_wrapper,
        async_func=async_wrapper,
        name=name,
        description=description,
        args_schema=BrowserUseInput,
    )
