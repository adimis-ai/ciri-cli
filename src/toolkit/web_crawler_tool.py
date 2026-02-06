import logging
import asyncio
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from crawl4ai.deep_crawling.bfs_strategy import BFSDeepCrawlStrategy
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, BrowserConfig

logger = logging.getLogger(__name__)


class WebCrawlerInput(BaseModel):
    url: str = Field(
        description="The primary URL of the webpage to crawl. Use this for extracting structured or textual content from known web pages."
    )
    word_count_threshold: Optional[int] = Field(
        default=10,
        description="Minimum word count required for a content block to be included. Higher values filter out noise like navigation menus.",
    )
    bypass_cache: Optional[bool] = Field(
        default=False,
        description="If True, ignores previously cached results and performs a fresh crawl. Use this for dynamic or rapidly changing content.",
    )
    deep_crawl: Optional[bool] = Field(
        default=False,
        description="Enables a Breadth-First Search (BFS) crawl starting from the initial URL to discover and extract content from linked internal pages.",
    )
    max_depth: Optional[int] = Field(
        default=2,
        description="The maximum number of link 'hops' from the starting URL (only applicable if deep_crawl=True).",
    )
    max_pages: Optional[int] = Field(
        default=10,
        description="The maximum total number of unique pages to crawl during a deep crawl operation (only applicable if deep_crawl=True).",
    )


def build_web_crawler_tool(
    name: str = "web_crawler",
    description: str = (
        "High-performance specialized web crawler for extracting clean, LLM-ready markdown content from websites. "
        "Best used when you need the textual content of specific pages or need to discover information across a site's structure via deep crawling (BFS). "
        "For interactive tasks like filling forms, clicking buttons, or navigating complex JS apps, use 'web_surfer_tool' instead."
    ),
    *,
    browser_config: Optional[BrowserConfig] = None,
) -> StructuredTool:
    """
    Builds a generic LangChain StructuredTool for web crawling using crawl4ai.
    Supports both sync and async execution.
    """

    async def arun_crawl(
        url: str,
        word_count_threshold: int = 10,
        bypass_cache: bool = False,
        deep_crawl: bool = False,
        max_depth: int = 2,
        max_pages: int = 10,
    ) -> str:
        """Async implementation of the web crawler tool."""

        deep_crawl_strategy = None
        if deep_crawl:
            deep_crawl_strategy = BFSDeepCrawlStrategy(
                max_depth=max_depth, max_pages=max_pages
            )

        config = CrawlerRunConfig(
            word_count_threshold=word_count_threshold,
            cache_mode=CacheMode.BYPASS if bypass_cache else CacheMode.ENABLED,
            deep_crawl_strategy=deep_crawl_strategy,
        )

        try:
            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await crawler.arun(url=url, config=config)
                if hasattr(result, "success") and result.success:
                    output = f"Primary URL: {url}\n\nContent:\n{result.markdown}"
                    if deep_crawl:
                        output += f"\n\n[Deep Crawl enabled: max_depth={max_depth}, max_pages={max_pages}]"
                    return output
                elif isinstance(result, list):
                    # Some versions might return a list for deep crawls
                    return "\n\n---\n\n".join([r.markdown for r in result if r.success])
                else:
                    error_msg = getattr(result, "error_message", "Unknown error")
                    return f"Failed to crawl {url}: {error_msg}"
        except Exception as e:
            logger.error(f"Error executing crawl4ai async crawl: {e}")
            return f"Error: {str(e)}"

    def run_crawl(
        url: str,
        word_count_threshold: int = 10,
        bypass_cache: bool = False,
        deep_crawl: bool = False,
        max_depth: int = 2,
        max_pages: int = 10,
    ) -> str:
        """Sync implementation of the web crawler tool."""
        try:
            return asyncio.run(
                arun_crawl(
                    url,
                    word_count_threshold,
                    bypass_cache,
                    deep_crawl,
                    max_depth,
                    max_pages,
                )
            )
        except Exception as e:
            logger.error(f"Error executing crawl4ai sync crawl: {e}")
            return f"Error: {str(e)}"

    return StructuredTool.from_function(
        func=run_crawl,
        coroutine=arun_crawl,
        name=name,
        description=description,
        args_schema=WebCrawlerInput,
    )
