import asyncio
import os
from pathlib import Path
from src.subagents.web_researcher import get_playwright_tools, TakeScreenshotTool


async def test_screenshot():
    print("Initializing Playwright tools...")
    # Use a dummy temp dir for profile
    import tempfile

    temp_dir = Path(tempfile.mkdtemp())

    tools = await get_playwright_tools(user_data_dir=temp_dir, headless=True)

    screenshot_tool = next((t for t in tools if t.name == "take_screenshot"), None)
    navigate_tool = next((t for t in tools if t.name == "navigate_browser"), None)

    if not screenshot_tool or not navigate_tool:
        print("Error: Tools not found")
        return

    print("Navigating to https://example.com...")
    await navigate_tool.arun("https://example.com")

    print("Taking screenshot...")
    result = await screenshot_tool.arun({})

    if result.startswith("data:image/png;base64,"):
        print("Success! Screenshot captured.")
        print(f"Result starts with: {result[:50]}...")
    else:
        print(f"Failure: Result dose not look like a base64 image: {result[:100]}")


if __name__ == "__main__":
    asyncio.run(test_screenshot())
