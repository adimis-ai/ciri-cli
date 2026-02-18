"""Shared retry helpers for ToolRetryMiddleware."""

from langgraph.errors import GraphInterrupt


def graphinterrupt_aware_failure(exc: Exception) -> str:
    """on_failure callable that re-raises GraphInterrupt but returns error messages for others.

    When used as `on_failure=graphinterrupt_aware_failure` in ToolRetryMiddleware,
    GraphInterrupt exceptions propagate correctly (enabling interrupt handling)
    while other tool errors are wrapped into a ToolMessage for the LLM.
    """
    if isinstance(exc, GraphInterrupt):
        raise exc
    exc_type = type(exc).__name__
    return f"Tool failed with {exc_type}: {exc}. Please try again."
