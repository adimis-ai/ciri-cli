from langgraph.types import Command, interrupt
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from typing import List, Annotated, Optional
from typing_extensions import TypedDict, NotRequired


class FollowUpQuery(TypedDict):
    question: str
    options: NotRequired[List[str]]


class FollowUpInterruptValue(TypedDict):
    queries: List[FollowUpQuery]


@tool
def follow_up_with_human(
    queries: List[FollowUpQuery],
    *,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> str:
    """Use this tool to ask the user one or more follow-up questions when you need clarification or input."""
    responses = interrupt({"type": "human_follow_up", "queries": queries})

    # Format the responses into a summary for the tool output
    formatted_responses = []
    for q, r in zip(queries, responses):
        formatted_responses.append(f"Question: {q['question']}\nResponse: {r}")

    content = "\n\n".join(formatted_responses)

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"User responses:\n{content}",
                    tool_call_id=tool_call_id,
                )
            ]
        }
    )
