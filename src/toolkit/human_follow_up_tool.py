from langgraph.types import Command, interrupt
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from typing import List, Annotated, Optional
from typing_extensions import TypedDict, NotRequired


class FollowUpInterruptValue(TypedDict):
    question: str
    options: NotRequired[List[str]]


@tool(
    infer_schema=True,
    return_direct=True,
    name_or_callable="follow_up_with_human",
    description="Use this tool to ask the user a follow-up question when you need clarification or input.",
)
def follow_up_with_human(
    question: str,
    options: Optional[List[str]] = None,
    *,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> str:
    response = interrupt(
        {"type": "human_follow_up", "question": question, "options": options}
    )

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"User response: {response}",
                    tool_call_id=tool_call_id,
                )
            ]
        }
    )
