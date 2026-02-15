import os
import json
import logging
import pickle
from pathlib import Path
from datetime import datetime, date
from functools import cached_property
from typing_extensions import NotRequired, TypedDict
from typing import (
    Any,
    Dict,
    Optional,
    Union,
    Sequence,
    Literal,
    Tuple,
    Mapping,
    NotRequired,
    List,
)

from langchain.agents import AgentState
from langgraph.types import StateSnapshot
from langchain_core.load import dumpd, loads
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.messages import AnyMessage, BaseMessage
from langchain.chat_models import BaseChatModel, init_chat_model
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langchain.agents.middleware.shell_tool import RedactionRule

from .toolkit.human_follow_up_tool import FollowUpInterruptValue

logger = logging.getLogger(__name__)
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class LLMConfig(BaseModel):
    """Configuration for language models."""

    model: str = Field(
        description="The language model to use, e.g. 'openai/gpt-oss-120b:free' or 'openai:gpt-4'."
    )
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    @cached_property
    def _parsed_model(self) -> Tuple[str | None, str]:
        if ":" in self.model:  # langchain direct provider:model
            return tuple(self.model.split(":", 1))
        if "/" in self.model:  # openrouter provider/model
            return tuple(self.model.split("/", 1))
        return None, self.model

    @cached_property
    def _is_openrouter(self) -> bool:
        # provider/model AND provider not explicitly specified via provider:model
        return "/" in self.model and ":" not in self.model

    @cached_property
    def _resolved_api_config(self) -> Dict[str, Any]:
        """
        Resolve API config once and cache.
        Avoid repeated env access + dict copies.
        """
        config = dict(self.model_kwargs)  # single copy

        # Fast path if api_key already provided
        if "api_key" in config:
            if self._is_openrouter and "base_url" not in config:
                config["base_url"] = os.getenv(
                    "OPENROUTER_API_BASE_URL", DEFAULT_OPENROUTER_BASE_URL
                )
            return config

        provider, _ = self._parsed_model

        if self._is_openrouter:
            config.setdefault(
                "base_url",
                os.getenv("OPENROUTER_API_BASE_URL", DEFAULT_OPENROUTER_BASE_URL),
            )
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY env variable not set.")
            config["api_key"] = api_key
            return config

        # Direct provider key lookup
        if not provider:
            raise ValueError(f"Provider missing in model: {self.model}")

        env_key = f"{provider.upper()}_API_KEY"
        api_key = os.getenv(env_key)
        if not api_key:
            raise ValueError(f"{env_key} env variable not set for {self.model}")

        config["api_key"] = api_key
        return config

    def init_langchain_model(self) -> BaseChatModel:
        """
        Initialize LangChain chat model.
        Zero redundant env reads or parsing.
        """
        provider, model_name = self._parsed_model
        config = self._resolved_api_config

        if self._is_openrouter:
            # avoid dict rebuild via pop-free filtering
            base_url = config.get("base_url")
            api_key = config["api_key"]

            extra = {
                k: v for k, v in config.items() if k not in ("api_key", "base_url")
            }

            return init_chat_model(
                model=model_name,
                model_provider="openai",
                base_url=base_url,
                api_key=api_key,
                **extra,
            )

        return init_chat_model(model=self.model, **config)


class ShellToolConfig(BaseModel):
    """Configuration for shell tool middleware."""

    env: Optional[Mapping[str, Any]] = None
    shell_command: Optional[Union[Sequence[str], str]] = None
    startup_commands: Optional[Union[tuple, list, str]] = None
    shutdown_commands: Optional[Union[tuple, list, str]] = None
    redaction_rules: tuple[RedactionRule, ...] | list[RedactionRule] | None = None


class SerializableSubAgent(BaseModel):
    """Serializable configuration for subagents."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    system_prompt: str
    use_parent_mcp_tools: bool = True
    include_follow_up_with_human_tool: bool = True
    include_shell_tool_middleware: bool = True
    llm_config: Optional[LLMConfig] = None
    interrupt_on: Optional[Dict[str, Any]] = (
        None  # Union[bool, InterruptOnConfig] - Any used for Pydantic compatibility
    )
    shell_tool_config: Optional[ShellToolConfig] = None
    mcp_connections: Optional[Dict[str, Any]] = None


class ActionRequest(TypedDict):
    name: str
    description: str
    arguments: NotRequired[dict[str, Any]]


class ReviewConfig(TypedDict):
    action_name: str
    allowed_decisions: list[Literal["approve", "edit", "reject"]]


class HumanInTheLoopInterrupt(TypedDict):
    review_configs: list[ReviewConfig]
    action_requests: list[ActionRequest]


class InterruptValue(TypedDict):
    value: Union[HumanInTheLoopInterrupt, FollowUpInterruptValue]


class ApprovalDecision(TypedDict):
    type: Literal["approve"]


class EditedAction(TypedDict):
    name: str
    args: dict[str, Any]


class EditDecision(TypedDict):
    type: Literal["edit"]
    edited_action: EditedAction


class RejectDecision(TypedDict):
    type: Literal["reject"]
    message: NotRequired[str]


class ApprovalDecisions(TypedDict):
    decisions: list[ApprovalDecision]


class EditDecisions(TypedDict):
    decisions: list[EditDecision]


class RejectDecisions(TypedDict):
    decisions: list[RejectDecision]


class ResumeCommand(TypedDict):
    resume: Union[ApprovalDecisions, EditDecisions, RejectDecisions]


class CiriState(
    AgentState[Any],
):
    __interrupt__: NotRequired[Optional[List[InterruptValue]]]


class CiriJsonPlusSerializer(JsonPlusSerializer):
    """Extended JsonPlusSerializer that handles non-serializable types gracefully.

    Overrides dumps_typed to catch pickle errors (e.g. _thread.lock objects)
    that occur when the state contains threading primitives or other
    unpickleable objects from middleware/tools.
    """

    def __init__(self, **kwargs):
        super().__init__(pickle_fallback=True, **kwargs)

    def _needs_precleaning(self, obj) -> bool:
        """Check if object type likely contains unpickleable items."""
        if obj is None or isinstance(obj, (str, int, float, bool, bytes)):
            return False

        type_name = type(obj).__name__

        # LangGraph types often contain locks
        if type_name in ("Send", "Command", "StateSnapshot"):
            return True

        # Check for module hints
        module = getattr(type(obj), "__module__", "")
        if module.startswith(("langgraph", "_thread", "threading")):
            return True

        # Dicts and lists might contain locks
        if isinstance(obj, (dict, list)):
            return True

        # Objects with __dict__ or __slots__ may have lock attributes
        if hasattr(obj, "__dict__") or hasattr(obj, "__slots__"):
            return True

        return False

    def dumps_typed(self, obj):
        # Pre-clean objects that likely contain unpickleable items
        if self._needs_precleaning(obj):
            obj = self._strip_unpickleable(obj)

        try:
            return super().dumps_typed(obj)
        except (TypeError, pickle.PicklingError, AttributeError) as e:
            # Fallback for any remaining edge cases
            logger.warning(
                "Pickle fallback failed for %s: %s — retrying after cleanup",
                type(obj).__name__,
                e,
            )
            cleaned = self._strip_unpickleable(obj)
            return super().dumps_typed(cleaned)

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        """Deserialize data, reconstructing LangGraph types from dict representations.

        This is called by LangGraph when loading checkpointed state. We need to
        reconstruct Send objects from their dict serialized form to avoid
        "Ignoring invalid packet type <class 'dict'> in pending sends" warnings.
        """
        result = super().loads_typed(data)
        return self._reconstruct_langgraph_types(result)

    @classmethod
    def _reconstruct_langgraph_types(cls, obj: Any) -> Any:
        """Recursively reconstruct LangGraph types from dict representations."""
        from langgraph.types import Send

        if obj is None or isinstance(obj, (str, int, float, bool, bytes)):
            return obj

        # Reconstruct Send objects from dicts with _type: "Send"
        if isinstance(obj, dict):
            if obj.get("_type") == "Send" and "node" in obj:
                node = obj.get("node")
                arg = cls._reconstruct_langgraph_types(obj.get("arg"))
                return Send(node=node, arg=arg)

            # Recursively process dict values
            return {k: cls._reconstruct_langgraph_types(v) for k, v in obj.items()}

        # Recursively process lists
        if isinstance(obj, list):
            return [cls._reconstruct_langgraph_types(item) for item in obj]

        # Recursively process tuples
        if isinstance(obj, tuple):
            return tuple(cls._reconstruct_langgraph_types(item) for item in obj)

        return obj

    @staticmethod
    def _strip_unpickleable(obj, _seen=None):
        """Recursively replace unpickleable values with their string repr.

        Proactively strips known unpickleable types (locks, threads, etc.)
        and recursively processes object attributes before attempting pickle test.
        """
        import threading

        if _seen is None:
            _seen = set()

        # Handle None and primitives early
        if obj is None or isinstance(obj, (str, int, float, bool, bytes)):
            return obj

        # Avoid circular references
        try:
            obj_id = id(obj)
            if obj_id in _seen:
                return None
            _seen.add(obj_id)
        except TypeError:
            # Some objects don't support id()
            pass

        # Handle known unpickleable types directly
        type_name = type(obj).__name__
        module_name = getattr(type(obj), "__module__", "")

        # Direct exclusion for thread-related types
        if (
            type_name in ("lock", "Lock", "RLock", "_thread.lock", "_RLock")
            or "lock" in type_name.lower()
            or module_name.startswith("_thread")
            or isinstance(obj, type(threading.Lock()))
        ):
            return f"<{type_name}>"

        # Handle dicts
        if isinstance(obj, dict):
            return {
                k: CiriJsonPlusSerializer._strip_unpickleable(v, _seen)
                for k, v in obj.items()
                if not (isinstance(k, str) and k.startswith("_lock"))  # Skip lock keys
            }

        # Handle lists and tuples
        if isinstance(obj, (list, tuple)):
            items = [CiriJsonPlusSerializer._strip_unpickleable(i, _seen) for i in obj]
            return tuple(items) if isinstance(obj, tuple) else items

        # Handle sets
        if isinstance(obj, (set, frozenset)):
            items = [CiriJsonPlusSerializer._strip_unpickleable(i, _seen) for i in obj]
            return frozenset(items) if isinstance(obj, frozenset) else set(items)

        # Handle LangGraph Send objects specially - extract only safe attributes
        if type_name == "Send" and hasattr(obj, "node") and hasattr(obj, "arg"):
            return {
                "_type": "Send",
                "node": CiriJsonPlusSerializer._strip_unpickleable(
                    getattr(obj, "node", None), _seen
                ),
                "arg": CiriJsonPlusSerializer._strip_unpickleable(
                    getattr(obj, "arg", None), _seen
                ),
            }

        # Handle objects with __dict__ or __slots__ - recursively clean their attributes
        cleaned_dict = {}
        has_internal_data = False

        if hasattr(obj, "__dict__"):
            has_internal_data = True
            for k, v in obj.__dict__.items():
                # Skip private lock-like attributes
                if isinstance(k, str) and ("lock" in k.lower() or k.startswith("_")):
                    continue
                cleaned_dict[k] = CiriJsonPlusSerializer._strip_unpickleable(v, _seen)

        if hasattr(obj, "__slots__"):
            has_internal_data = True
            for slot in obj.__slots__:
                if hasattr(obj, slot):
                    # Skip lock-like slots
                    if "lock" in slot.lower():
                        continue
                    cleaned_dict[slot] = CiriJsonPlusSerializer._strip_unpickleable(
                        getattr(obj, slot), _seen
                    )

        # If we extracted internal data, try to return a clean dict representation
        if has_internal_data and cleaned_dict:
            cleaned_dict["_type"] = type_name
            return cleaned_dict

        # Final fallback: try to pickle; if it fails, return string representation
        try:
            pickle.dumps(obj)
            return obj
        except (TypeError, pickle.PicklingError, AttributeError):
            return str(obj)


class CiriSerializer:
    """Comprehensive serializer for Ciri types with proper JSON handling."""

    @staticmethod
    def serialize_any_message(message: AnyMessage) -> Dict[str, Any]:
        """Serialize a LangChain message to JSON-compatible dict.

        Args:
            message: LangChain message of any type

        Returns:
            JSON-serializable dictionary representation
        """
        try:
            if isinstance(message, BaseMessage):
                return dumpd(message)
            elif hasattr(message, "model_dump"):
                return message.model_dump()
            elif hasattr(message, "dict"):
                return message.dict()
            else:
                # Fallback for dictionary-like messages
                return (
                    dict(message)
                    if hasattr(message, "__iter__")
                    else {"content": str(message)}
                )
        except Exception as e:
            logger.warning(f"Failed to serialize message {type(message)}: {e}")
            return {
                "content": str(message),
                "type": type(message).__name__,
                "serialization_error": str(e),
            }

    @staticmethod
    def deserialize_any_message(data: Dict[str, Any]) -> AnyMessage:
        """Deserialize a dict back to a LangChain message.

        Args:
            data: JSON dictionary representation

        Returns:
            Deserialized LangChain message
        """
        try:
            if "serialization_error" in data:
                # Handle previously failed serializations
                from langchain_core.messages import HumanMessage

                return HumanMessage(content=data.get("content", ""))
            return loads(data)
        except Exception as e:
            logger.warning(f"Failed to deserialize message data {data}: {e}")
            from langchain_core.messages import HumanMessage

            return HumanMessage(content=str(data.get("content", data)))

    @staticmethod
    def serialize_ciri_state(state: CiriState) -> Dict[str, Any]:
        """Serialize CiriState to JSON-compatible dict.

        Args:
            state: CiriState instance

        Returns:
            JSON-serializable dictionary representation
        """
        try:
            result = {}

            # Handle messages separately for proper serialization
            if "messages" in state:
                result["messages"] = [
                    CiriSerializer.serialize_any_message(msg)
                    for msg in state["messages"]
                ]

            # Handle interrupts
            if "__interrupt__" in state and state["__interrupt__"]:
                result["__interrupt__"] = [
                    CiriSerializer.serialize_interrupt(interrupt)
                    for interrupt in state["__interrupt__"]
                ]

            # Handle other fields
            for key, value in state.items():
                if key in ["messages", "__interrupt__"]:
                    continue  # Already handled above

                result[key] = CiriSerializer._serialize_value(value)

            return result
        except Exception as e:
            logger.error(f"Failed to serialize CiriState: {e}")
            return {"error": f"Serialization failed: {e}", "original_type": "CiriState"}

    @staticmethod
    def deserialize_ciri_state(data: Dict[str, Any]) -> CiriState:
        """Deserialize dict back to CiriState.

        Args:
            data: JSON dictionary representation

        Returns:
            Deserialized CiriState
        """
        try:
            result = {}

            # Handle messages
            if "messages" in data:
                result["messages"] = [
                    CiriSerializer.deserialize_any_message(msg_data)
                    for msg_data in data["messages"]
                ]
            else:
                result["messages"] = []

            # Handle interrupts
            if "__interrupt__" in data and data["__interrupt__"]:
                result["__interrupt__"] = [
                    CiriSerializer.deserialize_interrupt(interrupt_data)
                    for interrupt_data in data["__interrupt__"]
                ]

            # Handle other fields
            for key, value in data.items():
                if key not in ["messages", "__interrupt__"]:
                    result[key] = value

            return result  # Return dict directly since CiriState is a TypedDict
        except Exception as e:
            logger.error(f"Failed to deserialize CiriState: {e}")
            return {"messages": [], "error": f"Deserialization failed: {e}"}

    @staticmethod
    def serialize_state_snapshot(snapshot: StateSnapshot) -> Dict[str, Any]:
        """Serialize LangGraph StateSnapshot to JSON-compatible dict.

        Args:
            snapshot: LangGraph StateSnapshot instance

        Returns:
            JSON-serializable dictionary representation
        """
        try:
            # Handle created_at - it may already be a string or a datetime object
            created_at = None
            if snapshot.created_at:
                if isinstance(snapshot.created_at, str):
                    created_at = snapshot.created_at
                elif hasattr(snapshot.created_at, "isoformat"):
                    created_at = snapshot.created_at.isoformat()
                else:
                    created_at = str(snapshot.created_at)

            result = {
                "config": dict(snapshot.config) if snapshot.config else {},
                "metadata": dict(snapshot.metadata) if snapshot.metadata else {},
                "next": list(snapshot.next) if snapshot.next else [],
                "created_at": created_at,
                "parent_config": (
                    dict(snapshot.parent_config) if snapshot.parent_config else None
                ),
            }

            # Handle values (which should be CiriState)
            if snapshot.values:
                if isinstance(snapshot.values, dict) and "messages" in snapshot.values:
                    result["values"] = CiriSerializer.serialize_ciri_state(
                        snapshot.values
                    )
                else:
                    result["values"] = CiriSerializer._serialize_value(snapshot.values)
            else:
                result["values"] = None

            # Handle tasks
            if hasattr(snapshot, "tasks") and snapshot.tasks:
                result["tasks"] = [
                    CiriSerializer._serialize_value(task) for task in snapshot.tasks
                ]
            else:
                result["tasks"] = []

            # Handle interrupts (tasks may contain interrupts, or snapshot may have them)
            interrupts = []
            if hasattr(snapshot, "tasks") and snapshot.tasks:
                for task in snapshot.tasks:
                    if hasattr(task, "interrupts") and task.interrupts:
                        interrupts.extend(
                            CiriSerializer._serialize_value(interrupt)
                            for interrupt in task.interrupts
                        )
            result["interrupts"] = interrupts

            return result
        except Exception as e:
            logger.error(f"Failed to serialize StateSnapshot: {e}")
            return {
                "error": f"Serialization failed: {e}",
                "original_type": "StateSnapshot",
            }

    @staticmethod
    def serialize_interrupt(interrupt: InterruptValue) -> Dict[str, Any]:
        """Serialize interrupt value to JSON-compatible dict.

        Args:
            interrupt: InterruptValue instance

        Returns:
            JSON-serializable dictionary representation
        """
        try:
            return {"value": CiriSerializer._serialize_value(interrupt["value"])}
        except Exception as e:
            logger.error(f"Failed to serialize interrupt: {e}")
            return {
                "error": f"Serialization failed: {e}",
                "original_type": "InterruptValue",
            }

    @staticmethod
    def deserialize_interrupt(data: Dict[str, Any]) -> InterruptValue:
        """Deserialize dict back to InterruptValue.

        Args:
            data: JSON dictionary representation

        Returns:
            Deserialized InterruptValue
        """
        try:
            return {
                "value": data["value"]
            }  # Return dict directly since InterruptValue is a TypedDict
        except Exception as e:
            logger.error(f"Failed to deserialize interrupt: {e}")
            return {"value": {"error": f"Deserialization failed: {e}"}}

    @staticmethod
    def serialize_resume_command(command: ResumeCommand) -> Dict[str, Any]:
        """Serialize resume command to JSON-compatible dict.

        Args:
            command: ResumeCommand instance

        Returns:
            JSON-serializable dictionary representation
        """
        try:
            return {"resume": CiriSerializer._serialize_value(command["resume"])}
        except Exception as e:
            logger.error(f"Failed to serialize resume command: {e}")
            return {
                "error": f"Serialization failed: {e}",
                "original_type": "ResumeCommand",
            }

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """Recursively serialize any value to JSON-compatible format.

        Args:
            value: Any value to serialize

        Returns:
            JSON-serializable representation
        """
        if value is None:
            return None
        elif isinstance(value, (str, int, float, bool)):
            return value
        elif isinstance(value, (datetime, date)):
            return value.isoformat()
        elif isinstance(value, Path):
            return str(value)
        elif isinstance(value, BaseMessage):
            # Handle LangChain messages specifically
            return CiriSerializer.serialize_any_message(value)
        elif isinstance(value, BaseModel):
            return value.model_dump() if hasattr(value, "model_dump") else value.dict()
        elif hasattr(value, "model_dump"):
            return value.model_dump()
        elif hasattr(value, "dict"):
            return value.dict()
        # Handle LangGraph Send objects specifically
        elif (
            hasattr(value, "__class__")
            and value.__class__.__name__ == "Send"
            and hasattr(value, "node")
            and hasattr(value, "arg")
        ):
            return {
                "_type": "Send",
                "node": value.node,
                "arg": CiriSerializer._serialize_value(value.arg),
            }
        # Handle other LangGraph types
        elif (
            hasattr(value, "__class__")
            and value.__class__.__module__ == "langgraph.types"
        ):
            # Handle Command objects
            if value.__class__.__name__ == "Command":
                result = {"_type": "Command"}
                for field in ["update", "goto", "graph", "resume"]:
                    if hasattr(value, field):
                        field_value = getattr(value, field, None)
                        if field_value is not None:
                            result[field] = CiriSerializer._serialize_value(field_value)
                return result
            # Handle other LangGraph types generically
            else:
                result = {"_type": value.__class__.__name__}
                # Try to extract attributes from __slots__ or __dict__
                if hasattr(value, "__slots__"):
                    for slot in value.__slots__:
                        if hasattr(value, slot):
                            result[slot] = CiriSerializer._serialize_value(
                                getattr(value, slot)
                            )
                elif hasattr(value, "__dict__"):
                    for k, v in value.__dict__.items():
                        result[k] = CiriSerializer._serialize_value(v)
                return result
        elif isinstance(value, dict):
            return {k: CiriSerializer._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [CiriSerializer._serialize_value(item) for item in value]
        elif isinstance(value, set):
            return list(value)
        else:
            try:
                # Try standard JSON serialization
                json.dumps(value)
                return value
            except (TypeError, ValueError):
                # Fallback to string representation
                return str(value)


# Convenience functions for direct use
def serialize_any_message(message: AnyMessage) -> Dict[str, Any]:
    """Serialize a LangChain message to JSON-compatible dict."""
    return CiriSerializer.serialize_any_message(message)


def deserialize_any_message(data: Dict[str, Any]) -> AnyMessage:
    """Deserialize a dict back to a LangChain message."""
    return CiriSerializer.deserialize_any_message(data)


def serialize_state_snapshot(snapshot: StateSnapshot) -> Dict[str, Any]:
    """Serialize LangGraph StateSnapshot to JSON-compatible dict."""
    return CiriSerializer.serialize_state_snapshot(snapshot)


def serialize_interrupt(interrupt: InterruptValue) -> Dict[str, Any]:
    """Serialize interrupt value to JSON-compatible dict."""
    return CiriSerializer.serialize_interrupt(interrupt)


def deserialize_interrupt(data: Dict[str, Any]) -> InterruptValue:
    """Deserialize dict back to InterruptValue."""
    return CiriSerializer.deserialize_interrupt(data)


def serialize_resume_command(command: ResumeCommand) -> Dict[str, Any]:
    """Serialize resume command to JSON-compatible dict."""
    return CiriSerializer.serialize_resume_command(command)


class CiriJSONEncoder(json.JSONEncoder):
    """Enhanced JSON encoder for Ciri objects using comprehensive serializers."""

    def default(self, obj: Any) -> Any:
        # Use our comprehensive serializers for specific types
        try:
            # Handle CiriState
            if hasattr(obj, "__class__") and "CiriState" in str(type(obj)):
                return CiriSerializer.serialize_ciri_state(obj)

            # Handle AnyMessage (LangChain messages)
            if isinstance(obj, BaseMessage):
                return serialize_any_message(obj)

            # Handle StateSnapshot
            from langgraph.types import StateSnapshot

            if isinstance(obj, StateSnapshot):
                return serialize_state_snapshot(obj)

            # Handle LangGraph Send objects specifically
            if (
                hasattr(obj, "__class__")
                and obj.__class__.__name__ == "Send"
                and hasattr(obj, "node")
                and hasattr(obj, "arg")
            ):
                return {
                    "_type": "Send",
                    "node": obj.node,
                    "arg": CiriSerializer._serialize_value(obj.arg),
                }

            # Handle other LangGraph types (Send, Command, etc.)
            if (
                hasattr(obj, "__class__")
                and obj.__class__.__module__ == "langgraph.types"
            ):
                return CiriSerializer._serialize_value(obj)

            # Handle InterruptValue
            if isinstance(obj, dict) and "value" in obj and len(obj) == 1:
                try:
                    return serialize_interrupt(obj)
                except Exception:
                    pass  # Fall through to other handlers

            # Handle ResumeCommand
            if isinstance(obj, dict) and "resume" in obj and len(obj) == 1:
                try:
                    return serialize_resume_command(obj)
                except Exception:
                    pass  # Fall through to other handlers

            # Handle Pydantic models (v1 and v2)
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            if hasattr(obj, "dict"):
                return obj.dict()

            # Use CiriSerializer for complex objects
            return CiriSerializer._serialize_value(obj)

        except Exception as e:
            # Fallback to string representation - log to stderr
            logger.warning(f"Failed to serialize {type(obj)}: {e}")
            return str(obj)
