import json
import logging
from typing import Any, Dict, List, Optional, Union, Sequence, Callable, Literal
from datetime import datetime, date
from pathlib import Path

from pydantic import BaseModel
from langchain_core.load import dumpd, loads
from langchain_core.messages import AnyMessage, BaseMessage
from langgraph.types import StateSnapshot
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from .agent import (
    CiriState,
    ResumeCommand,
    InterruptValue,
    ApprovalDecisions,
    EditDecisions,
    RejectDecisions,
)

logger = logging.getLogger(__name__)


class CiriJsonPlusSerializer(JsonPlusSerializer):
    """Extended JsonPlusSerializer that can handle LangGraph Send objects."""

    def __init__(
        self,
        *,
        pickle_fallback: bool = False,
        allowed_json_modules: Sequence[tuple[str, ...]] | Literal[True] | None = None,
        __unpack_ext_hook__: Callable[[int, bytes], Any] | None = None,
    ):
        super().__init__(
            pickle_fallback=pickle_fallback,
            allowed_json_modules=allowed_json_modules,
            __unpack_ext_hook__=__unpack_ext_hook__,
        )

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        """Override dumps_typed to handle Send objects before msgpack serialization."""
        # Convert any Send objects to serializable form first
        serializable_obj = self._make_serializable(obj)
        return super().dumps_typed(serializable_obj)

    def _make_serializable(self, obj: Any) -> Any:
        """Recursively convert Send objects to serializable dictionaries."""
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif (
            hasattr(obj, "__class__")
            and obj.__class__.__name__ == "Send"
            and hasattr(obj, "node")
            and hasattr(obj, "arg")
        ):
            # Convert Send object to serializable dict
            return {
                "__send_object__": True,
                "node": obj.node,
                "arg": self._make_serializable(obj.arg),
            }
        elif hasattr(obj, "__class__") and obj.__class__.__name__ in [
            "lock",
            "_thread.lock",
            "Lock",
            "RLock",
        ]:
            # Skip thread locks and other thread-related objects
            return {"__thread_lock__": True, "type": obj.__class__.__name__}
        elif hasattr(obj, "__class__") and "Session" in obj.__class__.__name__:
            # Skip session objects that might contain locks
            return {"__session_object__": True, "type": obj.__class__.__name__}
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, "__dict__"):
            # Handle other objects with __dict__
            result = {}
            for k, v in obj.__dict__.items():
                try:
                    # Skip attributes that might contain locks or other non-serializable objects
                    if (
                        k.lower() in ["lock", "session", "tempdir", "policy"]
                        or "lock" in k.lower()
                    ):
                        result[k] = {"__skipped__": True, "reason": "non-serializable"}
                    else:
                        result[k] = self._make_serializable(v)
                except Exception:
                    result[k] = {"__skipped__": True, "reason": "serialization_error"}
            return result
        elif hasattr(obj, "__slots__"):
            # Handle objects with __slots__ (like Send)
            result = {}
            for slot in obj.__slots__:
                if hasattr(obj, slot):
                    try:
                        result[slot] = self._make_serializable(getattr(obj, slot))
                    except Exception:
                        result[slot] = {
                            "__skipped__": True,
                            "reason": "serialization_error",
                        }
            return result
        else:
            # Try to convert to string for non-serializable objects
            try:
                # Test if it's JSON serializable
                import json

                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return {"__non_serializable__": str(obj), "type": str(type(obj))}

    def _encode_constructor_args(self, constructor, args):
        """Override to handle Send objects specifically."""
        # Handle Send objects
        if (
            hasattr(constructor, "__module__")
            and hasattr(constructor, "__name__")
            and constructor.__module__ == "langgraph.types"
            and constructor.__name__ == "Send"
        ):
            # Send objects have node and arg attributes
            if len(args) >= 2:
                node, arg = args[0], args[1]
                return (node, self._serialize_value(arg))
            return args

        # Fall back to parent implementation
        return super()._encode_constructor_args(constructor, args)

    def _serialize_value(self, value: Any) -> Any:
        """Helper method to recursively serialize values, handling Send objects."""
        if value is None:
            return None
        elif isinstance(value, (str, int, float, bool)):
            return value
        elif hasattr(value, "__class__") and value.__class__.__name__ == "Send":
            # Convert Send object to a serializable form
            return {
                "_type": "Send",
                "node": value.node,
                "arg": self._serialize_value(value.arg),
            }
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(item) for item in value]
        else:
            return value


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
def serialize_ciri_state(state: CiriState) -> Dict[str, Any]:
    """Serialize CiriState to JSON-compatible dict."""
    return CiriSerializer.serialize_ciri_state(state)


def deserialize_ciri_state(data: Dict[str, Any]) -> CiriState:
    """Deserialize dict back to CiriState."""
    return CiriSerializer.deserialize_ciri_state(data)


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
                return serialize_ciri_state(obj)

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
