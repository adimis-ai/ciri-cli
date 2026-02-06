"""
Ciri Copilot

A sophisticated AI agent toolkit for various automation and analysis tasks.
"""

# Note: main function should be accessed directly from __main__ module
# when running with python -m src

__version__ = "1.0.0"

from .serializers import (
    CiriJSONEncoder,
    CiriSerializer,
    serialize_ciri_state,
    deserialize_ciri_state,
    serialize_any_message,
    deserialize_any_message,
    serialize_state_snapshot,
    serialize_interrupt,
    deserialize_interrupt,
    serialize_resume_command,
)

__all__ = [
    "CiriJSONEncoder",
    "CiriSerializer",
    "serialize_ciri_state",
    "deserialize_ciri_state",
    "serialize_any_message",
    "deserialize_any_message",
    "serialize_state_snapshot",
    "serialize_interrupt",
    "deserialize_interrupt",
    "serialize_resume_command",
]
