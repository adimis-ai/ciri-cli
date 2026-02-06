import re
from uuid import UUID
from langgraph.types import interrupt
from datetime import date, datetime, time
from langchain_core.tools import StructuredTool
from langchain.agents.middleware import AgentMiddleware
from typing import (
    NotRequired,
    Any,
    Optional,
    Type,
    Union,
    Literal,
    List,
)
from typing_extensions import TypedDict
from pydantic import (
    BaseModel,
    Field,
    create_model,
    model_validator,
    EmailStr,
    AnyUrl,
    IPvAnyAddress,
)


class JsonSchemaConverter:
    """
    Converts JSON Schema (draft-07) to Pydantic v2 models.

    Supports:
    - All basic types (string, integer, number, boolean, array, object, null)
    - String formats (date, date-time, time, email, uri, uuid, ipv4, ipv6, etc.)
    - String constraints (minLength, maxLength, pattern)
    - Number constraints (minimum, maximum, exclusiveMinimum, exclusiveMaximum, multipleOf)
    - Array constraints (minItems, maxItems, uniqueItems, items, prefixItems)
    - Object constraints (properties, additionalProperties, required)
    - Enums and const values
    - Type unions (anyOf, oneOf, allOf)
    - Nullable types (type: ["string", "null"])
    - $ref resolution with definitions/$defs
    - Nested object models
    - Default values
    """

    # Mapping from JSON Schema string formats to Python/Pydantic types
    STRING_FORMAT_MAP: dict[str, Type[Any]] = {
        "date": date,
        "date-time": datetime,
        "time": time,
        "email": EmailStr,
        "uri": AnyUrl,
        "url": AnyUrl,
        "uuid": UUID,
        "ipv4": IPvAnyAddress,
        "ipv6": IPvAnyAddress,
        "hostname": str,
        "idn-hostname": str,
        "idn-email": str,
        "uri-reference": str,
        "uri-template": str,
        "iri": str,
        "iri-reference": str,
        "json-pointer": str,
        "relative-json-pointer": str,
        "regex": str,
    }

    # Mapping from JSON Schema types to Python types
    TYPE_MAP: dict[str, Type[Any]] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }

    def __init__(self, schema: dict[str, Any], model_name: str = "DynamicModel"):
        self.root_schema = schema
        self.model_name = model_name
        self.definitions: dict[str, dict[str, Any]] = {}
        self._generated_models: dict[str, Type[BaseModel]] = {}
        self._model_counter = 0

        # Load definitions from schema (supports both $defs and definitions)
        self.definitions.update(schema.get("$defs", {}))
        self.definitions.update(schema.get("definitions", {}))

    def convert(self) -> Type[BaseModel]:
        """Convert the root schema to a Pydantic model."""
        return self._schema_to_model(self.root_schema, self.model_name)

    def _schema_to_model(
        self, schema: dict[str, Any], model_name: str
    ) -> Type[BaseModel]:
        """Convert a JSON Schema object to a Pydantic model."""
        # Handle $ref
        if "$ref" in schema:
            schema = self._resolve_ref(schema["$ref"])

        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        additional_properties = schema.get("additionalProperties", True)

        field_definitions: dict[str, Any] = {}

        for field_name, field_schema in properties.items():
            field_type, field_info = self._resolve_field(
                field_schema, field_name, field_name in required
            )
            field_definitions[field_name] = (field_type, field_info)

        # Create model config for Pydantic v2 (pass dict directly, not a type class)
        model_config_dict: dict[str, Any] = {}
        if additional_properties is False:
            model_config_dict["extra"] = "forbid"
        elif additional_properties is True:
            model_config_dict["extra"] = "allow"

        # Create the model - Pydantic v2 requires dict for __config__, not type()
        if model_config_dict:
            model = create_model(
                model_name,
                __config__=model_config_dict,
                **field_definitions,
            )
        else:
            model = create_model(
                model_name,
                **field_definitions,
            )

        return model

    def _resolve_ref(self, ref: str) -> dict[str, Any]:
        """Resolve a $ref pointer to its schema definition."""
        if ref.startswith("#/$defs/"):
            def_name = ref[8:]
            if def_name in self.definitions:
                return self.definitions[def_name]
        elif ref.startswith("#/definitions/"):
            def_name = ref[14:]
            if def_name in self.definitions:
                return self.definitions[def_name]
        elif ref == "#":
            return self.root_schema

        raise ValueError(f"Unable to resolve $ref: {ref}")

    def _resolve_field(
        self, schema: dict[str, Any], field_name: str, is_required: bool
    ) -> tuple[Type[Any], Any]:
        """Resolve a field schema to a Python type and Field definition."""
        # Handle $ref
        if "$ref" in schema:
            schema = self._resolve_ref(schema["$ref"])

        description = schema.get("description", "")
        default = schema.get("default", ...)
        title = schema.get("title", "")

        # Resolve the type
        field_type = self._resolve_type(schema, field_name)

        # Check if nullable
        is_nullable = self._is_nullable(schema)
        if is_nullable and field_type is not type(None):
            field_type = Optional[field_type]

        # Build Field kwargs
        field_kwargs: dict[str, Any] = {}
        if description:
            field_kwargs["description"] = description
        if title:
            field_kwargs["title"] = title

        # Handle default
        if default is not ...:
            field_kwargs["default"] = default
        elif not is_required:
            field_kwargs["default"] = None
            if not is_nullable:
                field_type = Optional[field_type]
        else:
            field_kwargs["default"] = ...

        # Add validation constraints for strings
        if schema.get("type") == "string":
            if "minLength" in schema:
                field_kwargs["min_length"] = schema["minLength"]
            if "maxLength" in schema:
                field_kwargs["max_length"] = schema["maxLength"]
            if "pattern" in schema:
                field_kwargs["pattern"] = schema["pattern"]

        # Add validation constraints for numbers
        if schema.get("type") in ("integer", "number"):
            if "minimum" in schema:
                field_kwargs["ge"] = schema["minimum"]
            if "maximum" in schema:
                field_kwargs["le"] = schema["maximum"]
            if "exclusiveMinimum" in schema:
                field_kwargs["gt"] = schema["exclusiveMinimum"]
            if "exclusiveMaximum" in schema:
                field_kwargs["lt"] = schema["exclusiveMaximum"]
            if "multipleOf" in schema:
                field_kwargs["multiple_of"] = schema["multipleOf"]

        # Add validation constraints for arrays
        if schema.get("type") == "array":
            if "minItems" in schema:
                field_kwargs["min_length"] = schema["minItems"]
            if "maxItems" in schema:
                field_kwargs["max_length"] = schema["maxItems"]

        return field_type, Field(**field_kwargs)

    def _resolve_type(
        self, schema: dict[str, Any], context_name: str = ""
    ) -> Type[Any]:
        """Resolve a JSON Schema to a Python type."""
        # Handle $ref
        if "$ref" in schema:
            ref = schema["$ref"]
            # Check if we've already generated this model
            if ref in self._generated_models:
                return self._generated_models[ref]

            resolved = self._resolve_ref(ref)
            # For object refs, generate a nested model
            if resolved.get("type") == "object" and "properties" in resolved:
                ref_name = ref.split("/")[-1]
                model = self._schema_to_model(resolved, self._make_model_name(ref_name))
                self._generated_models[ref] = model
                return model
            return self._resolve_type(resolved, context_name)

        # Handle const
        if "const" in schema:
            const_val = schema["const"]
            return Literal[const_val]

        # Handle enum
        if "enum" in schema:
            enum_values = schema["enum"]
            # Create a Literal type for the enum values
            if all(isinstance(v, str) for v in enum_values):
                return Literal[tuple(enum_values)]
            elif all(isinstance(v, int) for v in enum_values):
                return Literal[tuple(enum_values)]
            else:
                # Mixed types, fall back to Any
                return Any

        # Handle anyOf
        if "anyOf" in schema:
            return self._resolve_any_of(schema["anyOf"], context_name)

        # Handle oneOf
        if "oneOf" in schema:
            return self._resolve_one_of(schema["oneOf"], context_name)

        # Handle allOf
        if "allOf" in schema:
            return self._resolve_all_of(schema["allOf"], context_name)

        # Get the type(s)
        schema_type = schema.get("type")

        # Handle missing type - infer from other keywords
        if schema_type is None:
            if "properties" in schema:
                schema_type = "object"
            elif "items" in schema:
                schema_type = "array"
            elif "enum" in schema:
                return self._resolve_type({"enum": schema["enum"]}, context_name)
            else:
                return Any

        # Handle type arrays (e.g., ["string", "null"])
        if isinstance(schema_type, list):
            non_null_types = [t for t in schema_type if t != "null"]
            if len(non_null_types) == 0:
                return type(None)
            elif len(non_null_types) == 1:
                schema_type = non_null_types[0]
            else:
                # Multiple non-null types - create a Union
                types = [
                    self._resolve_type(
                        {"type": t, **{k: v for k, v in schema.items() if k != "type"}},
                        context_name,
                    )
                    for t in non_null_types
                ]
                return Union[tuple(types)]

        # Handle string with format
        if schema_type == "string":
            fmt = schema.get("format")
            if fmt and fmt in self.STRING_FORMAT_MAP:
                return self.STRING_FORMAT_MAP[fmt]
            return str

        # Handle integer
        if schema_type == "integer":
            return int

        # Handle number
        if schema_type == "number":
            return float

        # Handle boolean
        if schema_type == "boolean":
            return bool

        # Handle null
        if schema_type == "null":
            return type(None)

        # Handle array
        if schema_type == "array":
            return self._resolve_array_type(schema, context_name)

        # Handle object
        if schema_type == "object":
            return self._resolve_object_type(schema, context_name)

        # Fallback
        return self.TYPE_MAP.get(schema_type, Any)

    def _resolve_array_type(
        self, schema: dict[str, Any], context_name: str
    ) -> Type[Any]:
        """Resolve an array schema to a Python type."""
        items = schema.get("items")
        prefix_items = schema.get("prefixItems")

        # Handle tuple validation (prefixItems in draft 2020-12, or items as array in draft-07)
        if prefix_items:
            item_types = tuple(
                self._resolve_type(item_schema, f"{context_name}Item{i}")
                for i, item_schema in enumerate(prefix_items)
            )
            return tuple[item_types]

        # Handle items as array (tuple validation in draft-07)
        if isinstance(items, list):
            item_types = tuple(
                self._resolve_type(item_schema, f"{context_name}Item{i}")
                for i, item_schema in enumerate(items)
            )
            return tuple[item_types]

        # Handle single items schema
        if items:
            item_type = self._resolve_type(items, f"{context_name}Item")
            return list[item_type]

        # No items specified - list of Any
        return list[Any]

    def _resolve_object_type(
        self, schema: dict[str, Any], context_name: str
    ) -> Type[Any]:
        """Resolve an object schema to a Python type."""
        properties = schema.get("properties")
        additional_properties = schema.get("additionalProperties")

        # If has properties, generate a nested model
        if properties:
            model_name = self._make_model_name(context_name)
            return self._schema_to_model(schema, model_name)

        # If additionalProperties is a schema, use dict with typed values
        if isinstance(additional_properties, dict):
            value_type = self._resolve_type(
                additional_properties, f"{context_name}Value"
            )
            return dict[str, value_type]

        # Generic dict
        return dict[str, Any]

    def _resolve_any_of(
        self, schemas: list[dict[str, Any]], context_name: str
    ) -> Type[Any]:
        """Resolve anyOf to a Union type."""
        types = []
        for i, sub_schema in enumerate(schemas):
            resolved = self._resolve_type(sub_schema, f"{context_name}Option{i}")
            types.append(resolved)

        if len(types) == 1:
            return types[0]
        return Union[tuple(types)]

    def _resolve_one_of(
        self, schemas: list[dict[str, Any]], context_name: str
    ) -> Type[Any]:
        """Resolve oneOf to a Union type (same as anyOf for type purposes)."""
        return self._resolve_any_of(schemas, context_name)

    def _resolve_all_of(
        self, schemas: list[dict[str, Any]], context_name: str
    ) -> Type[Any]:
        """Resolve allOf by merging schemas."""
        merged: dict[str, Any] = {}

        for sub_schema in schemas:
            # Resolve refs first
            if "$ref" in sub_schema:
                sub_schema = self._resolve_ref(sub_schema["$ref"])

            # Merge properties
            if "properties" in sub_schema:
                merged.setdefault("properties", {}).update(sub_schema["properties"])

            # Merge required
            if "required" in sub_schema:
                existing_required = set(merged.get("required", []))
                existing_required.update(sub_schema["required"])
                merged["required"] = list(existing_required)

            # Merge other keys (last wins)
            for key, value in sub_schema.items():
                if key not in ("properties", "required"):
                    merged[key] = value

        # Set type to object if we have properties
        if "properties" in merged and "type" not in merged:
            merged["type"] = "object"

        return self._resolve_type(merged, context_name)

    def _is_nullable(self, schema: dict[str, Any]) -> bool:
        """Check if a schema allows null values."""
        schema_type = schema.get("type")

        # Check type array
        if isinstance(schema_type, list) and "null" in schema_type:
            return True

        # Check anyOf/oneOf for null
        for combinator in ("anyOf", "oneOf"):
            if combinator in schema:
                for sub_schema in schema[combinator]:
                    if sub_schema.get("type") == "null":
                        return True

        return False

    def _make_model_name(self, context: str) -> str:
        """Generate a unique model name."""
        # Clean the context string
        clean = re.sub(r"[^a-zA-Z0-9]", "", context)
        if not clean:
            self._model_counter += 1
            return f"{self.model_name}Nested{self._model_counter}"
        return f"{clean}Model"


def json_schema_to_pydantic_model(
    schema: dict[str, Any], model_name: str = "DynamicModel"
) -> Type[BaseModel]:
    """
    Converts a JSON Schema (draft-07) to a Pydantic v2 model class.

    This is a convenience function that creates a JsonSchemaConverter and
    converts the schema. For more control, use JsonSchemaConverter directly.

    Args:
        schema: JSON Schema dict (draft-07 compatible)
        model_name: Name for the generated model class

    Returns:
        A dynamically created Pydantic BaseModel class

    Supports:
    - All basic types (string, integer, number, boolean, array, object, null)
    - String formats (date, date-time, time, email, uri, uuid, ipv4, ipv6, etc.)
    - String constraints (minLength, maxLength, pattern)
    - Number constraints (minimum, maximum, exclusiveMinimum, exclusiveMaximum, multipleOf)
    - Array constraints (minItems, maxItems, items, prefixItems)
    - Object constraints (properties, additionalProperties, required)
    - Enums and const values
    - Type unions (anyOf, oneOf, allOf)
    - Nullable types (type: ["string", "null"])
    - $ref resolution with definitions/$defs
    - Nested object models
    - Default values
    """
    converter = JsonSchemaConverter(schema, model_name)
    return converter.convert()


class FrontendTool(BaseModel):
    """
    Serializable frontend tool definition.
    Matches the SerializableFrontendTool interface from the frontend SDK.
    """

    id: str = Field(..., description="Unique identifier for the tool")
    name: str = Field(..., description="Display name of the tool")
    return_direct: Optional[bool] = Field(
        default=None,
        alias="returnDirect",
        description="Whether to return the tool output directly without further processing",
    )
    description: str = Field(..., description="Description of what the tool does")
    input_schema: dict[str, Any] = Field(
        ...,
        alias="inputSchema",
        description="JSON Schema (draft-07) defining the tool's input parameters",
    )
    input_schema_model: Optional[Type[BaseModel]] = Field(
        default=None,
        exclude=True,
        description="Auto-generated Pydantic model from input_schema for validation",
    )
    response_schema: Optional[dict[str, Any]] = Field(
        default=None,
        alias="responseSchema",
        description="JSON Schema (draft-07) defining the tool's response format",
    )
    response_schema_model: Optional[Type[BaseModel]] = Field(
        default=None,
        exclude=True,
        description="Auto-generated Pydantic model from response_schema for validation",
    )

    model_config = {"populate_by_name": True, "arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _build_input_schema_model(self) -> "FrontendTool":
        """Auto-generate a Pydantic model from the input_schema JSON Schema."""
        if self.input_schema_model is None and self.input_schema:
            self.input_schema_model = json_schema_to_pydantic_model(
                self.input_schema, model_name=f"{self.name.replace(' ', '')}Input"
            )

        if self.response_schema_model is None and self.response_schema is not None:
            self.response_schema_model = json_schema_to_pydantic_model(
                self.response_schema,
                model_name=f"{self.name.replace(' ', '')}Response",
            )

        return self


class FrontendToolsInterruptValue(TypedDict):
    type: Literal["frontend_tool_call"]
    tool_name: str
    tool_call_id: str
    parameters: NotRequired[dict[str, Any]]


class FrontendToolResponse(TypedDict):
    success: bool
    response: NotRequired[Any]
    error_message: NotRequired[str]


def build_frontend_tool(tool: FrontendTool) -> StructuredTool:
    """Build a LangChain StructuredTool from a FrontendTool definition."""

    def _tool(**kwargs):
        response = interrupt(
            {
                "type": "frontend_tool_call",
                "tool_name": tool.name,
                "parameters": kwargs,
            }
        )

        if response.get("success"):
            return response
        else:
            return {
                "success": False,
                "response": response.get("error_message", "Unknown error"),
            }

    return StructuredTool.from_function(
        name=tool.name,
        description=tool.description,
        args_schema=tool.input_schema_model,
        return_direct=tool.return_direct if tool.return_direct is not None else False,
        func=_tool,
    )


def _extract_frontend_tools(context: Any) -> List[Any]:
    """Extract frontend_tools from context, handling dict, BaseModel, and dataclass types."""
    if context is None:
        return []

    # Handle dictionary
    if isinstance(context, dict):
        return context.get("frontend_tools", []) or []

    # Handle objects with attributes (BaseModel, dataclass, etc.)
    try:
        return getattr(context, "frontend_tools", []) or []
    except (AttributeError, TypeError):
        return []


def build_frontend_tools(
    context: Any,
    *,
    exclude_tools: Optional[List[str]] = None,
    include_tools: Union[Literal["*"], List[str]] = "*",
) -> list[StructuredTool]:
    """Build a list of LangChain StructuredTools from a list of FrontendTool definitions.

    - `include_tools` can be "*" (default) to include all tools, or a list of tool
      names to include.
    - `exclude_tools` is an optional list of tool names to always exclude.
    """

    if exclude_tools is None:
        exclude_tools = []

    frontend_tools = _extract_frontend_tools(context)
    tools: List[StructuredTool] = []

    for tool in frontend_tools:
        name = getattr(tool, "name", None)
        if not name:
            continue

        # Skip excluded tools
        if name in exclude_tools:
            continue

        # If include_tools is a list, only include those names
        if include_tools != "*" and name not in include_tools:
            continue

        tools.append(build_frontend_tool(tool))

    return tools


class FrontendToolsMiddleware(AgentMiddleware):
    """Agent middleware to add frontend tools to the agent's toolset."""

    def __init__(
        self,
        exclude_tools: Optional[List[str]] = None,
        include_tools: Union[Literal["*"], List[str]] = "*",
    ):
        super().__init__()
        self.exclude_tools = exclude_tools
        self.include_tools = include_tools
        self.tools: List[StructuredTool] = []

    def before_agent(self, state, runtime):
        context = runtime.context
        frontend_tools = build_frontend_tools(
            context,
            exclude_tools=self.exclude_tools,
            include_tools=self.include_tools,
        )

        # Ensure uniqueness by tool name
        existing_names = {tool.name for tool in self.tools}
        unique_frontend_tools = [
            tool for tool in frontend_tools if tool.name not in existing_names
        ]
        self.tools = self.tools + unique_frontend_tools
        return super().before_agent(state, runtime)
