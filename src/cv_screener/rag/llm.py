"""Shared JSON-parsing helpers for RAG node models."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from cv_screener.config import RAGModelSettings

if TYPE_CHECKING:
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.runnables import Runnable, RunnableConfig

MIN_FENCED_BLOCK_LINES = 3


def build_structured_output_model[StructuredOutputT: BaseModel](
    schema: type[StructuredOutputT],
    *,
    settings: RAGModelSettings | None = None,
) -> Runnable[LanguageModelInput, StructuredOutputT]:
    """Build a chat model wrapper that returns schema-validated JSON."""
    resolved_settings = settings or RAGModelSettings()
    llm = ChatOpenAI(
        model=resolved_settings.rag_model,
        base_url=resolved_settings.openai_base_url,
        api_key=resolved_settings.openai_api_key,
        temperature=resolved_settings.rag_temperature,
        max_retries=resolved_settings.rag_max_retries,
    )

    def invoke_json_model(
        input_value: LanguageModelInput,
        config: RunnableConfig | None = None,
    ) -> StructuredOutputT:
        return _invoke_json_model(
            llm=llm,
            schema=schema,
            input_value=input_value,
            config=config,
        )

    return RunnableLambda(invoke_json_model)


def invoke_structured_output[StructuredOutputT: BaseModel](
    input_value: LanguageModelInput,
    *,
    model: Runnable[LanguageModelInput, StructuredOutputT],
    schema: type[StructuredOutputT],
    label: str,
    config: RunnableConfig | None = None,
) -> StructuredOutputT:
    """Invoke a structured-output model and validate the result."""
    result = model.invoke(input_value, config=config)
    try:
        return schema.model_validate(result)
    except ValidationError as error:
        msg = f"{label} returned invalid structured output: {error}"
        raise ValueError(msg) from error


def _invoke_json_model[StructuredOutputT: BaseModel](
    *,
    llm: ChatOpenAI,
    schema: type[StructuredOutputT],
    input_value: LanguageModelInput,
    config: RunnableConfig | None,
) -> StructuredOutputT:
    response = llm.invoke(
        _augment_input_with_json_instruction(input_value, schema), config=config
    )
    try:
        return _parse_llm_response(response, schema)
    except ValueError:
        repaired_response = llm.invoke(
            _build_repair_input(schema, response),
            config=config,
        )
        return _parse_llm_response(repaired_response, schema)


def _augment_input_with_json_instruction(
    input_value: LanguageModelInput,
    schema: type[BaseModel],
) -> LanguageModelInput:
    instruction = SystemMessage(content=_json_output_instruction(schema))
    if isinstance(input_value, list):
        return [*input_value, instruction]
    if isinstance(input_value, tuple):
        return [*input_value, instruction]
    if isinstance(input_value, str):
        return "\n\n".join([input_value, _json_output_instruction(schema)])
    return input_value


def _build_repair_input(
    schema: type[BaseModel],
    response: BaseMessage | str | object,
) -> list[SystemMessage | HumanMessage]:
    raw_text = _coerce_text_content(response)
    return [
        SystemMessage(content=_json_output_instruction(schema)),
        HumanMessage(
            content=(
                "Rewrite the following content as valid JSON that matches the schema "
                "exactly. Return JSON only.\n\n"
                f"{raw_text}"
            )
        ),
    ]


def _parse_llm_response[StructuredOutputT: BaseModel](
    response: BaseMessage | str | StructuredOutputT | dict[str, object] | object,
    schema: type[StructuredOutputT],
) -> StructuredOutputT:
    if isinstance(response, schema):
        return response
    if isinstance(response, BaseMessage):
        json_text = _extract_json_text(_coerce_text_content(response))
        try:
            return schema.model_validate_json(json_text)
        except ValidationError as error:
            msg = f"Invalid JSON response for {schema.__name__}: {error}"
            raise ValueError(msg) from error
    if isinstance(response, dict) and _looks_like_schema_payload(response, schema):
        return schema.model_validate(response)
    if isinstance(response, BaseModel):
        dumped = response.model_dump()
        if isinstance(dumped, dict) and _looks_like_schema_payload(dumped, schema):
            return schema.model_validate(dumped)

    json_text = _extract_json_text(_coerce_text_content(response))
    try:
        return schema.model_validate_json(json_text)
    except ValidationError as error:
        msg = f"Invalid JSON response for {schema.__name__}: {error}"
        raise ValueError(msg) from error


def _coerce_text_content(response: BaseMessage | str | object) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        dict_content = _coerce_content_value(response.get("content"))
        if dict_content is not None:
            return dict_content
    content = getattr(response, "content", None)
    object_content = _coerce_content_value(content)
    if object_content is not None:
        return object_content
    return str(response)


def _coerce_content_value(content: object) -> str | None:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            parts.append(item["text"])
    if not parts:
        return None
    return "\n".join(parts)


def _extract_json_text(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= MIN_FENCED_BLOCK_LINES and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].lstrip()
    if _is_valid_json(stripped):
        return stripped

    object_start = stripped.find("{")
    object_end = stripped.rfind("}")
    if object_start != -1 and object_end != -1 and object_end > object_start:
        candidate = stripped[object_start : object_end + 1]
        if _is_valid_json(candidate):
            return candidate

    array_start = stripped.find("[")
    array_end = stripped.rfind("]")
    if array_start != -1 and array_end != -1 and array_end > array_start:
        candidate = stripped[array_start : array_end + 1]
        if _is_valid_json(candidate):
            return candidate

    msg = "Model response did not contain valid JSON"
    raise ValueError(msg)


def _is_valid_json(candidate: str) -> bool:
    try:
        json.loads(candidate)
    except json.JSONDecodeError:
        return False
    return True


def _looks_like_schema_payload(
    response: dict[str, object],
    schema: type[BaseModel],
) -> bool:
    required_fields = set(schema.model_fields)
    return required_fields.issubset(response.keys())


def _json_output_instruction(schema: type[BaseModel]) -> str:
    schema_json = json.dumps(schema.model_json_schema(), indent=2, sort_keys=True)
    return (
        "Return only valid JSON. Do not call tools. Do not add markdown fences, "
        "commentary, or prose. The JSON must match this schema exactly:\n"
        f"{schema_json}"
    )
