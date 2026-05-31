"""Shared structured-output helpers for RAG node models."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from cv_screener.config import RAGModelSettings

if TYPE_CHECKING:
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.runnables import Runnable, RunnableConfig


def build_structured_output_model[StructuredOutputT: BaseModel](
    schema: type[StructuredOutputT],
    *,
    settings: RAGModelSettings | None = None,
) -> Runnable[LanguageModelInput, StructuredOutputT]:
    """Build a structured-output chat model for the configured backend."""
    resolved_settings = settings or RAGModelSettings()
    llm = ChatOpenAI(
        model=resolved_settings.rag_model,
        base_url=resolved_settings.openai_base_url,
        api_key=resolved_settings.openai_api_key,
        temperature=resolved_settings.rag_temperature,
        max_retries=resolved_settings.rag_max_retries,
    )
    return cast(
        "Runnable[LanguageModelInput, StructuredOutputT]",
        llm.with_structured_output(schema, method="function_calling"),
    )


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
