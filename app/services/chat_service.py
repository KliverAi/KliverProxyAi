"""Chat service for handling chat logic"""
import time
import json
from typing import Tuple, Optional

from app.models import ChatRequest, ChatResponse, AiResponse, ChatRole, AIProvider, TokenAiServiceUsageInfo
from app.services.llm_service import create_llm, convert_to_langchain_message, extract_token_usage
from app.config import logger, tracer


async def process_chat_request(request: ChatRequest) -> AiResponse:
    """
    Process a chat request and return the AI response.

    Args:
        request: The chat request containing model, messages, and configuration

    Returns:
        AiResponse containing the chat response and token usage

    Raises:
        ValueError: If the request is invalid
        Exception: If there's an error processing the request
    """
    request_start_time = time.time()

    is_structured = request.output_schema is not None
    model = request.model
    api_key = request.api_key
    provider = request.get_provider()

    # Log request info
    request_info = f"🚀 {provider.value} {model} request"
    if is_structured:
        request_info += f" (structured: {list(request.output_schema.keys())})"
    if request.context_cache_name:
        request_info += f" (Cache: {request.context_cache_name})"
    logger.info(request_info)

    # Initialize the appropriate LLM client
    llm = create_llm(
        provider=provider,
        model=model,
        api_key=api_key,
        temperature=request.temperature,
        context_cache_name=request.context_cache_name,
    )

    # Convert ChatMessage objects to LangChain message format
    langchain_messages = [
        convert_to_langchain_message(msg)
        for msg in request.messages
    ]

    # Process based on whether it's structured or regular chat
    if is_structured:
        response_content, token_usage = await _process_structured_chat(
            llm=llm,
            messages=langchain_messages,
            schema=request.output_schema,
            provider=provider,
            model=model,
            temperature=request.temperature
        )
    else:
        response_content, token_usage = await _process_regular_chat(
            llm=llm,
            messages=langchain_messages,
            provider=provider,
            model=model,
            temperature=request.temperature,
            cache_name=request.context_cache_name
        )

    # Log final summary with telemetry
    request_duration = time.time() - request_start_time

    if tracer:
        with tracer.start_as_current_span("ai_request_summary") as span:
            span.set_attribute("ai.total_duration_seconds", request_duration)
            span.set_attribute("ai.provider", provider.value)
            span.set_attribute("ai.model", model)
            span.set_attribute("ai.total_tokens", token_usage.total_token_count if token_usage else 0)

    return AiResponse(
        response=ChatResponse(
            content=response_content,
            model=model,
            role=ChatRole.ASSISTANT
        ),
        token_usage=token_usage
    )


async def _process_structured_chat(
    llm,
    messages: list,
    schema: dict,
    provider: AIProvider,
    model: str,
    temperature: float
) -> Tuple[str, Optional[TokenAiServiceUsageInfo]]:
    """Process a structured chat request"""
    json_schema = schema
    schema_name = "StructuredOutput"

    if "output_schema" in json_schema and isinstance(json_schema["output_schema"], dict):
        schema_name = json_schema.get("name", "StructuredOutput")
        json_schema = json_schema["output_schema"]

    if "title" not in json_schema:
        json_schema["title"] = schema_name
    if "description" not in json_schema:
        json_schema["description"] = f"Structured output schema for {schema_name}"

    # Different providers support different methods for structured output
    structured_llm = _create_structured_llm(llm, json_schema, provider)

    run_config = {
        "run_name": f"{schema_name}",
        "tags": [provider.value, model, "structured_output", schema_name],
        "metadata": {
            "provider": provider.value,
            "model": model,
            "schema_name": schema_name,
            "temperature": temperature
        }
    }

    llm_start_time = time.time()

    # Execute with telemetry
    if tracer:
        response_obj = await _execute_with_telemetry(
            structured_llm,
            messages,
            run_config,
            provider,
            model,
            temperature,
            schema_name=schema_name
        )
    else:
        response_obj = await structured_llm.ainvoke(messages, config=run_config)

    llm_duration = time.time() - llm_start_time
    logger.info(f"✅ Structured response in {llm_duration:.2f}s")

    # Serialize response
    if isinstance(response_obj, (dict, list)):
        response_content = json.dumps(response_obj, indent=2)
    elif hasattr(response_obj, 'model_dump_json'):
        response_content = response_obj.model_dump_json(indent=2)
    else:
        response_content = str(response_obj)

    return response_content, None  # Often unavailable for structured output


async def _process_regular_chat(
    llm,
    messages: list,
    provider: AIProvider,
    model: str,
    temperature: float,
    cache_name: Optional[str]
) -> Tuple[str, TokenAiServiceUsageInfo]:
    """Process a regular chat request"""
    run_config = {
        "run_name": f"{provider.value}_{model}_chat",
        "tags": [provider.value, model, "chat"],
        "metadata": {
            "provider": provider.value,
            "model": model,
            "temperature": temperature,
            "message_count": len(messages),
            "context_cache_name": cache_name or "N/A"
        }
    }

    llm_start_time = time.time()

    # Execute with telemetry
    if tracer:
        response = await _execute_with_telemetry(
            llm,
            messages,
            run_config,
            provider,
            model,
            temperature,
            cache_name=cache_name,
            message_count=len(messages)
        )
    else:
        response = await llm.ainvoke(messages, config=run_config)

    llm_duration = time.time() - llm_start_time
    token_usage = extract_token_usage(response, provider)
    logger.info(f"✅ Response in {llm_duration:.2f}s | Tokens: {token_usage.input_token_count}→{token_usage.output_token_count} ({token_usage.total_token_count})")

    return response.content, token_usage


def _create_structured_llm(llm, json_schema: dict, provider: AIProvider):
    """Create a structured LLM based on provider"""
    try:
        if provider == AIProvider.OPENAI:
            return llm.with_structured_output(json_schema, method="json_schema")
        elif provider == AIProvider.GEMINI:
            try:
                return llm.with_structured_output(json_schema)
            except Exception:
                return llm.with_structured_output(json_schema, method="json_mode")
        elif provider == AIProvider.CLAUDE:
            return llm.with_structured_output(json_schema)
        else:
            return llm.with_structured_output(json_schema)
    except Exception as e:
        logger.error(f"Failed to create structured output with provider {provider}: {e}")
        raise ValueError(f"Structured output not supported for {provider.value} with this configuration: {str(e)}")


async def _execute_with_telemetry(
    llm,
    messages: list,
    run_config: dict,
    provider: AIProvider,
    model: str,
    temperature: float,
    schema_name: Optional[str] = None,
    cache_name: Optional[str] = None,
    message_count: Optional[int] = None
):
    """Execute LLM invocation with telemetry tracking"""
    span_name = f"ai_structured_call_{provider.value}" if schema_name else f"ai_chat_call_{provider.value}"

    with tracer.start_as_current_span(span_name) as span:
        span.set_attribute("ai.provider", provider.value)
        span.set_attribute("ai.model", model)
        span.set_attribute("ai.temperature", temperature)

        if schema_name:
            span.set_attribute("ai.schema_name", schema_name)
        if cache_name:
            span.set_attribute("ai.cache_name", cache_name)
        if message_count:
            span.set_attribute("ai.message_count", message_count)

        start_time = time.time()
        response = await llm.ainvoke(messages, config=run_config)
        duration = time.time() - start_time

        span.set_attribute("ai.duration_seconds", duration)

        return response
