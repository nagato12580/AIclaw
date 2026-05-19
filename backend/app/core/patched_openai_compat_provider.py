from __future__ import annotations

from typing import Any

from nanobot.providers.openai_compat_provider import OpenAICompatProvider
from nanobot.providers.base import LLMResponse

from app.trace import build_error_attributes, build_usage_attributes, trace_service


class PatchedOpenAICompatProvider(OpenAICompatProvider):
    _MAX_COMPLETION_TOKEN_MODELS = ("gpt-5", "o1", "o3", "o4")

    def _build_kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        max_tokens: int,
        temperature: float,
        reasoning_effort: str | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> dict[str, Any]:
        kwargs = super()._build_kwargs(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )

        model_name = (model or self.default_model or "").lower()
        spec = self._spec
        supports_max_completion_tokens = bool(
            spec and getattr(spec, "supports_max_completion_tokens", False)
        )
        should_use_max_completion_tokens = supports_max_completion_tokens or any(
            token in model_name for token in self._MAX_COMPLETION_TOKEN_MODELS
        )

        if should_use_max_completion_tokens and "max_tokens" in kwargs:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")

        return kwargs

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        effective_model = model or self.default_model
        attributes = self._trace_attributes(
            effective_model=effective_model,
            stream=False,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )
        with trace_service.start_generation(
            "llm.chat",
            model=effective_model,
            attributes=attributes,
            input_payload={"messages": messages},
        ) as generation:
            response = await super().chat(
                messages=messages,
                tools=tools,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                tool_choice=tool_choice,
            )
            self._update_generation(generation, response)
            return response

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta=None,
    ) -> LLMResponse:
        effective_model = model or self.default_model
        attributes = self._trace_attributes(
            effective_model=effective_model,
            stream=True,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )
        with trace_service.start_generation(
            "llm.chat_stream",
            model=effective_model,
            attributes=attributes,
            input_payload={"messages": messages},
        ) as generation:
            response = await super().chat_stream(
                messages=messages,
                tools=tools,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                tool_choice=tool_choice,
                on_content_delta=on_content_delta,
            )
            self._update_generation(generation, response)
            return response

    def _trace_attributes(
        self,
        *,
        effective_model: str,
        stream: bool,
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
        temperature: float,
        reasoning_effort: str | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "component": "llm_provider",
            "provider": getattr(self._spec, "name", None) or "openai_compat",
            "model": effective_model,
            "api_base": self.api_base,
            "stream": stream,
            "tools.count": len(tools or []),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "reasoning_effort": reasoning_effort,
            "tool_choice": tool_choice,
        }

    def _update_generation(self, generation: Any, response: LLMResponse) -> None:
        generation.set_attributes(
            {
                "finish_reason": response.finish_reason,
                "response.has_tool_calls": bool(response.tool_calls),
                "response.tool_calls.count": len(response.tool_calls),
                "response.has_reasoning": bool(response.reasoning_content),
            }
        )
        generation.set_attributes(build_usage_attributes(response.usage))
        update_payload: dict[str, Any] = {
            "output": {
                "content": response.content,
                "tool_calls": [tool_call.name for tool_call in response.tool_calls],
                "usage": response.usage,
            }
        }
        if response.usage:
            update_payload["usage_details"] = {
                "input_tokens": response.usage.get("prompt_tokens"),
                "output_tokens": response.usage.get("completion_tokens"),
                "total_tokens": response.usage.get("total_tokens"),
            }
        generation.update(**update_payload)
        if response.finish_reason == "error":
            generation.set_attributes(build_error_attributes(Exception(response.content or "LLM error"), stage="llm"))
