from __future__ import annotations

import logging
import os
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, Iterator, Mapping, Optional

from app.trace.attributes import sanitize_attributes

logger = logging.getLogger(__name__)


class _NoopSpan:
    def set_attributes(self, _attributes: Optional[Mapping[str, Any]] = None) -> None:
        return None

    def update(self, **_kwargs: Any) -> None:
        return None

    def update_trace(self, **_kwargs: Any) -> None:
        return None

    def record_error(self, _exc: Exception, *, stage: str = "unknown") -> None:
        return None


class _SpanAdapter:
    def __init__(self, raw_span: Any) -> None:
        self._raw_span = raw_span

    def set_attributes(self, attributes: Optional[Mapping[str, Any]] = None) -> None:
        payload = sanitize_attributes(attributes)
        if not payload:
            return
        set_attribute = getattr(self._raw_span, "set_attribute", None)
        if callable(set_attribute):
            for key, value in payload.items():
                set_attribute(key, value)
            return
        update = getattr(self._raw_span, "update", None)
        if callable(update):
            update(metadata=payload)

    def update(self, **kwargs: Any) -> None:
        update = getattr(self._raw_span, "update", None)
        if callable(update):
            update(**kwargs)

    def update_trace(self, **kwargs: Any) -> None:
        update_trace = getattr(self._raw_span, "update_trace", None)
        if callable(update_trace):
            update_trace(**kwargs)

    def record_error(self, exc: Exception, *, stage: str = "unknown") -> None:
        self.set_attributes(
            {
                "error": True,
                "error.stage": stage,
                "error.type": exc.__class__.__name__,
                "error.message": str(exc),
            }
        )
        self.update(level="ERROR", status_message=str(exc))


class TraceService:
    def __init__(self) -> None:
        self._client: Any = None
        self._enabled = False
        self._initialized = False
        self._httpx_instrumented = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def initialized(self) -> bool:
        return self._initialized

    def flush(self) -> None:
        if not self._enabled or not self._client:
            return
        flush = getattr(self._client, "flush", None)
        if callable(flush):
            try:
                flush()
            except Exception as exc:
                logger.warning("Langfuse flush failed: %s", exc)

    def _read_config(self) -> Dict[str, Optional[str]]:
        return {
            "public_key": os.getenv("LANGFUSE_PUBLIC_KEY"),
            "secret_key": os.getenv("LANGFUSE_SECRET_KEY"),
            "base_url": os.getenv("LANGFUSE_BASE_URL", "http://localhost:3000"),
        }

    def initialize(self) -> bool:
        if self._initialized:
            return self._enabled
            
        enable_tracing = os.getenv("ENABLE_TRACING", "false").lower() in ("true", "1", "t", "yes")
        if not enable_tracing:
            self._initialized = True
            self._enabled = False
            return False
            
        self._initialized = True
        cfg = self._read_config()
        if not cfg["public_key"] or not cfg["secret_key"]:
            logger.info("Langfuse tracing disabled: missing LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY")
            return False
        try:
            from langfuse import Langfuse
        except Exception as exc:
            logger.warning("Langfuse tracing disabled: SDK import failed: %s", exc)
            return False
        try:
            self._client = Langfuse(
                public_key=cfg["public_key"],
                secret_key=cfg["secret_key"],
                host=cfg["base_url"],
            )
            self._enabled = True
            logger.info("Langfuse tracing enabled, host=%s", cfg["base_url"])
        except Exception as exc:
            logger.warning("Langfuse tracing initialization failed, fallback to no-op: %s", exc)
            self._client = None
            self._enabled = False
            return False

        try:
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

            HTTPXClientInstrumentor().instrument()
            self._httpx_instrumented = True
        except Exception as exc:
            logger.warning("HTTPX OTEL instrumentation unavailable: %s", exc)
        return True

    def shutdown(self) -> None:
        if self._enabled and self._client:
            self.flush()
            close = getattr(self._client, "shutdown", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        if self._httpx_instrumented:
            try:
                from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

                HTTPXClientInstrumentor().uninstrument()
            except Exception:
                pass
        self._client = None
        self._enabled = False
        self._initialized = False
        self._httpx_instrumented = False

    @contextmanager
    def start_span(
        self,
        name: str,
        *,
        attributes: Optional[Mapping[str, Any]] = None,
        input_payload: Optional[Any] = None,
    ) -> Iterator[_SpanAdapter | _NoopSpan]:
        with self._start_observation(name, as_type="span", attributes=attributes, input_payload=input_payload) as span:
            yield span

    @contextmanager
    def start_generation(
        self,
        name: str,
        *,
        model: Optional[str] = None,
        attributes: Optional[Mapping[str, Any]] = None,
        input_payload: Optional[Any] = None,
    ) -> Iterator[_SpanAdapter | _NoopSpan]:
        observation_kwargs: Dict[str, Any] = {}
        if model:
            observation_kwargs["model"] = model
        with self._start_observation(
            name,
            as_type="generation",
            attributes=attributes,
            input_payload=input_payload,
            observation_kwargs=observation_kwargs,
        ) as generation:
            yield generation

    @contextmanager
    def _start_observation(
        self,
        name: str,
        *,
        as_type: str,
        attributes: Optional[Mapping[str, Any]] = None,
        input_payload: Optional[Any] = None,
        observation_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[_SpanAdapter | _NoopSpan]:
        if not self._enabled or not self._client:
            yield _NoopSpan()
            return
        try:
            start_observation = getattr(self._client, "start_as_current_observation", None)
            if callable(start_observation):
                ctx = start_observation(
                    name=name,
                    as_type=as_type,
                    **dict(observation_kwargs or {}),
                )
            else:
                start_span = getattr(self._client, "start_as_current_span", None)
                start_generation = getattr(self._client, "start_as_current_generation", None)
                if as_type == "generation" and callable(start_generation):
                    ctx = start_generation(name=name, **dict(observation_kwargs or {}))
                elif callable(start_span):
                    ctx = start_span(name=name)
                else:
                    yield _NoopSpan()
                    return
        except Exception as exc:
            logger.warning("Langfuse observation start failed (%s): %s", name, exc)
            yield _NoopSpan()
            return
        try:
            with ctx as raw_span:
                span = _SpanAdapter(raw_span)
                if attributes:
                    span.set_attributes(attributes)
                if input_payload is not None:
                    span.update(input=input_payload)
                propagation_ctx = nullcontext()
                session_id = attributes.get("session_id") if attributes else None
                if isinstance(session_id, str) and session_id:
                    try:
                        from langfuse import propagate_attributes

                        propagation_ctx = propagate_attributes(session_id=session_id)
                    except Exception as exc:
                        logger.warning("Langfuse attribute propagation unavailable: %s", exc)
                with propagation_ctx:
                    yield span
        except Exception as exc:
            logger.warning("Langfuse observation failure (%s): %s", name, exc)
            yield _NoopSpan()
        finally:
            self.flush()


trace_service = TraceService()
