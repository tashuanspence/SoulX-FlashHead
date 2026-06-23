import json
import os
import sys
import logging
from contextlib import contextmanager

from loguru import logger

try:
    from ddtrace import patch, tracer
except ImportError:
    patch = None
    tracer = None

DD_TRACE_ENABLED = os.environ.get("DD_TRACE_ENABLED", "true").lower() == "true"
VERBOSE = os.environ.get("SOULX_VERBOSE", os.environ.get("VERBOSE", "false")).lower() in {"1", "true", "yes", "on"}
server_started = False


class HealthCheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if VERBOSE:
            return True
        msg = record.getMessage()
        for path in ["GET / ", "GET /status", "GET /api/status", "GET /runpod/health", "GET /api/runpod/health", "GET /health", "GET /healthz"]:
            if path in msg:
                return False
        return True


def setup_uvicorn_filters():
    for logger_name in ("uvicorn", "uvicorn.access"):
        l = logging.getLogger(logger_name)
        # Prevent duplicate filters
        if not any(isinstance(f, HealthCheckFilter) for f in l.filters):
            l.addFilter(HealthCheckFilter())


setup_uvicorn_filters()


def _log_filter(record):
    # Always allow WARNING, ERROR, CRITICAL
    if record["level"].no >= 30:
        return True

    # If verbose is enabled, show everything
    if VERBOSE:
        return True

    # If the server is in startup/warmup phase, allow detailed logs
    if not server_started:
        return True

    # After startup, only show explicit [REQUEST] logs or specific summaries
    message = record["message"]
    if "[REQUEST]" in message:
        return True

    # Filter out upstream SoulX-FlashHead timing logs
    if "[generate]" in message:
        return False

    return False


class FilteredStdout:
    """Wraps sys.stdout to suppress upstream [generate] print() timing logs
    after the server has started. The upstream flash_head_pipeline.py uses
    print() for per-step timing, which bypasses loguru entirely."""

    def __init__(self, stream):
        self._stream = stream

    def write(self, data):
        if data and server_started and not VERBOSE and "[generate]" in data:
            return
        return self._stream.write(data)

    def flush(self):
        return self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


sys.stdout = FilteredStdout(sys.stdout)

logger.remove()
logger.add(sys.stdout, filter=_log_filter)

if DD_TRACE_ENABLED and patch is not None:
    patch(fastapi=True, aiohttp=True)


def json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    return str(value)


def log_event(event: str, **fields):
    # If not verbose, we don't need to spam stdout with telemetry JSON strings
    if not VERBOSE and server_started:
        return
    payload = {
        "service": os.environ.get("DD_SERVICE", "soulx-flashhead"),
        "env": os.environ.get("DD_ENV", "unknown"),
        "event": event,
        **{k: json_safe(v) for k, v in fields.items()},
    }
    logger.info(json.dumps(payload, sort_keys=True))


@contextmanager
def traced_span(name: str, **tags):
    if not DD_TRACE_ENABLED or tracer is None:
        yield None
        return

    with tracer.trace(name) as span:
        for key, value in tags.items():
            span.set_tag(key, json_safe(value))
        yield span


def generation_log_fields(request_id: str, generation_type: str, model_type: str, **extra):
    return {
        "request_id": request_id,
        "generation_type": generation_type,
        "model_type": model_type,
        **extra,
    }


def log_request_summary(
    endpoint: str,
    request_id: str,
    client_ip: str,
    media_info: dict,
    args: dict,
    status: str = "STARTED",
    duration_ms: float = None,
    error: str = None,
):
    media_str = ", ".join(f"{k}={v}" for k, v in media_info.items() if v is not None)
    args_str = ", ".join(f"{k}={v}" for k, v in args.items() if v is not None)
    
    if status == "STARTED":
        logger.info(
            f"[REQUEST] {endpoint} STARTED [id={request_id}] from={client_ip} | media=({media_str}) | args=({args_str})"
        )
    elif status == "COMPLETED":
        logger.info(
            f"[REQUEST] {endpoint} COMPLETED [id={request_id}] from={client_ip} duration={duration_ms:.1f}ms | media=({media_str}) | args=({args_str})"
        )
    elif status == "FAILED":
        logger.error(
            f"[REQUEST] {endpoint} FAILED [id={request_id}] from={client_ip} duration={duration_ms:.1f}ms error='{error}' | media=({media_str}) | args=({args_str})"
        )
