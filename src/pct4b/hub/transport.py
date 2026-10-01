"""Explicit upload timeouts and credential-safe diagnostic logging."""

from __future__ import annotations

import logging
import re


def redact(message: str) -> str:
    message = re.sub(r"https?://[^\s<>]+", "[URL omitted]", message)
    return re.sub(r"hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{20,}", "[credential omitted]", message)


class SafeTransferLog(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = ()
        return True


def configure_transport(settings: dict) -> None:
    import httpx
    from huggingface_hub import set_client_factory

    connect = float(settings.get("connect_timeout_seconds", 180))
    transfer = float(settings.get("transfer_timeout_seconds", 1800))
    if connect <= 0 or transfer <= 0:
        raise ValueError("HTTP timeouts must be positive")
    timeout = httpx.Timeout(connect=connect, read=transfer, write=transfer, pool=connect)

    def enforce_timeouts(request: httpx.Request) -> None:
        # Some Hub calls supply their own short timeout; apply the upload policy there too.
        request.extensions["timeout"] = timeout.as_dict()

    def factory() -> httpx.Client:
        return httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
            event_hooks={"request": [enforce_timeouts]},
        )

    set_client_factory(factory)
    for name in ("huggingface_hub.utils._http", "httpx", "httpcore"):
        logger = logging.getLogger(name)
        if not any(isinstance(f, SafeTransferLog) for f in logger.filters):
            logger.addFilter(SafeTransferLog())
