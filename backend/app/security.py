"""Request-level protections for the /api routes: rate limit, optional API key, size cap.

This runs BEFORE the upload is parsed, so abusive requests are rejected cheaply.
"""
import secrets
import time
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import JSONResponse

from app.config import settings

RATE_WINDOW_SECONDS = 60
MAX_TRACKED_CLIENTS = 10_000  # stops the in-memory table from growing forever
BODY_OVERHEAD_BYTES = 1024 * 1024  # multipart framing around the file itself

# client address -> times of its recent requests (in-memory: resets when the server restarts)
_recent_requests: dict[str, deque] = defaultdict(deque)


def reset_rate_limiter() -> None:
    _recent_requests.clear()


def _error(status_code: int, detail: str, headers: dict | None = None) -> JSONResponse:
    headers = {"Cache-Control": "no-store", **(headers or {})}
    return JSONResponse(status_code=status_code, content={"success": False, "detail": detail}, headers=headers)


def _seconds_until_allowed(client: str) -> int | None:
    """Record this request. Return None if allowed, else how many seconds to wait."""
    limit = settings.rate_limit_per_minute
    if limit <= 0:
        return None

    now = time.monotonic()
    times = _recent_requests[client]
    while times and now - times[0] > RATE_WINDOW_SECONDS:
        times.popleft()
    if len(times) >= limit:
        return int(RATE_WINDOW_SECONDS - (now - times[0])) + 1
    times.append(now)

    if len(_recent_requests) > MAX_TRACKED_CLIENTS:  # forget clients with no recent requests
        for address in [a for a, t in _recent_requests.items() if not t or now - t[-1] > RATE_WINDOW_SECONDS]:
            del _recent_requests[address]
    return None


async def security_middleware(request: Request, call_next):
    is_api_call = request.url.path.startswith("/api/")

    if is_api_call and request.method == "POST":
        # Behind a reverse proxy every request shares the proxy's address; see README.
        client = request.client.host if request.client else "unknown"

        wait_seconds = _seconds_until_allowed(client)
        if wait_seconds is not None:
            return _error(429, "Too many requests. Please slow down and try again shortly.",
                          {"Retry-After": str(wait_seconds)})

        if settings.api_key:
            supplied = request.headers.get("x-api-key", "")
            # compare_digest avoids leaking the key through response timing
            if not secrets.compare_digest(supplied.encode(), settings.api_key.encode()):
                return _error(401, "Invalid or missing API key.")

        declared_length = request.headers.get("content-length", "")
        if declared_length.isdigit() and int(declared_length) > settings.max_upload_mb * 1024 * 1024 + BODY_OVERHEAD_BYTES:
            return _error(413, f"File too large. Maximum size is {settings.max_upload_mb} MB.")

    response = await call_next(request)
    if is_api_call:
        response.headers["Cache-Control"] = "no-store"  # results contain sensitive health data
    return response
