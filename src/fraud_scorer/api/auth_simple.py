"""Per-browser signed sessions for the web interface."""
import base64
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
import time

SESSION_COOKIE = "fraud_scorer_session"
SESSION_MAX_AGE = 8 * 60 * 60


def _configuration():
    username = os.environ.get("FS_AUTH_USERNAME", "")
    password = os.environ.get("FS_AUTH_PASSWORD", "")
    secret = os.environ.get("FS_SESSION_SECRET", "")
    if not username or not password or len(secret) < 32:
        return None
    return username, password, secret


def authenticate(username: str, password: str) -> bool:
    config = _configuration()
    return bool(config and isinstance(username, str) and isinstance(password, str)
                and hmac.compare_digest(username.encode(), config[0].encode())
                and hmac.compare_digest(password.encode(), config[1].encode()))


def mark_authenticated(username: str, response) -> None:
    config = _configuration()
    if not config or username != config[0]:
        raise ValueError("Authentication is not configured")
    payload = base64.urlsafe_b64encode(json.dumps({
        "user": username, "issued": int(time.time()),
    }, separators=(",", ":")).encode()).decode()
    signature = hmac.new(config[2].encode(), payload.encode(), hashlib.sha256).hexdigest()
    response.set_cookie(
        SESSION_COOKIE, f"{payload}.{signature}", max_age=SESSION_MAX_AGE,
        httponly=True, secure=True, samesite="strict", path="/",
    )


def clear_authentication(response) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/", secure=True, httponly=True, samesite="strict")


def _session(request):
    config = _configuration()
    token = request.cookies.get(SESSION_COOKIE, "")
    if not config or not isinstance(token, str) or len(token) > 4096:
        return None
    try:
        payload, signature = token.split(".")
        expected = hmac.new(config[2].encode(), payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return None
        session = json.loads(base64.b64decode(payload, altchars=b"-_", validate=True))
        issued = session.get("issued")
        now = time.time()
        if (session.get("user") != config[0] or type(issued) is not int
                or not 0 <= now - issued < SESSION_MAX_AGE):
            return None
        return session
    except (ValueError, TypeError, AttributeError, UnicodeError):
        return None


def is_authenticated(request) -> bool:
    return _session(request) is not None


def current_user(request):
    session = _session(request)
    return session["user"] if session else None


def last_login(request):
    session = _session(request)
    return datetime.fromtimestamp(session["issued"], timezone.utc) if session else None
