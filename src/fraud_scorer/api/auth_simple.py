"""Simple authentication helper with fixed credentials."""
from datetime import datetime
from typing import Optional

AUTH_USERNAME = "mclovin"
AUTH_PASSWORD = "blacknox"

_authenticated = False
_authenticated_user: Optional[str] = None
_last_login_at: Optional[datetime] = None

def authenticate(username: str, password: str) -> bool:
    """Return True when the provided credentials match the allowed pair."""
    return username == AUTH_USERNAME and password == AUTH_PASSWORD

def mark_authenticated(username: str) -> None:
    """Mark the application as authenticated and track active user metadata."""
    global _authenticated, _authenticated_user, _last_login_at
    _authenticated = True
    _authenticated_user = username
    _last_login_at = datetime.now()

def clear_authentication() -> None:
    """Reset the authentication state."""
    global _authenticated, _authenticated_user, _last_login_at
    _authenticated = False
    _authenticated_user = None
    _last_login_at = None

def is_authenticated() -> bool:
    """Report whether credentials have been validated."""
    return _authenticated

def current_user() -> Optional[str]:
    """Return the authenticated user if there is one."""
    return _authenticated_user


def last_login() -> Optional[datetime]:
    """Return the timestamp of the last successful authentication, if any."""
    return _last_login_at
