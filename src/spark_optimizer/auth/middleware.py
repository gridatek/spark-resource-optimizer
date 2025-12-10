"""Authentication middleware for protecting API routes."""

import hashlib
from functools import wraps
from typing import Callable, List, Optional, Union

from flask import request, jsonify, g, current_app

from .models import User, UserRole
from .utils import verify_token


def get_current_user() -> Optional[User]:
    """Get the current authenticated user from the request context.

    Returns:
        Current User object or None if not authenticated
    """
    return getattr(g, "current_user", None)


def _extract_token_from_header() -> Optional[str]:
    """Extract JWT token from Authorization header.

    Returns:
        Token string or None
    """
    auth_header = request.headers.get("Authorization", "")

    if auth_header.startswith("Bearer "):
        return auth_header[7:]

    return None


def _extract_api_key() -> Optional[str]:
    """Extract API key from request header.

    Returns:
        API key or None
    """
    return request.headers.get("X-API-Key")


def _authenticate_with_token(token: str) -> Optional[User]:
    """Authenticate user with JWT token.

    Args:
        token: JWT token

    Returns:
        User object if token is valid, None otherwise
    """
    from spark_optimizer.storage.database import Database

    payload = verify_token(token)
    if payload is None:
        return None

    # Check token type
    if payload.get("type") != "access":
        return None

    user_id = payload.get("sub")
    if user_id is None:
        return None

    # Get database from app context
    db_url = current_app.config.get("DATABASE_URL", "sqlite:///spark_optimizer.db")
    db = Database(db_url)

    with db.get_session() as session:
        user = session.query(User).filter(User.id == int(user_id)).first()

        if user is None:
            return None

        if not user.is_active:
            return None

        # Detach from session for use outside context
        session.expunge(user)
        return user


def _authenticate_with_api_key(api_key: str) -> Optional[User]:
    """Authenticate user with API key.

    Args:
        api_key: API key

    Returns:
        User object if API key is valid, None otherwise
    """
    from spark_optimizer.storage.database import Database

    # Hash the API key to compare with stored hash
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    db_url = current_app.config.get("DATABASE_URL", "sqlite:///spark_optimizer.db")
    db = Database(db_url)

    with db.get_session() as session:
        user = session.query(User).filter(User.api_key == api_key_hash).first()

        if user is None:
            return None

        if not user.is_active:
            return None

        # Detach from session for use outside context
        session.expunge(user)
        return user


def require_auth(f: Callable) -> Callable:
    """Decorator to require authentication for a route.

    Supports both JWT tokens (via Authorization header) and API keys (via X-API-Key header).

    Usage:
        @app.route("/protected")
        @require_auth
        def protected_route():
            user = get_current_user()
            return jsonify({"message": f"Hello, {user.username}!"})
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = None

        # Try JWT token first
        token = _extract_token_from_header()
        if token:
            user = _authenticate_with_token(token)

        # Try API key if no token
        if user is None:
            api_key = _extract_api_key()
            if api_key:
                user = _authenticate_with_api_key(api_key)

        if user is None:
            return (
                jsonify(
                    {
                        "error": "Authentication required",
                        "message": "Please provide a valid JWT token or API key",
                    }
                ),
                401,
            )

        # Store user in request context
        g.current_user = user
        return f(*args, **kwargs)

    return decorated_function


def require_roles(roles: Union[UserRole, List[UserRole]]) -> Callable:
    """Decorator to require specific roles for a route.

    Args:
        roles: Single role or list of roles that are allowed

    Usage:
        @app.route("/admin")
        @require_auth
        @require_roles(UserRole.ADMIN)
        def admin_route():
            return jsonify({"message": "Admin only!"})

        @app.route("/manage")
        @require_auth
        @require_roles([UserRole.ADMIN, UserRole.USER])
        def manage_route():
            return jsonify({"message": "Admin or User!"})
    """
    if isinstance(roles, UserRole):
        roles = [roles]

    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user = get_current_user()

            if user is None:
                return (
                    jsonify(
                        {
                            "error": "Authentication required",
                            "message": "Please authenticate first",
                        }
                    ),
                    401,
                )

            # Check if user has any of the required roles
            user_has_permission = any(user.has_permission(role) for role in roles)

            if not user_has_permission:
                return (
                    jsonify(
                        {
                            "error": "Insufficient permissions",
                            "message": f"This action requires one of the following roles: {', '.join(r.value for r in roles)}",
                            "your_role": user.role.value,
                        }
                    ),
                    403,
                )

            return f(*args, **kwargs)

        return decorated_function

    return decorator


def optional_auth(f: Callable) -> Callable:
    """Decorator that attempts authentication but doesn't require it.

    Use this for routes where authentication is optional but provides additional features.

    Usage:
        @app.route("/public")
        @optional_auth
        def public_route():
            user = get_current_user()
            if user:
                return jsonify({"message": f"Hello, {user.username}!"})
            return jsonify({"message": "Hello, guest!"})
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = None

        # Try JWT token first
        token = _extract_token_from_header()
        if token:
            user = _authenticate_with_token(token)

        # Try API key if no token
        if user is None:
            api_key = _extract_api_key()
            if api_key:
                user = _authenticate_with_api_key(api_key)

        # Store user in request context (may be None)
        g.current_user = user
        return f(*args, **kwargs)

    return decorated_function
