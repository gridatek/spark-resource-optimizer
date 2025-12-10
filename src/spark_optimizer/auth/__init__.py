"""Authentication and authorization module for Spark Resource Optimizer."""

from .models import User, UserRole
from .utils import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    verify_token,
)
from .middleware import require_auth, require_roles, get_current_user
from .routes import auth_bp

__all__ = [
    "User",
    "UserRole",
    "hash_password",
    "verify_password",
    "create_access_token",
    "create_refresh_token",
    "verify_token",
    "require_auth",
    "require_roles",
    "get_current_user",
    "auth_bp",
]
