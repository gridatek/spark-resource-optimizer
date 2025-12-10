"""Authentication routes for user management and token handling."""

import logging
import re
from datetime import datetime
from typing import Optional

from flask import Blueprint, request, jsonify, current_app, g

from .models import User, UserRole, RefreshToken
from .utils import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    verify_token,
    generate_api_key,
    hash_token,
)
from .middleware import require_auth, require_roles, get_current_user

logger = logging.getLogger(__name__)

auth_bp = Blueprint("auth", __name__)


def get_db():
    """Get database instance from app context."""
    from spark_optimizer.storage.database import Database

    if "auth_db" not in g:
        db_url = current_app.config.get("DATABASE_URL", "sqlite:///spark_optimizer.db")
        g.auth_db = Database(db_url)
    return g.auth_db


def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None


def validate_password(password: str) -> tuple[bool, Optional[str]]:
    """Validate password strength.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r"\d", password):
        return False, "Password must contain at least one digit"
    return True, None


@auth_bp.route("/register", methods=["POST"])
def register():
    """Register a new user.

    Request body:
        {
            "username": "johndoe",
            "email": "john@example.com",
            "password": "SecurePass123",
            "full_name": "John Doe",  // optional
            "organization": "Acme Inc"  // optional
        }

    Returns:
        User data and tokens on success
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Request body is required"}), 400

        # Validate required fields
        required_fields = ["username", "email", "password"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        username = data["username"].strip().lower()
        email = data["email"].strip().lower()
        password = data["password"]

        # Validate username
        if len(username) < 3 or len(username) > 50:
            return jsonify({"error": "Username must be between 3 and 50 characters"}), 400
        if not re.match(r"^[a-z0-9_]+$", username):
            return (
                jsonify(
                    {
                        "error": "Username can only contain lowercase letters, numbers, and underscores"
                    }
                ),
                400,
            )

        # Validate email
        if not validate_email(email):
            return jsonify({"error": "Invalid email format"}), 400

        # Validate password
        is_valid, error_msg = validate_password(password)
        if not is_valid:
            return jsonify({"error": error_msg}), 400

        db = get_db()

        with db.get_session() as session:
            # Check if username already exists
            existing_user = (
                session.query(User).filter(User.username == username).first()
            )
            if existing_user:
                return jsonify({"error": "Username already exists"}), 409

            # Check if email already exists
            existing_email = session.query(User).filter(User.email == email).first()
            if existing_email:
                return jsonify({"error": "Email already registered"}), 409

            # Create new user
            user = User(
                username=username,
                email=email,
                password_hash=hash_password(password),
                role=UserRole.USER,  # Default role
                full_name=data.get("full_name", "").strip() or None,
                organization=data.get("organization", "").strip() or None,
                is_active=True,
                is_verified=False,  # Would need email verification in production
            )

            session.add(user)
            session.flush()  # Get the user ID

            # Create tokens
            access_token, access_expires = create_access_token(
                user.id, user.username, user.role.value
            )
            refresh_token, refresh_expires, refresh_hash = create_refresh_token(user.id)

            # Store refresh token
            token_record = RefreshToken(
                user_id=user.id,
                token_hash=refresh_hash,
                expires_at=refresh_expires,
                user_agent=request.headers.get("User-Agent"),
                ip_address=request.remote_addr,
            )
            session.add(token_record)
            session.commit()

            logger.info(f"New user registered: {username}")

            return (
                jsonify(
                    {
                        "message": "User registered successfully",
                        "user": user.to_dict(),
                        "tokens": {
                            "access_token": access_token,
                            "refresh_token": refresh_token,
                            "token_type": "bearer",
                            "expires_in": int(
                                (access_expires - datetime.utcnow()).total_seconds()
                            ),
                        },
                    }
                ),
                201,
            )

    except Exception as e:
        logger.error(f"Error registering user: {e}", exc_info=True)
        return jsonify({"error": "Registration failed", "message": str(e)}), 500


@auth_bp.route("/login", methods=["POST"])
def login():
    """Authenticate user and return tokens.

    Request body:
        {
            "username": "johndoe",  // or email
            "password": "SecurePass123"
        }

    Returns:
        User data and tokens on success
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Request body is required"}), 400

        # Support both username and email for login
        identifier = data.get("username") or data.get("email")
        password = data.get("password")

        if not identifier or not password:
            return jsonify({"error": "Username/email and password are required"}), 400

        identifier = identifier.strip().lower()

        db = get_db()

        with db.get_session() as session:
            # Find user by username or email
            user = (
                session.query(User)
                .filter((User.username == identifier) | (User.email == identifier))
                .first()
            )

            if user is None:
                return jsonify({"error": "Invalid credentials"}), 401

            if not user.is_active:
                return jsonify({"error": "Account is disabled"}), 403

            if not verify_password(password, user.password_hash):
                return jsonify({"error": "Invalid credentials"}), 401

            # Update last login
            user.last_login = datetime.utcnow()

            # Create tokens
            access_token, access_expires = create_access_token(
                user.id, user.username, user.role.value
            )
            refresh_token, refresh_expires, refresh_hash = create_refresh_token(user.id)

            # Store refresh token
            token_record = RefreshToken(
                user_id=user.id,
                token_hash=refresh_hash,
                expires_at=refresh_expires,
                user_agent=request.headers.get("User-Agent"),
                ip_address=request.remote_addr,
            )
            session.add(token_record)
            session.commit()

            user_dict = user.to_dict()

            logger.info(f"User logged in: {user.username}")

            return jsonify(
                {
                    "message": "Login successful",
                    "user": user_dict,
                    "tokens": {
                        "access_token": access_token,
                        "refresh_token": refresh_token,
                        "token_type": "bearer",
                        "expires_in": int(
                            (access_expires - datetime.utcnow()).total_seconds()
                        ),
                    },
                }
            )

    except Exception as e:
        logger.error(f"Error during login: {e}", exc_info=True)
        return jsonify({"error": "Login failed", "message": str(e)}), 500


@auth_bp.route("/refresh", methods=["POST"])
def refresh():
    """Refresh access token using refresh token.

    Request body:
        {
            "refresh_token": "..."
        }

    Returns:
        New access token
    """
    try:
        data = request.get_json()

        if not data or "refresh_token" not in data:
            return jsonify({"error": "Refresh token is required"}), 400

        refresh_token = data["refresh_token"]

        # Verify refresh token
        payload = verify_token(refresh_token)
        if payload is None:
            return jsonify({"error": "Invalid or expired refresh token"}), 401

        if payload.get("type") != "refresh":
            return jsonify({"error": "Invalid token type"}), 401

        user_id = int(payload["sub"])
        token_hash = hash_token(refresh_token)

        db = get_db()

        with db.get_session() as session:
            # Check if refresh token exists and is valid
            token_record = (
                session.query(RefreshToken)
                .filter(
                    RefreshToken.user_id == user_id,
                    RefreshToken.token_hash == token_hash,
                    RefreshToken.revoked_at.is_(None),
                )
                .first()
            )

            if token_record is None or not token_record.is_valid:
                return jsonify({"error": "Invalid or revoked refresh token"}), 401

            # Get user
            user = session.query(User).filter(User.id == user_id).first()

            if user is None or not user.is_active:
                return jsonify({"error": "User not found or inactive"}), 401

            # Create new access token
            access_token, access_expires = create_access_token(
                user.id, user.username, user.role.value
            )

            return jsonify(
                {
                    "access_token": access_token,
                    "token_type": "bearer",
                    "expires_in": int(
                        (access_expires - datetime.utcnow()).total_seconds()
                    ),
                }
            )

    except Exception as e:
        logger.error(f"Error refreshing token: {e}", exc_info=True)
        return jsonify({"error": "Token refresh failed", "message": str(e)}), 500


@auth_bp.route("/logout", methods=["POST"])
@require_auth
def logout():
    """Logout user and revoke refresh token.

    Request body (optional):
        {
            "refresh_token": "..."  // If provided, revoke this specific token
        }
    """
    try:
        user = get_current_user()
        data = request.get_json() or {}
        refresh_token = data.get("refresh_token")

        db = get_db()

        with db.get_session() as session:
            if refresh_token:
                # Revoke specific token
                token_hash = hash_token(refresh_token)
                token_record = (
                    session.query(RefreshToken)
                    .filter(
                        RefreshToken.user_id == user.id,
                        RefreshToken.token_hash == token_hash,
                    )
                    .first()
                )
                if token_record:
                    token_record.revoked_at = datetime.utcnow()
            else:
                # Revoke all tokens for this user
                session.query(RefreshToken).filter(
                    RefreshToken.user_id == user.id,
                    RefreshToken.revoked_at.is_(None),
                ).update({"revoked_at": datetime.utcnow()})

            session.commit()

        logger.info(f"User logged out: {user.username}")

        return jsonify({"message": "Logged out successfully"})

    except Exception as e:
        logger.error(f"Error during logout: {e}", exc_info=True)
        return jsonify({"error": "Logout failed", "message": str(e)}), 500


@auth_bp.route("/me", methods=["GET"])
@require_auth
def get_profile():
    """Get current user's profile."""
    user = get_current_user()
    return jsonify(user.to_dict())


@auth_bp.route("/me", methods=["PATCH"])
@require_auth
def update_profile():
    """Update current user's profile.

    Request body:
        {
            "full_name": "John Doe",
            "organization": "Acme Inc"
        }
    """
    try:
        user = get_current_user()
        data = request.get_json()

        if not data:
            return jsonify({"error": "Request body is required"}), 400

        db = get_db()

        with db.get_session() as session:
            db_user = session.query(User).filter(User.id == user.id).first()

            if db_user is None:
                return jsonify({"error": "User not found"}), 404

            # Update allowed fields
            if "full_name" in data:
                db_user.full_name = data["full_name"].strip() or None

            if "organization" in data:
                db_user.organization = data["organization"].strip() or None

            session.commit()

            return jsonify(
                {"message": "Profile updated successfully", "user": db_user.to_dict()}
            )

    except Exception as e:
        logger.error(f"Error updating profile: {e}", exc_info=True)
        return jsonify({"error": "Profile update failed", "message": str(e)}), 500


@auth_bp.route("/me/password", methods=["POST"])
@require_auth
def change_password():
    """Change current user's password.

    Request body:
        {
            "current_password": "OldPass123",
            "new_password": "NewSecurePass456"
        }
    """
    try:
        user = get_current_user()
        data = request.get_json()

        if not data:
            return jsonify({"error": "Request body is required"}), 400

        current_password = data.get("current_password")
        new_password = data.get("new_password")

        if not current_password or not new_password:
            return (
                jsonify({"error": "Current password and new password are required"}),
                400,
            )

        # Validate new password
        is_valid, error_msg = validate_password(new_password)
        if not is_valid:
            return jsonify({"error": error_msg}), 400

        db = get_db()

        with db.get_session() as session:
            db_user = session.query(User).filter(User.id == user.id).first()

            if db_user is None:
                return jsonify({"error": "User not found"}), 404

            # Verify current password
            if not verify_password(current_password, db_user.password_hash):
                return jsonify({"error": "Current password is incorrect"}), 401

            # Update password
            db_user.password_hash = hash_password(new_password)

            # Revoke all refresh tokens (force re-login)
            session.query(RefreshToken).filter(
                RefreshToken.user_id == user.id,
                RefreshToken.revoked_at.is_(None),
            ).update({"revoked_at": datetime.utcnow()})

            session.commit()

            logger.info(f"Password changed for user: {user.username}")

            return jsonify(
                {
                    "message": "Password changed successfully. Please log in again with your new password."
                }
            )

    except Exception as e:
        logger.error(f"Error changing password: {e}", exc_info=True)
        return jsonify({"error": "Password change failed", "message": str(e)}), 500


@auth_bp.route("/me/api-key", methods=["POST"])
@require_auth
def generate_user_api_key():
    """Generate a new API key for the current user.

    This will invalidate any existing API key.
    """
    try:
        user = get_current_user()

        db = get_db()

        with db.get_session() as session:
            db_user = session.query(User).filter(User.id == user.id).first()

            if db_user is None:
                return jsonify({"error": "User not found"}), 404

            # Generate new API key
            api_key, api_key_hash = generate_api_key()

            db_user.api_key = api_key_hash
            db_user.api_key_created_at = datetime.utcnow()

            session.commit()

            logger.info(f"API key generated for user: {user.username}")

            return jsonify(
                {
                    "message": "API key generated successfully",
                    "api_key": api_key,
                    "warning": "Store this key securely. It will not be shown again.",
                }
            )

    except Exception as e:
        logger.error(f"Error generating API key: {e}", exc_info=True)
        return jsonify({"error": "API key generation failed", "message": str(e)}), 500


@auth_bp.route("/me/api-key", methods=["DELETE"])
@require_auth
def revoke_api_key():
    """Revoke the current user's API key."""
    try:
        user = get_current_user()

        db = get_db()

        with db.get_session() as session:
            db_user = session.query(User).filter(User.id == user.id).first()

            if db_user is None:
                return jsonify({"error": "User not found"}), 404

            db_user.api_key = None
            db_user.api_key_created_at = None

            session.commit()

            logger.info(f"API key revoked for user: {user.username}")

            return jsonify({"message": "API key revoked successfully"})

    except Exception as e:
        logger.error(f"Error revoking API key: {e}", exc_info=True)
        return jsonify({"error": "API key revocation failed", "message": str(e)}), 500


# Admin routes


@auth_bp.route("/users", methods=["GET"])
@require_auth
@require_roles(UserRole.ADMIN)
def list_users():
    """List all users (admin only).

    Query parameters:
        - limit: Number of users to return (default: 50)
        - offset: Offset for pagination (default: 0)
        - role: Filter by role
        - is_active: Filter by active status
    """
    try:
        limit = request.args.get("limit", 50, type=int)
        offset = request.args.get("offset", 0, type=int)
        role = request.args.get("role")
        is_active = request.args.get("is_active")

        db = get_db()

        with db.get_session() as session:
            query = session.query(User)

            if role:
                try:
                    role_enum = UserRole(role)
                    query = query.filter(User.role == role_enum)
                except ValueError:
                    pass

            if is_active is not None:
                is_active_bool = is_active.lower() in ("true", "1", "yes")
                query = query.filter(User.is_active == is_active_bool)

            total = query.count()
            users = query.order_by(User.created_at.desc()).offset(offset).limit(limit).all()

            return jsonify(
                {
                    "users": [u.to_dict() for u in users],
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                }
            )

    except Exception as e:
        logger.error(f"Error listing users: {e}", exc_info=True)
        return jsonify({"error": "Failed to list users", "message": str(e)}), 500


@auth_bp.route("/users/<int:user_id>", methods=["GET"])
@require_auth
@require_roles(UserRole.ADMIN)
def get_user(user_id: int):
    """Get a specific user (admin only)."""
    try:
        db = get_db()

        with db.get_session() as session:
            user = session.query(User).filter(User.id == user_id).first()

            if user is None:
                return jsonify({"error": "User not found"}), 404

            return jsonify(user.to_dict(include_sensitive=True))

    except Exception as e:
        logger.error(f"Error getting user: {e}", exc_info=True)
        return jsonify({"error": "Failed to get user", "message": str(e)}), 500


@auth_bp.route("/users/<int:user_id>", methods=["PATCH"])
@require_auth
@require_roles(UserRole.ADMIN)
def update_user(user_id: int):
    """Update a user (admin only).

    Request body:
        {
            "role": "user",
            "is_active": true,
            "is_verified": true
        }
    """
    try:
        current_user = get_current_user()
        data = request.get_json()

        if not data:
            return jsonify({"error": "Request body is required"}), 400

        db = get_db()

        with db.get_session() as session:
            user = session.query(User).filter(User.id == user_id).first()

            if user is None:
                return jsonify({"error": "User not found"}), 404

            # Prevent admin from demoting themselves
            if user.id == current_user.id and "role" in data:
                if data["role"] != UserRole.ADMIN.value:
                    return (
                        jsonify({"error": "Cannot demote yourself from admin role"}),
                        400,
                    )

            # Update fields
            if "role" in data:
                try:
                    user.role = UserRole(data["role"])
                except ValueError:
                    return jsonify({"error": f"Invalid role: {data['role']}"}), 400

            if "is_active" in data:
                user.is_active = bool(data["is_active"])

            if "is_verified" in data:
                user.is_verified = bool(data["is_verified"])

            session.commit()

            logger.info(
                f"User {user_id} updated by admin {current_user.username}: {data}"
            )

            return jsonify(
                {"message": "User updated successfully", "user": user.to_dict()}
            )

    except Exception as e:
        logger.error(f"Error updating user: {e}", exc_info=True)
        return jsonify({"error": "Failed to update user", "message": str(e)}), 500


@auth_bp.route("/users/<int:user_id>", methods=["DELETE"])
@require_auth
@require_roles(UserRole.ADMIN)
def delete_user(user_id: int):
    """Delete a user (admin only)."""
    try:
        current_user = get_current_user()

        if user_id == current_user.id:
            return jsonify({"error": "Cannot delete your own account"}), 400

        db = get_db()

        with db.get_session() as session:
            user = session.query(User).filter(User.id == user_id).first()

            if user is None:
                return jsonify({"error": "User not found"}), 404

            username = user.username

            # Delete refresh tokens
            session.query(RefreshToken).filter(RefreshToken.user_id == user_id).delete()

            # Delete user
            session.delete(user)
            session.commit()

            logger.info(f"User {username} deleted by admin {current_user.username}")

            return jsonify({"message": f"User {username} deleted successfully"})

    except Exception as e:
        logger.error(f"Error deleting user: {e}", exc_info=True)
        return jsonify({"error": "Failed to delete user", "message": str(e)}), 500
