"""User model and role definitions for authentication."""

from datetime import datetime
from enum import Enum
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Enum as SQLEnum
from spark_optimizer.storage.database import Base


class UserRole(str, Enum):
    """User roles for Role-Based Access Control (RBAC)."""

    ADMIN = "admin"  # Full access to all features
    USER = "user"  # Can use recommendations, view jobs, submit feedback
    VIEWER = "viewer"  # Read-only access to jobs and stats


class User(Base):  # type: ignore[misc,valid-type]
    """User model for authentication and authorization."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role: UserRole = Column(  # type: ignore[assignment]
        SQLEnum(UserRole), default=UserRole.USER, nullable=False
    )

    # Profile information
    full_name = Column(String(255), nullable=True)
    organization = Column(String(255), nullable=True)

    # Account status
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    # API key for programmatic access (optional)
    api_key = Column(String(64), unique=True, nullable=True, index=True)
    api_key_created_at = Column(DateTime, nullable=True)

    def to_dict(self, include_sensitive: bool = False) -> dict:
        """Convert user to dictionary representation.

        Args:
            include_sensitive: Whether to include sensitive fields like api_key

        Returns:
            Dictionary representation of the user
        """
        data = {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role.value,
            "full_name": self.full_name,
            "organization": self.organization,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }

        if include_sensitive and self.api_key:
            data["api_key"] = self.api_key
            data["api_key_created_at"] = (
                self.api_key_created_at.isoformat() if self.api_key_created_at else None
            )

        return data

    def has_permission(self, required_role: UserRole) -> bool:
        """Check if user has at least the required role level.

        Role hierarchy: ADMIN > USER > VIEWER

        Args:
            required_role: The minimum role required

        Returns:
            True if user has sufficient permissions
        """
        role_hierarchy = {
            UserRole.VIEWER: 0,
            UserRole.USER: 1,
            UserRole.ADMIN: 2,
        }

        user_level = role_hierarchy.get(self.role, 0)  # type: ignore[arg-type]
        required_level = role_hierarchy.get(required_role, 0)

        return user_level >= required_level

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username='{self.username}', role='{self.role.value}')>"


class RefreshToken(Base):  # type: ignore[misc,valid-type]
    """Model for storing refresh tokens."""

    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    token_hash = Column(String(255), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    revoked_at = Column(DateTime, nullable=True)
    user_agent = Column(String(255), nullable=True)
    ip_address = Column(String(45), nullable=True)

    @property
    def is_valid(self) -> bool:
        """Check if the refresh token is still valid."""
        if self.revoked_at is not None:
            return False
        return datetime.utcnow() < self.expires_at

    def __repr__(self) -> str:
        return f"<RefreshToken(id={self.id}, user_id={self.user_id}, valid={self.is_valid})>"
