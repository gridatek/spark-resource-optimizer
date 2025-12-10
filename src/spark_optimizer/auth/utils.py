"""Authentication utilities for password hashing and JWT token management."""

import hashlib
import hmac
import secrets
import base64
import json
import os
from datetime import datetime, timedelta
from typing import Optional, Tuple, Any


# Configuration defaults (can be overridden via environment variables)
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_hex(32))
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# Password hashing configuration
PBKDF2_ITERATIONS = 100000
SALT_LENGTH = 32
HASH_LENGTH = 32


def hash_password(password: str) -> str:
    """Hash a password using PBKDF2-SHA256.

    Args:
        password: Plain text password to hash

    Returns:
        Base64-encoded string containing salt and hash
    """
    salt = secrets.token_bytes(SALT_LENGTH)
    pw_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS, dklen=HASH_LENGTH
    )
    # Store as: iterations$salt$hash (all base64 encoded)
    return f"{PBKDF2_ITERATIONS}${base64.b64encode(salt).decode()}${base64.b64encode(pw_hash).decode()}"


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash.

    Args:
        password: Plain text password to verify
        password_hash: Stored password hash

    Returns:
        True if password matches, False otherwise
    """
    try:
        parts = password_hash.split("$")
        if len(parts) != 3:
            return False

        iterations = int(parts[0])
        salt = base64.b64decode(parts[1])
        stored_hash = base64.b64decode(parts[2])

        computed_hash = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, iterations, dklen=len(stored_hash)
        )

        return hmac.compare_digest(computed_hash, stored_hash)
    except (ValueError, TypeError):
        return False


def _base64url_encode(data: bytes) -> str:
    """Encode bytes to base64url string without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _base64url_decode(data: str) -> bytes:
    """Decode base64url string to bytes."""
    padding = 4 - len(data) % 4
    if padding != 4:
        data += "=" * padding
    return base64.urlsafe_b64decode(data)


def _create_jwt(payload: dict, secret: str) -> str:
    """Create a JWT token with HS256 algorithm.

    Args:
        payload: Token payload
        secret: Secret key for signing

    Returns:
        JWT token string
    """
    header = {"alg": JWT_ALGORITHM, "typ": "JWT"}

    header_b64 = _base64url_encode(json.dumps(header).encode())
    payload_b64 = _base64url_encode(json.dumps(payload).encode())

    message = f"{header_b64}.{payload_b64}"
    signature = hmac.new(
        secret.encode(), message.encode(), hashlib.sha256
    ).digest()
    signature_b64 = _base64url_encode(signature)

    return f"{message}.{signature_b64}"


def _verify_jwt(token: str, secret: str) -> Optional[dict]:
    """Verify a JWT token and return its payload.

    Args:
        token: JWT token string
        secret: Secret key for verification

    Returns:
        Token payload if valid, None otherwise
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None

        header_b64, payload_b64, signature_b64 = parts

        # Verify signature
        message = f"{header_b64}.{payload_b64}"
        expected_signature = hmac.new(
            secret.encode(), message.encode(), hashlib.sha256
        ).digest()

        actual_signature = _base64url_decode(signature_b64)

        if not hmac.compare_digest(expected_signature, actual_signature):
            return None

        # Decode payload
        payload = json.loads(_base64url_decode(payload_b64))

        return payload
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def create_access_token(
    user_id: int,
    username: str,
    role: str,
    expires_delta: Optional[timedelta] = None,
) -> Tuple[str, datetime]:
    """Create a JWT access token.

    Args:
        user_id: User's database ID
        username: User's username
        role: User's role
        expires_delta: Custom expiration time, defaults to ACCESS_TOKEN_EXPIRE_MINUTES

    Returns:
        Tuple of (token_string, expiration_datetime)
    """
    if expires_delta is None:
        expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    now = datetime.utcnow()
    expire = now + expires_delta

    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "type": "access",
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }

    token = _create_jwt(payload, JWT_SECRET_KEY)
    return token, expire


def create_refresh_token(
    user_id: int,
    expires_delta: Optional[timedelta] = None,
) -> Tuple[str, datetime, str]:
    """Create a refresh token.

    Args:
        user_id: User's database ID
        expires_delta: Custom expiration time, defaults to REFRESH_TOKEN_EXPIRE_DAYS

    Returns:
        Tuple of (token_string, expiration_datetime, token_hash_for_storage)
    """
    if expires_delta is None:
        expires_delta = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    now = datetime.utcnow()
    expire = now + expires_delta

    # Generate a random token
    random_part = secrets.token_hex(32)

    payload = {
        "sub": str(user_id),
        "type": "refresh",
        "jti": random_part,  # JWT ID for unique identification
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }

    token = _create_jwt(payload, JWT_SECRET_KEY)

    # Hash the token for storage (so we don't store the actual token)
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    return token, expire, token_hash


def verify_token(token: str) -> Optional[dict]:
    """Verify a JWT token and return its payload.

    Args:
        token: JWT token string

    Returns:
        Token payload if valid and not expired, None otherwise
    """
    payload = _verify_jwt(token, JWT_SECRET_KEY)

    if payload is None:
        return None

    # Check expiration
    exp = payload.get("exp")
    if exp is None:
        return None

    if datetime.utcnow().timestamp() > exp:
        return None

    return payload


def generate_api_key() -> Tuple[str, str]:
    """Generate an API key for programmatic access.

    Returns:
        Tuple of (api_key, api_key_hash_for_storage)
    """
    api_key = f"sro_{secrets.token_hex(24)}"  # sro = spark resource optimizer
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    return api_key, api_key_hash


def verify_api_key(api_key: str, stored_hash: str) -> bool:
    """Verify an API key against its stored hash.

    Args:
        api_key: API key to verify
        stored_hash: Stored hash of the API key

    Returns:
        True if API key is valid
    """
    computed_hash = hashlib.sha256(api_key.encode()).hexdigest()
    return hmac.compare_digest(computed_hash, stored_hash)


def hash_token(token: str) -> str:
    """Hash a token for storage.

    Args:
        token: Token to hash

    Returns:
        SHA256 hash of the token
    """
    return hashlib.sha256(token.encode()).hexdigest()
