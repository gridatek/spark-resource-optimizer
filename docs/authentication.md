# Authentication & Authorization Guide

This guide covers the authentication and authorization features of Spark Resource Optimizer.

## Overview

Spark Resource Optimizer includes a comprehensive authentication system with:

- **JWT-based authentication** for secure API access
- **Role-Based Access Control (RBAC)** with three user roles
- **API key support** for programmatic access
- **Refresh tokens** for session management

## User Roles

| Role | Permissions |
|------|-------------|
| `admin` | Full access to all features, user management |
| `user` | Use recommendations, view jobs, submit feedback |
| `viewer` | Read-only access to jobs and statistics |

## Authentication Methods

### 1. JWT Tokens

JWT tokens are the primary authentication method for interactive use.

#### Register a New User

```bash
curl -X POST http://localhost:8080/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "johndoe",
    "email": "john@example.com",
    "password": "SecurePass123",
    "full_name": "John Doe",
    "organization": "Acme Inc"
  }'
```

Response:
```json
{
  "message": "User registered successfully",
  "user": {
    "id": 1,
    "username": "johndoe",
    "email": "john@example.com",
    "role": "user",
    "full_name": "John Doe",
    "organization": "Acme Inc",
    "is_active": true,
    "is_verified": false,
    "created_at": "2024-01-15T10:30:00"
  },
  "tokens": {
    "access_token": "eyJ...",
    "refresh_token": "eyJ...",
    "token_type": "bearer",
    "expires_in": 1800
  }
}
```

#### Login

```bash
curl -X POST http://localhost:8080/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "johndoe",
    "password": "SecurePass123"
  }'
```

#### Using Access Tokens

Include the access token in the `Authorization` header:

```bash
curl -X GET http://localhost:8080/api/v1/jobs \
  -H "Authorization: Bearer eyJ..."
```

#### Refreshing Tokens

When the access token expires, use the refresh token to get a new one:

```bash
curl -X POST http://localhost:8080/api/v1/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "eyJ..."
  }'
```

### 2. API Keys

API keys are ideal for automated scripts and CI/CD pipelines.

#### Generate an API Key

```bash
curl -X POST http://localhost:8080/api/v1/auth/me/api-key \
  -H "Authorization: Bearer eyJ..."
```

Response:
```json
{
  "message": "API key generated successfully",
  "api_key": "sro_abc123...",
  "warning": "Store this key securely. It will not be shown again."
}
```

#### Using API Keys

Include the API key in the `X-API-Key` header:

```bash
curl -X POST http://localhost:8080/api/v1/recommend \
  -H "X-API-Key: sro_abc123..." \
  -H "Content-Type: application/json" \
  -d '{"input_size_bytes": 10737418240}'
```

#### Revoke an API Key

```bash
curl -X DELETE http://localhost:8080/api/v1/auth/me/api-key \
  -H "Authorization: Bearer eyJ..."
```

## User Management

### Get Current Profile

```bash
curl -X GET http://localhost:8080/api/v1/auth/me \
  -H "Authorization: Bearer eyJ..."
```

### Update Profile

```bash
curl -X PATCH http://localhost:8080/api/v1/auth/me \
  -H "Authorization: Bearer eyJ..." \
  -H "Content-Type: application/json" \
  -d '{
    "full_name": "John D. Doe",
    "organization": "New Company"
  }'
```

### Change Password

```bash
curl -X POST http://localhost:8080/api/v1/auth/me/password \
  -H "Authorization: Bearer eyJ..." \
  -H "Content-Type: application/json" \
  -d '{
    "current_password": "OldPass123",
    "new_password": "NewSecurePass456"
  }'
```

### Logout

```bash
curl -X POST http://localhost:8080/api/v1/auth/logout \
  -H "Authorization: Bearer eyJ..." \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "eyJ..."
  }'
```

## Admin Operations

Admin users have access to additional endpoints for user management.

### List All Users

```bash
curl -X GET "http://localhost:8080/api/v1/auth/users?limit=50&offset=0" \
  -H "Authorization: Bearer eyJ..."
```

### Get User Details

```bash
curl -X GET http://localhost:8080/api/v1/auth/users/1 \
  -H "Authorization: Bearer eyJ..."
```

### Update User Role

```bash
curl -X PATCH http://localhost:8080/api/v1/auth/users/1 \
  -H "Authorization: Bearer eyJ..." \
  -H "Content-Type: application/json" \
  -d '{
    "role": "admin",
    "is_active": true
  }'
```

### Delete User

```bash
curl -X DELETE http://localhost:8080/api/v1/auth/users/2 \
  -H "Authorization: Bearer eyJ..."
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `JWT_SECRET_KEY` | Secret key for signing JWT tokens | Auto-generated |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Access token lifetime in minutes | `30` |
| `REFRESH_TOKEN_EXPIRE_DAYS` | Refresh token lifetime in days | `7` |

### Disabling Authentication

To disable authentication (not recommended for production):

```python
run_server(
    db_url="sqlite:///spark_optimizer.db",
    enable_auth=False
)
```

Or via CLI:
```bash
# Authentication is enabled by default
spark-optimizer serve --host 0.0.0.0 --port 8080
```

## Password Requirements

Passwords must meet the following criteria:

- Minimum 8 characters
- At least one uppercase letter (A-Z)
- At least one lowercase letter (a-z)
- At least one digit (0-9)

## Security Best Practices

1. **Use HTTPS** in production to protect tokens in transit
2. **Set strong JWT secret** via environment variable
3. **Rotate API keys** regularly
4. **Use short-lived access tokens** (default: 30 minutes)
5. **Store tokens securely** - never in local storage for web apps
6. **Implement rate limiting** at the load balancer level
7. **Monitor failed login attempts** for suspicious activity

## Error Responses

### 401 Unauthorized

```json
{
  "error": "Authentication required",
  "message": "Please provide a valid JWT token or API key"
}
```

### 403 Forbidden

```json
{
  "error": "Insufficient permissions",
  "message": "This action requires one of the following roles: admin",
  "your_role": "user"
}
```

### 409 Conflict

```json
{
  "error": "Username already exists"
}
```

## Integration Examples

### Python

```python
import requests

class SparkOptimizerClient:
    def __init__(self, base_url, api_key=None):
        self.base_url = base_url
        self.api_key = api_key
        self.access_token = None

    def login(self, username, password):
        response = requests.post(
            f"{self.base_url}/api/v1/auth/login",
            json={"username": username, "password": password}
        )
        data = response.json()
        self.access_token = data["tokens"]["access_token"]
        return data

    def _get_headers(self):
        if self.api_key:
            return {"X-API-Key": self.api_key}
        elif self.access_token:
            return {"Authorization": f"Bearer {self.access_token}"}
        return {}

    def recommend(self, input_size_bytes, **kwargs):
        response = requests.post(
            f"{self.base_url}/api/v1/recommend",
            headers=self._get_headers(),
            json={"input_size_bytes": input_size_bytes, **kwargs}
        )
        return response.json()

# Usage with API key
client = SparkOptimizerClient(
    "http://localhost:8080",
    api_key="sro_abc123..."
)
recommendation = client.recommend(10 * 1024**3)  # 10 GB

# Usage with login
client = SparkOptimizerClient("http://localhost:8080")
client.login("johndoe", "SecurePass123")
recommendation = client.recommend(10 * 1024**3)
```

### JavaScript/TypeScript

```typescript
class SparkOptimizerClient {
  private baseUrl: string;
  private apiKey?: string;
  private accessToken?: string;

  constructor(baseUrl: string, apiKey?: string) {
    this.baseUrl = baseUrl;
    this.apiKey = apiKey;
  }

  async login(username: string, password: string) {
    const response = await fetch(`${this.baseUrl}/api/v1/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });
    const data = await response.json();
    this.accessToken = data.tokens.access_token;
    return data;
  }

  private getHeaders(): HeadersInit {
    if (this.apiKey) {
      return { 'X-API-Key': this.apiKey };
    }
    if (this.accessToken) {
      return { 'Authorization': `Bearer ${this.accessToken}` };
    }
    return {};
  }

  async recommend(inputSizeBytes: number, options: Record<string, any> = {}) {
    const response = await fetch(`${this.baseUrl}/api/v1/recommend`, {
      method: 'POST',
      headers: {
        ...this.getHeaders(),
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ input_size_bytes: inputSizeBytes, ...options }),
    });
    return response.json();
  }
}
```
