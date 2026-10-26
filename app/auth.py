"""Stage 6: authentication and role-based access control (RBAC).

- Passwords are stored as bcrypt hashes, never as plain text.
- Logging in returns a JWT: a signed token holding the user's id and role.
- Every protected endpoint reads the token and knows who is asking.
- ROLE_LEVELS says which document access levels each role may search.
"""
import os
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
import psycopg
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

ROLE_LEVELS = {
    "student":   ["public", "student"],
    "professor": ["public", "student", "professor"],
    "admin":     ["public", "student", "professor", "admin"],
}
TOKEN_MINUTES = 60
ALGORITHM = "HS256"

# tells FastAPI (and the /docs page) to expect "Authorization: Bearer <token>"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


class User(BaseModel):
    id: int
    email: str
    role: str

    @property
    def levels(self) -> list[str]:
        return ROLE_LEVELS[self.role]


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def _secret() -> str:
    secret = os.environ.get("JWT_SECRET")
    if not secret:
        raise RuntimeError("JWT_SECRET is missing from .env")
    return secret


def authenticate(email: str, password: str) -> User | None:
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        row = conn.execute(
            "SELECT id, email, role, password_hash FROM users WHERE email = %s",
            (email.lower(),),
        ).fetchone()
    if not row or not verify_password(password, row[3]):
        return None   # same answer for "no such user" and "wrong password"
    return User(id=row[0], email=row[1], role=row[2])


def create_token(user: User) -> str:
    payload = {
        "sub": str(user.id), "email": user.email, "role": user.role,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=TOKEN_MINUTES),
    }
    return jwt.encode(payload, _secret(), algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """FastAPI dependency: turns the request's token into a User, or 401."""
    try:
        payload = jwt.decode(token, _secret(), algorithms=[ALGORITHM])
        return User(id=int(payload["sub"]), email=payload["email"], role=payload["role"])
    except (jwt.InvalidTokenError, KeyError, ValueError):
        # covers expired, tampered and malformed tokens
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_role(*roles: str):
    """Dependency factory: only the given roles may call the endpoint, else 403."""
    def check(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                                detail="Not allowed for your role")
        return user
    return check
