"""
auth.py — Simple session-based authentication for SreeGPT.

Uses:
  - SQLite (via SQLAlchemy) to store users
  - passlib/bcrypt for password hashing
  - itsdangerous.URLSafeTimedSerializer for signed session cookies
"""

import os
from datetime import datetime
from pathlib import Path

from fastapi import Request, HTTPException, status
from fastapi.responses import RedirectResponse
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from passlib.context import CryptContext
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

# ── Storage ───────────────────────────────────────────────────────────────────
Path("data").mkdir(exist_ok=True)

AUTH_DB_URL = "sqlite:///data/auth.db"
_engine = create_engine(AUTH_DB_URL, connect_args={"check_same_thread": False})
_Session = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
_Base = declarative_base()

# ── Models ────────────────────────────────────────────────────────────────────
class User(_Base):
    __tablename__ = "users"
    id         = Column(Integer, primary_key=True, index=True)
    username   = Column(String(64),  unique=True, nullable=False, index=True)
    email      = Column(String(255), unique=True, nullable=False, index=True)
    password   = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_auth_db():
    _Base.metadata.create_all(bind=_engine)


# ── Password hashing ──────────────────────────────────────────────────────────
_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")

def _truncate(plain: str) -> str:
    """bcrypt silently truncates or errors after 72 bytes — truncate explicitly."""
    return plain.encode("utf-8")[:72].decode("utf-8", errors="ignore")

def hash_password(plain: str) -> str:
    return _pwd.hash(_truncate(plain))

def verify_password(plain: str, hashed: str) -> bool:
    return _pwd.verify(_truncate(plain), hashed)


# ── Signed session cookies ────────────────────────────────────────────────────
SECRET_KEY = os.getenv("SESSION_SECRET", "change-me-in-production-please-32chars")
_serializer = URLSafeTimedSerializer(SECRET_KEY)
COOKIE_NAME = "sreegpt_session"
SESSION_MAX_AGE = 60 * 60 * 24 * 7  # 7 days


def create_session_token(username: str) -> str:
    return _serializer.dumps(username, salt="session")


def decode_session_token(token: str) -> str | None:
    try:
        return _serializer.loads(token, salt="session", max_age=SESSION_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return None


# ── User CRUD ─────────────────────────────────────────────────────────────────
def get_user_by_username(username: str) -> User | None:
    db = _Session()
    try:
        return db.query(User).filter(User.username == username).first()
    finally:
        db.close()


def get_user_by_email(email: str) -> User | None:
    db = _Session()
    try:
        return db.query(User).filter(User.email == email).first()
    finally:
        db.close()


def create_user(username: str, email: str, plain_password: str) -> User:
    db = _Session()
    try:
        user = User(
            username=username,
            email=email.lower().strip(),
            password=hash_password(plain_password),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    finally:
        db.close()


# ── FastAPI helpers ───────────────────────────────────────────────────────────
def get_current_user(request: Request) -> str | None:
    """Return username from cookie, or None if not logged in."""
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    return decode_session_token(token)


def require_login(request: Request) -> str:
    """Dependency — raises 302 redirect to /login if not authenticated."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_302_FOUND,
            headers={"Location": "/login"},
        )
    return user
