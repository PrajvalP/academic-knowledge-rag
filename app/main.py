"""Stage 6: the FastAPI backend.

Run:   uvicorn main:app --app-dir app --reload
Docs:  http://127.0.0.1:8000/docs
"""
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

import psycopg  # noqa: E402
from fastapi import Depends, FastAPI, HTTPException, status  # noqa: E402
from fastapi.security import OAuth2PasswordRequestForm  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from auth import User, authenticate, create_token, get_current_user, require_role  # noqa: E402
from generation import answer  # noqa: E402
from rerank import get_reranker  # noqa: E402
from retrieval import get_model  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load both models at startup, so the first question isn't slow
    get_model()
    get_reranker()
    yield


app = FastAPI(title="Academic Knowledge RAG", version="0.1", lifespan=lifespan)


# ---------- request and response shapes (Pydantic validates them) ----------

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class Citation(BaseModel):
    n: int
    title: str
    course_code: str | None
    year: int | None
    section: str | None
    page: int | None
    url: str | None


class AskResponse(BaseModel):
    answer: str
    status: str                  # answered | i_dont_know
    reason: str | None           # why "I don't know", if it was
    citations: list[Citation]
    warnings: list[str]


class Document(BaseModel):
    file: str
    title: str
    doc_type: str
    course_code: str | None
    year: int | None
    valid_from: str | None
    source_url: str | None


# ---------- endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/auth/login", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends()):
    """Log in with email (in the 'username' field) and password."""
    user = authenticate(form.username, form.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Wrong email or password",
                            headers={"WWW-Authenticate": "Bearer"})
    return Token(access_token=create_token(user), role=user.role)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, user: User = Depends(get_current_user)):
    """Answer a question using only documents this user's role may see."""
    result = answer(req.question, levels=user.levels)
    return AskResponse(
        answer=result["answer"], status=result["status"], reason=result["reason"],
        citations=result["citations"], warnings=result["warnings"],
    )


@app.get("/documents", response_model=list[Document])
def documents(user: User = Depends(get_current_user)):
    """Current documents this user may access."""
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        rows = conn.execute("""
            SELECT file, title, doc_type, course_code, year, valid_from::text, source_url
            FROM documents
            WHERE access_level = ANY(%s) AND superseded_by IS NULL
            ORDER BY doc_type, title
        """, (user.levels,)).fetchall()
    fields = Document.model_fields.keys()
    return [Document(**dict(zip(fields, r))) for r in rows]


@app.get("/admin/stats")
def admin_stats(user: User = Depends(require_role("admin"))):
    """Admin only: what's in the database."""
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        q = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
        return {
            "documents": q("SELECT count(*) FROM documents"),
            "superseded": q("SELECT count(*) FROM documents WHERE superseded_by IS NOT NULL"),
            "chunks": q("SELECT count(*) FROM chunks"),
            "chunks_without_embedding": q("SELECT count(*) FROM chunks WHERE embedding IS NULL"),
            "users": q("SELECT count(*) FROM users"),
        }
