"""Stage 6 tests: prove the API's access rules hold.

Run: python -m pytest tests -v
Needs: Postgres running, demo users seeded, DEMO_PASSWORD and JWT_SECRET in .env.
None of these tests call the LLM, so they cost nothing.
"""
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
from main import app  # noqa: E402

PASSWORD = os.environ.get("DEMO_PASSWORD", "")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:   # 'with' runs startup (loads the models once)
        yield c


def login(client, role: str) -> dict:
    r = client.post("/auth/login", data={"username": f"{role}@ju.test", "password": PASSWORD})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_backend"] in ("local", "api")


def test_ask_requires_login(client):
    assert client.post("/ask", json={"question": "How is TARI29 graded?"}).status_code == 401


def test_wrong_password_is_rejected(client):
    r = client.post("/auth/login", data={"username": "student@ju.test", "password": "wrong"})
    assert r.status_code == 401


def test_tampered_token_is_rejected(client):
    headers = login(client, "student")
    headers["Authorization"] = headers["Authorization"][:-3] + "abc"
    assert client.get("/documents", headers=headers).status_code == 401


def test_student_cannot_use_admin_endpoint(client):
    assert client.get("/admin/stats", headers=login(client, "student")).status_code == 403


def test_admin_can_use_admin_endpoint(client):
    r = client.get("/admin/stats", headers=login(client, "admin"))
    assert r.status_code == 200 and r.json()["documents"] > 0


def test_document_lists_follow_roles(client):
    def files(role):
        return {d["file"] for d in client.get("/documents", headers=login(client, role)).json()}
    student, professor, admin = files("student"), files("professor"), files("admin")
    assert not any("synthetic" in f for f in student)
    assert {f for f in professor if "synthetic" in f} == {
        "synthetic_staff_grading_guidelines_2026.pdf",
        "synthetic_staff_reexam_scheduling_2026.pdf",
        "synthetic_staff_plagiarism_case_handling_2026.pdf",
    }
    assert "synthetic_admin_course_evaluation_procedures_2026.pdf" in admin
    assert student < professor < admin   # each role sees strictly more


def test_student_gets_no_answer_from_staff_documents(client):
    # stopped at Gate 1 (the staff document is invisible), so no LLM call
    r = client.post("/ask", headers=login(client, "student"),
                    json={"question": "What share of points do examiners require for grade 5?"})
    assert r.status_code == 200
    assert r.json()["status"] == "i_dont_know"


def test_question_too_short_is_rejected(client):
    r = client.post("/ask", headers=login(client, "student"), json={"question": "hi"})
    assert r.status_code == 422   # Pydantic validation error
