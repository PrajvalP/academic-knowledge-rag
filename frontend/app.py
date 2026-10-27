"""Stage 10: student-facing frontend (Streamlit).

It only talks to the FastAPI backend over HTTP, exactly like any other
client would: log in -> get a token -> send the token with every request.
Access control stays in the backend.

Run (with the backend already running):
    streamlit run frontend/app.py
"""
import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = 120  # seconds; a sleeping free server can take ~1 min to wake up

st.set_page_config(page_title="Academic Knowledge Assistant", page_icon="🎓")


# ---------- talking to the API ----------

def api(method: str, path: str, **kwargs):
    """Call the backend with the user's token. Returns the response."""
    headers = {}
    if st.session_state.get("token"):
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    try:
        return requests.request(method, f"{API_URL}{path}", headers=headers,
                                timeout=TIMEOUT, **kwargs)
    except requests.Timeout:
        st.warning("The server is waking up (free hosting sleeps when unused). "
                   "Please try again in a minute.")
        st.stop()
    except requests.ConnectionError:
        st.error(f"Can't reach the backend at {API_URL}. Is it running?")
        st.stop()


def logout(message: str | None = None):
    for key in ["token", "role", "email", "messages"]:
        st.session_state.pop(key, None)
    if message:
        st.session_state.flash = message
    st.rerun()


# ---------- login page ----------

def login_page():
    st.title("🎓 Academic Knowledge Assistant")
    st.caption("Ask questions about Jönköping University course syllabi and academic "
               "policies. Every answer cites its source.")
    st.caption("Free hosting: the first request after a quiet period can take up to a "
               "minute while the server wakes up.")
    if msg := st.session_state.pop("flash", None):
        st.info(msg)

    with st.form("login"):
        email = st.text_input("Email", placeholder="student@ju.test")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in", type="primary")

    if submitted:
        # OAuth2 login expects form fields named username and password
        r = api("POST", "/auth/login", data={"username": email, "password": password})
        if r.status_code == 200:
            body = r.json()
            st.session_state.token = body["access_token"]
            st.session_state.role = body["role"]
            st.session_state.email = email
            st.session_state.messages = []
            st.rerun()
        else:
            st.error("Wrong email or password.")


# ---------- showing one answer ----------

def show_answer(result: dict):
    if result["status"] == "i_dont_know":
        st.info(f"🤷 {result['answer']}\n\nTry rephrasing, or ask the student "
                "service office.")
        return

    st.markdown(result["answer"])
    for warning in result.get("warnings", []):
        st.warning(warning, icon="⚠️")

    if result["citations"]:
        st.caption("Sources")
        for c in result["citations"]:
            code = f" ({c['course_code']})" if c.get("course_code") else ""
            label = f"[{c['n']}] {c['title']}{code}, {c['year']}, page {c['page']}"
            with st.expander(label):
                st.write(f"Section: {c.get('section') or '-'}")
                if c.get("url"):
                    # '#page=N' makes most browsers open the PDF at that page
                    st.link_button("Open the PDF", f"{c['url']}#page={c['page']}")


# ---------- main page ----------

def chat_page():
    with st.sidebar:
        st.markdown(f"**{st.session_state.email}**")
        st.caption(f"Role: {st.session_state.role}")
        if st.button("Log out"):
            logout()

        st.divider()
        st.markdown("**Documents you can search**")
        r = api("GET", "/documents")
        if r.status_code == 401:
            logout("Your session expired. Please log in again.")
        docs = r.json()
        st.caption(f"{len(docs)} current documents")
        for d in docs:
            code = f"{d['course_code']} " if d.get("course_code") else ""
            st.markdown(f"- {code}{d['title']} ({d['year']})")

    st.title("🎓 Academic Knowledge Assistant")

    # replay the conversation so far (Streamlit reruns the script on every click)
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.markdown(m["content"])
            else:
                show_answer(m["content"])

    if question := st.chat_input("Ask about a course or a university rule..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the documents..."):
                r = api("POST", "/ask", json={"question": question})
            if r.status_code == 401:
                logout("Your session expired. Please log in again.")
            elif r.status_code == 422:
                st.error("Please write a question of at least 3 characters.")
            elif r.status_code != 200:
                st.error(f"Something went wrong (error {r.status_code}).")
            else:
                result = r.json()
                show_answer(result)
                st.session_state.messages.append({"role": "assistant", "content": result})


if st.session_state.get("token"):
    chat_page()
else:
    login_page()
