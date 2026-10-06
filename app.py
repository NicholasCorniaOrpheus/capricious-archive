"""Streamlit GUI.   streamlit run app.py"""
import os
from pathlib import Path
import streamlit as st
from corpus import Index, load_corpus
import llm
import hmac

st.set_page_config(page_title="Capricious Archive", page_icon="📜", layout="wide")


def secret(name, default=None):
    try:  # 1. Streamlit secrets (Cloud settings or local secrets.toml)
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)  # 2. environment variable


# ---- password gate: FAILS CLOSED (no password configured = app stays locked)
pw = secret("APP_PASSWORD")
if not pw:
    st.error("APP_PASSWORD is not configured, so the app is locked.")
    st.stop()
if not st.session_state.get("ok"):
    entered = st.text_input("Password", type="password")
    if entered:
        if hmac.compare_digest(entered.encode(), str(pw).encode()):
            st.session_state.ok = True
            st.rerun()
        else:
            st.error("Wrong password")
    st.stop()

if not secret("ANTHROPIC_API_KEY"):
    st.error("ANTHROPIC_API_KEY is not configured.")
    st.stop()
os.environ["ANTHROPIC_API_KEY"] = secret("ANTHROPIC_API_KEY")


@st.cache_resource
def load():
    idx = Index(load_corpus("data"))
    p = Path("profile.md")
    return idx, (p.read_text(encoding="utf-8") if p.exists() else "")


@st.cache_resource
def ledger():
    return {"spent": 0.0}  # shared by all sessions; resets when the app restarts


idx, profile = load()
budget, led = float(secret("BUDGET_USD", 4.0)), ledger()

with st.sidebar:
    st.title("📜 Capricious Archive")
    models = {"Haiku 4.5 — cheap drafts": llm.CHEAP, "Sonnet 5.5 — polish": llm.STRONG}
    model = models[st.radio("Model", list(models))]
    n_poems = st.slider("Exemplar poems retrieved", 2, 10, 6)
    max_tokens = st.slider("Max output tokens", 300, 3000, 1200, 100)
    show_ctx = st.toggle("Show retrieved context", True)
    s = idx.stats()
    st.caption(f"{s['entities']} entities · {s['poems']} poems indexed")
    st.caption(
        "Style profile: "
        + ("loaded ✅" if profile else "missing ⚠️ (run `python cli.py profile`)")
    )
    st.progress(
        min(1.0, led["spent"] / budget),
        f"Shared budget ~${led['spent']:.2f} / ${budget:.2f}",
    )
    if st.button("New conversation"):
        st.session_state.msgs = []
        st.rerun()

st.session_state.setdefault("msgs", [])
for m in st.session_state.msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("meta") and show_ctx:
            with st.expander(
                f"Retrieved: {', '.join(m['meta']['terms'])}  ·  ~${m['meta']['cost']:.4f}"
            ):
                st.text(m["meta"]["context"])

if q := st.chat_input(
    "e.g. A madrigal in hendecasyllables on a lover abandoned, with eye and light imagery"
):
    st.session_state.msgs.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        if led["spent"] >= budget:
            st.error("Shared budget exhausted. Ask the administrator.")
        else:
            with st.spinner("Searching the corpus and writing…"):
                hist = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.msgs[:-1]
                ][-6:]
                r = llm.run_request(idx, profile, q, hist, model, max_tokens, n_poems)
            led["spent"] += r["cost"]
            st.markdown(r["text"])
            st.session_state.msgs.append(
                {"role": "assistant", "content": r["text"], "meta": r}
            )
            st.rerun()
