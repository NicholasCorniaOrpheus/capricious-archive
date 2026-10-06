"""Streamlit GUI.   streamlit run app.py"""
import os
from pathlib import Path
import streamlit as st
from corpus import Index, load_corpus
import llm
import tomllib

st.set_page_config(page_title="Capricious Archive", page_icon="📜", layout="wide")


def secret(name, default=None):
    key = os.environ.get("APP_PASSWORD")  # 1. environment variable wins
    if key:
        return key
    p = (
        Path(__file__).parent / ".streamlit" / "secrets.toml"
    )  # 2. fall back to secrets.toml
    if p.exists():
        return tomllib.loads(p.read_text(encoding="utf-8")).get("APP_PASSWORD")
    return None


# ---- password gate (skipped if APP_PASSWORD is not set, e.g. local dev)
pw = secret("APP_PASSWORD")
if pw and not st.session_state.get("ok"):
    entered = st.text_input("Password", type="password")
    if entered:
        if entered == pw:
            st.session_state.ok = True
            st.rerun()
        else:
            st.error("Wrong password")
    st.stop()

if secret("ANTHROPIC_API_KEY"):
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
    st.title("📜 Cinquecento Poet")
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
