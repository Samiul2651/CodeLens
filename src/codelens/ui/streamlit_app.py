"""CodeLens demo page. Run (with the API running):

uv run streamlit run src/codelens/ui/streamlit_app.py
"""

import streamlit as st

from codelens.api.schemas import AskResponse
from codelens.config import get_settings
from codelens.core.models import AnswerMode
from codelens.ui import client as api
from codelens.ui.view import MODE_LABELS, cost_rows, notes, source_rows

INGEST_HINT = "uv run codelens ingest path/to/repo --repo-id my-repo"


@st.cache_resource
def get_client() -> api.CodeLensClient:
    """One HTTP client for the whole page server."""
    return api.connect(get_settings().api_url)


def show_answer(response: AskResponse) -> None:
    for note in notes(response):
        st.warning(note)
    st.markdown(response.answer)

    rows = source_rows(response)
    if rows:
        st.subheader("Sources")
        for row in rows:
            st.markdown(f"- `{row.source}`" + (" **(cited)**" if row.cited else ""))

    if response.cost is not None:
        cost = response.cost
        left, middle, right = st.columns(3)
        left.metric("Cost", f"${cost.cost_usd:.4f}")
        middle.metric("Time", f"{cost.latency_seconds:.1f} s")
        right.metric("Input tokens", f"{cost.input_tokens + cost.cache_read_tokens:,}")
        with st.expander("Cost details"):
            st.table({label: [value] for label, value in cost_rows(cost)})


def main() -> None:
    st.set_page_config(page_title="CodeLens", layout="centered")
    st.title("CodeLens")
    st.caption("Ask a question about a codebase. Compare RAG with long-context.")

    client = get_client()
    try:
        repos = client.repos()
    except api.ApiError as error:
        st.error(str(error))
        return
    if not repos:
        st.info(f"No repos are ingested yet. Ingest one first:\n\n`{INGEST_HINT}`")
        return

    with st.form("ask"):
        repo_id = st.selectbox("Repo", repos)
        mode = st.radio(
            "Mode", list(AnswerMode), format_func=lambda m: MODE_LABELS[m], horizontal=True
        )
        question = st.text_area("Question", max_chars=2_000, placeholder="Where is login checked?")
        submitted = st.form_submit_button("Ask")

    if not submitted:
        return
    if not question.strip() or repo_id is None or mode is None:
        st.warning("Type a question first.")
        return
    with st.spinner("Asking Claude..."):
        try:
            response = client.ask(repo_id, question.strip(), mode)
        except api.ApiError as error:
            st.error(str(error))
            return
    show_answer(response)


if __name__ == "__main__":
    main()
