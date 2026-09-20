"""
app.py
------
The Streamlit user interface. This is the file you run locally with:

    streamlit run app.py

and the file Streamlit Cloud runs when you deploy.
"""

import streamlit as st
from agent import run_research

st.set_page_config(page_title="AI Research Agent", page_icon="🔎", layout="centered")


def get_secret(key: str):
    """Safely read a value from Streamlit secrets without crashing if
    no secret with that name has been configured."""
    try:
        return st.secrets[key]
    except Exception:
        return None


st.title("🔎 AI Research Agent")
st.caption("Single-agent researcher built with CrewAI · Groq (openai/gpt-oss-120b) · DuckDuckGo Search")

# ---------------------------------------------------------------------------
# API key: read ONLY from Streamlit secrets.
# On Streamlit Community Cloud this is set under: App settings -> Secrets, as
#   GROQ_API_KEY = "your_real_key"
# There is no fallback text box — the app expects the secret to be configured.
# ---------------------------------------------------------------------------
groq_api_key = get_secret("GROQ_API_KEY")

with st.sidebar:
    st.header("⚙️ About")
    if groq_api_key:
        st.success("Groq API key loaded from secrets ✅")
    else:
        st.error("GROQ_API_KEY secret not found.")
        st.caption(
            "Add it in Streamlit Cloud under **App settings → Secrets** as:\n\n"
            "```toml\nGROQ_API_KEY = \"your_real_key\"\n```"
        )
    st.markdown("---")
    st.markdown("**Model:** `openai/gpt-oss-120b` (via Groq)")
    st.markdown("**Search:** DuckDuckGo (free, no key needed)")
    st.markdown("---")
    st.caption(
        "This app sends your topic to a Groq-hosted LLM and does live "
        "DuckDuckGo web searches. Don't enter sensitive information."
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
topic = st.text_input(
    "What topic should the agent research?",
    placeholder="e.g. The impact of AI agents on software jobs in 2026",
)

run_clicked = st.button("Run Research", type="primary", use_container_width=True)

if run_clicked:
    if not groq_api_key:
        st.error(
            "GROQ_API_KEY is not set. Add it in Streamlit Cloud under "
            "App settings → Secrets, then reload the app."
        )
    elif not topic.strip():
        st.error("Please enter a research topic.")
    else:
        with st.spinner("Researching... this usually takes 30-90 seconds ⏳"):
            try:
                report = run_research(topic.strip(), groq_api_key)
                st.session_state["last_report"] = report
                st.session_state["last_topic"] = topic.strip()
            except Exception as e:
                st.error(f"Something went wrong: {e}")

# ---------------------------------------------------------------------------
# Show the last report (persists across reruns thanks to session_state)
# ---------------------------------------------------------------------------
if "last_report" in st.session_state:
    st.markdown("---")
    st.subheader(f"📄 Report: {st.session_state['last_topic']}")
    st.markdown(st.session_state["last_report"])
    st.download_button(
        "⬇️ Download report as Markdown",
        data=st.session_state["last_report"],
        file_name="research_report.md",
        mime="text/markdown",
    )
