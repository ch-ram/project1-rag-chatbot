"""
Production Streamlit frontend for RAG chatbot.
"""

from __future__ import annotations

import streamlit as st

from api_client import APIClient
from styles import load_css
from components import (
    render_header,
    render_sidebar_settings,
    render_uploaded_docs,
    render_empty_state,
    render_chat_message,
    render_sources,
    render_status,
)


# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="PSR RAG Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# IMPORTANT
st.markdown(
    load_css(),
    unsafe_allow_html=True,
)


# ==========================================================
# SESSION STATE
# ==========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []

if "session_id" not in st.session_state:
    st.session_state.session_id = "default"

if "api_client" not in st.session_state:
    st.session_state.api_client = APIClient()

client = st.session_state.api_client


# ==========================================================
# SIDEBAR
# ==========================================================
with st.sidebar:

    st.title("Workspace")

    try:
        is_online = client.health_check()
    except Exception:
        is_online = False

    render_status(is_online)

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload Document",
        type=[
            "pdf",
            "docx",
            "txt",
            "md",
            "html",
        ],
    )

    if uploaded_file:

        with st.spinner(
            "Processing document..."
        ):

            try:
                result = (
                    client.upload_document(
                        uploaded_file
                    )
                )

                st.session_state.uploaded_docs.append(
                    result
                )

                st.success(
                    f"Uploaded: {result.get('file')}"
                )

            except Exception as e:
                st.error(str(e))

    st.divider()

    settings = (
        render_sidebar_settings()
    )

    st.divider()

    render_uploaded_docs(
        st.session_state.uploaded_docs
    )

    st.divider()

    if st.button(
        "Clear Conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()


# ==========================================================
# MAIN UI
# ==========================================================
render_header()

if (
    not st.session_state.messages
    and not st.session_state.uploaded_docs
):
    render_empty_state()


# ==========================================================
# CHAT HISTORY
# ==========================================================
for message in st.session_state.messages:

    render_chat_message(
        message["role"],
        message["content"],
    )

    if (
        message["role"]
        == "assistant"
        and "sources" in message
    ):
        render_sources(
            message["sources"]
        )


# ==========================================================
# USER INPUT
# ==========================================================
prompt = st.chat_input(
    "Ask a question about your documents"
)

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    st.rerun()


# ==========================================================
# RESPONSE
# ==========================================================
if (
    st.session_state.messages
    and st.session_state.messages[-1]["role"]
    == "user"
):

    user_prompt = (
        st.session_state.messages[-1]
        ["content"]
    )

    with st.spinner(
        "Generating response..."
    ):

        try:

            response = client.query(
                question=user_prompt,
                session_id=(
                    st.session_state.session_id
                ),
                top_k=settings["top_k"],
                stream=settings["stream"],
                use_cot=settings["use_cot"],
            )

            answer = response.get(
                "answer",
                "No response generated.",
            )

            sources = response.get(
                "sources",
                [],
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )

            st.rerun()

        except Exception as e:
            st.error(
                f"Query failed: {str(e)}"
            )