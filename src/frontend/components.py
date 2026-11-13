"""
Reusable frontend UI components.
"""

from __future__ import annotations

import streamlit as st

from constants import (
    APP_NAME,
    APP_SUBTITLE,
    SUGGESTED_PROMPTS,
    SOURCE_PREVIEW_LENGTH,
)


def render_header():

    st.markdown(
        f"""
        <div class="app-header">

            <div class="app-title">
                {APP_NAME}
            </div>

            <div class="app-subtitle">
                {APP_SUBTITLE}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status(
    is_online: bool,
):

    css_class = (
        "status-connected"
        if is_online
        else "status-offline"
    )

    label = (
        "Connected"
        if is_online
        else "Offline"
    )

    st.markdown(
        f"""
        <div class="{css_class}">
            {label}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_settings():

    st.sidebar.markdown(
        "### Settings"
    )

    top_k = st.sidebar.slider(
        "Retrieval Depth",
        1,
        20,
        5,
    )

    use_cot = st.sidebar.checkbox(
        "Reasoning Mode",
        value=False,
    )

    stream = st.sidebar.checkbox(
        "Streaming",
        value=True,
    )

    return {
        "top_k": top_k,
        "use_cot": use_cot,
        "stream": stream,
    }


def render_uploaded_docs(
    uploaded_docs: list,
):

    st.sidebar.markdown(
        "### Knowledge Base"
    )

    if not uploaded_docs:
        st.sidebar.caption(
            "No documents uploaded"
        )
        return

    for doc in uploaded_docs:

        file_name = doc.get(
            "file",
            "document",
        )

        chunks = doc.get(
            "chunks_indexed",
            0,
        )

        st.sidebar.markdown(
            f"""
            <div class="source-card">

                <div class="source-title">
                    {file_name}
                </div>

                <div class="source-meta">
                    {chunks} chunks indexed
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


def render_empty_state():

    prompts_html = "".join(
        [
            f"<li>{prompt}</li>"
            for prompt in SUGGESTED_PROMPTS
        ]
    )

    st.markdown(
        f"""
        <div class="empty-state">

            <div class="empty-title">
                Upload documents to begin
            </div>

            <div class="empty-text">

                Ask grounded questions about
                PDFs, DOCX, TXT, Markdown,
                and HTML files.

                <br><br>

                Suggested prompts:

                <ul style="
                    display:inline-block;
                    text-align:left;
                    margin-top:10px;
                ">
                    {prompts_html}
                </ul>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat_message(
    role: str,
    content: str,
):

    css_class = (
        "chat-user"
        if role == "user"
        else "chat-assistant"
    )

    role_label = (
        "You"
        if role == "user"
        else "Assistant"
    )

    st.markdown(
        f"""
        <div class="{css_class}">

            <div class="message-role">
                {role_label}
            </div>

            <div class="message-content">
                {content}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(
    sources: list,
):

    if not sources:
        return

    st.markdown("### Sources")

    for source in sources:

        metadata = source.get(
            "metadata",
            {},
        )

        file_name = metadata.get(
            "file_name",
            "document",
        )

        page = metadata.get(
            "page",
            "N/A",
        )

        score = round(
            source.get(
                "score",
                0.0,
            ),
            4,
        )

        content = source.get(
            "content",
            "",
        )

        preview = content[
            :SOURCE_PREVIEW_LENGTH
        ]

        with st.expander(file_name):

            st.write(
                f"Page: {page}"
            )

            st.write(
                f"Relevance: {score}"
            )

            st.write(preview)