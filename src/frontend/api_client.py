"""
Frontend API client for FastAPI backend.
Handles uploads, queries, health checks,
streaming, and session management.
"""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st


class APIClient:
    """FastAPI backend client."""

    def __init__(self):

        self.base_url = os.getenv(
            "RAG_API_URL",
            "http://localhost:8000",
        )

        self.timeout = 300

    # ==================================================
    # Health Check
    # ==================================================
    def health_check(
        self,
    ) -> bool:
        """Check backend status."""

        try:

            response = requests.get(
                f"{self.base_url}/health",
                timeout=10,
            )

            return (
                response.status_code
                == 200
            )

        except Exception:
            return False

    # ==================================================
    # Upload Document
    # ==================================================
    def upload_document(
        self,
        uploaded_file,
    ) -> dict[str, Any]:
        """Upload file to backend."""

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file,
                uploaded_file.type,
            )
        }

        response = requests.post(
            f"{self.base_url}/api/v1/upload",
            files=files,
            timeout=self.timeout,
        )

        if response.status_code != 200:

            try:
                detail = (
                    response.json()
                    .get(
                        "detail",
                        "Upload failed",
                    )
                )

            except Exception:
                detail = (
                    response.text
                )

            raise Exception(
                f"Upload failed: {detail}"
            )

        return response.json()

    # ==================================================
    # Query
    # ==================================================
    def query(
        self,
        question: str,
        session_id: str = "default",
        top_k: int = 5,
        stream: bool = True,
        use_cot: bool = False,
    ) -> dict[str, Any]:
        """Query RAG backend."""

        payload = {
            "question":
                question,
            "session_id":
                session_id,
            "use_cot":
                use_cot,
            "stream":
                stream,
            "top_k":
                top_k,
        }

        response = requests.post(
            f"{self.base_url}/api/v1/query",
            json=payload,
            timeout=self.timeout,
        )

        if response.status_code != 200:

            try:
                detail = (
                    response.json()
                    .get(
                        "detail",
                        "Query failed",
                    )
                )

            except Exception:
                detail = (
                    response.text
                )

            raise Exception(
                f"Query failed: {detail}"
            )

        return response.json()

    # ==================================================
    # Stream Query
    # ==================================================
    def stream_query(
        self,
        question: str,
        session_id: str = "default",
        top_k: int = 5,
        use_cot: bool = False,
    ):
        """Streaming response generator."""

        payload = {
            "question":
                question,
            "session_id":
                session_id,
            "use_cot":
                use_cot,
            "stream":
                True,
            "top_k":
                top_k,
        }

        with requests.post(
            f"{self.base_url}/api/v1/query",
            json=payload,
            stream=True,
            timeout=self.timeout,
        ) as response:

            response.raise_for_status()

            for line in (
                response
                .iter_lines()
            ):

                if line:

                    decoded = (
                        line.decode(
                            "utf-8"
                        )
                    )

                    yield decoded

    # ==================================================
    # Clear Session
    # ==================================================
    def clear_session(
        self,
        session_id: str,
    ) -> bool:
        """Delete backend session."""

        response = requests.delete(
            f"{self.base_url}/api/v1/sessions/{session_id}",
            timeout=60,
        )

        return (
            response.status_code
            == 200
        )

    # ==================================================
    # Get API Info
    # ==================================================
    def get_api_info(
        self,
    ) -> dict:

        try:

            response = requests.get(
                f"{self.base_url}/docs",
                timeout=10,
            )

            return {
                "status":
                    "available",
                "url":
                    self.base_url,
            }

        except Exception:

            return {
                "status":
                    "offline",
                "url":
                    self.base_url,
            }