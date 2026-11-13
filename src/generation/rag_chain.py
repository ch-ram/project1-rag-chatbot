"""
Production RAG chain with OpenAI generation.
"""

from __future__ import annotations

from typing import AsyncIterator, Optional
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)
from langchain_core.output_parsers import (
    StrOutputParser,
)
from langchain_openai import ChatOpenAI

from src.generation.prompts import (
    get_condense_question_prompt,
    get_rag_prompt,
    get_rag_prompt_with_cot,
)
from src.retrieval.hybrid_retriever import (
    HybridRetriever,
)
from src.utils.config import (
    get_settings,
)
from src.utils.logging import (
    get_logger,
)

logger = get_logger(__name__)


# ==================================================
# Memory
# ==================================================
class ConversationMemory:

    def __init__(
        self,
        window_size: int = 10,
    ):
        self.window_size = (
            window_size
        )

        self.messages = []

    def add_user_message(
        self,
        content: str,
    ):
        self.messages.append(
            HumanMessage(
                content=content
            )
        )
        self._trim()

    def add_ai_message(
        self,
        content: str,
    ):
        self.messages.append(
            AIMessage(
                content=content
            )
        )
        self._trim()

    def get_history(
        self,
    ):
        return (
            self.messages.copy()
        )

    def clear(
        self,
    ):
        self.messages.clear()

    def _trim(
        self,
    ):
        limit = (
            self.window_size * 2
        )

        if (
            len(self.messages)
            > limit
        ):
            self.messages = (
                self.messages[
                    -limit:
                ]
            )


# ==================================================
# RAG Chain
# ==================================================
class RAGChain:

    def __init__(
        self,
        retriever: HybridRetriever,
    ):
        settings = (
            get_settings()
        )

        self.retriever = (
            retriever
        )

        self.sessions = {}

        self._output_parser = (
            StrOutputParser()
        )

        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=(
                settings
                .openai_temperature
            ),
            max_tokens=(
                settings
                .openai_max_tokens
            ),
            streaming=True,
        )

        logger.info(
            "generation_provider_initialized",
            provider="openai",
            model=(
                settings
                .openai_model
            ),
        )

    # ==================================================
    # Session
    # ==================================================
    def get_or_create_session(
        self,
        session_id: Optional[
            str
        ] = None,
    ) -> str:

        sid = (
            session_id
            or str(uuid4())
        )

        if (
            sid
            not in self.sessions
        ):
            settings = (
                get_settings()
            )

            self.sessions[
                sid
            ] = (
                ConversationMemory(
                    window_size=(
                        settings
                        .memory_window_size
                    )
                )
            )

        return sid

    # ==================================================
    # Query
    # ==================================================
    async def aquery(
        self,
        question: str,
        session_id: Optional[
            str
        ] = None,
        use_cot: bool = False,
        top_k: int = 5,
    ) -> dict:

        sid = (
            self
            .get_or_create_session(
                session_id
            )
        )

        memory = (
            self.sessions[sid]
        )

        effective_question = (
            question
        )

        if (
            memory.get_history()
        ):
            effective_question = (
                await self
                ._condense_question(
                    question,
                    memory.get_history(),
                )
            )

        results = (
            self.retriever
            .retrieve(
                effective_question,
                top_k=top_k,
            )
        )

        context = (
            self
            ._format_context(
                results
            )
        )

        prompt = (
            get_rag_prompt_with_cot()
            if use_cot
            else get_rag_prompt()
        )

        chain = (
            prompt
            | self.llm
            | self._output_parser
        )

        answer = (
            await chain
            .ainvoke(
                {
                    "context":
                        context,
                    "question":
                        question,
                    "chat_history":
                        memory.get_history(),
                }
            )
        )

        memory.add_user_message(
            question
        )

        memory.add_ai_message(
            answer
        )

        # ==================================================
        # Sources + Citations
        # ==================================================
        sources = []

        for (
            doc,
            score,
        ) in results:

            metadata = (
                doc.metadata
            )

            source_name = (
                metadata.get(
                    "file_name",
                    "Unknown"
                )
            )

            page = metadata.get(
                "page",
                "N/A"
            )

            citation = (
                f"(Page {page}, "
                f"{source_name})"
            )

            sources.append(
                {
                    "content":
                        doc.page_content[
                            :300
                        ],
                    "metadata":
                        metadata,
                    "citation":
                        citation,
                    "score":
                        float(score),
                }
            )

        return {
            "answer":
                answer,
            "sources":
                sources,
            "session_id":
                sid,
            "retrieval_scores":
                [
                    float(score)
                    for _, score
                    in results
                ],
        }

    # ==================================================
    # Streaming
    # ==================================================
    async def astream(
        self,
        question: str,
        session_id: Optional[
            str
        ] = None,
        use_cot: bool = False,
        top_k: int = 5,
    ) -> AsyncIterator[
        str
    ]:

        response = (
            await self.aquery(
                question=question,
                session_id=session_id,
                use_cot=use_cot,
                top_k=top_k,
            )
        )

        yield response[
            "answer"
        ]

    # ==================================================
    # Condense Question
    # ==================================================
    async def _condense_question(
        self,
        question: str,
        chat_history: list,
    ) -> str:

        prompt = (
            get_condense_question_prompt()
        )

        chain = (
            prompt
            | self.llm
            | self._output_parser
        )

        history_text = (
            "\n".join(
                f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
                for m in chat_history[-6:]
            )
        )

        return (
            await chain
            .ainvoke(
                {
                    "chat_history":
                        history_text,
                    "question":
                        question,
                }
            )
        )

    # ==================================================
    # Context Formatting
    # ==================================================
    @staticmethod
    def _format_context(
        results,
    ) -> str:

        if not results:
            return (
                "No relevant documents found."
            )

        parts = []

        for i, (
            doc,
            score,
        ) in enumerate(
            results,
            start=1,
        ):

            metadata = (
                doc.metadata
            )

            source = metadata.get(
                "file_name",
                "Unknown",
            )

            page = metadata.get(
                "page",
                "N/A",
            )

            text = (
                doc.page_content
                .replace("\n", " ")
                .strip()
            )

            text = " ".join(
                text.split()
            )

            if len(text) < 80:
                continue

            parts.append(
                f"""
DOCUMENT {i}
Source: {source}
Page: {page}

CONTENT:
{text}
"""
            )

        return "\n\n".join(parts)

    # ==================================================
    # Session Cleanup
    # ==================================================
    def clear_session(
        self,
        session_id: str,
    ) -> bool:

        if (
            session_id
            in self.sessions
        ):
            self.sessions[
                session_id
            ].clear()

            del self.sessions[
                session_id
            ]

            return True

        return False