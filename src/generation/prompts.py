"""
Production prompt templates with few-shot examples
and chain-of-thought.
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

# ==================================================
# System Prompt
# ==================================================

RAG_SYSTEM_PROMPT = """
You are an expert research and document analysis assistant.

Your task is to answer questions using ONLY
the retrieved document context.

Instructions:
1. Read all retrieved context carefully.
2. Use retrieved content as the PRIMARY source.
3. You may combine information across multiple retrieved sections.
4. If the answer is partially available,
   provide the best grounded answer.
5. Summarize related evidence naturally.
6. Include page citations naturally when evidence exists.
7. When answering from documents,
   cite evidence like:
   (Page X, filename.pdf)
8. Be concise but informative.
9. Only say:
   "I don't have enough information in the provided documents."
   when there is truly no relevant evidence.

Rules:
- Do NOT hallucinate facts.
- Do NOT contradict retrieved evidence.
- Prioritize retrieved context over prior knowledge.
- Infer reasonable structure if multiple chunks together imply an answer.
- Research papers often distribute methodology across sections
  like preprocessing, feature engineering, experiments, and evaluation.

Retrieved Context:
{context}
"""


CONDENSED_QUESTION_PROMPT = """
Given the conversation history
and a follow-up question,
rewrite the follow-up question
as a standalone question.

Chat History:
{chat_history}

Follow-up Question:
{question}

Standalone Question:
"""


# ==================================================
# Prompt Templates
# ==================================================
def get_rag_prompt() -> ChatPromptTemplate:
    return (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    RAG_SYSTEM_PROMPT,
                ),

                MessagesPlaceholder(
                    variable_name="chat_history",
                    optional=True,
                ),

                (
                    "human",
                    "{question}",
                ),
            ]
        )
    )


def get_rag_prompt_with_cot() -> ChatPromptTemplate:
    cot_system = (
        RAG_SYSTEM_PROMPT
        + """

Before answering:

1. Identify what the user is asking.
2. Find the most relevant retrieved sections.
3. Combine related evidence if needed.
4. Form a grounded answer.
5. Include citations when possible.
6. Mention uncertainty only if evidence is missing.

Think internally step by step,
but return ONLY the final answer.
"""
    )

    return (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    cot_system,
                ),

                MessagesPlaceholder(
                    variable_name="chat_history",
                    optional=True,
                ),

                (
                    "human",
                    "{question}",
                ),
            ]
        )
    )


def get_condense_question_prompt(
) -> ChatPromptTemplate:

    return (
        ChatPromptTemplate.from_template(
            CONDENSED_QUESTION_PROMPT
        )
    )