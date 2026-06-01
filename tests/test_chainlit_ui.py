from cv_screener.chainlit.ui import format_chat_response
from cv_screener.rag.schema import AnswerCitation, AnswerOutput
from cv_screener.rag.service import RAGQueryResult


def test_format_chat_response_prefers_structured_answer_with_sources() -> None:
    result = RAGQueryResult(
        final_text="stale fallback",
        state={
            "answer": AnswerOutput(
                answer="Ada Lovelace has Python backend experience.",
                citations=[
                    AnswerCitation(
                        rank=1,
                        candidate_name="Ada Lovelace",
                        source_file="ada.pdf",
                        page=2,
                        section="Experience",
                    ),
                    AnswerCitation(
                        rank=1,
                        candidate_name="Ada Lovelace",
                        source_file="ada.pdf",
                        page=2,
                        section="Experience",
                    ),
                ],
            )
        },
    )

    assert format_chat_response(result) == (
        "Ada Lovelace has Python backend experience.\n\n"
        "Sources:\n"
        "[1] Ada Lovelace - ada.pdf (page 2, Experience)"
    )


def test_format_chat_response_falls_back_to_final_text() -> None:
    result = RAGQueryResult(
        final_text="Hello there",
        state={"final_text": "Hello there"},
    )

    assert format_chat_response(result) == "Hello there"
