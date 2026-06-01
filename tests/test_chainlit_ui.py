from cv_screener.chainlit.ui import (
    candidate_reference_clarification,
    candidate_references_from_answer,
    candidate_references_from_result,
    format_chat_response,
    rewrite_query_with_candidate_reference,
)
from cv_screener.rag.schema import AnswerCitation, AnswerOutput, TargetedLookupOutput
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


def test_rewrite_query_with_candidate_reference_replaces_ordinals() -> None:
    rewritten = rewrite_query_with_candidate_reference(
        "Give me the second candidate's cv",
        ["Ada Lovelace", "Grace Hopper", "Hedy Lamarr"],
    )

    assert rewritten == "Give me Grace Hopper cv"


def test_rewrite_query_with_candidate_reference_replaces_pronouns() -> None:
    rewritten = rewrite_query_with_candidate_reference(
        "Give me his cv",
        ["MARCO ANTONIO RAMÍREZ"],
    )

    assert rewritten == "Give me the CV of MARCO ANTONIO RAMÍREZ"


def test_candidate_references_from_answer_preserves_order() -> None:
    answer = AnswerOutput(
        answer="Candidates found.",
        citations=[
            AnswerCitation(
                rank=2,
                candidate_name="Grace Hopper",
                source_file="grace.pdf",
                page=1,
                section="Experience",
            ),
            AnswerCitation(
                rank=1,
                candidate_name="Ada Lovelace",
                source_file="ada.pdf",
                page=1,
                section="Experience",
            ),
            AnswerCitation(
                rank=3,
                candidate_name="Grace Hopper",
                source_file="grace.pdf",
                page=2,
                section="Experience",
            ),
        ],
    )

    assert candidate_references_from_answer(answer) == [
        "Grace Hopper",
        "Ada Lovelace",
    ]


def test_candidate_reference_clarification_for_ambiguous_followup() -> None:
    clarification = candidate_reference_clarification("Give me his cv", [])

    assert clarification == "Which candidate do you mean?"


def test_candidate_references_from_targeted_lookup_result() -> None:
    result = RAGQueryResult(
        final_text="Candidates who match the query: Ada Lovelace [1], Grace Hopper [2]",
        state={
            "targeted_lookup": TargetedLookupOutput(
                candidate_names=["Ada Lovelace", "Grace Hopper"],
                response_mode="list_candidates",
            )
        },
    )

    assert candidate_references_from_result(result) == [
        "Ada Lovelace",
        "Grace Hopper",
    ]
