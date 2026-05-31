from cv_screener.rag.graph import GraphDependencies
from cv_screener.rag.schema import (
    AnswerCitation,
    AnswerOutput,
    BriefAnswerOutput,
    PlannerOutput,
    ReviewOutput,
    ReviewVerdict,
    RouteDecision,
    RouteTarget,
)
from cv_screener.rag.service import RAGQueryService
from cv_screener.retrieval.schema import RetrievedChunk
from tests.rag.test_graph import FakeReranker, FakeRetriever, make_runnable


def test_rag_query_service_returns_final_graph_output() -> None:
    chunk = RetrievedChunk(
        candidate_name="Ada Lovelace",
        source_file="ada-lovelace.pdf",
        document_title="Ada Lovelace CV",
        page=2,
        section="Experience",
        text="Built Python data pipelines and internal APIs.",
        score=0.72,
        rank=1,
    )
    service = RAGQueryService(
        dependencies=GraphDependencies(
            router_model=make_runnable(
                RouteDecision(
                    route=RouteTarget.CV_QUERY,
                    reasoning="The user is asking about candidate skills.",
                )
            ),
            planner_model=make_runnable(
                PlannerOutput(primary_query="python backend engineer")
            ),
            brief_answer_model=make_runnable(BriefAnswerOutput(text="unused")),
            retriever=FakeRetriever([chunk]),
            reranker=FakeReranker([chunk]),
            answer_model=make_runnable(
                AnswerOutput(
                    answer="Ada Lovelace has Python backend experience.",
                    citations=[
                        AnswerCitation(
                            source_file="ada-lovelace.pdf",
                            page=2,
                            section="Experience",
                        )
                    ],
                )
            ),
            reviewer_model=make_runnable(
                ReviewOutput(
                    verdict=ReviewVerdict.APPROVE,
                    reasoning="The cited chunk supports the answer.",
                )
            ),
        )
    )

    result = service.run("Who has Python backend experience?")

    assert result.final_text == (
        "Ada Lovelace has Python backend experience.\n\n"
        "Sources:\n"
        "- ada-lovelace.pdf (page 2, Experience)"
    )
    assert result.state["final_text"] == result.final_text
