import asyncio

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
from tests.rag.test_graph import (
    FakeLookupService,
    FakeReranker,
    FakeRetriever,
    make_runnable,
)


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
            lookup_service=FakeLookupService(),
            retriever=FakeRetriever([chunk]),
            reranker=FakeReranker([chunk]),
            answer_model=make_runnable(
                AnswerOutput(
                    answer="Ada Lovelace has Python backend experience.",
                    citations=[
                        AnswerCitation(
                            rank=1,
                            candidate_name="Ada Lovelace",
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
        "[1] Ada Lovelace - ada-lovelace.pdf (page 2, Experience)"
    )
    assert result.state["final_text"] == result.final_text


def test_rag_query_service_async_stream_yields_per_node_events() -> None:
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
            lookup_service=FakeLookupService(),
            retriever=FakeRetriever([chunk]),
            reranker=FakeReranker([chunk]),
            answer_model=make_runnable(
                AnswerOutput(
                    answer="Ada Lovelace has Python backend experience.",
                    citations=[
                        AnswerCitation(
                            rank=1,
                            candidate_name="Ada Lovelace",
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

    async def _collect() -> list[tuple[str, dict[str, object]]]:
        result: list[tuple[str, dict[str, object]]] = []
        async for node_name, state_update in service.async_stream(
            "Who has Python experience?"
        ):
            result.append((node_name, state_update))
        return result

    events: list[tuple[str, dict[str, object]]] = asyncio.run(_collect())

    node_names = [name for name, _ in events]
    assert node_names == [
        "router",
        "planner",
        "retrieve",
        "rerank",
        "answer",
        "review",
        "finalize",
    ]

    _name, final_update = events[-1]
    final_text = final_update.get("final_text", "")
    assert isinstance(final_text, str)
    assert "Ada Lovelace" in final_text
