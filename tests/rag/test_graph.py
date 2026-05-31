from typing import cast

from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.base import Runnable

from cv_screener.rag.graph import GraphDependencies, build_rag_graph
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
from cv_screener.retrieval.schema import RetrievedChunk


class FakeRetriever:
    def __init__(self, response: list[RetrievedChunk]) -> None:
        self.response = response
        self.queries: list[str] = []

    def retrieve(self, query_text: str) -> list[RetrievedChunk]:
        self.queries.append(query_text)
        return self.response


class FakeReranker:
    def __init__(self, response: list[RetrievedChunk]) -> None:
        self.response = response
        self.calls: list[tuple[str, list[RetrievedChunk], int | None]] = []

    def rerank(
        self,
        query_text: str,
        chunks: list[RetrievedChunk],
        *,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append((query_text, chunks, top_k))
        return self.response


def make_runnable[T](response: T) -> Runnable[LanguageModelInput, T]:
    return cast("Runnable[LanguageModelInput, T]", RunnableLambda(lambda _: response))


def test_graph_bypasses_planner_and_retrieval_for_small_talk() -> None:
    retriever = FakeRetriever(
        [
            RetrievedChunk(
                source_file="ignored.pdf",
                document_title="Ignored",
                page=1,
                section="Summary",
                text="ignored",
                score=0.1,
                rank=1,
            )
        ]
    )
    reranker = FakeReranker([])
    dependencies = GraphDependencies(
        router_model=make_runnable(
            RouteDecision(
                route=RouteTarget.SMALL_TALK,
                reasoning="Greeting detected.",
            )
        ),
        planner_model=make_runnable(PlannerOutput(primary_query="unused")),
        brief_answer_model=make_runnable(
            BriefAnswerOutput(text="Hi. Ask me about the CVs when you're ready.")
        ),
        retriever=retriever,
        reranker=reranker,
        answer_model=make_runnable(
            AnswerOutput(
                answer="unused",
                citations=[
                    AnswerCitation(
                        rank=1,
                        source_file="ignored.pdf",
                        page=1,
                        section="Summary",
                    )
                ],
            )
        ),
        reviewer_model=make_runnable(
            ReviewOutput(verdict=ReviewVerdict.APPROVE, reasoning="unused")
        ),
    )
    graph = build_rag_graph()

    result = graph.invoke({"user_query": "hello"}, context=dependencies)

    assert result.get("final_text") == "Hi. Ask me about the CVs when you're ready."
    assert "planner" not in result
    assert result.get("brief_answer") == BriefAnswerOutput(
        text="Hi. Ask me about the CVs when you're ready."
    )
    assert "retrieved_chunks" not in result
    assert "reranked_chunks" not in result
    assert "answer" not in result
    assert retriever.queries == []
    assert reranker.calls == []


def test_graph_routes_cv_queries_through_rerank_answer_and_review() -> None:
    retrieved_chunks = [
        RetrievedChunk(
            candidate_name="Ada Lovelace",
            source_file="ada-lovelace.pdf",
            document_title="Ada Lovelace CV",
            page=2,
            section="Experience",
            text="Built Python data pipelines and internal APIs.",
            score=0.72,
            rank=2,
        ),
        RetrievedChunk(
            candidate_name="Grace Hopper",
            source_file="grace-hopper.pdf",
            document_title="Grace Hopper CV",
            page=3,
            section="Experience",
            text="Led compiler modernization initiatives.",
            score=0.81,
            rank=1,
        ),
    ]
    retriever = FakeRetriever(retrieved_chunks)
    reranked_chunks = [retrieved_chunks[0]]
    reranker = FakeReranker(reranked_chunks)
    dependencies = GraphDependencies(
        router_model=make_runnable(
            RouteDecision(
                route=RouteTarget.CV_QUERY,
                reasoning="The user is asking about candidate skills.",
            )
        ),
        planner_model=make_runnable(
            PlannerOutput(
                primary_query="python backend engineer",
                alternate_queries=["python api engineer"],
            )
        ),
        brief_answer_model=make_runnable(BriefAnswerOutput(text="unused")),
        retriever=retriever,
        reranker=reranker,
        answer_model=make_runnable(
            AnswerOutput(
                answer="Ada Lovelace has Python backend experience.",
                citations=[
                    AnswerCitation(
                        rank=2,
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
    graph = build_rag_graph()

    result = graph.invoke(
        {"user_query": "Who has Python backend experience?"},
        context=dependencies,
    )

    route = result.get("route")
    planner = result.get("planner")
    assert route is not None
    assert route.route is RouteTarget.CV_QUERY
    assert planner is not None
    assert planner.primary_query == "python backend engineer"
    merged_chunks = [
        RetrievedChunk(
            candidate_name="Grace Hopper",
            source_file="grace-hopper.pdf",
            document_title="Grace Hopper CV",
            page=3,
            section="Experience",
            text="Led compiler modernization initiatives.",
            score=0.81,
            rank=1,
        ),
        RetrievedChunk(
            candidate_name="Ada Lovelace",
            source_file="ada-lovelace.pdf",
            document_title="Ada Lovelace CV",
            page=2,
            section="Experience",
            text="Built Python data pipelines and internal APIs.",
            score=0.72,
            rank=2,
        ),
    ]
    assert result.get("retrieved_chunks") == merged_chunks
    assert result.get("reranked_chunks") == reranked_chunks
    assert result.get("answer") == AnswerOutput(
        answer="Ada Lovelace has Python backend experience.",
        citations=[
            AnswerCitation(
                rank=2,
                candidate_name="Ada Lovelace",
                source_file="ada-lovelace.pdf",
                page=2,
                section="Experience",
            )
        ],
    )
    assert result.get("review") == ReviewOutput(
        verdict=ReviewVerdict.APPROVE,
        reasoning="The cited chunk supports the answer.",
    )
    assert result.get("final_text") == (
        "Ada Lovelace has Python backend experience.\n\n"
        "Sources:\n"
        "[2] Ada Lovelace - ada-lovelace.pdf (page 2, Experience)"
    )
    assert retriever.queries == [
        "python backend engineer",
        "python api engineer",
    ]
    assert reranker.calls == [
        ("python backend engineer", merged_chunks, None)
    ]
