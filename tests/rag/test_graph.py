from typing import cast

from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.base import Runnable

from cv_screener.persistence.lookup import CandidateMatch, DocumentMatch
from cv_screener.rag.graph import GraphDependencies, build_rag_graph
from cv_screener.rag.schema import (
    AnswerCitation,
    AnswerOutput,
    BriefAnswerOutput,
    FullCVOutput,
    PlannerOutput,
    ReviewOutput,
    ReviewVerdict,
    RouteDecision,
    RouteTarget,
    TargetedLookupOutput,
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


class FakeLookupCandidate(CandidateMatch):
    pass


class FakeLookupDocument(DocumentMatch):
    pass


class FakeLookupService:
    def __init__(
        self,
        *,
        candidate_by_name: CandidateMatch | None = None,
        candidates_by_skill: list[CandidateMatch] | None = None,
        candidates_by_education: list[CandidateMatch] | None = None,
        document_by_name: DocumentMatch | None = None,
        hydrated_chunks: list[RetrievedChunk] | None = None,
    ) -> None:
        self._candidate_by_name = candidate_by_name
        self._candidates_by_skill = candidates_by_skill or []
        self._candidates_by_education = candidates_by_education or []
        self._document_by_name = document_by_name
        self._hydrated_chunks = hydrated_chunks or []

    def find_candidate_by_name(self, name: str) -> CandidateMatch | None:
        _ = name
        return self._candidate_by_name

    def find_candidates_by_skill(self, skill_name: str) -> list[CandidateMatch]:
        _ = skill_name
        return self._candidates_by_skill

    def find_candidates_by_education(self, institution: str) -> list[CandidateMatch]:
        _ = institution
        return self._candidates_by_education

    def find_document_by_candidate_name(self, name: str) -> DocumentMatch | None:
        _ = name
        return self._document_by_name

    def fetch_chunks(
        self,
        *,
        candidate_ids: list[str],
        sections: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        _ = (candidate_ids, sections)
        return self._hydrated_chunks


def make_runnable[T](response: T) -> Runnable[LanguageModelInput, T]:
    return cast("Runnable[LanguageModelInput, T]", RunnableLambda(lambda _: response))


def make_sequence_runnable[T](responses: list[T]) -> Runnable[LanguageModelInput, T]:
    remaining = list(responses)

    def invoke(_: object) -> T:
        if not remaining:
            msg = "No responses remaining for sequence runnable"
            raise AssertionError(msg)
        return remaining.pop(0)

    return cast("Runnable[LanguageModelInput, T]", RunnableLambda(invoke))


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
        lookup_service=FakeLookupService(),
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
        lookup_service=FakeLookupService(),
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
    assert reranker.calls == [("python backend engineer", merged_chunks, None)]


def test_graph_retries_review_once_after_revise_verdict() -> None:
    retrieved_chunk = RetrievedChunk(
        candidate_name="Ada Lovelace",
        source_file="ada-lovelace.pdf",
        document_title="Ada Lovelace CV",
        page=2,
        section="Experience",
        text="Built Python APIs for internal platforms.",
        score=0.92,
        rank=1,
    )
    retriever = FakeRetriever([retrieved_chunk])
    reranker = FakeReranker([retrieved_chunk])
    dependencies = GraphDependencies(
        router_model=make_runnable(
            RouteDecision(
                route=RouteTarget.CV_QUERY,
                reasoning="The user is asking about candidate skills.",
            )
        ),
        planner_model=make_runnable(PlannerOutput(primary_query="python api engineer")),
        brief_answer_model=make_runnable(BriefAnswerOutput(text="unused")),
        lookup_service=FakeLookupService(),
        retriever=retriever,
        reranker=reranker,
        answer_model=make_runnable(
            AnswerOutput(
                answer="Ada Lovelace led platform strategy and built Python APIs.",
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
        reviewer_model=make_sequence_runnable(
            [
                ReviewOutput(
                    verdict=ReviewVerdict.REVISE,
                    reasoning="The chunk supports Python APIs but not platform strategy.",
                    revised_answer="Ada Lovelace built Python APIs.",
                ),
                ReviewOutput(
                    verdict=ReviewVerdict.APPROVE,
                    reasoning="The revised answer is grounded in the cited evidence.",
                ),
            ]
        ),
    )
    graph = build_rag_graph()

    result = graph.invoke(
        {"user_query": "Who has Python API experience?"},
        context=dependencies,
    )

    assert result.get("answer") == AnswerOutput(
        answer="Ada Lovelace built Python APIs.",
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
    assert result.get("review") == ReviewOutput(
        verdict=ReviewVerdict.APPROVE,
        reasoning="The revised answer is grounded in the cited evidence.",
    )
    assert result.get("review_attempts") == 1
    assert result.get("nodes_executed") == [
        "router",
        "planner",
        "retrieve",
        "rerank",
        "answer",
        "review",
        "review",
        "finalize",
    ]


def test_graph_routes_full_cv_queries_to_return_cv() -> None:
    dependencies = GraphDependencies(
        router_model=make_runnable(
            RouteDecision(route=RouteTarget.FULL_CV, reasoning="Direct CV request.")
        ),
        planner_model=make_runnable(PlannerOutput(primary_query="unused")),
        brief_answer_model=make_runnable(BriefAnswerOutput(text="unused")),
        lookup_service=FakeLookupService(
            document_by_name=FakeLookupDocument(
                candidate_id="cand-1",
                full_name="José Luis Zayas Alcaide",
                source_file="jose-luis-zayas-alcaide.pdf",
                document_title="José Luis Zayas Alcaide CV",
                pdf_path="/tmp/jose-luis-zayas-alcaide.pdf",  # noqa: S108
                parsed_markdown="## **SUMMARY**\n\nAI engineer.",
            )
        ),
        retriever=FakeRetriever([]),
        reranker=FakeReranker([]),
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

    result = graph.invoke(
        {"user_query": "Give me the CV of Jose Luis"}, context=dependencies
    )

    assert result.get("full_cv") == FullCVOutput(
        candidate_name="José Luis Zayas Alcaide",
        source_file="jose-luis-zayas-alcaide.pdf",
        document_title="José Luis Zayas Alcaide CV",
        pdf_path="/tmp/jose-luis-zayas-alcaide.pdf",  # noqa: S108
        parsed_markdown="## **SUMMARY**\n\nAI engineer.",
    )
    assert (
        result.get("final_text")
        == "José Luis Zayas Alcaide - jose-luis-zayas-alcaide.pdf\n\n"
        "## **SUMMARY**\n\nAI engineer.\n\n"
        "PDF: /tmp/jose-luis-zayas-alcaide.pdf"
    )
    assert result.get("nodes_executed") == ["router", "return_cv", "finalize"]


def test_graph_routes_targeted_profile_queries_through_hydrate() -> None:
    hydrated = [
        RetrievedChunk(
            candidate_name="Jane Doe",
            source_file="jane-doe.pdf",
            document_title="Jane Doe CV",
            page=1,
            section="SUMMARY",
            text="Backend engineer.",
            score=1.0,
            rank=1,
        )
    ]
    dependencies = GraphDependencies(
        router_model=make_runnable(
            RouteDecision(
                route=RouteTarget.TARGETED_LOOKUP,
                reasoning="Deterministic profile lookup.",
            )
        ),
        planner_model=make_runnable(PlannerOutput(primary_query="unused")),
        brief_answer_model=make_runnable(BriefAnswerOutput(text="unused")),
        lookup_service=FakeLookupService(
            candidate_by_name=FakeLookupCandidate("cand-1", "Jane Doe"),
            hydrated_chunks=hydrated,
        ),
        retriever=FakeRetriever([]),
        reranker=FakeReranker(hydrated),
        answer_model=make_runnable(
            AnswerOutput(
                answer="Jane Doe is a backend engineer.",
                citations=[
                    AnswerCitation(
                        rank=1,
                        candidate_name="Jane Doe",
                        source_file="jane-doe.pdf",
                        page=1,
                        section="SUMMARY",
                    )
                ],
            )
        ),
        reviewer_model=make_runnable(
            ReviewOutput(
                verdict=ReviewVerdict.APPROVE,
                reasoning="The summary chunk supports the answer.",
            )
        ),
    )
    graph = build_rag_graph()

    result = graph.invoke({"user_query": "Summarize Jane Doe"}, context=dependencies)

    assert result.get("targeted_lookup") == TargetedLookupOutput(
        candidate_ids=["cand-1"],
        candidate_names=["Jane Doe"],
        sections=["PROFILE", "SUMMARY", "EXPERIENCE", "EDUCATION", "SKILLS"],
        response_mode="profile",
        fallback_to_semantic=False,
    )
    assert result.get("retrieved_chunks") == hydrated
    assert result.get("nodes_executed") == [
        "router",
        "targeted_lookup",
        "hydrate",
        "answer",
        "review",
        "finalize",
    ]


def test_graph_routes_targeted_skill_queries_directly_from_sql() -> None:
    hydrated = [
        RetrievedChunk(
            candidate_name="Marie Curie",
            source_file="marie-curie.pdf",
            document_title="Marie Curie CV",
            page=1,
            section="SKILLS",
            text="Python, data analysis, research.",
            score=1.0,
            rank=1,
        ),
        RetrievedChunk(
            candidate_name="Ada Lovelace",
            source_file="ada-lovelace.pdf",
            document_title="Ada Lovelace CV",
            page=1,
            section="SKILLS",
            text="Python, algorithms, analytics.",
            score=1.0,
            rank=2,
        ),
    ]
    dependencies = GraphDependencies(
        router_model=make_runnable(
            RouteDecision(
                route=RouteTarget.TARGETED_LOOKUP,
                reasoning="Deterministic skill lookup.",
            )
        ),
        planner_model=make_runnable(PlannerOutput(primary_query="unused")),
        brief_answer_model=make_runnable(BriefAnswerOutput(text="unused")),
        lookup_service=FakeLookupService(
            candidates_by_skill=[
                FakeLookupCandidate("cand-1", "Marie Curie"),
                FakeLookupCandidate("cand-2", "Ada Lovelace"),
            ],
            hydrated_chunks=hydrated,
        ),
        retriever=FakeRetriever([]),
        reranker=FakeReranker([]),
        answer_model=make_runnable(
            AnswerOutput(
                answer="unused",
                citations=[
                    AnswerCitation(
                        rank=1,
                        source_file="ignored.pdf",
                        page=1,
                        section="Skills",
                    )
                ],
            )
        ),
        reviewer_model=make_runnable(
            ReviewOutput(verdict=ReviewVerdict.APPROVE, reasoning="unused")
        ),
    )
    graph = build_rag_graph()

    result = graph.invoke({"user_query": "Who knows Python?"}, context=dependencies)

    assert result.get("targeted_lookup") == TargetedLookupOutput(
        candidate_ids=["cand-1", "cand-2"],
        candidate_names=["Marie Curie", "Ada Lovelace"],
        sections=["SKILLS", "EXPERIENCE", "PROJECTS"],
        response_mode="list_candidates",
        fallback_to_semantic=False,
    )
    assert result.get("final_text") == (
        "Candidates who match the query: Marie Curie [1], Ada Lovelace [2]\n\n"
        "Sources:\n"
        "[1] Marie Curie - marie-curie.pdf (page 1, SKILLS)\n"
        "[2] Ada Lovelace - ada-lovelace.pdf (page 1, SKILLS)"
    )
    assert result.get("nodes_executed") == [
        "router",
        "targeted_lookup",
        "hydrate",
        "finalize",
    ]
