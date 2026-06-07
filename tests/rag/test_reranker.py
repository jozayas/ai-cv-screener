from cv_screener.rag.nodes.rerank import LocalReranker, ScoreReranker, reranker_node
from cv_screener.rag.schema import PlannerOutput
from cv_screener.retrieval.schema import RetrievedChunk


class FakeCrossEncoder:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.inputs: list[list[str]] = []

    def predict(
        self,
        inputs: list[list[str]],
        *,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        convert_to_numpy: bool = True,
    ) -> list[float]:
        _ = (batch_size, show_progress_bar, convert_to_numpy)
        self.inputs = inputs
        return self.scores


class FakeGraphReranker:
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


def test_local_reranker_loads_model_only_when_used() -> None:
    calls: list[str] = []
    cross_encoder = FakeCrossEncoder([1.0])

    def model_factory(model_name: str) -> FakeCrossEncoder:
        calls.append(model_name)
        return cross_encoder

    reranker = LocalReranker(model_factory=model_factory)

    assert calls == []

    _ = reranker.rerank(
        "python",
        [
            RetrievedChunk(
                source_file="ada.pdf",
                document_title="Ada Lovelace CV",
                page=1,
                section="Experience",
                text="Python.",
                score=0.7,
                rank=1,
            )
        ],
    )

    assert calls == ["cross-encoder/ms-marco-MiniLM-L6-v2"]


def test_score_reranker_sorts_by_existing_retrieval_score() -> None:
    reranker = ScoreReranker(top_k=2)
    chunks = [
        RetrievedChunk(
            source_file="low.pdf",
            document_title="Low CV",
            page=1,
            section="Experience",
            text="Low score.",
            score=0.2,
            rank=3,
        ),
        RetrievedChunk(
            source_file="high.pdf",
            document_title="High CV",
            page=1,
            section="Experience",
            text="High score.",
            score=0.9,
            rank=1,
        ),
        RetrievedChunk(
            source_file="mid.pdf",
            document_title="Mid CV",
            page=1,
            section="Experience",
            text="Mid score.",
            score=0.5,
            rank=2,
        ),
    ]

    reranked = reranker.rerank("ignored", chunks)

    assert [chunk.source_file for chunk in reranked] == ["high.pdf", "mid.pdf"]
    assert [chunk.rank for chunk in reranked] == [1, 2]


def test_local_reranker_sorts_by_cross_encoder_score() -> None:
    cross_encoder = FakeCrossEncoder([0.2, 0.9, 0.4])
    reranker = LocalReranker(
        top_k=2,
        model_factory=lambda _: cross_encoder,
    )
    chunks = [
        RetrievedChunk(
            source_file="ada.pdf",
            document_title="Ada Lovelace CV",
            page=1,
            section="Summary",
            text="Python and APIs.",
            score=0.1,
            rank=1,
        ),
        RetrievedChunk(
            source_file="grace.pdf",
            document_title="Grace Hopper CV",
            page=2,
            section="Experience",
            text="Compiler engineering and COBOL.",
            score=0.2,
            rank=2,
        ),
        RetrievedChunk(
            source_file="linus.pdf",
            document_title="Linus Torvalds CV",
            page=3,
            section="Projects",
            text="Kernel work and C systems programming.",
            score=0.3,
            rank=3,
        ),
    ]

    reranked = reranker.rerank("systems engineer", chunks)

    assert [chunk.source_file for chunk in reranked] == ["grace.pdf", "linus.pdf"]
    assert [chunk.score for chunk in reranked] == [0.9, 0.4]
    assert [chunk.rank for chunk in reranked] == [1, 2]
    assert cross_encoder.inputs == [
        ["systems engineer", "Python and APIs."],
        ["systems engineer", "Compiler engineering and COBOL."],
        ["systems engineer", "Kernel work and C systems programming."],
    ]


def test_local_reranker_limits_cross_encoder_input() -> None:
    cross_encoder = FakeCrossEncoder([0.9, 0.8])
    reranker = LocalReranker(
        top_k=2,
        max_input_chunks=2,
        model_factory=lambda _: cross_encoder,
    )
    chunks = [
        RetrievedChunk(
            source_file=f"candidate-{index}.pdf",
            document_title=f"Candidate {index} CV",
            page=1,
            section="Experience",
            text=f"Candidate {index} experience.",
            score=float(10 - index),
            rank=index,
        )
        for index in range(1, 6)
    ]

    reranked = reranker.rerank("backend engineer", chunks)

    assert [chunk.source_file for chunk in reranked] == [
        "candidate-1.pdf",
        "candidate-2.pdf",
    ]
    assert cross_encoder.inputs == [
        ["backend engineer", "Candidate 1 experience."],
        ["backend engineer", "Candidate 2 experience."],
    ]


def test_reranker_node_uses_planner_primary_query() -> None:
    response = [
        RetrievedChunk(
            source_file="ada.pdf",
            document_title="Ada Lovelace CV",
            page=2,
            section="Experience",
            text="Built Python pipelines.",
            score=0.88,
            rank=1,
        )
    ]
    reranker = FakeGraphReranker(response)
    retrieved_chunks = [
        RetrievedChunk(
            source_file="grace.pdf",
            document_title="Grace Hopper CV",
            page=1,
            section="Summary",
            text="Distributed systems and leadership.",
            score=0.7,
            rank=1,
        )
    ]

    update = reranker_node(
        {
            "planner": PlannerOutput(primary_query="python backend engineer"),
            "retrieved_chunks": retrieved_chunks,
        },
        reranker=reranker,
        top_k=3,
    )

    assert update == {"reranked_chunks": response}
    assert reranker.calls == [("python backend engineer", retrieved_chunks, 3)]
