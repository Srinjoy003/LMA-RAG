"""Search index over the question bank (spec section 15).

Every question is embedded once with a sentence-transformers model; vectors are
saved to `index/embeddings.npy` and the row order + bookkeeping to `index/meta.json`.

The embedding function is injectable: any callable `list[str] -> array (n, d)`.
Tests pass a small deterministic fake so no model is downloaded.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .bank import ROOT, Bank, load_bank, question_text

MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_INDEX_DIR = ROOT / "index"

EmbedFn = Callable[[list], np.ndarray]


def default_embedder(model_name: str = MODEL_NAME) -> EmbedFn:
    """Return an embedding function backed by sentence-transformers (CPU)."""
    from sentence_transformers import SentenceTransformer  # imported lazily: it is slow

    model = SentenceTransformer(model_name, device="cpu")

    def embed(texts: list) -> np.ndarray:
        return np.asarray(model.encode(list(texts), normalize_embeddings=True,
                                       show_progress_bar=False))

    return embed


def _normalise(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def find_near_duplicates(ids: list, embeddings: np.ndarray, threshold: float = 0.9) -> list:
    """Pairs (id_a, id_b, cosine) with cosine above `threshold`, most similar first."""
    if len(ids) < 2:
        return []
    unit = _normalise(embeddings)
    sims = unit @ unit.T
    rows, cols = np.triu_indices(len(ids), k=1)
    pairs = [(ids[i], ids[j], float(sims[i, j]))
             for i, j in zip(rows, cols) if sims[i, j] > threshold]
    return sorted(pairs, key=lambda p: -p[2])


class QuestionIndex:
    def __init__(self, questions: list, embeddings: np.ndarray, embed_fn: EmbedFn | None = None,
                 model_name: str = MODEL_NAME, bank_hash: str = ""):
        if len(questions) != len(embeddings):
            raise ValueError("questions and embeddings must have the same length")
        self.questions = list(questions)
        self.ids = [q["id"] for q in self.questions]
        self.embeddings = _normalise(embeddings) if len(embeddings) else np.zeros((0, 1), np.float32)
        self.model_name = model_name
        self.bank_hash = bank_hash
        self._embed_fn = embed_fn

    # ---- building and saving ------------------------------------------
    @classmethod
    def build(cls, bank: Bank, embed_fn: EmbedFn | None = None,
              model_name: str = MODEL_NAME) -> "QuestionIndex":
        embed_fn = embed_fn or default_embedder(model_name)
        questions = sorted(bank.questions, key=lambda q: q["id"])
        texts = [question_text(q, bank.topic_name) for q in questions]
        embeddings = embed_fn(texts) if texts else np.zeros((0, 1), np.float32)
        return cls(questions, embeddings, embed_fn, model_name, bank.content_hash())

    def save(self, index_dir: str | Path = DEFAULT_INDEX_DIR) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "embeddings.npy", self.embeddings)
        meta = {"question_ids": self.ids, "model": self.model_name, "bank_hash": self.bank_hash}
        (index_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_dir: str | Path = DEFAULT_INDEX_DIR, bank: Bank | None = None,
             embed_fn: EmbedFn | None = None) -> "QuestionIndex":
        """Load a saved index. Questions are read from the bank (default: the project bank).

        `embed_fn` is only needed for text queries; by default the sentence-transformers
        model is loaded on the first `search(query=...)`.
        """
        index_dir = Path(index_dir)
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        embeddings = np.load(index_dir / "embeddings.npy")
        bank = bank or load_bank()
        missing = [i for i in meta["question_ids"] if bank.get_question(i) is None]
        if missing:
            raise ValueError(f"index is stale: {len(missing)} question ids are not in the bank "
                             f"(e.g. {missing[0]}). Re-run scripts/build_index.py")
        questions = [bank.get_question(i) for i in meta["question_ids"]]
        index = cls(questions, embeddings, embed_fn, meta["model"], meta["bank_hash"])
        if index.is_stale(bank):
            warnings.warn("the question bank changed since the index was built; "
                          "re-run scripts/build_index.py", stacklevel=2)
        return index

    def is_stale(self, bank: Bank) -> bool:
        return self.bank_hash != bank.content_hash()

    # ---- searching -------------------------------------------------------
    def _embedder(self) -> EmbedFn:
        if self._embed_fn is None:
            self._embed_fn = default_embedder(self.model_name)
        return self._embed_fn

    def search(self, query: str | None = None, skill: str | None = None,
               levels: list | None = None, prefer_topics: list | None = None,
               exclude_ids: Iterable[str] = frozenset(), k: int = 5) -> list:
        """Return up to k (question, score) pairs.

        Filter by skill, levels and exclude_ids first. Then rank by how many of
        `prefer_topics` the question covers (more is better), and among equals by
        cosine similarity to `query` (if given), then by id.

        The score is the cosine similarity when a query is given, otherwise the
        topic overlap count.
        """
        exclude = set(exclude_ids)
        prefer = set(prefer_topics or [])
        level_set = set(levels) if levels is not None else None

        candidates = [
            i for i, q in enumerate(self.questions)
            if (skill is None or q["skill"] == skill)
            and (level_set is None or q["level"] in level_set)
            and q["id"] not in exclude
        ]
        if not candidates:
            return []

        if query:
            query_vec = _normalise(self._embedder()([query]))[0]
            cosine = {i: float(self.embeddings[i] @ query_vec) for i in candidates}
        else:
            cosine = {i: 0.0 for i in candidates}
        overlap = {i: len(prefer & set(self.questions[i]["topics"])) for i in candidates}

        ranked = sorted(candidates,
                        key=lambda i: (-overlap[i], -cosine[i], self.questions[i]["id"]))
        return [(self.questions[i], cosine[i] if query else float(overlap[i]))
                for i in ranked[:k]]

    def near_duplicates(self, threshold: float = 0.9) -> list:
        return find_near_duplicates(self.ids, self.embeddings, threshold)
