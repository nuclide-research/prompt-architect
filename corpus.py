"""Runtime RAG over the bundled O'Reilly book corpus.

Pure-Python, dependency-free, offline, deterministic. Indexes the per-book
`_combined.md` extractions under `books/` with BM25 and answers passage queries.

Why BM25 and not embeddings: it needs no model download, no network, and no
third-party package, so `PromptArchitect` stays a self-contained import. The
`Retriever` protocol below lets an embedding backend (chromadb, sentence-
transformers, an API) be swapped in later without touching the architect.

Design grounding: lexical retrieval over a curated corpus is the baseline
retrieval-augmentation pattern (Pai, *Designing LLM Applications*, RAG chapters);
chunking with overlap preserves cross-paragraph context.
"""

from __future__ import annotations

import hashlib
import math
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Protocol, runtime_checkable

from .models import Retrieved

# Small, deliberately conservative stopword set. Keeping it short avoids dropping
# domain terms that matter for prompt-engineering queries ("format", "output").
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have in into is it its of on or
    that the their then there these this to was were will with we you your""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# BM25 tuning. k1 controls term-frequency saturation; b controls length
# normalization. 1.5 / 0.75 are the long-standing defaults.
_K1 = 1.5
_B = 0.75

_CACHE_VERSION = 2  # bump when chunking or index format changes


def tokenize(text: str) -> List[str]:
    """Lowercase, split on word boundaries, drop stopwords and 1-char tokens."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in _STOPWORDS]


@runtime_checkable
class Retriever(Protocol):
    """Pluggable retrieval backend. Swap BM25 for embeddings by implementing this."""

    def search(self, query: str, k: int = 5, book: Optional[str] = None) -> List[Retrieved]:
        ...


class _Chunk:
    """One indexed passage. __slots__ keeps thousands of these lightweight."""

    __slots__ = ("book", "text", "tf", "length")

    def __init__(self, book: str, text: str, tf: Dict[str, int], length: int) -> None:
        self.book = book
        self.text = text
        self.tf = tf
        self.length = length


class BookCorpus:
    """BM25 index over the `books/*/_combined.md` extractions.

    Builds on first use, caches to disk keyed on the source files' mtimes plus the
    chunking parameters, and answers `search()` queries. Implements `Retriever`.
    """

    def __init__(
        self,
        books_dir: Optional[str] = None,
        chunk_words: int = 220,
        overlap: int = 40,
        cache: bool = True,
    ) -> None:
        self.books_dir = Path(books_dir) if books_dir else Path(__file__).parent / "books"
        self.chunk_words = chunk_words
        self.overlap = overlap
        self.cache = cache

        self._chunks: List[_Chunk] = []
        self._idf: Dict[str, float] = {}
        self._avgdl: float = 0.0
        self._built = False

    # ------------------------------------------------------------------ #
    # Index lifecycle
    # ------------------------------------------------------------------ #

    def build(self) -> "BookCorpus":
        """Build (or load from cache) the BM25 index. Idempotent."""
        if self._built:
            return self
        if not self.books_dir.is_dir():
            raise FileNotFoundError(f"books_dir not found: {self.books_dir}")

        combined = sorted(self.books_dir.glob("*/_combined.md"))
        if not combined:
            raise FileNotFoundError(
                f"no '*/_combined.md' files under {self.books_dir}; run colophon with -combined"
            )

        if self.cache and self._load_cache(combined):
            self._built = True
            return self

        self._index(combined)
        if self.cache:
            self._save_cache(combined)
        self._built = True
        return self

    def _index(self, combined: List[Path]) -> None:
        chunks: List[_Chunk] = []
        df: Dict[str, int] = {}
        for path in combined:
            book = path.parent.name
            text = path.read_text(encoding="utf-8", errors="replace")
            for passage in self._chunk(text):
                tokens = tokenize(passage)
                if not tokens:
                    continue
                tf: Dict[str, int] = {}
                for tok in tokens:
                    tf[tok] = tf.get(tok, 0) + 1
                chunks.append(_Chunk(book, passage, tf, len(tokens)))
                for term in tf:
                    df[term] = df.get(term, 0) + 1

        n = len(chunks)
        if n == 0:
            raise ValueError("corpus produced zero indexable chunks")
        # BM25+ idf, floored at a small positive value so common terms still rank.
        self._idf = {
            term: math.log(1 + (n - d + 0.5) / (d + 0.5)) for term, d in df.items()
        }
        self._avgdl = sum(c.length for c in chunks) / n
        self._chunks = chunks

    def _chunk(self, text: str) -> List[str]:
        """Word-window chunks with overlap, split on blank-line paragraph runs.

        Markdown headings and short fragments are folded into the running window
        so each chunk carries enough context to be independently meaningful.
        """
        # Normalize whitespace within paragraphs; keep paragraph breaks.
        paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
        words: List[str] = []
        for p in paras:
            if p:
                words.extend(p.split(" "))
        chunks: List[str] = []
        step = max(1, self.chunk_words - self.overlap)
        for start in range(0, len(words), step):
            window = words[start : start + self.chunk_words]
            if not window:
                break
            chunks.append(" ".join(window))
            if start + self.chunk_words >= len(words):
                break
        return chunks

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #

    def search(self, query: str, k: int = 5, book: Optional[str] = None) -> List[Retrieved]:
        """Return the top-k passages for `query`, optionally scoped to one book."""
        if not self._built:
            self.build()
        q_terms = tokenize(query)
        if not q_terms:
            return []
        # Pre-resolve idf for query terms once.
        q_idf = {t: self._idf.get(t, 0.0) for t in set(q_terms)}

        scored: List[tuple] = []
        for cid, chunk in enumerate(self._chunks):
            if book is not None and chunk.book != book:
                continue
            score = self._bm25(q_idf, chunk)
            if score > 0:
                scored.append((score, cid, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        results: List[Retrieved] = []
        for score, cid, chunk in scored[:k]:
            results.append(
                Retrieved(
                    book=chunk.book,
                    chunk_id=cid,
                    score=round(float(score), 4),
                    text=chunk.text,
                )
            )
        return results

    def _bm25(self, q_idf: Dict[str, float], chunk: _Chunk) -> float:
        score = 0.0
        norm = _K1 * (1 - _B + _B * chunk.length / self._avgdl)
        tf = chunk.tf
        for term, idf in q_idf.items():
            if idf == 0.0:
                continue
            f = tf.get(term, 0)
            if f == 0:
                continue
            score += idf * (f * (_K1 + 1)) / (f + norm)
        return score

    # ------------------------------------------------------------------ #
    # Cache (pickle keyed on source mtimes + params)
    # ------------------------------------------------------------------ #

    def _cache_path(self) -> Path:
        return self.books_dir / ".rag_cache.pkl"

    def _signature(self, combined: List[Path]) -> str:
        parts = [str(_CACHE_VERSION), str(self.chunk_words), str(self.overlap)]
        for p in combined:
            st = p.stat()
            parts.append(f"{p.name}:{st.st_size}:{int(st.st_mtime)}")
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    def _load_cache(self, combined: List[Path]) -> bool:
        path = self._cache_path()
        if not path.is_file():
            return False
        try:
            with path.open("rb") as fh:
                blob = pickle.load(fh)
            if blob.get("sig") != self._signature(combined):
                return False
            self._chunks = blob["chunks"]
            self._idf = blob["idf"]
            self._avgdl = blob["avgdl"]
            return True
        except Exception:
            # Corrupt or incompatible cache: rebuild silently.
            return False

    def _save_cache(self, combined: List[Path]) -> None:
        blob = {
            "sig": self._signature(combined),
            "chunks": self._chunks,
            "idf": self._idf,
            "avgdl": self._avgdl,
        }
        try:
            tmp = self._cache_path().with_suffix(".pkl.tmp")
            with tmp.open("wb") as fh:
                pickle.dump(blob, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, self._cache_path())
        except OSError:
            # Read-only books dir: skip caching, not fatal.
            pass

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def num_chunks(self) -> int:
        if not self._built:
            self.build()
        return len(self._chunks)

    def books(self) -> List[str]:
        if not self._built:
            self.build()
        return sorted({c.book for c in self._chunks})


if __name__ == "__main__":
    corpus = BookCorpus().build()
    print(f"indexed {corpus.num_chunks} chunks across {len(corpus.books())} books")
    for q in ("output format guardrails json schema", "context window token limit",
              "few-shot examples improve output", "reduce hallucination uncertainty"):
        print(f"\n### {q}")
        for r in corpus.search(q, k=2):
            print(f"  [{r['score']:.2f}] {r['book']}")
            print("    " + r["text"][:160].replace("\n", " ") + "...")
