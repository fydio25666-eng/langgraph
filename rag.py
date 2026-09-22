"""Small offline Chroma index for the local customer-service manual."""
import hashlib
import math
import re
from pathlib import Path
from typing import Any

import chromadb

ROOT = Path(__file__).resolve().parent
MANUAL = ROOT / "商品售后手册.txt"
DB_PATH = ROOT / ".chroma"
COLLECTION_NAME = "support_manual"
VECTOR_SIZE = 256
VECTOR_DISTANCE_THRESHOLD = 1.5


class HashEmbedding:
    """Deterministic local embedding; no network or model download is required."""

    @staticmethod
    def name() -> str:
        """Chroma 1.x uses this identifier to validate collection compatibility."""
        return "hash-embedding-v1"

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._encode(text) for text in input]

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        """Provide Chroma's document embedding interface for inserts/upserts."""
        return [self._encode(text) for text in input]

    def embed_query(self, input: str | list[str]) -> list[list[float]]:
        """Provide Chroma 1.x's query embedding interface for single/batch input."""
        texts = [input] if isinstance(input, str) else input
        return [self._encode(text) for text in texts]

    @staticmethod
    def _encode(text: str) -> list[float]:
        vector = [0.0] * VECTOR_SIZE
        tokens = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "big") % VECTOR_SIZE
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def _chunks(text: str, size: int = 800) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 2 > size:
            chunks.append(current)
            current = ""
        current = f"{current}\n\n{paragraph}".strip()
    if current:
        chunks.append(current)
    return chunks


def _collection():
    """Open the persistent collection and initialize it when it is empty."""
    if not MANUAL.exists():
        return None
    DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=HashEmbedding(),
        metadata={"hnsw:space": "cosine"},
    )
    chunks = _chunks(MANUAL.read_text(encoding="utf-8"))
    # Upsert is idempotent and also repairs a missing/partially-built index.
    if chunks and collection.count() != len(chunks):
        collection.upsert(
            ids=[f"manual-{index}" for index in range(len(chunks))],
            documents=chunks,
            metadatas=[{"source": f"商品售后手册.txt#chunk-{index + 1}"} for index in range(len(chunks))],
        )
    return collection


def _keywords(text: str) -> set[str]:
    """Extract searchable terms, including Chinese character bigrams."""
    terms: set[str] = set(re.findall(r"[A-Za-z0-9_]+", text.lower()))
    for sequence in re.findall(r"[\u4e00-\u9fff]+", text):
        terms.update(sequence)
        terms.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return terms


def _keyword_score(question: str, document: str) -> float:
    query_terms = _keywords(question)
    if not query_terms:
        return 0.0
    document_terms = _keywords(document)
    return len(query_terms & document_terms) / len(query_terms)


def retrieve(question: str, top_k: int = 3) -> tuple[str, list[str]]:
    """Hybrid retrieval: combine Chroma vector candidates with keyword matches."""
    try:
        collection = _collection()
        if collection is None or collection.count() == 0:
            return "暂无可用手册内容。", []

        candidate_count = min(max(top_k * 3, 6), collection.count())
        vector_result: dict[str, Any] = collection.query(
            query_texts=[question],
            n_results=candidate_count,
            include=["documents", "metadatas", "distances"],
        )

        documents = (vector_result.get("documents") or [[]])[0]
        metadatas = (vector_result.get("metadatas") or [[]])[0]
        distances = (vector_result.get("distances") or [[]])[0]
        ranked: dict[str, tuple[float, str, dict[str, Any]]] = {}
        for index, (document, metadata) in enumerate(zip(documents, metadatas)):
            distance = float(distances[index]) if index < len(distances) else 1.0
            vector_score = max(0.0, 1.0 - distance / 2.0)
            keyword_score = _keyword_score(question, document)
            ranked[document] = (0.65 * vector_score + 0.35 * keyword_score, document, metadata)

        # Keyword retrieval scans the local manual and can recover exact terms
        # even when a vector candidate falls below the relaxed distance cutoff.
        all_rows = collection.get(include=["documents", "metadatas"])
        for document, metadata in zip(
            all_rows.get("documents", []), all_rows.get("metadatas", [])
        ):
            keyword_score = _keyword_score(question, document)
            if keyword_score <= 0:
                continue
            current = ranked.get(document)
            vector_score = current[0] if current else 0.0
            score = max(vector_score, 0.35 * keyword_score)
            ranked[document] = (score, document, metadata)

        selected = sorted(ranked.values(), key=lambda item: item[0], reverse=True)[:top_k]
        # Do not discard all candidates on a low-similarity question; the
        # threshold only removes weak vector-only matches when stronger results exist.
        if not selected:
            return "暂无匹配手册内容。", []
        selected = [item for item in selected if item[0] >= (1 - VECTOR_DISTANCE_THRESHOLD / 2) or _keyword_score(question, item[1]) > 0] or selected[:1]
        sources = [item[2].get("source", "本地售后手册") for item in selected]
        context = "\n\n".join(
            f"[来源：{source}]\n{document}"
            for source, (_, document, _) in zip(sources, selected)
        )
        return context or "暂无匹配手册内容。", sources
    except Exception:
        # Retrieval is an enhancement; a temporary local-index failure must not crash chat.
        return "知识库暂时不可用，请仅依据已确认信息回答。", []
