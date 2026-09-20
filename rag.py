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
    if not MANUAL.exists():
        return None
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=HashEmbedding(),
        metadata={"hnsw:space": "cosine"},
    )
    chunks = _chunks(MANUAL.read_text(encoding="utf-8"))
    if chunks:
        collection.upsert(
            ids=[f"manual-{index}" for index in range(len(chunks))],
            documents=chunks,
            metadatas=[{"source": f"商品售后手册.txt#chunk-{index + 1}"} for index in range(len(chunks))],
        )
    return collection


def retrieve(question: str, top_k: int = 3) -> tuple[str, list[str]]:
    """Retrieve manual passages and return context plus citation source IDs."""
    try:
        collection = _collection()
        if collection is None or collection.count() == 0:
            return "暂无可用手册内容。", []
        result: dict[str, Any] = collection.query(
            query_texts=[question],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas"],
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        sources = [item.get("source", "本地售后手册") for item in metadatas]
        context = "\n\n".join(
            f"[来源：{source}]\n{document}"
            for source, document in zip(sources, documents)
        )
        return context or "暂无匹配手册内容。", sources
    except Exception:
        # Retrieval is an enhancement; a temporary local-index failure must not crash chat.
        return "知识库暂时不可用，请仅依据已确认信息回答。", []
