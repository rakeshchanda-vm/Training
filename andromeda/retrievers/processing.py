from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Set

from andromeda.retrievers.core import Document


TextNormalizer = Callable[[str], str]


@dataclass
class RawDocument:
    """Raw payload prior to chunking or normalization."""

    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    mime_type: Optional[str] = None
    source: Optional[str] = None


@dataclass
class ChunkingConfig:
    """
    Controls how text is chunked.

    strategy: chooses the splitter used.
    size: maximum characters or tokens per chunk.
    overlap: overlap between adjacent chunks to preserve context.
    separators: optional separators used by recursive splitting.
    min_length: chunks shorter than this are skipped to avoid noise.
    max_chunks: optional guardrail to prevent runaway splits on huge docs.
    merge_shorter_than: merge adjacent chunks shorter than this threshold.
    sentence_overlap: overlap in sentences for the sentence_window strategy.
    """

    strategy: Literal["recursive", "sentence", "sentence_window", "markdown", "tokens", "fixed"] = "recursive"
    size: int = 800
    overlap: int = 120
    separators: Optional[Sequence[str]] = None
    min_length: int = 40
    max_chunks: Optional[int] = None
    merge_shorter_than: Optional[int] = None
    sentence_overlap: int = 1


def _collapse_ws(text: str) -> str:
    """
    Whitespace normalizer that preserves newlines.

    - Collapses runs of spaces/tabs within a line
    - Normalizes CRLF to LF
    - Collapses 3+ consecutive newlines to 2
    - Strips trailing whitespace on each line
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", ln).rstrip() for ln in text.split("\n")]
    text = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def _strip_control_chars(text: str) -> str:
    # Removes non printable control chars that break some vector DBs, but preserves newlines/tabs.
    return "".join(ch for ch in text if ch.isprintable() or ch in {"\n", "\t"})


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _naive_recursive_split(
    text: str, size: int, overlap: int, separators: Optional[Sequence[str]]
) -> List[str]:
    """Fallback recursive splitter when LangChain splitters are unavailable."""
    seps = list(separators) if separators else ["\n\n", "\n", ". ", " "]
    chunks: List[str] = [text]
    for sep in seps:
        next_chunks: List[str] = []
        for chunk in chunks:
            if len(chunk) <= size:
                next_chunks.append(chunk)
                continue
            parts = chunk.split(sep)
            buf: List[str] = []
            for part in parts:
                candidate = (sep if buf else "").join(buf + [part])
                if len(candidate) >= size:
                    if buf:
                        next_chunks.append(sep.join(buf))
                    buf = [part]
                else:
                    buf.append(part)
            if buf:
                next_chunks.append(sep.join(buf))
        chunks = next_chunks
    # Final pass to enforce overlap
    with_overlap: List[str] = []
    for chunk in chunks:
        start = 0
        while start < len(chunk):
            end = start + size
            with_overlap.append(chunk[start:end])
            if end >= len(chunk):
                break
            start = end - overlap
    return with_overlap


def _sentence_split(text: str, size: int, overlap: int) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    buf: List[str] = []
    current_len = 0
    for sent in sentences:
        if current_len + len(sent) + 1 > size and buf:
            chunk = " ".join(buf).strip()
            chunks.append(chunk)
            buf = [sent[-overlap:]] if overlap else []
            current_len = len(" ".join(buf))
        else:
            buf.append(sent)
            current_len += len(sent) + 1
    if buf:
        chunks.append(" ".join(buf).strip())
    return chunks


def _markdown_split(text: str, size: int, overlap: int) -> List[str]:
    sections: List[str] = []
    current: List[str] = []
    for line in text.splitlines():
        if line.startswith("#") and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current).strip())

    chunks: List[str] = []
    for sec in sections:
        if len(sec) <= size:
            chunks.append(sec)
            continue
        # If section still too long, fall back to recursive split
        chunks.extend(
            _naive_recursive_split(sec, size=size, overlap=overlap, separators=None)
        )
    return chunks


def _token_split(text: str, size: int, overlap: int) -> List[str]:
    """
    Token aware splitting using tiktoken if available, otherwise falls back to word windows.
    """
    try:
        import tiktoken  # type: ignore
    except ImportError:  # pragma: no cover - optional dep
        words = text.split()
        chunks: List[str] = []
        step = size - overlap if size > overlap else size
        for i in range(0, len(words), step):
            chunk_words = words[i : i + size]
            chunks.append(" ".join(chunk_words))
        return chunks

    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks: List[str] = []
    step = size - overlap if size > overlap else size
    for i in range(0, len(tokens), step):
        token_slice = tokens[i : i + size]
        chunks.append(enc.decode(token_slice))
    return chunks


def _sentence_window_split(text: str, size: int, overlap_sentences: int) -> List[str]:
    """
    Builds overlapping windows of sentences to keep context across boundaries.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    windows: List[str] = []
    if not sentences:
        return windows
    start = 0
    while start < len(sentences):
        buf: List[str] = []
        length = 0
        idx = start
        while idx < len(sentences) and length + len(sentences[idx]) <= size:
            buf.append(sentences[idx])
            length += len(sentences[idx]) + 1
            idx += 1
        if buf:
            windows.append(" ".join(buf).strip())
        if idx == start:  # single very long sentence
            idx += 1
        start = max(idx - overlap_sentences, start + 1)
    return windows


def _merge_short_chunks(chunks: List[str], min_length: int, merge_threshold: Optional[int], size: int) -> List[str]:
    """
    Merges adjacent small chunks to avoid over-fragmentation while respecting max size.
    """
    if not merge_threshold:
        return chunks
    merged: List[str] = []
    buf: List[str] = []
    buf_len = 0
    for chunk in chunks:
        c = chunk.strip()
        if not c:
            continue
        if len(c) >= merge_threshold:
            if buf:
                merged_chunk = " ".join(buf).strip()
                if merged_chunk:
                    merged.append(merged_chunk)
                buf, buf_len = [], 0
            merged.append(c[:size])
            continue
        # accumulate small chunks
        if buf_len + len(c) <= size:
            buf.append(c)
            buf_len += len(c) + 1
        else:
            merged_chunk = " ".join(buf).strip()
            if merged_chunk:
                merged.append(merged_chunk[:size])
            buf, buf_len = [c], len(c)
    if buf:
        merged_chunk = " ".join(buf).strip()
        if merged_chunk:
            merged.append(merged_chunk[:size])
    # Final pass: drop anything still below min_length
    return [m for m in merged if len(m) >= min_length]


class DocumentProcessingEngine:
    """
    Runs text normalization, chunking, and optional deduplication before inserting into a store.
    """

    def __init__(
        self,
        chunking: ChunkingConfig | None = None,
        normalizers: Optional[Sequence[TextNormalizer]] = None,
        deduplicate: bool = True,
    ):
        self.chunking = chunking or ChunkingConfig()
        self.normalizers = list(normalizers) if normalizers else [_strip_control_chars, _collapse_ws]
        self._deduplicate = deduplicate
        self._seen_hashes: Set[str] = set() if deduplicate else set()

        self._splitter = self._build_splitter(self.chunking)

    def _build_splitter(self, cfg: ChunkingConfig) -> Callable[[str], List[str]]:
        if cfg.strategy == "recursive":
            try:
                # Prefer the standalone package but fall back if unavailable
                try:
                    from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
                except ImportError:  # pragma: no cover
                    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=cfg.size,
                    chunk_overlap=cfg.overlap,
                    separators=list(cfg.separators) if cfg.separators else None,
                )
                return splitter.split_text
            except Exception:
                # Fallback to naive splitter without adding hard dependency
                return lambda text: _naive_recursive_split(
                    text, size=cfg.size, overlap=cfg.overlap, separators=cfg.separators
                )
        if cfg.strategy == "sentence":
            return lambda text: _sentence_split(text, size=cfg.size, overlap=cfg.overlap)
        if cfg.strategy == "sentence_window":
            return lambda text: _sentence_window_split(
                text, size=cfg.size, overlap_sentences=cfg.sentence_overlap
            )
        if cfg.strategy == "markdown":
            return lambda text: _markdown_split(text, size=cfg.size, overlap=cfg.overlap)
        if cfg.strategy == "tokens":
            return lambda text: _token_split(text, size=cfg.size, overlap=cfg.overlap)
        if cfg.strategy == "fixed":
            return lambda text: _naive_recursive_split(
                text, size=cfg.size, overlap=cfg.overlap, separators=[" "]
            )
        raise ValueError(f"Unknown chunking strategy {cfg.strategy}")

    def _normalize(self, text: str) -> str:
        normalized = text
        for fn in self.normalizers:
            normalized = fn(normalized)
        return normalized

    def process(self, docs: Iterable[RawDocument]) -> List[Document]:
        """Normalize + chunk raw documents into the canonical Document shape."""
        processed: List[Document] = []
        seen_hashes: Set[str] = set() if self._deduplicate else set()
        for raw in docs:
            normalized = self._normalize(raw.text)
            if not normalized:
                continue

            parts = self._splitter(normalized)
            # Merge over-fragmented chunks and cap total chunks for safety.
            parts = _merge_short_chunks(
                parts,
                min_length=self.chunking.min_length,
                merge_threshold=self.chunking.merge_shorter_than or 0,
                size=self.chunking.size,
            )
            if self.chunking.max_chunks and len(parts) > self.chunking.max_chunks:
                parts = parts[: self.chunking.max_chunks]
            for idx, chunk in enumerate(parts):
                if len(chunk) < self.chunking.min_length:
                    continue

                if self._deduplicate:
                    digest = _hash_text(chunk)
                    if digest in seen_hashes:
                        continue
                    seen_hashes.add(digest)

                meta = dict(raw.metadata or {})
                meta.update(
                    {
                        "source_id": raw.id,
                        "chunk_index": idx,
                        "mime_type": raw.mime_type,
                        "source": raw.source or meta.get("source"),
                        "chunk_strategy": self.chunking.strategy,
                    }
                )
                processed.append(
                    Document(
                        id=f"{raw.id}::chunk-{idx}",
                        text=chunk,
                        metadata=meta,
                    )
                )
        return processed
