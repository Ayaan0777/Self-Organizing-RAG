import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from services.llm_factory import get_llm


def rechunk_semantic(
    text: str,
    source: str,
    chunk_size: int = 250,
    chunk_overlap: int = 80,
) -> list[Document]:
    """
    Strategy A — Configurable chunk size with overlap.
    Default: 250 chars, 80 overlap. The decision engine may pass
    different values (e.g., 256 for precision, 512 for context).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = chunk_size,
        chunk_overlap = chunk_overlap,
        separators    = ["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )
    return splitter.create_documents(
        [text], metadatas=[{"source": source, "strategy": "semantic",
                           "chunk_size": chunk_size, "chunk_overlap": chunk_overlap}]
    )


def rechunk_llm(text: str, source: str, chunk_size: int = 250, chunk_overlap: int = 80) -> list[Document]:
    """
    Strategy B — Ask the LLM to find topic boundaries, split there.
    Use when content mixes multiple distinct topics in one document.
    Falls back to rechunk_semantic on any parse failure.
    """
    llm    = get_llm()
    prompt = (
        "Identify natural topic-boundary sentence indices in the text below. "
        "Return ONLY a JSON array of integers — the 0-indexed sentence numbers "
        "where a new topic begins. Example output: [3, 7, 12]\n\nText:\n"
        + text[:3000]
    )
    try:
        raw        = llm.invoke(prompt).content.strip()
        split_idxs = set(json.loads(raw))
        sentences  = [s.strip() for s in text.replace(". ", ".|").split("|") if s.strip()]
        chunks, buf = [], []
        for i, sent in enumerate(sentences):
            buf.append(sent)
            if i in split_idxs and buf:
                chunks.append(Document(
                    page_content = " ".join(buf),
                    metadata     = {"source": source, "strategy": "llm"},
                ))
                buf = []
        if buf:
            chunks.append(Document(
                page_content = " ".join(buf),
                metadata     = {"source": source, "strategy": "llm"},
            ))
        return chunks if len(chunks) > 1 else rechunk_semantic(text, source, chunk_size, chunk_overlap)
    except Exception:
        return rechunk_semantic(text, source, chunk_size, chunk_overlap)


def rechunk_entropy(text: str, source: str, chunk_size: int = 250, chunk_overlap: int = 80) -> list[Document]:
    """
    Strategy C — Split at sentences with high vocabulary novelty (topic shifts).
    Use when content is long and topics drift gradually without clear headings.
    Falls back to rechunk_semantic if fewer than 2 chunks produced.
    """
    sentences          = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    vocab, chunks, buf = set(), [], []

    for sent in sentences:
        words     = set(sent.lower().split())
        new_words = words - vocab
        novelty   = len(new_words) / max(len(words), 1)
        vocab    |= words

        if novelty > 0.6 and buf:   # high novelty = topic shift = start new chunk
            chunks.append(Document(
                page_content = ". ".join(buf),
                metadata     = {"source": source, "strategy": "entropy"},
            ))
            buf = []
        buf.append(sent)

    if buf:
        chunks.append(Document(
            page_content = ". ".join(buf),
            metadata     = {"source": source, "strategy": "entropy"},
        ))

    return chunks if len(chunks) > 1 else rechunk_semantic(text, source, chunk_size, chunk_overlap)
