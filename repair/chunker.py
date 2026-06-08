import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from services.llm_factory import get_llm


def rechunk_semantic(text: str, source: str) -> list[Document]:
    """
    Strategy A — Smaller chunks with more overlap.
    Best default. Use when top score is low but chunks look vaguely relevant.
    Produces chunks of ~250 chars with 80-char overlap.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = 250,
        chunk_overlap = 80,
        separators    = ["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )
    return splitter.create_documents(
        [text], metadatas=[{"source": source, "strategy": "semantic"}]
    )


def rechunk_llm(text: str, source: str) -> list[Document]:
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
        return chunks if len(chunks) > 1 else rechunk_semantic(text, source)
    except Exception:
        return rechunk_semantic(text, source)


def rechunk_entropy(text: str, source: str) -> list[Document]:
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

    return chunks if len(chunks) > 1 else rechunk_semantic(text, source)
