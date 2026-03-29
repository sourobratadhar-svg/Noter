"""
Chunking Module — Semantic Text Splitting
==========================================
Splits text into chunks optimized for embedding and retrieval.
Strategy: paragraph boundaries → sentence boundaries → hard split.

Usage:
    from chunking import chunk_text
    chunks = chunk_text("long text...", max_tokens=400)
"""

import re
from typing import List


def chunk_text(text: str, max_tokens: int = 400, overlap: int = 50) -> List[str]:
    """
    Split text into semantically meaningful chunks (300-500 token range).

    Algorithm:
    1. Split by paragraph boundaries (double newlines)
    2. Accumulate paragraphs until token budget reached
    3. For oversized paragraphs, split by sentence boundaries
    4. Tokens estimated at ~4 chars each

    Args:
        text: Input text to chunk
        max_tokens: Maximum tokens per chunk (estimated at 4 chars/token)
        overlap: Reserved for future overlap implementation

    Returns:
        List of text chunks, each under max_tokens estimated size
    """
    text = text.strip()
    if not text:
        return []

    # Split by double-newlines (paragraph boundaries)
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        combined = (current_chunk + "\n\n" + para).strip() if current_chunk else para
        estimated_tokens = len(combined) // 4

        if estimated_tokens <= max_tokens:
            current_chunk = combined
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # Long paragraph: split by sentences
            if len(para) // 4 > max_tokens:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                sub_chunk = ""
                for sent in sentences:
                    combined_sent = (sub_chunk + " " + sent).strip() if sub_chunk else sent
                    if len(combined_sent) // 4 <= max_tokens:
                        sub_chunk = combined_sent
                    else:
                        if sub_chunk:
                            chunks.append(sub_chunk)
                        sub_chunk = sent
                if sub_chunk:
                    current_chunk = sub_chunk
                else:
                    current_chunk = ""
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    # Filter out tiny fragments that lack semantic value
    return [c for c in chunks if len(c) > 20]
