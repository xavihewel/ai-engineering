#!/usr/bin/env python3
"""Compare two texts using OpenAI embeddings and cosine similarity."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from app.embedding_pipeline.embedder import OpenAIEmbedder  # noqa: E402


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return the dot product divided by the product of the vector norms."""
    if not a or len(a) != len(b):
        raise ValueError("Vectors must be non-empty and have the same length.")
    norm_a = math.sqrt(sum(value * value for value in a))
    norm_b = math.sqrt(sum(value * value for value in b))
    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")
    return sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two texts using cosine similarity.")
    parser.add_argument("--text-a", required=True, help="First text.")
    parser.add_argument("--text-b", required=True, help="Second text.")
    args = parser.parse_args(argv)

    load_dotenv(ROOT / ".env")
    embedder = OpenAIEmbedder()
    vector_a = embedder.embed_one(args.text_a)
    vector_b = embedder.embed_one(args.text_b)
    similarity = cosine_similarity(vector_a, vector_b)
    print(f"Text A: {args.text_a}")
    print(f"Text B: {args.text_b}")
    print(f"Cosine similarity: {similarity:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
