"""Detailed notes on the Phase 2 evaluation package.

This package exists to make retrieval quality measurable instead of relying on
eyeballing top results. Its two modules have separate jobs:

- `labeled_queries.py` is the hand-written ground truth. It contains natural
  language prompts and the wiki ids that should count as relevant.
- `run_eval.py` is the harness. It loads the corpus, embeds it once, asks one
  `VectorStore` implementation for each labeled query, computes recall@k and
  MRR, and writes a JSON artifact under `eval/results/`.

Keeping labels separate from harness code lets us add queries without touching
retrieval logic, and lets later phases (filters, hybrid search, RAG) rerun the
exact same comparison. This module intentionally has no runtime imports so
importing `eval` never has side effects.
"""
