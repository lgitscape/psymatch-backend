# ann_helper.py – Approximate Nearest‑Neighbour utilities for PsyMatch
# ==================================================================
"""ANNIndex + convenience `shortlist` used by `recommender.py`.

Highlights
~~~~~~~~~~
* **Cosine similarity** (inner‑product after L2 normalisation).
* Uses **FAISS** (`IndexFlatIP`) if available; GPU build supported if
  `faiss-gpu` is installed.
* Falls back to a pure‑NumPy brute‑force search when FAISS is missing –
  good for ≤5 000 therapists or local testing.
* Convenience `shortlist()` builds the same feature matrix the core
  recommender uses, then does ANN on the average feature vector.
"""
from __future__ import annotations

from typing import List, Sequence, Optional, Callable
import numpy as np

try:
    import faiss  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    faiss = None  # type: ignore

__all__ = ["ANNIndex", "shortlist"]


class ANNIndex:
    """Cosine‑similarity ANN; ID‑preserving."""

    def __init__(self, dim: int) -> None:
        self._use_faiss = faiss is not None
        self._dim = dim
        if self._use_faiss:
            self._index = faiss.IndexFlatIP(dim)
        else:
            self._vecs: np.ndarray = np.empty((0, dim), dtype=np.float32)
        self._ids: List[str] = []

    def build(self, vecs: np.ndarray, ids: List[str]) -> None:
        """Build the index from feature matrix `vecs` and associated `ids`."""
        if vecs.shape[0] != len(ids):
            raise ValueError("vecs and ids length mismatch")
        vecs = vecs.astype("float32")
        if self._use_faiss:
            faiss.normalize_L2(vecs)
            self._index.reset()
            self._index.add(vecs)
        else:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._vecs = vecs / norms
        self._ids = ids

    def query(self, q_vec: np.ndarray, k: int = 200) -> List[str]:
        """Return the top-`k` most similar IDs to `q_vec` via cosine similarity."""
        if q_vec.shape[-1] != self._dim:
            raise ValueError("Query vector dimensionality mismatch")
        q_vec = q_vec.astype("float32")
        if self._use_faiss:
            q = q_vec.reshape(1, -1)
            faiss.normalize_L2(q)
            _, idx = self._index.search(q, k)
            return [self._ids[i] for i in idx[0] if i != -1]
        # Brute-force fallback
        q_norm = q_vec / max(np.linalg.norm(q_vec), 1e-9)
        sims = self._vecs @ q_norm
        top = np.argsort(-sims)[:k]
        return [self._ids[i] for i in top]


def shortlist(
    client,
    therapists: Sequence,
    *,
    k: int = 200,
    feature_fn: Optional[Callable] = None
) -> List:
    """Return at most `k` therapist objects via cosine ANN search.

    Parameters
    ----------
    client : ClientProfile
        The client profile.
    therapists : Sequence[TherapistProfile]
        Sequence of therapist profiles.
    k : int, optional
        Number of nearest neighbours to return (default: 200).
    feature_fn : callable, optional
        A function(client, therapists) -> pd.DataFrame with a 'th_idx'
        column and numeric features. If None, falls back to binary
        bag-of-topics embedding.
    """
    # Small lists skip ANN
    if len(therapists) <= k:
        return list(therapists)

    # Build feature matrix
    if feature_fn is None:
        # Simple binary bag-of-topics
        topic_set = sorted({t for th in therapists for t in th.topics})
        t2i = {t: i for i, t in enumerate(topic_set)}
        vecs = np.zeros((len(therapists), len(topic_set)), dtype=np.float32)
        for idx, th in enumerate(therapists):
            for t in th.topics:
                vecs[idx, t2i[t]] = 1.0
        ids = [th.id for th in therapists]
    else:
        import pandas as pd  # local import
        df = feature_fn(client, therapists)
        ids = df.th_idx.tolist()
        vecs = df.drop(columns=["th_idx"]).astype("float32").to_numpy()

    ann = ANNIndex(vecs.shape[1])
    ann.build(vecs, ids)
    query_vec = vecs.mean(axis=0)
    keep_ids = set(ann.query(query_vec, k=k))
    return [th for th in therapists if th.id in keep_ids]
