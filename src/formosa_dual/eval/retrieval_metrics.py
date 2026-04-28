"""formosa_dual.eval.retrieval_metrics — image↔tag retrieval evaluation (§5.22).

Computes R@K (K=1,5,10), mAP per category, and cluster purity.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from formosa_dual.data.tag_vocab import TagVocabulary
from formosa_dual.utils.logging import get_logger

logger = get_logger(__name__)


def recall_at_k(
    query_embs: torch.Tensor,
    gallery_embs: torch.Tensor,
    query_labels: list[int],
    gallery_labels: list[int],
    k_list: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Compute R@K for each K.

    Args:
        query_embs: ``[N_q, d]`` L2-normalised embeddings.
        gallery_embs: ``[N_g, d]`` L2-normalised embeddings.
        query_labels: Integer labels for each query.
        gallery_labels: Integer labels for each gallery item.
        k_list: List of K values.

    Returns:
        Dict ``{R@1: float, R@5: float, ...}``.
    """
    sim = query_embs @ gallery_embs.T  # [N_q, N_g]
    results: dict[str, float] = {}
    for k in k_list:
        hits = 0
        for i, q_label in enumerate(query_labels):
            topk_indices = sim[i].topk(min(k, sim.size(1))).indices.tolist()
            topk_labels = [gallery_labels[j] for j in topk_indices]
            if q_label in topk_labels:
                hits += 1
        results[f"R@{k}"] = hits / max(len(query_labels), 1)
    return results


def map_per_category(
    query_embs: torch.Tensor,
    gallery_embs: torch.Tensor,
    query_labels: list[int],
    gallery_labels: list[int],
) -> dict[str, float]:
    """Compute mAP per category and overall mAP.

    Args:
        query_embs: ``[N_q, d]`` L2-normalised.
        gallery_embs: ``[N_g, d]`` L2-normalised.
        query_labels: Integer label for each query.
        gallery_labels: Integer label for each gallery item.

    Returns:
        Dict with ``mAP`` (overall) and ``AP_{label}`` per category.
    """
    sim = query_embs @ gallery_embs.T  # [N_q, N_g]
    # Group queries by label
    from collections import defaultdict
    label_queries: dict[int, list[int]] = defaultdict(list)
    for i, label in enumerate(query_labels):
        label_queries[label].append(i)

    ap_per_label: dict[int, float] = {}
    for label, query_indices in label_queries.items():
        aps = []
        for qi in query_indices:
            ranked = sim[qi].argsort(descending=True).tolist()
            n_relevant = sum(1 for gl in gallery_labels if gl == label)
            if n_relevant == 0:
                continue
            hits = 0
            precision_sum = 0.0
            for rank, gi in enumerate(ranked, start=1):
                if gallery_labels[gi] == label:
                    hits += 1
                    precision_sum += hits / rank
            aps.append(precision_sum / n_relevant)
        ap_per_label[label] = sum(aps) / max(len(aps), 1)

    overall_map = sum(ap_per_label.values()) / max(len(ap_per_label), 1)
    result = {"mAP": overall_map}
    for label, ap in ap_per_label.items():
        result[f"AP_{label}"] = ap
    return result


def cluster_purity(
    embs: torch.Tensor,
    true_labels: list[int],
    n_clusters: int | None = None,
) -> float:
    """Compute cluster purity via k-means + homogeneity_score.

    Args:
        embs: ``[N, d]`` embeddings.
        true_labels: Ground-truth integer labels.
        n_clusters: Number of clusters (defaults to number of unique labels).

    Returns:
        Homogeneity score in [0, 1].
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import homogeneity_score
    except ImportError as exc:
        raise ImportError("scikit-learn is required for cluster_purity") from exc

    n_unique = len(set(true_labels))
    k = n_clusters or n_unique
    X = embs.cpu().numpy()
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    predicted = km.fit_predict(X)
    return float(homogeneity_score(true_labels, predicted))
