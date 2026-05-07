"""formosa_dual.eval.retrieval_metrics — image↔tag retrieval evaluation (§5.22).

Computes image→tag R@K, tag→image R@K, mean average precision, and cluster
purity.  The high-level helpers operate on the contrastive embeddings emitted by
``DualObjectiveModel.forward`` and rank against the full tag vocabulary, not only
the sampled in-batch negatives.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from formosa_dual.data.tag_vocab import TagVocabulary
from formosa_dual.utils.logging import get_logger

logger = get_logger(__name__)


def compute_retrieval_metrics(
    visual_embs: torch.Tensor,
    tag_embs: torch.Tensor,
    pos_tag_ids: torch.Tensor,
    pos_tag_mask: torch.Tensor,
    image_to_tag_k: tuple[int, ...] = (1, 5, 10),
    tag_to_image_k: tuple[int, ...] = (1, 5),
) -> dict[str, float]:
    """Compute full-vocabulary image↔tag retrieval metrics.

    Args:
        visual_embs: Image embeddings with shape ``[N, d]``.
        tag_embs: Full tag gallery embeddings with shape ``[K, d]`` where row
            index equals tag id.
        pos_tag_ids: Padded positive tag ids with shape ``[N, P]``.
        pos_tag_mask: Boolean valid-positive mask with shape ``[N, P]``.
        image_to_tag_k: K values for image→tag recall.
        tag_to_image_k: K values for tag→image recall.

    Returns:
        Flat metric dict containing ``image_to_tag_R@1/R@5/R@10``,
        ``tag_to_image_R@1/R@5``, ``image_to_tag_mAP``, ``tag_to_image_mAP``,
        and ``mAP`` (alias for image→tag mAP).
    """
    if visual_embs.numel() == 0 or tag_embs.numel() == 0:
        return _empty_metrics(image_to_tag_k, tag_to_image_k)

    if visual_embs.dim() != 2:
        raise ValueError(f"visual_embs must be [N, d], got {tuple(visual_embs.shape)}")
    if tag_embs.dim() != 2:
        raise ValueError(f"tag_embs must be [K, d], got {tuple(tag_embs.shape)}")
    if pos_tag_ids.shape != pos_tag_mask.shape:
        raise ValueError(
            "pos_tag_ids and pos_tag_mask must have identical shape, got "
            f"{tuple(pos_tag_ids.shape)} and {tuple(pos_tag_mask.shape)}"
        )
    if pos_tag_ids.dim() != 2:
        raise ValueError(f"pos_tag_ids must be [N, P], got {tuple(pos_tag_ids.shape)}")
    if visual_embs.size(0) != pos_tag_ids.size(0):
        raise ValueError(
            "visual_embs and pos_tag_ids disagree on N: "
            f"{visual_embs.size(0)} vs {pos_tag_ids.size(0)}"
        )
    if visual_embs.size(1) != tag_embs.size(1):
        raise ValueError(
            "visual_embs and tag_embs disagree on embedding dim: "
            f"{visual_embs.size(1)} vs {tag_embs.size(1)}"
        )

    visual_embs = F.normalize(visual_embs.float(), p=2, dim=-1)
    tag_embs = F.normalize(tag_embs.float(), p=2, dim=-1)

    valid_pos_mask = (
        pos_tag_mask.bool() & (pos_tag_ids >= 0) & (pos_tag_ids < tag_embs.size(0))
    )
    sim = visual_embs @ tag_embs.T

    metrics: dict[str, float] = {}
    metrics.update(
        _image_to_tag_recall(sim, pos_tag_ids, valid_pos_mask, image_to_tag_k)
    )
    image_map = _image_to_tag_map(sim, pos_tag_ids, valid_pos_mask)
    metrics["image_to_tag_mAP"] = image_map

    tag_metrics, tag_map, n_eval_tags = _tag_to_image_metrics(
        sim=sim,
        pos_tag_ids=pos_tag_ids,
        pos_tag_mask=valid_pos_mask,
        k_list=tag_to_image_k,
    )
    metrics.update(tag_metrics)
    metrics["tag_to_image_mAP"] = tag_map
    metrics["mAP"] = image_map
    metrics["n_images"] = float(visual_embs.size(0))
    metrics["n_tags"] = float(tag_embs.size(0))
    metrics["n_eval_tags"] = float(n_eval_tags)
    return metrics


def embed_all_tag_embeddings(
    model,
    vocab: TagVocabulary,
    device: torch.device,
    chunk_size: int = 512,
) -> torch.Tensor | None:
    """Embed every vocabulary tag using ``model.tag_projector``.

    Args:
        model: ``DualObjectiveModel`` or a wrapped module exposing
            ``tag_projector``.
        vocab: Tag vocabulary whose ids define the gallery order.
        device: Device where the model lives.
        chunk_size: Number of tags embedded per call.

    Returns:
        ``[K, d]`` CPU tensor, or ``None`` if the model has no usable tag
        projector.
    """
    unwrapped = _unwrap_model(model)
    tag_projector = getattr(unwrapped, "tag_projector", None)
    if tag_projector is None:
        return None

    try:
        n_tags = len(vocab)
    except TypeError:
        logger.warning("Cannot compute retrieval tag gallery: vocab has no length")
        return None
    if n_tags <= 0:
        return None

    chunks: list[torch.Tensor] = []
    try:
        for start in range(0, n_tags, chunk_size):
            end = min(start + chunk_size, n_tags)
            tag_ids = torch.arange(
                start, end, device=device, dtype=torch.long
            ).unsqueeze(0)
            if hasattr(tag_projector, "get_tag_embeddings"):
                emb = tag_projector.get_tag_embeddings(tag_ids)
            else:
                emb = tag_projector(tag_ids)
            chunks.append(emb.squeeze(0).detach().cpu())
    except (RuntimeError, TypeError, AttributeError) as exc:
        logger.warning("Cannot compute retrieval tag gallery: %s", exc)
        return None

    return torch.cat(chunks, dim=0) if chunks else None


def evaluate_retrieval_loader(
    model,
    dataloader: DataLoader,
    vocab: TagVocabulary,
    device: torch.device,
    desc: str = "retrieval",
    show_progress: bool = True,
) -> dict[str, float]:
    """Run a retrieval-only pass over a dataloader.

    Args:
        model: ``DualObjectiveModel``.
        dataloader: Evaluation dataloader using ``DualCollator``.
        vocab: Full tag vocabulary.
        device: Target device.
        desc: Progress-bar label.
        show_progress: Whether to render a progress bar.

    Returns:
        Retrieval metric dict from :func:`compute_retrieval_metrics`.
    """
    tag_gallery = embed_all_tag_embeddings(model, vocab, device=device)
    if tag_gallery is None:
        return _empty_metrics((1, 5, 10), (1, 5))

    visual_batches: list[torch.Tensor] = []
    pos_id_batches: list[torch.Tensor] = []
    pos_mask_batches: list[torch.Tensor] = []

    model.eval()
    progress = tqdm(
        dataloader,
        desc=desc,
        unit="batch",
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )
    with torch.no_grad():
        for batch in progress:
            batch = _move_batch_to_device(batch, device)
            out = model(batch)
            visual_emb = out.get("visual_emb")
            if visual_emb is None:
                continue
            visual_batches.append(visual_emb.detach().cpu())
            pos_id_batches.append(batch["pos_tag_ids"].detach().cpu())
            pos_mask_batches.append(batch["pos_tag_mask"].detach().cpu())

    if not visual_batches:
        return _empty_metrics((1, 5, 10), (1, 5))

    return compute_retrieval_metrics(
        visual_embs=torch.cat(visual_batches, dim=0),
        tag_embs=tag_gallery,
        pos_tag_ids=torch.cat(pos_id_batches, dim=0),
        pos_tag_mask=torch.cat(pos_mask_batches, dim=0),
    )


def _image_to_tag_recall(
    sim: torch.Tensor,
    pos_tag_ids: torch.Tensor,
    pos_tag_mask: torch.Tensor,
    k_list: tuple[int, ...],
) -> dict[str, float]:
    valid_queries = pos_tag_mask.any(dim=1)
    denom = int(valid_queries.sum().item())
    results: dict[str, float] = {}
    if denom == 0:
        for k in k_list:
            results[f"image_to_tag_R@{k}"] = 0.0
        return results

    max_gallery = sim.size(1)
    for k in k_list:
        topk = sim.topk(min(k, max_gallery), dim=1).indices
        hits = 0
        for i in torch.nonzero(valid_queries, as_tuple=True)[0].tolist():
            positives = set(pos_tag_ids[i][pos_tag_mask[i]].tolist())
            if positives.intersection(topk[i].tolist()):
                hits += 1
        results[f"image_to_tag_R@{k}"] = hits / denom
    return results


def _image_to_tag_map(
    sim: torch.Tensor,
    pos_tag_ids: torch.Tensor,
    pos_tag_mask: torch.Tensor,
) -> float:
    aps: list[float] = []
    valid_queries = torch.nonzero(pos_tag_mask.any(dim=1), as_tuple=True)[0].tolist()
    for i in valid_queries:
        relevant = torch.zeros(sim.size(1), dtype=torch.bool)
        relevant[pos_tag_ids[i][pos_tag_mask[i]].long().cpu()] = True
        aps.append(_average_precision(sim[i].detach().cpu(), relevant))
    return float(sum(aps) / len(aps)) if aps else 0.0


def _tag_to_image_metrics(
    sim: torch.Tensor,
    pos_tag_ids: torch.Tensor,
    pos_tag_mask: torch.Tensor,
    k_list: tuple[int, ...],
) -> tuple[dict[str, float], float, int]:
    valid_ids = pos_tag_ids[pos_tag_mask].long().cpu()
    if valid_ids.numel() == 0:
        return {f"tag_to_image_R@{k}": 0.0 for k in k_list}, 0.0, 0

    unique_tag_ids = sorted(set(valid_ids.tolist()))
    results: dict[str, float] = {}
    sim_tag_to_image = sim.T.detach().cpu()
    pos_ids_cpu = pos_tag_ids.cpu()
    pos_mask_cpu = pos_tag_mask.cpu()

    hit_counts = {k: 0 for k in k_list}
    aps: list[float] = []
    for tag_id in unique_tag_ids:
        relevant_images = ((pos_ids_cpu == tag_id) & pos_mask_cpu).any(dim=1)
        if not bool(relevant_images.any()):
            continue
        scores = sim_tag_to_image[tag_id]
        for k in k_list:
            topk = scores.topk(min(k, scores.numel())).indices
            if bool(relevant_images[topk].any()):
                hit_counts[k] += 1
        aps.append(_average_precision(scores, relevant_images))

    denom = len(unique_tag_ids)
    for k in k_list:
        results[f"tag_to_image_R@{k}"] = hit_counts[k] / max(denom, 1)
    tag_map = float(sum(aps) / len(aps)) if aps else 0.0
    return results, tag_map, denom


def _average_precision(scores: torch.Tensor, relevant: torch.Tensor) -> float:
    relevant = relevant.bool()
    n_relevant = int(relevant.sum().item())
    if n_relevant == 0:
        return 0.0
    order = scores.argsort(descending=True)
    ranked_relevant = relevant[order].float()
    cumulative_hits = torch.cumsum(ranked_relevant, dim=0)
    ranks = torch.arange(1, ranked_relevant.numel() + 1, dtype=torch.float32)
    precision_at_rank = cumulative_hits / ranks
    ap = (precision_at_rank * ranked_relevant).sum() / n_relevant
    return float(ap.item())


def _empty_metrics(
    image_to_tag_k: tuple[int, ...],
    tag_to_image_k: tuple[int, ...],
) -> dict[str, float]:
    metrics = {f"image_to_tag_R@{k}": 0.0 for k in image_to_tag_k}
    metrics.update({f"tag_to_image_R@{k}": 0.0 for k in tag_to_image_k})
    metrics.update(
        {
            "image_to_tag_mAP": 0.0,
            "tag_to_image_mAP": 0.0,
            "mAP": 0.0,
            "n_images": 0.0,
            "n_tags": 0.0,
            "n_eval_tags": 0.0,
        }
    )
    return metrics


def _unwrap_model(model):
    return getattr(model, "module", model)


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


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
