"""formosa_dual.eval.chair_pope — hallucination metrics (§5.22).

CHAIR_i and CHAIR_s with the cultural tag vocabulary as the object set.
POPE-style yes/no probing.
"""
from __future__ import annotations

import re

from formosa_dual.data.tag_vocab import TagVocabulary
from formosa_dual.utils.logging import get_logger

logger = get_logger(__name__)


def chair_scores(
    hypotheses: list[str],
    references: list[dict],
    vocab: TagVocabulary,
) -> dict[str, float]:
    """Compute CHAIR_i and CHAIR_s.

    CHAIR_i = fraction of image descriptions with at least one hallucinated tag.
    CHAIR_s = fraction of mentioned tags that are hallucinated.

    A tag is "mentioned" if found in *hypothesis* via Aho-Corasick.
    A tag is "hallucinated" if it is mentioned but NOT in the reference tags.

    Args:
        hypotheses: List of generated captions.
        references: List of reference dicts with key ``culture_tags``.
        vocab: :class:`~formosa_dual.data.tag_vocab.TagVocabulary`.

    Returns:
        Dict: ``{CHAIR_i, CHAIR_s, n_images, n_mentioned, n_hallucinated}``.
    """
    from formosa_dual.data.tag_annotator import TagAnnotator
    annotator = TagAnnotator(vocab=vocab)

    n_images = len(hypotheses)
    n_hal_images = 0
    n_mentioned = 0
    n_hallucinated = 0

    for hyp, ref in zip(hypotheses, references):
        record = {"caption": hyp, "id": "__chair__", "metadata": {}}
        predicted_ids = annotator.annotate(record)
        predicted_tags = {vocab.decode(tid) for tid in predicted_ids if vocab.decode(tid)}
        ref_tags = set(ref.get("culture_tags", []))

        hallucinated = predicted_tags - ref_tags
        n_mentioned += len(predicted_tags)
        n_hallucinated += len(hallucinated)
        if hallucinated:
            n_hal_images += 1

    chair_i = n_hal_images / max(n_images, 1)
    chair_s = n_hallucinated / max(n_mentioned, 1)

    return {
        "CHAIR_i": chair_i,
        "CHAIR_s": chair_s,
        "n_images": n_images,
        "n_mentioned": n_mentioned,
        "n_hallucinated": n_hallucinated,
    }


def pope_probe(
    model_answer_fn,
    probe_list: list[dict],
) -> dict[str, float]:
    """Compute POPE yes/no accuracy.

    Each item in *probe_list*:
        ``{"image": PIL.Image, "question": "Is there a X in the image?", "label": "yes"|"no"}``

    Args:
        model_answer_fn: Callable ``(image, question) -> str`` returning model answer.
        probe_list: List of probe dicts.

    Returns:
        Dict: ``{accuracy, precision, recall, f1}``.
    """
    tp = fp = tn = fn = 0
    for probe in probe_list:
        answer = model_answer_fn(probe["image"], probe["question"]).strip().lower()
        predicted_yes = answer.startswith("yes")
        ground_yes = probe["label"].strip().lower() == "yes"
        if predicted_yes and ground_yes:
            tp += 1
        elif predicted_yes and not ground_yes:
            fp += 1
        elif not predicted_yes and not ground_yes:
            tn += 1
        else:
            fn += 1

    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
