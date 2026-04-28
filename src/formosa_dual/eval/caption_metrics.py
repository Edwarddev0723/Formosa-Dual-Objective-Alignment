"""formosa_dual.eval.caption_metrics — standard text generation evaluation metrics.

Wraps sacrebleu (BLEU-4), rouge-score (ROUGE-L), bert-score (BERTScore),
and a Chinese-CLIP-based CLIPScore.

Note: CIDEr (pycocoevalcap) requires loading the full annotation file.
This module provides a best-effort wrapper; if pycocoevalcap is unavailable,
the function raises ImportError with a clear install message.
"""
from __future__ import annotations

from formosa_dual.utils.logging import get_logger

logger = get_logger(__name__)


def bleu4(hypotheses: list[str], references: list[str]) -> float:
    """Compute corpus-level BLEU-4.

    Args:
        hypotheses: List of generated captions.
        references: List of reference captions (one per hypothesis).

    Returns:
        BLEU-4 score as a float in [0, 100].
    """
    from sacrebleu.metrics import BLEU
    bleu = BLEU(effective_order=True)
    result = bleu.corpus_score(hypotheses, [references])
    return result.score


def rouge_l(hypotheses: list[str], references: list[str]) -> float:
    """Compute mean ROUGE-L F1.

    Args:
        hypotheses: List of generated captions.
        references: List of reference captions.

    Returns:
        Mean ROUGE-L F1 as a float in [0, 1].
    """
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [
        scorer.score(ref, hyp)["rougeL"].fmeasure
        for hyp, ref in zip(hypotheses, references)
    ]
    return sum(scores) / max(len(scores), 1)


def bertscore(hypotheses: list[str], references: list[str], lang: str = "zh") -> float:
    """Compute mean BERTScore F1 using bert-base-chinese.

    Args:
        hypotheses: List of generated captions.
        references: List of reference captions.
        lang: Language code passed to bert_score.

    Returns:
        Mean BERTScore F1 as a float in [0, 1].
    """
    from bert_score import score as bs_score
    _, _, F1 = bs_score(hypotheses, references, lang=lang, verbose=False)
    return float(F1.mean().item())


def cider(hypotheses: list[str], references: list[str]) -> float:
    """Compute CIDEr score.

    Requires ``pycocoevalcap`` to be installed.

    Args:
        hypotheses: List of generated captions.
        references: List of reference captions.

    Returns:
        CIDEr score as a float.

    Raises:
        ImportError: If pycocoevalcap is not installed.
    """
    try:
        from pycocoevalcap.cider.cider import Cider
    except ImportError as exc:
        raise ImportError(
            "pycocoevalcap is required for CIDEr. Install with: pip install pycocoevalcap"
        ) from exc

    gts = {i: [ref] for i, ref in enumerate(references)}
    res = {i: [hyp] for i, hyp in enumerate(hypotheses)}
    scorer = Cider()
    score, _ = scorer.compute_score(gts, res)
    return float(score)


def clipscore_chinese(
    hypotheses: list[str],
    images,
    clip_model_name: str = "OFA-Sys/chinese-clip-vit-base-patch16",
) -> float:
    """Compute mean CLIPScore using Chinese-CLIP.

    Args:
        hypotheses: List of generated captions.
        images: List of PIL Images or file paths.
        clip_model_name: Chinese-CLIP HuggingFace model ID.

    Returns:
        Mean cosine similarity between image and text embeddings, scaled to [0, 100].
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(clip_model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(clip_model_name, trust_remote_code=True)
    model.eval()

    from PIL import Image as PILImage
    from transformers import CLIPProcessor

    try:
        processor = CLIPProcessor.from_pretrained(clip_model_name, trust_remote_code=True)
    except Exception:
        processor = None

    scores = []
    with torch.no_grad():
        for hyp, img in zip(hypotheses, images):
            if isinstance(img, (str, bytes)):
                img = PILImage.open(img).convert("RGB")

            text_inputs = tokenizer([hyp], return_tensors="pt", padding=True, truncation=True, max_length=64)
            text_feat = model.get_text_features(**text_inputs)
            text_feat = F.normalize(text_feat, p=2, dim=-1)

            if processor is not None:
                vision_inputs = processor(images=img, return_tensors="pt")
                image_feat = model.get_image_features(**vision_inputs)
            else:
                # Fallback: use model.visual_projection if available
                raise NotImplementedError("CLIPProcessor required for image encoding")

            image_feat = F.normalize(image_feat, p=2, dim=-1)
            sim = (image_feat * text_feat).sum(dim=-1).item()
            scores.append(sim)

    return sum(scores) / max(len(scores), 1) * 100.0
