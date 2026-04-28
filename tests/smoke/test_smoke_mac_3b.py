"""tests/smoke/test_smoke_mac_3b.py — real Qwen2.5-VL-3B loads and takes 1 step.

Must complete in <5 minutes on Mac.
Marked slow — not run in CI by default.
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip(
    "formosa_dual.utils.synthetic",
    reason="legacy smoke test depends on a synthetic helper that no longer exists; "
    "use tests/smoke/test_smoke_mac_synthetic.py instead",
)
# Skip at collection if the legacy helper is missing.
try:
    from formosa_dual.utils.synthetic import make_synthetic_dataset  # type: ignore # noqa: F401
except ImportError:
    pytest.skip(
        "make_synthetic_dataset removed; superseded by test_smoke_mac_synthetic.py",
        allow_module_level=True,
    )


@pytest.mark.smoke
@pytest.mark.slow
def test_mac_3b_loads_and_steps(tmp_path):
    """Real backbone loads, takes one forward+backward step, saves+loads checkpoint."""
    from formosa_dual.config.schema import (
        RunConfig, ModelConfig, LoRAConfig, AuxModulesConfig,
        ContrastiveConfig, CaptionConfig, DataConfig, OptimConfig,
        TrainingConfig, DeviceConfig, LoggingConfig, SmokeConfig,
    )
    from formosa_dual.utils.seeding import set_seed
    from formosa_dual.utils.synthetic import make_synthetic_dataset
    from formosa_dual.data.manifest import write_manifest
    from formosa_dual.data.tag_vocab import TagVocabulary
    from formosa_dual.data.negative_sampler import NegativeSampler
    from formosa_dual.data.dataset import FormosaDataset
    from formosa_dual.data.collator import DualCollator
    from formosa_dual.models.dual_model import DualObjectiveModel
    from formosa_dual.losses.dual_objective import DualObjectiveLoss
    from formosa_dual.training.checkpoint import save_checkpoint, load_checkpoint
    from torch.utils.data import DataLoader
    from transformers import AutoProcessor

    set_seed(42)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    vocab, records = make_synthetic_dataset(num_samples=4, num_tags=16, output_dir=data_dir, seed=42)
    vocab_path = data_dir / "vocab.json"
    vocab.save(vocab_path)
    manifest_path = data_dir / "train.jsonl"
    write_manifest(records, manifest_path)

    cfg = RunConfig(
        model=ModelConfig(
            name="Qwen/Qwen2.5-VL-3B-Instruct",
            freeze_vit=True,
            freeze_merger=True,
            lora=LoRAConfig(r=4, alpha=8, dropout=0.05),
            aux_modules=AuxModulesConfig(
                attention_pooler=True,
                projection_head=True,
                tag_projector=True,
                tag_proj_clip_model="OFA-Sys/chinese-clip-vit-base-patch16",
                projection_dim=256,
            ),
        ),
        contrastive=ContrastiveConfig(
            enabled=True,
            tau=0.07,
            num_negatives=2,
            neg_sampling="uniform",
            lambda_schedule="constant",
            lambda_value=0.5,
        ),
        caption=CaptionConfig(enabled=True, label_smoothing=0.0),
        data=DataConfig(
            vocab_path=str(vocab_path),
            train_manifest=str(manifest_path),
            val_manifest=str(manifest_path),
            image_root=str(data_dir / "images"),
            max_caption_tokens=64,
            max_pos_tags=4,
            num_workers=0,
        ),
        optim=OptimConfig(lr=1e-4, weight_decay=0.01, scheduler="constant_with_warmup", warmup_steps=0),
        training=TrainingConfig(num_epochs=1, per_device_batch_size=1, gradient_accumulation_steps=1, seed=42),
        device=DeviceConfig(mixed_precision="no"),
        logging=LoggingConfig(output_dir=str(tmp_path / "output"), run_name="3b_smoke"),
        smoke=SmokeConfig(enabled=True, max_steps=1),
    )

    processor = AutoProcessor.from_pretrained(cfg.model.name, trust_remote_code=True)
    sampler = NegativeSampler(vocab=vocab, strategy="uniform", num_negatives=2, seed=42)
    collator = DualCollator(
        processor=processor,
        vocab=vocab,
        negative_sampler=sampler,
        max_caption_tokens=64,
        max_pos_tags=4,
    )

    ds = FormosaDataset(manifest_path=manifest_path, vocab=vocab, image_root=data_dir / "images",
                        smoke_max_samples=4)
    loader = DataLoader(ds, batch_size=1, collate_fn=collator)

    model = DualObjectiveModel(cfg=cfg, vocab=vocab, processor=processor)
    loss_fn = DualObjectiveLoss(cfg=cfg, total_steps=1)

    batch = next(iter(loader))
    out = model(batch)
    loss_dict = loss_fn(out, batch, step=0)
    loss_dict["loss"].backward()

    assert torch.isfinite(loss_dict["loss"]), "Loss must be finite."

    # Checkpoint roundtrip
    ckpt_dir = tmp_path / "ckpt"
    save_checkpoint(model=model, optimizer=None, scheduler=None, training_state={"step": 1, "epoch": 0},
                    cfg=cfg, checkpoint_dir=ckpt_dir)
    load_checkpoint(model=model, optimizer=None, scheduler=None, checkpoint_dir=ckpt_dir)
