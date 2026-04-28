"""formosa_dual.training.trainer — DualTrainer training loop.

Uses HuggingFace Accelerate for device + gradient accumulation abstraction.
Does NOT subclass transformers.Trainer.

Public methods:
    train() -> None
    evaluate(test_set_name: str) -> dict
    save_checkpoint(name: str) -> Path
    load_checkpoint(path: Path) -> None
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from formosa_dual.config.schema import RunConfig
from formosa_dual.data.tag_vocab import TagVocabulary
from formosa_dual.losses.dual_objective import DualObjectiveLoss
from formosa_dual.training.callbacks import CheckpointCallback, EvalCallback, LoggingCallback
from formosa_dual.training.checkpoint import load_checkpoint, save_checkpoint
from formosa_dual.utils.logging import get_logger
from formosa_dual.utils.seeding import set_seed

logger = get_logger(__name__)


class DualTrainer:
    """Orchestrates the dual-objective training loop.

    Args:
        cfg: Full :class:`~formosa_dual.config.schema.RunConfig`.
        model: :class:`~formosa_dual.models.dual_model.DualObjectiveModel`.
        loss_fn: :class:`~formosa_dual.losses.dual_objective.DualObjectiveLoss`.
        train_loader: Training :class:`~torch.utils.data.DataLoader`.
        val_loader: Validation :class:`~torch.utils.data.DataLoader`.
        accelerator: ``accelerate.Accelerator`` instance.
        vocab: :class:`~formosa_dual.data.tag_vocab.TagVocabulary`.
    """

    def __init__(
        self,
        cfg: RunConfig,
        model,
        loss_fn: DualObjectiveLoss,
        train_loader: DataLoader,
        val_loader: DataLoader,
        accelerator: Any,
        vocab: TagVocabulary,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.accelerator = accelerator
        self.vocab = vocab

        set_seed(cfg.training.seed)

        # Build optimizer and scheduler
        self.optimizer = _build_optimizer(cfg, model)
        total_steps = _compute_total_steps(cfg, train_loader)
        self.scheduler = _build_scheduler(cfg, self.optimizer, total_steps)

        # Prepare with accelerator
        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.scheduler,
        ) = accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler
        )

        # Internal state
        self._global_step: int = 0
        self._current_epoch: int = 0
        self._last_val_metrics: dict = {}
        self._best_metric: float | None = None

        # Callbacks
        output_dir = Path(cfg.logging.output_dir) / (cfg.logging.run_name or "run")
        best_key, higher = _best_metric_key(cfg)
        self._callbacks = [
            LoggingCallback(logging_steps=cfg.training.logging_steps),
            EvalCallback(eval_steps=cfg.training.eval_steps),
            CheckpointCallback(
                save_steps=cfg.training.save_steps,
                save_total_limit=cfg.training.save_total_limit,
                output_dir=output_dir,
                best_metric_key=best_key,
                higher_is_better=higher,
            ),
        ]

        logger.info(
            "DualTrainer initialised: total_steps=%d, output_dir=%s",
            total_steps,
            output_dir,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop."""
        cfg = self.cfg
        smoke = cfg.smoke.enabled
        max_steps = cfg.smoke.max_steps if smoke else None

        for epoch in range(cfg.training.num_epochs):
            self._current_epoch = epoch
            logger.info("Epoch %d / %d", epoch + 1, cfg.training.num_epochs)

            for batch in self.train_loader:
                self._global_step += 1

                # Hard negative refresh
                if (
                    cfg.contrastive.enabled
                    and cfg.contrastive.neg_sampling == "hard"
                    and self._global_step % cfg.contrastive.hard_neg_refresh_every_steps == 0
                ):
                    self._refresh_hard_negatives()

                # Forward + loss
                with self.accelerator.accumulate(self.model):
                    model_output = self.model(batch)
                    loss_dict = self.loss_fn(model_output, batch, self._global_step)
                    loss = loss_dict["loss"]

                    if not loss.isfinite():
                        logger.error("NaN/Inf loss detected at step %d — aborting.", self._global_step)
                        sys.exit(2)

                    self.accelerator.backward(loss)

                    if cfg.optim.max_grad_norm > 0:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), cfg.optim.max_grad_norm
                        )

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                # Build metrics dict for callbacks
                metrics = {
                    "loss": loss.item(),
                    "loss_caption": _to_scalar(loss_dict["loss_caption"]),
                    "loss_contrast": _to_scalar(loss_dict["loss_contrast"]),
                    "lambda": loss_dict["lambda"],
                    "epoch": epoch + 1,
                }
                for g in self.optimizer.param_groups:
                    if "name" in g:
                        metrics[f"lr_{g['name']}"] = g["lr"]

                # Fire callbacks
                for cb in self._callbacks:
                    if hasattr(cb, "on_step_end"):
                        cb.on_step_end(self, self._global_step, metrics)

                # Smoke cap
                if max_steps is not None and self._global_step >= max_steps:
                    logger.info("Smoke mode: max_steps=%d reached, stopping.", max_steps)
                    break

            else:
                # Epoch exhausted without smoke break
                continue
            # Smoke break propagated
            break

        # Final eval + checkpoint
        val_metrics = self.evaluate("val")
        self._last_val_metrics = val_metrics
        for cb in self._callbacks:
            if hasattr(cb, "on_train_end"):
                cb.on_train_end(self, self._global_step)
        logger.info("Training complete at step %d.", self._global_step)

    def evaluate(self, test_set_name: str = "val") -> dict:
        """Run evaluation over the validation loader.

        Args:
            test_set_name: Name tag used for logging (e.g. ``"val"``).

        Returns:
            Dict with keys: ``val_loss_caption``, ``val_loss_contrast``,
            ``val_perplexity``, ``val_retrieval_r5``.
        """
        self.model.eval()
        total_loss_caption = 0.0
        total_loss_contrast = 0.0
        n_steps = 0

        with torch.no_grad():
            for batch in self.val_loader:
                model_output = self.model(batch)
                loss_dict = self.loss_fn(model_output, batch, self._global_step)
                total_loss_caption += _to_scalar(loss_dict["loss_caption"])
                total_loss_contrast += _to_scalar(loss_dict["loss_contrast"])
                n_steps += 1

        self.model.train()
        n_steps = max(n_steps, 1)
        avg_caption = total_loss_caption / n_steps
        avg_contrast = total_loss_contrast / n_steps

        metrics = {
            "val_loss_caption": avg_caption,
            "val_loss_contrast": avg_contrast,
            "val_perplexity": math.exp(min(avg_caption, 20)),
            "val_retrieval_r5": 0.0,  # ASSUMPTION: full retrieval eval not implemented here
        }
        logger.info(
            "%s | val_loss_caption=%.4f | val_perplexity=%.2f",
            test_set_name,
            avg_caption,
            metrics["val_perplexity"],
        )
        return metrics

    def save_checkpoint(self, name: str) -> Path:
        """Save a named checkpoint.

        Args:
            name: Checkpoint name suffix.

        Returns:
            Path to checkpoint directory.
        """
        output_dir = Path(self.cfg.logging.output_dir) / (self.cfg.logging.run_name or "run")
        return save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            accelerator=self.accelerator,
            cfg=self.cfg,
            step=self._global_step,
            epoch=self._current_epoch,
            best_metric=self._best_metric,
            output_dir=output_dir,
        )

    def load_checkpoint(self, path: Path) -> None:
        """Restore training state from a checkpoint directory.

        Args:
            path: Path to the checkpoint directory.
        """
        state = load_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            checkpoint_dir=path,
            accelerator=self.accelerator,
        )
        self._global_step = state.get("step", 0)
        self._current_epoch = state.get("epoch", 0)
        self._best_metric = state.get("best_metric")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _refresh_hard_negatives(self) -> None:
        """Refresh hard negative index in the NegativeSampler."""
        # ASSUMPTION: NegativeSampler is accessible via train_loader.dataset.sampler
        dataset = self.train_loader.dataset
        sampler = getattr(dataset, "sampler", None)
        if sampler is None:
            return
        if hasattr(sampler, "refresh_hard_neg_index"):
            try:
                sampler.refresh_hard_neg_index(self.model, self.train_loader)
            except NotImplementedError:
                logger.warning("NegativeSampler.refresh_hard_neg_index not implemented; skipping.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_total_steps(cfg: RunConfig, train_loader: DataLoader) -> int:
    steps_per_epoch = math.ceil(len(train_loader) / cfg.training.gradient_accumulation_steps)
    if cfg.smoke.enabled:
        return cfg.smoke.max_steps
    return steps_per_epoch * cfg.training.num_epochs


def _build_optimizer(cfg: RunConfig, model):
    param_groups = model.get_trainable_param_groups()
    if not param_groups:
        # Fallback: all trainable params at lr_lora
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad]}]

    optimizer_name = cfg.optim.optimizer
    if optimizer_name == "adamw_8bit":
        from formosa_dual.training.device import has_bitsandbytes
        if not has_bitsandbytes():
            raise RuntimeError("adamw_8bit requires bitsandbytes which is not installed.")
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(
            param_groups,
            betas=(cfg.optim.adam_beta1, cfg.optim.adam_beta2),
            eps=cfg.optim.adam_epsilon,
        )
    else:
        return torch.optim.AdamW(
            param_groups,
            betas=(cfg.optim.adam_beta1, cfg.optim.adam_beta2),
            eps=cfg.optim.adam_epsilon,
        )


def _build_scheduler(cfg: RunConfig, optimizer, total_steps: int):
    from transformers import get_scheduler
    warmup_steps = int(cfg.optim.warmup_ratio * total_steps)
    return get_scheduler(
        cfg.optim.scheduler,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


def _best_metric_key(cfg: RunConfig) -> tuple[str, bool]:
    """Return (metric_key, higher_is_better)."""
    if cfg.caption.enabled:
        return "val_loss_caption", False
    return "val_retrieval_r5", True


def _to_scalar(val) -> float:
    if isinstance(val, torch.Tensor):
        return val.item()
    return float(val)
