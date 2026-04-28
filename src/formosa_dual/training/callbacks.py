"""formosa_dual.training.callbacks — lightweight training event hooks.

Callbacks are called by :class:`~formosa_dual.training.trainer.DualTrainer`
at the appropriate points. They are plain Python classes (not nn.Module).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from formosa_dual.utils.logging import get_logger

if TYPE_CHECKING:
    from formosa_dual.training.trainer import DualTrainer

logger = get_logger(__name__)


class LoggingCallback:
    """Emit per-step training metrics to the logger.

    Args:
        logging_steps: Emit every N global steps.
    """

    def __init__(self, logging_steps: int = 10) -> None:
        self.logging_steps = logging_steps

    def on_step_end(self, trainer: "DualTrainer", step: int, metrics: dict) -> None:
        if step % self.logging_steps != 0:
            return
        parts = [f"step={step}"]
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        logger.info("train | %s", " | ".join(parts))


class EvalCallback:
    """Trigger validation at regular intervals.

    Args:
        eval_steps: Run eval every N global steps.
    """

    def __init__(self, eval_steps: int = 500) -> None:
        self.eval_steps = eval_steps

    def on_step_end(self, trainer: "DualTrainer", step: int, metrics: dict) -> None:
        if step % self.eval_steps != 0:
            return
        val_metrics = trainer.evaluate("val")
        for k, v in val_metrics.items():
            logger.info("eval | step=%d | %s=%.4f", step, k, v)
        trainer._last_val_metrics = val_metrics


class CheckpointCallback:
    """Save checkpoints at regular intervals.

    Maintains a rolling window of ``save_total_limit`` checkpoints.
    Always writes ``checkpoint-best/`` and ``checkpoint-latest/``.

    Args:
        save_steps: Save every N global steps.
        save_total_limit: Maximum rolling checkpoints to keep (0 = unlimited).
        output_dir: Base output directory.
        best_metric_key: Metric name for best-checkpoint selection.
        higher_is_better: True if higher metric is better.
    """

    def __init__(
        self,
        save_steps: int,
        save_total_limit: int,
        output_dir: Path,
        best_metric_key: str = "val_loss_caption",
        higher_is_better: bool = False,
    ) -> None:
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit
        self.output_dir = Path(output_dir)
        self.best_metric_key = best_metric_key
        self.higher_is_better = higher_is_better
        self._saved_checkpoints: list[Path] = []
        self._best_metric: float | None = None

    def on_step_end(self, trainer: "DualTrainer", step: int, metrics: dict) -> None:
        if step % self.save_steps != 0:
            return
        self._do_save(trainer, step)

    def on_train_end(self, trainer: "DualTrainer", step: int) -> None:
        self._do_save(trainer, step)

    def _do_save(self, trainer: "DualTrainer", step: int) -> None:
        from formosa_dual.training.checkpoint import save_checkpoint
        import shutil

        val_metrics = getattr(trainer, "_last_val_metrics", {})
        metric_val = val_metrics.get(self.best_metric_key)

        ckpt_dir = save_checkpoint(
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            accelerator=trainer.accelerator,
            cfg=trainer.cfg,
            step=step,
            epoch=trainer._current_epoch,
            best_metric=self._best_metric,
            output_dir=self.output_dir,
        )
        self._saved_checkpoints.append(ckpt_dir)

        # Copy to checkpoint-latest/
        latest = self.output_dir / "checkpoint-latest"
        if latest.exists():
            shutil.rmtree(latest)
        shutil.copytree(str(ckpt_dir), str(latest))

        # Best checkpoint
        if metric_val is not None:
            is_better = (
                (self.higher_is_better and metric_val > (self._best_metric or float("-inf")))
                or (not self.higher_is_better and metric_val < (self._best_metric or float("inf")))
            )
            if is_better:
                self._best_metric = metric_val
                best = self.output_dir / "checkpoint-best"
                if best.exists():
                    shutil.rmtree(best)
                shutil.copytree(str(ckpt_dir), str(best))
                logger.info("New best checkpoint (%s=%.4f) saved to %s", self.best_metric_key, metric_val, best)

        # Rolling window eviction
        if self.save_total_limit > 0 and len(self._saved_checkpoints) > self.save_total_limit:
            evict = self._saved_checkpoints.pop(0)
            if evict.exists():
                shutil.rmtree(evict)
                logger.info("Evicted old checkpoint: %s", evict)
