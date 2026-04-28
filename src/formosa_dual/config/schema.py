"""formosa_dual.config.schema — Pydantic configuration models.

Every field, default, and validator is derived directly from the construction
spec §4.1.  Do not alter types, defaults, or validators without spec approval.
"""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _StrictBase(BaseModel):
    """Base model that rejects unknown keys."""

    model_config = ConfigDict(extra="forbid")


class ModelConfig(_StrictBase):
    name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    revision: Optional[str] = None
    torch_dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    attn_implementation: Literal["sdpa", "flash_attention_2", "eager"] = "sdpa"
    freeze_vit: bool = True
    freeze_merger: bool = True
    unfreeze_vit_last_n: int = 0


class LoRAConfig(_StrictBase):
    enabled: bool = True
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05
    target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    bias: Literal["none", "all", "lora_only"] = "none"


class AuxModulesConfig(_StrictBase):
    proj_dim: int = 256
    pooler_type: Literal["mean", "attention"] = "attention"
    pooler_num_heads: int = 8
    proj_hidden: int = 1024
    tag_init: Literal["random", "lm_token_avg", "chinese_clip"] = "chinese_clip"
    chinese_clip_model: str = "OFA-Sys/chinese-clip-vit-base-patch16"
    freeze_tag_base: bool = True


class ContrastiveConfig(_StrictBase):
    enabled: bool = True
    lambda_value: float = 0.2
    lambda_schedule: Literal["constant", "warmup", "warmup_anneal"] = "warmup"
    lambda_warmup_ratio: float = 0.1
    lambda_anneal_ratio: float = 0.0
    lambda_floor: float = 0.0
    tau: float = 0.07
    negatives_per_image: int = 256
    neg_sampling: Literal["uniform", "inverse_freq", "hard"] = "uniform"
    hard_neg_refresh_every_steps: int = 200


class CaptionConfig(_StrictBase):
    enabled: bool = True
    max_caption_tokens: int = 384
    label_smoothing: float = 0.0


class DataConfig(_StrictBase):
    train_manifest: str
    val_manifest: str
    test_manifests: dict[str, str] = {}
    vocab_path: str
    image_root: str
    max_pixels: int = 802816  # 1024*28*28
    min_pixels: int = 200704  # 256*28*28
    num_workers: int = 4
    pin_memory: bool = True
    curriculum: Optional[dict] = None  # populated only by V4


class OptimConfig(_StrictBase):
    lr_lora: float = 2e-4
    lr_aux: float = 1e-3
    weight_decay_lora: float = 0.0
    weight_decay_aux: float = 0.05
    weight_decay_tag_proj: float = 0.1
    optimizer: Literal["adamw", "adamw_8bit"] = "adamw"
    scheduler: Literal["cosine", "linear", "constant_with_warmup"] = "cosine"
    warmup_ratio: float = 0.05
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0


class TrainingConfig(_StrictBase):
    num_epochs: int = 3
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    gradient_checkpointing: bool = True
    seed: int = 42
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 20
    save_total_limit: int = 3
    early_stopping_patience: int = 0


class DeviceConfig(_StrictBase):
    auto_detect: bool = True
    force: Optional[Literal["cuda", "mps", "cpu"]] = None
    mixed_precision: Literal["bf16", "fp16", "no"] = "bf16"


class LoggingConfig(_StrictBase):
    backend: Literal["wandb", "tensorboard", "none"] = "tensorboard"
    project: str = "formosa-dual"
    run_name: Optional[str] = None
    output_dir: str = "outputs/default"


class SmokeConfig(_StrictBase):
    enabled: bool = False
    max_train_samples: int = 16
    max_eval_samples: int = 8
    max_steps: int = 5
    max_pixels_override: Optional[int] = 200704
    max_caption_tokens_override: Optional[int] = 64
    skip_eval: bool = False


class RunConfig(_StrictBase):
    profile: Literal["dev_mac", "dev_smoke", "prod_gb10"] = "prod_gb10"
    experiment: str = "v3_hero"

    model: ModelConfig
    lora: LoRAConfig
    aux: AuxModulesConfig
    contrastive: ContrastiveConfig
    caption: CaptionConfig
    data: DataConfig
    optim: OptimConfig
    training: TrainingConfig
    device: DeviceConfig
    logging: LoggingConfig
    smoke: SmokeConfig = Field(default_factory=SmokeConfig)

    @model_validator(mode="after")
    def at_least_one_loss_enabled(self) -> "RunConfig":
        if not self.contrastive.enabled and not self.caption.enabled:
            raise ValueError("At least one of caption/contrastive must be enabled")
        return self
