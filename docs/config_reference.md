# Configuration Reference — formosa-dual

Every key in `RunConfig` (spec §4.1). Defaults match `configs/base.yaml`.
Overrides flow: `base.yaml < profiles/<p>.yaml < experiments/<e>.yaml < --override`.

## `model.*`

| Key | Type | Default | Valid values | Description |
|---|---|---|---|---|
| `model.name` | str | `Qwen/Qwen2.5-VL-7B-Instruct` | HF model id | Backbone identifier. |
| `model.revision` | Optional[str] | `null` | git ref | Pin a specific HF revision. |
| `model.torch_dtype` | str | `bf16` | `bf16` / `fp16` / `fp32` | Compute dtype. |
| `model.attn_implementation` | str | `sdpa` | `sdpa` / `flash_attention_2` / `eager` | Attention backend. |
| `model.freeze_vit` | bool | `true` | — | Freeze the vision encoder. |
| `model.freeze_merger` | bool | `true` | — | Freeze the vision-LM merger. |
| `model.unfreeze_vit_last_n` | int | `0` | ≥0 | Unfreeze the last N ViT transformer layers. |

## `lora.*`

| Key | Type | Default | Valid values | Description |
|---|---|---|---|---|
| `lora.enabled` | bool | `true` | — | Apply LoRA to the LM. |
| `lora.r` | int | `32` | ≥1 | LoRA rank. |
| `lora.alpha` | int | `64` | ≥1 | LoRA alpha. |
| `lora.dropout` | float | `0.05` | [0,1) | LoRA dropout. |
| `lora.target_modules` | list[str] | `[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]` | — | LM target modules. |
| `lora.bias` | str | `none` | `none` / `all` / `lora_only` | LoRA bias policy. |

## `aux.*`

| Key | Type | Default | Valid values | Description |
|---|---|---|---|---|
| `aux.proj_dim` | int | `256` | ≥1 | Shared contrastive embedding dimension. |
| `aux.pooler_type` | str | `attention` | `mean` / `attention` | Visual-token pooling strategy. |
| `aux.pooler_num_heads` | int | `8` | ≥1 | Heads in attention pooler. |
| `aux.proj_hidden` | int | `1024` | ≥1 | Hidden width of projection head. |
| `aux.tag_init` | str | `chinese_clip` | `random` / `lm_token_avg` / `chinese_clip` | Tag embedding initialiser. |
| `aux.chinese_clip_model` | str | `OFA-Sys/chinese-clip-vit-base-patch16` | HF model id | Chinese-CLIP base model. |
| `aux.freeze_tag_base` | bool | `true` | — | Freeze the base CLIP tag embeddings. |

## `contrastive.*`

| Key | Type | Default | Valid values | Description |
|---|---|---|---|---|
| `contrastive.enabled` | bool | `true` | — | Enable the contrastive loss. |
| `contrastive.lambda_value` | float | `0.2` | ≥0 | λ peak. |
| `contrastive.lambda_schedule` | str | `warmup` | `constant` / `warmup` / `warmup_anneal` | λ trajectory. |
| `contrastive.lambda_warmup_ratio` | float | `0.1` | [0,1] | Fraction of total steps spent warming up λ. |
| `contrastive.lambda_anneal_ratio` | float | `0.0` | [0,1] | Final fraction of steps for cosine anneal. |
| `contrastive.lambda_floor` | float | `0.0` | ≥0 | Floor value of λ after anneal. |
| `contrastive.tau` | float | `0.07` | >0 | InfoNCE temperature. |
| `contrastive.negatives_per_image` | int | `256` | ≥1 | M for each anchor. |
| `contrastive.neg_sampling` | str | `uniform` | `uniform` / `inverse_freq` / `hard` | Negative sampling strategy. |
| `contrastive.hard_neg_refresh_every_steps` | int | `200` | ≥1 | Refresh interval for `hard`. |

## `caption.*`

| Key | Type | Default | Valid values | Description |
|---|---|---|---|---|
| `caption.enabled` | bool | `true` | — | Enable the caption loss. |
| `caption.max_caption_tokens` | int | `384` | ≥1 | Max LM caption length. |
| `caption.label_smoothing` | float | `0.0` | [0,1) | CE label smoothing. |

## `data.*`

| Key | Type | Default | Description |
|---|---|---|---|
| `data.train_manifest` | str | required | Path to JSONL train manifest. |
| `data.val_manifest` | str | required | Path to JSONL validation manifest. |
| `data.test_manifests` | dict[str,str] | `{}` | Map name → JSONL path. |
| `data.vocab_path` | str | required | Path to `vocab_T_*.json`. |
| `data.image_root` | str | required | Image root directory. |
| `data.max_pixels` | int | `802816` | Max pixel budget (1024·28·28). |
| `data.min_pixels` | int | `200704` | Min pixel budget (256·28·28). |
| `data.num_workers` | int | `4` | DataLoader workers. |
| `data.pin_memory` | bool | `true` | DataLoader `pin_memory`. |
| `data.curriculum` | dict | `null` | V4 only — phase schedule. |

## `optim.*`

| Key | Type | Default | Description |
|---|---|---|---|
| `optim.lr_lora` | float | `2e-4` | LoRA learning rate. |
| `optim.lr_aux` | float | `1e-3` | Aux modules learning rate. |
| `optim.weight_decay_lora` | float | `0.0` | WD on LoRA params. |
| `optim.weight_decay_aux` | float | `0.05` | WD on aux params. |
| `optim.weight_decay_tag_proj` | float | `0.1` | WD on tag projector. |
| `optim.optimizer` | str | `adamw` | `adamw` / `adamw_8bit` (GB10 only). |
| `optim.scheduler` | str | `cosine` | `cosine` / `linear` / `constant_with_warmup`. |
| `optim.warmup_ratio` | float | `0.05` | Fraction of steps used for warmup. |
| `optim.adam_beta1` | float | `0.9` | β1. |
| `optim.adam_beta2` | float | `0.95` | β2. |
| `optim.adam_epsilon` | float | `1e-8` | ε. |
| `optim.max_grad_norm` | float | `1.0` | Clip-grad norm; ≤0 disables. |

## `training.*`

| Key | Type | Default | Description |
|---|---|---|---|
| `training.num_epochs` | int | `3` | Total epochs. |
| `training.per_device_batch_size` | int | `2` | Micro-batch per device. |
| `training.gradient_accumulation_steps` | int | `16` | Grad accumulation. |
| `training.gradient_checkpointing` | bool | `true` | Always on per spec §1. |
| `training.seed` | int | `42` | Global seed. |
| `training.eval_steps` | int | `500` | Eval frequency. |
| `training.save_steps` | int | `1000` | Checkpoint frequency. |
| `training.logging_steps` | int | `20` | Per-step metric log frequency. |
| `training.save_total_limit` | int | `3` | Rolling checkpoint window. |
| `training.early_stopping_patience` | int | `0` | Disabled when 0. |

## `device.*`

| Key | Type | Default | Description |
|---|---|---|---|
| `device.auto_detect` | bool | `true` | Use cuda > mps > cpu when no `force`. |
| `device.force` | Optional[str] | `null` | `cuda` / `mps` / `cpu`. |
| `device.mixed_precision` | str | `bf16` | `bf16` / `fp16` / `no`. |

## `logging.*`

| Key | Type | Default | Description |
|---|---|---|---|
| `logging.backend` | str | `tensorboard` | `wandb` / `tensorboard` / `none`. |
| `logging.project` | str | `formosa-dual` | wandb project. |
| `logging.run_name` | Optional[str] | `null` | Run name. |
| `logging.output_dir` | str | `outputs/default` | Output base directory. |

## `smoke.*`

| Key | Type | Default | Description |
|---|---|---|---|
| `smoke.enabled` | bool | `false` | Toggle smoke mode. |
| `smoke.max_train_samples` | int | `16` | Cap train set. |
| `smoke.max_eval_samples` | int | `8` | Cap eval set. |
| `smoke.max_steps` | int | `5` | Cap total optimizer steps. |
| `smoke.max_pixels_override` | Optional[int] | `200704` | Smaller pixel budget. |
| `smoke.max_caption_tokens_override` | Optional[int] | `64` | Shorter captions. |
| `smoke.skip_eval` | bool | `false` | Skip eval at end of smoke run. |

## Cross-key validators

- `at_least_one_loss_enabled` — at least one of `caption.enabled` /
  `contrastive.enabled` must be true. The `--dry-run` flag of `train_dual.py`
  bypasses this validator with a warning so V0 zero-shot configs validate.
