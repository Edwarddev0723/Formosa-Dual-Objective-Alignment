# Failure Modes — formosa-dual

Operational debugging table mapping symptoms to diagnoses and fixes.
Each row references the relevant spec section.

| Symptom | Diagnosis | Fix |
|---|---|---|
| `L_con` collapses to ~0 in epoch 1, accuracy on retrieval drops | Contrastive collapse: visual tokens map to a single point; the projection head loses rank. (Spec §1, §5.16.) | Lower `contrastive.tau` from 0.07 → 0.05; raise `contrastive.lambda_value` slightly; verify `aux.proj_dim=256`; check `pos_tag_mask` is non-empty. |
| `L_cap` is consistently higher than the V1 (caption-only) baseline | Contrastive interference: λ too high or warmup too short. (Spec §1, §5.17.) | Reduce `contrastive.lambda_value` (0.2 → 0.1); increase `contrastive.lambda_warmup_ratio` to 0.2. |
| V3 strong on `test_id`, weak on `test_source_holdout` | Source-specific overfit: training distribution dominated by one corpus. (Spec §6.5.) | Inspect `splits_manifest.json`; rebalance via `scripts/build_splits.py --stratify-by source`. |
| Auto Culturalness rises but human eval is flat | "Cultural-sounding" without grounding: model parroting tag tokens. (Spec §5.21.) | Inspect F1 vs S_IDF vs E_NLI individually; if E_NLI flat, claims aren't entailed by article — revisit retrieval target or expand premise. |
| First-step `NaN` loss on MPS | bf16 instability on Apple Silicon for some attention shapes. (Spec §7.3.) | Set `device.mixed_precision: "no"` in the active profile (already set in `dev_smoke`). |
| `OOM` on Mac at `per_device_batch_size=1` | Image tokens × LM hidden state too large. (Spec §7.4.) | Lower `smoke.max_pixels_override` (200704 → 50176); confirm `gradient_checkpointing=true`. |
| Tokenizer hangs or crashes mid-run on Mac | `num_workers>0` is unstable on macOS. (Spec §9.5.) | Set `data.num_workers: 0`. |
| `AutoVideoProcessor requires the Torchvision library` | `torchvision` missing in active env. (Spec §7.5 `requirements/base.txt`.) | `pip install -r requirements/mac.txt`. |
| `flash-attn` import error on Mac | flash-attn is CUDA-only. (Spec §7.4.) | Set `model.attn_implementation: sdpa`; the device validator does this automatically (`validation.validate_config_for_device`). |
| 8-bit optimizer error on Mac | `bitsandbytes` is CUDA-only. (Spec §7.4.) | Set `optim.optimizer: adamw`; `validate_config_for_device` raises `ConfigError` if `adamw_8bit` is requested on MPS. |
| Best metric never updates / `checkpoint-best` missing | Caption disabled (V2/V0): the trainer falls back to `val_retrieval_r5` (higher = better). (Spec §5.19 contract 4.) | If retrieval R@5 is also 0, the eval loop is not feeding retrieval inputs; check `cfg.contrastive.enabled` for the V2 case. |
| `at_least_one_loss_enabled` validator raises for `v0_zero_shot` | Both losses disabled by design. (Spec §4.3 V0 note.) | Use `--dry-run` (the loader allows the bypass with a warning) or evaluate via `eval/zero_shot.py` instead of `train_dual.py`. |
| Resuming from a checkpoint silently re-randomises | RNG state file missing or path wrong. (Spec §5.20.) | Confirm `rng_state.pt` exists in checkpoint dir; pass the full directory path to `--resume-from`. |
| `pyahocorasick` build error during `pip install` | Xcode CLT missing. | `xcode-select --install` then re-install. |
| `HF_HOME` runs out of space mid-download | The 7B Qwen2.5-VL weights are ~16 GB. | `export HF_HOME=/path/with/space` before re-running `scripts/download_models.py`. |
