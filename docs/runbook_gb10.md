# GB10 Runbook — formosa-dual

This runbook covers production training on a GB10 (Blackwell) system.
For Mac development, see [runbook_mac.md](runbook_mac.md).

## Option A — Native conda install

```bash
conda create -n formosa-dual python=3.11 -y
conda activate formosa-dual

pip install -r requirements/gb10.txt   # includes bitsandbytes, flash-attn, vllm, wandb
pip install -e .
```

If `flash-attn` build fails, retry with:

```bash
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

## Option B — NeMo container install

```bash
docker run --gpus all -it --rm \
    -v $PWD:/workspace -w /workspace \
    nvcr.io/nvidia/nemo:24.07
pip install -r requirements/gb10.txt
pip install -e .
```

## Verify

```bash
python scripts/verify_environment.py
```

Expected: exit 0 with `device=cuda`, `compute_capability=9.0` or `12.x`
(depending on the GB10 software stack), `bf16_supported=True`,
`bitsandbytes_available=True`, `flash_attn_available=True`.

On GB10-class Blackwell systems, `compute_capability=12.x` is normal.
If `flash_attn_available=False`, use SDPA until the local CUDA / flash-attn
stack supports the installed driver:

```bash
python train_dual.py --profile prod_gb10 --experiment v3_hero --smoke \
    --override model.attn_implementation=sdpa logging.backend=none
```

## First Production Run

```bash
# 1. Pre-fetch backbone + Chinese-CLIP into HF cache
python scripts/download_models.py \
    --models Qwen/Qwen2.5-VL-7B-Instruct OFA-Sys/chinese-clip-vit-base-patch16

# 2. Build vocab + annotations + splits (run once per dataset version)
python scripts/prepare_hf_dataset.py \
    --dataset renhehuang/formosa-vlm-caption-v1 \
    --split train \
    --output-dir data/raw

python scripts/build_tag_vocab.py \
    --tier1 data/sources/tier1.txt \
    --tier2 data/sources/tier2_*.txt \
    --tier3-from-captions data/raw/captions.txt \
    --target-size 800 --min-freq 5 \
    --output data/vocab/vocab_T_v1.json
# If there are no tier-2 tag files yet, omit the "--tier2 ..." line.
# V3 needs non-empty culture tags; for caption-only sanity checks use v1_caption_only.

python scripts/annotate_tags.py \
    --input data/raw/manifest.jsonl \
    --vocab data/vocab/vocab_T_v1.json \
    --use-aho-corasick --use-metadata --max-tags 10 \
    --output data/annotated/manifest.jsonl --num-workers 8

python scripts/build_splits.py \
    --annotations data/annotated/manifest.jsonl \
    --train-ratio 0.80 --dev-ratio 0.10 --test-ratio 0.10 \
    --group-by article_url --stratify-by source \
    --output-dir data/splits

# 3. Hero training (V3)
python train_dual.py --profile prod_gb10 --experiment v3_hero
```

## Monitoring

- `wandb` is enabled by `prod_gb10.yaml`. Make sure `WANDB_API_KEY` is set.
- Use `nvidia-smi` cautiously: GB10's unified memory makes `Used` numbers less
  meaningful than `free -h` for the host.
- Per-step training metrics (`loss`, `loss_caption`, `loss_contrast`, `lambda`,
  `lr_lora`, `lr_aux`) are emitted every `cfg.training.logging_steps`.

## Resume Training

```bash
python train_dual.py --profile prod_gb10 --experiment v3_hero \
    --resume-from outputs/v3_hero/checkpoint-1000
```

`load_checkpoint` restores: PEFT adapter, aux modules (pooler, proj_head,
tag_projector.projector), optimizer, scheduler, RNG state, and
`training_state.json`.
