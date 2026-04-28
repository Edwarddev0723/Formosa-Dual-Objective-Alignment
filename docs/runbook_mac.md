# Mac Runbook — formosa-dual

This runbook covers Mac (Apple Silicon) development and smoke testing.
Production training happens on GB10; see [runbook_gb10.md](runbook_gb10.md).

## Prerequisites

- macOS 14 (Sonoma) or later — required for stable `bf16` on MPS.
- Apple Silicon (M-series). Intel Macs are unsupported.
- Python 3.11 (spec §3 `.python-version`).
- Xcode Command Line Tools: `xcode-select --install` (needed by `pyahocorasick`).
- ~8 GB free disk for the 3B Qwen2.5-VL backbone weights and HF cache.

## Setup

Preferred (uv):

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements/mac.txt
uv pip install -e .
```

Pip fallback:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements/mac.txt
pip install -e .
```

## Verify

```bash
python scripts/verify_environment.py
```

Expected: exit 0 with `device=mps`, `bf16_supported=True`,
`bitsandbytes_available=False`, `flash_attn_available=False`.

## Daily Smoke

```bash
python train_dual.py --profile dev_mac --experiment v3_hero --smoke
```

Expected wall-clock: <15 min on M3 Pro / 36 GB.

For the fastest iteration loop, use the synthetic-only path:

```bash
python scripts/make_synthetic_data.py --num-train 8 --num-val 4 --output-dir data/synthetic
pytest tests/smoke/test_smoke_mac_synthetic.py -v
```

This must complete in <60 s and exercises every module on a stub backbone.

## Known Issues

- **MPS memory leak**: long Mac runs may slowly accumulate unreleased memory.
  If observed, set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` and restart between
  experiments.
- **`pyahocorasick` build fails**: install Xcode CLT first; reinstall with
  `pip install --no-binary pyahocorasick pyahocorasick`.
- **HF cache fills the disk**: the 3B model is ~8 GB on disk. Move `HF_HOME`
  to an external SSD if needed: `export HF_HOME=/Volumes/SSD/.cache/huggingface`.
- **Tokenizer hangs**: set `cfg.data.num_workers=0` (already done in `dev_smoke`
  profile). On Mac, multiprocess DataLoader is slower than single-worker.
- **NaN loss on first step**: very rare with bf16 on MPS — fall back to
  `mixed_precision: "no"` (already used by `dev_smoke`).
