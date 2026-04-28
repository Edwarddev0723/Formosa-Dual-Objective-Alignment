# Formosa-Dual

Formosa-VLM dual-objective alignment: caption SFT + cultural contrastive auxiliary.

## Quick Start (Mac smoke)

```bash
pip install -e ".[dev]"
python scripts/verify_environment.py
python scripts/make_synthetic_data.py --num-train 8 --num-val 4 --output-dir data/synthetic
python train_dual.py --profile dev_smoke --experiment v3_hero --smoke
```

## Quick Start (GB10 production)

```bash
pip install -r requirements/gb10.txt
pip install -e .
python scripts/verify_environment.py
python scripts/download_models.py --models Qwen/Qwen2.5-VL-7B-Instruct OFA-Sys/chinese-clip-vit-base-patch16
python train_dual.py --profile prod_gb10 --experiment v3_hero
```

## Repository Layout

```
formosa-dual/
├── configs/          # YAML configs (base, profiles, experiments, ablations)
├── src/formosa_dual/ # Library package
├── scripts/          # Offline preprocessing and setup scripts
├── train_dual.py     # Single training entrypoint
├── eval/             # Evaluation scripts
├── tests/            # Unit / integration / smoke tests
├── docs/             # Runbooks and references
├── data/             # gitignored; populated by scripts
└── outputs/          # gitignored; written by training
```

## Documentation

- Research spec: `docs/FORMOSA_VLM_DUAL_OBJECTIVE_SPEC.md` (companion; not in this repo)
- Construction spec: `docs/FORMOSA_DUAL_CONSTRUCTION_SPEC.md`
- Mac runbook: `docs/runbook_mac.md`
- GB10 runbook: `docs/runbook_gb10.md`
- Config reference: `docs/config_reference.md`
- Failure modes: `docs/failure_modes.md`

## License

Apache-2.0
