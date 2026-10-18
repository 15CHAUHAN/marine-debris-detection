---
name: cli-and-config
description: Structure every script in scripts/ as a deterministic, flag-driven CLI with validated YAML config, explicit seeds, logging, and exit codes.
---

# CLI & Config Conventions

## 1. Every `scripts/` Entrypoint Gets the Same Shape
The project has 30+ scripts under `scripts/` (`train_one.py`, `evaluate_test.py`,
`prepare_taco.py`, `make_demo_video.py`, ...). They must be callable, scriptable,
and reproducible without editing source:

```python
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, overrides=args.set)
    ...
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

- `main()` returns an int exit code; `raise SystemExit` propagates it.
- `argv` is injectable so tests can call `main(["--config", ...])`.
- No argument parsing at import time; importing a script must be side-effect free.

## 2. Standard Flags (adopt these names everywhere)
| Flag | Purpose |
|---|---|
| `--config PATH` | YAML config (required for training/eval scripts) |
| `--set k=v` (repeatable) | override a config leaf without editing the file |
| `--out DIR` | artifact directory; default under `artifacts/` |
| `--seed INT` | default `0`; seeds torch/numpy/random |
| `--device {cpu,cuda}` | default `cuda` if available, else `cpu` |
| `--dry-run` | validate config + data paths, print the plan, exit 0 |
| `--force` | allow overwriting an existing artifact directory |
| `--verbose` / `-v` | DEBUG logging |
| `--limit INT` | cap samples/epochs for a smoke test |

Consistency matters more than elegance: a flag present in one script should mean
the same thing in every script.

## 3. Config Files
- YAML only, one file per experiment, under `configs/` (create it).
- Validate the parsed mapping against an explicit schema and raise on unknown keys
  — a typo'd key that is silently ignored is a wasted GPU hour.
- The config must fully determine the run. If a value is not in the config, it must
  be a constant in `src/`, not a flag-default hidden in the script.
- Never put absolute machine paths in a config; use paths relative to the project
  root and resolve them once at load time.

```yaml
# configs/unet_marida.yaml
seed: 0
data:
  root: .            # relative to project root
  splits: splits/
  size: 256
  num_classes: 11
model:
  name: unet_attention
  encoder: resnet34
train:
  epochs: 60
  batch_size: 8
  lr: 0.0003
  amp: true
out: artifacts/unet_marida
```

## 4. Determinism
- `--seed` sets `random`, `numpy`, `torch.manual_seed`, and
  `torch.cuda.manual_seed_all`, and writes the seed into the run manifest.
- Note in the run manifest when `torch.use_deterministic_algorithms(True)` is on,
  because it can raise for some ops; do not enable it silently.
- Report in the manifest whether cuDNN benchmarking was enabled
  (`torch.backends.cudnn.benchmark`) since it changes run-to-run variance.

## 5. Paths
- One project-root constant per script: `ROOT = Path(__file__).resolve().parents[1]`.
- Resolve every input/output path from it; accept overrides as `Path`.
- Fail fast with a clear message when an input path is missing, listing the path
  and how to prepare it (e.g. "run scripts/prepare_taco.py first").

## 6. Logging, Not `print`
- `logging.getLogger(__name__)`, configured once in `main()` with a level from `-v`.
- Log: config path, resolved output dir, device, dataset sizes, and per-epoch
  metrics. Keep prints out of `src/` modules entirely.
- Write machine-readable results to files (`metrics.json`, `run.json`); logs are
  for humans and must not be the only record.

## 7. Exit Codes
| Code | Meaning |
|---|---|
| 0 | success |
| 1 | config/schema error |
| 2 | missing input data or artifact |
| 3 | runtime/compute failure |

Scripts chained in PowerShell (`if ($LASTEXITCODE -eq 0)`) depend on this being
consistent.

## 8. Anti-Patterns
- Hardcoded absolute paths or dates in scripts (`D:\TECH-ATHON\artifacts\run3`).
- Overwriting an existing run directory without `--force`.
- `sys.argv` indexing instead of argparse.
- Config keys read but never used, or used but never documented in the template.
- Wrapping argparse defaults in `if args.x is None: args.x = CONFIG_DEFAULT` —
  ambiguous precedence. Define one precedence: CLI > config > code default.
