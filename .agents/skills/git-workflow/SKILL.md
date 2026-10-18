---
name: git-workflow
description: Initialize and operate the TECH-ATHON repository safely, with a .gitignore that excludes 43 GB of datasets, artifacts, and weights while tracking all source code.
---

# Git Workflow

## 1. Current State
`D:\TECH-ATHON` is **not** a git repository and has no `.gitignore`. The working
tree is 43.79 GB, of which roughly 36 GB is datasets and 13.7 GB is `artifacts/`.
Git is installed (2.53) and the global config already sets `user.name`/`user.email`
and a global git-lfs filter. `gh` (GitHub CLI) is **not** installed.

Never run `git add -A` or `git add .` in this tree. Stage paths explicitly.

## 2. Initialize
```powershell
cd D:\TECH-ATHON
git init -b main
# create .gitignore FIRST (see §3), then:
git add .gitignore AGENTS.md CONTEXT.md requirements.txt src scripts tests dashboard docs splits .agents
git status --short          # review before committing
git commit -m "chore: initialize repository with source, tests and skills"
```

If a stray `git add` already staged large files, reset without touching the working
tree: `git reset` (staged only) — never `git checkout .` or `git clean -fd` here,
because untracked data is irreplaceable.

## 3. Mandatory `.gitignore`
Datasets, artifacts, weights, and the venv must never be tracked. See
`references/gitignore.md` for the full file. Summary of what is excluded:

| Excluded | Size |
|---|---|
| `.venv/` | 5.2 GB (reinstallable from `requirements.txt`) |
| `20210626/`, `patches/`, `MARIDA/`, `marine-debris-fls-datasets-master/`, `taco_raw/`, `TACO-master/`, `utd2.v8i.yolo26/`, `shapefiles/` | ~36 GB |
| `artifacts/`, `runs/`, `weights/` | 13.8 GB |
| `*.pt`, `*.pth`, `*.ckpt`, `*.zip`, `*.mp4`, `*.tif`, `*.tiff`, `*.npy`, `*.npz` | weights/data |
| `__pycache__/`, `.pytest_cache/`, `*.pyc` | build noise |

Tracked: `src/`, `scripts/`, `tests/`, `dashboard/`, `docs/`, `splits/` (manifests
only), `.agents/skills/`, `requirements.txt`, `AGENTS.md`, `CONTEXT.md`.

## 4. Size Guard Before Every Commit
```powershell
git status --short
git diff --cached --stat | Select-Object -Last 5
git ls-files -z | ForEach-Object { $_ -split "`0" } | Where-Object { $_ } |
  ForEach-Object { [PSCustomObject]@{ MB = [math]::Round((Get-Item $_ -ErrorAction SilentlyContinue).Length / 1MB, 2); Path = $_ } } |
  Sort-Object MB -Descending | Select-Object -First 10
```
Any tracked file above ~5 MB is a mistake unless it is deliberately added to
git-lfs. Repository total should stay well under the ~1 GB hosting comfort zone.

## 5. Commit Convention
`type(scope): imperative summary` — scope names the area, not the file path.

| Type | Use |
|---|---|
| `feat` | new model, dataset loader, dashboard tab |
| `fix` | bug fix in pipeline/trainer/dashboard |
| `data` | split manifests, label mapping, ingestion changes |
| `test` | tests only |
| `docs` | README, CONTEXT.md, report text |
| `chore` | deps, config, repo hygiene |
| `refactor` | no behaviour change |

Examples: `feat(models): add unet_effnet_transformer variant`,
`fix(splits): reject scenes crossing train/val boundary`,
`data(taco): regenerate 7-class label mapping`, `test(leakage): assert GAN ids stay in train`.

Body: explain **why**, and reference the run_id when the change follows an
experiment (`Refs: unet_attention_marida_20261018-1430_a1b2c3d`).

## 6. Branches and Review Milestones
- `main` holds reviewed, working state.
- Feature branches: `feat/<area>-<topic>`, `fix/<area>-<topic>`,
  `data/<topic>`, `exp/<model>-<dataset>` for throwaway experiments.
- Tag the exact reviewed revision for faculty evaluation:
  `git tag -a v0.1-faculty-review -m "Reviewed revision for project evaluation"`.
- Experiments that should not merge stay on `exp/*`; squash them if they do.

## 7. Repository Layout Protection
- Never commit inside `patches/`, `MARIDA/`, `artifacts/`, `runs/`, or `.venv/`.
- `splits/` is tracked (manifests are small and must be versioned with the code);
  the patch arrays they reference are not.
- The "no data leakage" directive is enforced by tracked tests
  (`tests/test_leakage_audit.py`, `tests/test_splits.py`) — they must remain in the
  repository and in CI.
- Regenerating data must not change tracked `splits/` files silently; if counts
  change, that is a deliberate commit with a clear message.

## 8. If Pushing Remotely Later
- Code only: the repository is small (well under 100 MB), so no LFS is needed for
  source. Use LFS only if a small, genuinely required binary is added.
- GitHub's hard limits: 100 MB per file, ~1 GB recommended per repository. Datasets
  belong on a drive or a dataset host, not in the repo.
- Do not push by turning the 43 GB tree into a repository and then "cleaning up";
  the .gitignore must exist **before** the first `git add`.

## 9. Anti-Patterns
- `git add -A` / `git add .` / `git commit -a` in this tree.
- Committing `.pt` weights, checkpoints, or dataset archives.
- `git clean -fd` or `git checkout .` — destroys untracked data.
- Editing `.gitignore` to un-ignore a dataset "just once".
- Deleting a run directory to keep git happy instead of ignoring it.
