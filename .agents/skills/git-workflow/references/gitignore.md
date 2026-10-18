# Reference: .gitignore, recovery and diagnostics

## 1. Full `.gitignore` for D:\TECH-ATHON

Create this file **before** the first `git add`.

```gitignore
# ---------- Python ----------
__pycache__/
*.py[cod]
*.so
.Python
build/
dist/
*.egg-info/
.pytest_cache/
.coverage
htmlcov/
.mypy_cache/
.ruff_cache/

# ---------- Environments ----------
.venv/
venv/
env/
.conda/

# ---------- Datasets (never commit) ----------
20210626/
MARIDA/
patches/
taco_raw/
TACO-master/
marine-debris-fls-datasets-master/
utd2.v8i.yolo26/
shapefiles/
data/
*.npz
*.npy
*.tif
*.tiff
*.tif.aux.xml
*.zip

# ---------- Artifacts / runs / weights ----------
artifacts/
runs/
weights/
mlruns/
wandb/
*.pt
*.pth
*.ckpt
*.onnx
*.engine
*.pkl

# ---------- Media ----------
*.mp4
*.mov
*.avi
*.gif

# ---------- Notebooks / editors ----------
.ipynb_checkpoints/
.vscode/
.idea/
*.swp

# ---------- OS ----------
Thumbs.db
desktop.ini
.DS_Store

# ---------- Local secrets ----------
.env
.env.*
*.key
*.pem
credentials.json
```

Exceptions worth considering (uncomment only if the file is small and essential):

```gitignore
# !splits/           # splits/ is tracked: manifests are small and versioned
# !docs/*.png        # small documentation figures
```

## 2. Verify what git actually sees

```powershell
git status --short
git check-ignore -v .venv patches artifacts runs weights yolov8s.pt TACO-master.zip
git ls-files | Measure-Object -Line
git count-objects -vH
```

`git check-ignore -v` prints the matching rule — use it when something unexpected
is tracked or missing.

## 3. Pre-commit size guard (optional, no dependencies)

`.git/hooks/pre-commit`:

```sh
#!/bin/sh
# Block accidental staging of files over 5 MB
limit=$((5 * 1024 * 1024))
fail=0
for f in $(git diff --cached --name-only --diff-filter=ACM); do
  [ -f "$f" ] || continue
  size=$(wc -c <"$f")
  if [ "$size" -gt "$limit" ]; then
    echo "REFUSED: $f is $((size / 1024 / 1024)) MB (>5 MB). Add to .gitignore or use git-lfs."
    fail=1
  fi
done
exit $fail
```

On Windows, run it from Git Bash: `chmod +x .git/hooks/pre-commit`. If the hook
blocks a legitimate small binary, prefer adding it to git-lfs over bypassing with
`--no-verify`.

## 4. Recovery: large file already committed

Only if the commit has **not** been pushed:

```powershell
git rm --cached -r artifacts runs .venv           # unstage/untrack, keep on disk
git commit -m "chore: stop tracking generated artifacts and datasets"
```

If a huge blob is already in history and the repo is local-only, the cleanest fix
is to re-initialize rather than rewrite history:

```powershell
Remove-Item -Recurse -Force .git
git init -b main     # ensure .gitignore is correct first, then re-add
```

Do not run `git filter-branch`/`filter-repo` on a repository whose only copy is
this machine without a backup.

## 5. Safe staging commands for this repo

```powershell
git add .gitignore AGENTS.md CONTEXT.md requirements.txt skills-lock.json
git add src scripts tests dashboard docs splits .agents
git status --short
git commit -m "feat(models): add attention UNet variant"
```

Then confirm the tree stayed small:

```powershell
git ls-files -z | ForEach-Object { $_ -split "`0" } | Where-Object { $_ } |
  Measure-Object Length -Sum |
  ForEach-Object { "{0:N2} MB tracked" -f ($_.Sum / 1MB) }
```

Target: tracked source well under 100 MB. If that number is gigabytes, a dataset
directory slipped past `.gitignore` — fix it with §4 before making more commits.

## 6. Review milestone tagging

```powershell
git tag -a v0.1-faculty-review -m "Reviewed revision submitted for evaluation"
git show --stat v0.1-faculty-review
git archive --format=zip -o TECH-ATHON-v0.1-source.zip v0.1-faculty-review
```

`git archive` produces a clean source-only zip for submission — it contains
tracked files only, so no 36 GB of data leaks into the deliverable.
