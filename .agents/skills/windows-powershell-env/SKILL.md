---
name: windows-powershell-env
description: Execution rules and environment constraints for Windows PowerShell, virtual environment, and CUDA on D: drive.
---

# Windows PowerShell & Environment Guidelines

1. **Working Directory & Paths:**
   - Always run commands with `D:\TECH-ATHON` as the working directory.
   - Use Windows PowerShell syntax. Never use Linux bash commands like `export`, `source`, `grep`, or `cat` (use `Get-Content`, `Select-String`, `$env:VAR = ...`).
   - Use Windows path conventions or forward slashes in Python `Path()` objects.

2. **Python Environment Execution:**
   - Always execute Python via the project's virtual environment:
     `D:\TECH-ATHON\.venv\Scripts\python.exe`
   - Never call bare `python` or `python3` to avoid picking up the global Windows Python interpreter.

3. **Multiprocessing / DataLoader Safety on Windows:**
   - In PyTorch and Ultralytics DataLoaders, **ALWAYS** set `num_workers=0` (or `workers=0`).
   - On Windows, process spawning (`spawn` start method) causes memory deadlocks or pickling errors if workers > 0.

4. **Hardware & GPU Acceleration:**
   - Target GPU: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM).
   - Mixed Precision: Always enable automatic mixed precision (`amp=True` or `torch.cuda.amp.autocast()`).
   - Keep batch size <= 16 for 256x256 segmentation or 640x640 detection to prevent CUDA Out Of Memory (OOM) errors.
