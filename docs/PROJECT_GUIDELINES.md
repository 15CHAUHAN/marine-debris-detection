# Project Guidelines

Project: Problem Statement 17 - AI-Based Marine Plastic Pollution Detection

These are the mandatory working rules for every task performed in this project. They apply to the main agent and to every subagent spawned during development.

## 1. Documentation First

- Always keep documentation of all changes, decisions, experiments, and outcomes.
- If there is any doubt, never assume anything. Clarify with the user first, then proceed.
- Decisions must be recorded with the reasoning behind them, the date, and any alternatives considered.
- The project documentation should have clear explanations of all files, how they connect, and how data travels through the pipeline.

## 2. Code Quality

- Keep the code clean, modular, and minimalistic.
- Never write logic from scratch if a prebuilt library already does the same work. Prefer established libraries (OpenCV, PyTorch, torchvision, scikit-image, albumentations, etc.).
- Reuse existing code and follow the patterns already present in the project.

## 3. Documentation Structure

- All documentation must live under the `docs/` directory.
- Every module or pipeline stage must be documented with:
  - Purpose
  - Inputs and outputs
  - How it connects to other modules
  - Data flow description

## 4. Writing Style

- Never use em dashes (mdashes) or any generic AI watermark phrases.
- Write clear, technical, and factual text.

## 5. Development Methodology

- Follow A/B testing methodology for experiments and model choices.
- Keep security essentials in mind at every stage.
- Follow testing based development strategy: tests are written together with the implementation.

## 6. Task Division and Parallel Development

- Divide the work into smaller, well-defined goals.
- Deploy subagents wherever necessary to follow parallel development and integration.
- Each subagent is spawned with a defined scope and a clear deliverable.

## 7. Subagent Contracts

Every subagent task must define:
- Goal: what the subagent must achieve
- Tasks: the concrete steps to perform
- Success metrics: how the subagent's output is measured and accepted

## 8. Numbers and Results

- Never create numbers or results on your own.
- All numbers and results must come from actual testing and development.
- Report metrics (IoU, mAP, F1-Score, loss curves, etc.) only from real runs logged in the project.

## 9. Order of Execution

- These instructions were written into this file first.
- Any work starts only after the user provides explicit instructions.