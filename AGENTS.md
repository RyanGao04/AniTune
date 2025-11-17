# Repository Guidelines

## Project Structure & Module Organization
- Keep the repository root lean: retain top-level docs like `README.md`, `LICENSE`, and planning artifacts (e.g., `CS539_proj_proposal_v3.pdf`).
- Place reusable library code in `src/anitune/`. Add runnable entry points in `scripts/` (e.g., `train.py`, `infer.py`).
- Store experiments in `notebooks/`; keep committed notebooks small and reproducible. Use `data/` (gitignored) for raw/large inputs and `assets/` for lightweight examples.
- Mirror package structure in `tests/` (`tests/augmentations/test_resize.py` alongside `src/anitune/augmentations/resize.py`).

## Build, Test, and Development Commands
- Create a virtual environment: `python -m venv .venv && source .venv/bin/activate`.
- Install dependencies once `requirements.txt` (or `pyproject.toml`) is defined: `python -m pip install -r requirements.txt`.
- Format and lint before pushing: `python -m black src tests` and `python -m ruff check src tests` (pin versions in `pyproject.toml`).
- Run automated tests: `python -m pytest`.
- Typical training run once entry points exist: `python -m anitune.train --config configs/base.yaml --data-dir data/processed`.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indents; target line length 100. Prefer type hints and docstrings for public functions/classes.
- Names: packages/modules/functions use `snake_case`; classes use `PascalCase`; constants use `UPPER_SNAKE_CASE`.
- Keep configs in `configs/` (YAML/JSON). CLI flags should be kebab-case and map cleanly to config keys.
- Isolate model checkpoints and artifacts under `runs/` or `outputs/` (gitignored) to keep the repo clean.

## Testing Guidelines
- Use `pytest`; place tests in `tests/` with filenames `test_*.py`. Keep tests deterministic (set seeds) and fast; mock I/O/GPU-heavy calls.
- Aim for broad coverage (≥80%). Add regression tests for every bug fix and sanity checks for new datasets or augmentations.
- When introducing new configs or scripts, include a smoke test (small batch/epoch) and document expected runtime.

## Commit & Pull Request Guidelines
- Use concise, Conventional Commit-style prefixes: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`.
- Scope each PR to one logical change. Provide a description, rationale, and test evidence (`pytest` output or screenshots/metrics for model changes).
- Link to issues/tasks when available. Update `README.md`, configs, and notebooks when behavior changes. Include before/after metrics for training-related updates.
