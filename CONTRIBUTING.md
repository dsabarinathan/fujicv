# Contributing to FujiCV

Thank you for considering a contribution! FujiCV is an open-source library and
all improvements — bug fixes, new features, documentation, and tests — are
welcome.

---

## Quick start

```bash
# 1. Fork the repo on GitHub, then clone your fork
git clone https://github.com/<your-username>/fujicv.git
cd fujicv

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install in editable mode with dev dependencies
pip install -e ".[dev]"

# 4. Install pre-commit hooks
pip install pre-commit
pre-commit install
```

---

## Running tests

```bash
# Run the full test suite
pytest

# Run a specific file
pytest tests/test_trainer.py -v

# Run with coverage
pytest --cov=fujicv --cov-report=term-missing
```

All PRs must pass `pytest` with **zero failures** before merging.

---

## Code style

FujiCV uses **ruff** for linting and formatting:

```bash
# Check
ruff check fujicv tests

# Auto-fix
ruff check --fix fujicv tests
```

The pre-commit hooks run `ruff` automatically on every commit.

---

## Type checking

```bash
mypy fujicv
```

Aim for full coverage on new public APIs.  `Any` is acceptable for internal
helpers but avoid it in public function signatures.

---

## Branching model

| Branch | Purpose |
|--------|---------|
| `main` | Latest stable release |
| `feat/<name>` | New feature branches |
| `fix/<name>` | Bug fix branches |

Open a PR against `main`. Please keep PRs focused — one logical change per PR.

---

## Adding a new feature

1. Create your branch: `git checkout -b feat/my-feature`
2. Add the implementation under the appropriate `fujicv/` sub-package.
3. Export the new symbol from the sub-package `__init__.py`.
4. Write tests in `tests/test_<feature>.py` (aim for ≥ 5 meaningful tests).
5. Update `CHANGELOG.md` under an `## [Unreleased]` heading.
6. Open a PR with a clear description of what the feature does and why.

---

## Reporting bugs

Please open a GitHub Issue with:

- **Environment**: OS, Python version, PyTorch version, GPU (if applicable).
- **Minimal reproducible example**: the smallest snippet that triggers the bug.
- **Expected vs. actual behaviour**.

---

## Security

Do **not** commit API keys, tokens, or credentials anywhere in the source tree.
The `detect-secrets` pre-commit hook will block such commits automatically.

---

## License

By contributing you agree that your contributions will be licensed under the
Apache 2.0 License that covers this project.
