---
name: Good first issue
about: A well-scoped task suitable for a first-time contributor
title: "[GOOD FIRST ISSUE] "
labels: good first issue
assignees: ''
---

## What needs to be done

<!-- Clear, one-paragraph description of the task. -->

## Why it matters

<!-- Explain the user impact so the contributor understands context. -->

## How to get started

1. `git clone https://github.com/dsabarinathan/fujicv.git && cd fujicv`
2. `pip install -e ".[dev]" && pre-commit install`
3. Find the relevant file: <!-- e.g. fujicv/training/swa.py -->
4. <!-- Step-by-step hint for what to change -->
5. Run `pytest tests/<relevant_file>.py -v` to verify your change.
6. Open a PR with a short description.

## Acceptance criteria

- [ ] <!-- Criterion 1 -->
- [ ] `pytest` passes with no new failures.
- [ ] New behaviour is covered by at least one test.

## Difficulty

🟢 Easy — <!-- estimated time: e.g. 1–2 hours -->
