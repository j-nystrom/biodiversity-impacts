# AGENTS.md

## Scope

- Applies to the entire repository unless a subdirectory-specific `AGENTS.md` overrides it.
- Prefer repo-local conventions and guidelines over generic agent behavior.


## Goals of this project

The goal of this project is to provide a robust, flexible and reproducible machine learning workflow for researchers working on spatial biodiversity models.
- Robust: The code should be accurate, reliable, validated and well-tested so that users can trust the output.
- Flexible: It should be easy to extend and adapt the codebase, e.g. implementing new biodiversity metrics, features, evaluation metrics or models.
- Reproducible: The settings and exact code used for every run should be documented through appropriate logs, metadata and file exports.


## Core principles

Based on [andrej-karpathy-skills](https://github.com/forrestchang/andrej-karpathy-skills/tree/main).

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a better or simpler approach exists, say so. Push back when warranted. Don't agree just to please the user.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan before execution:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.


## Environment and commands

- Create the environment with `conda env create -f environment.yaml` and
  activate `sbm_pipe`.
- Add contributor tools with `conda env update -n sbm_pipe -f environment-dev.yaml`.
- Install commit hooks with `pre-commit install`.
- Run hooks for changed files with `pre-commit run --files <path1> <path2>`.
- Run hooks across the repo with `pre-commit run --all-files`.
- Run pipeline entrypoints with `python -m src.dags.dags <dag_name>`.


## Repository map

- Main pipeline code lives under `src/`.
- Pipeline entrypoints live under `src/dags/`.
- Data ingestion and preprocessing code lives under `src/data/`.
- Biodiversity metrics and feature code lives under `src/features/`.
- Model training and validation code lives under `src/models/`.
- Shared path definitions live in `src/paths.py`.
- Shared utilities live under `src/utils/`.
- Runtime validation helpers live under `src/validation/`.
- Automated tests and fixtures live under `src/tests/`.
- User-facing documentation lives in `README.md`, `CONTRIBUTING.md`, and
  `docs/`.


## Implementation rules

1. Inspect the real entrypoints first. Check all relevant DAGs, tasks, configs,
   and affected input/output paths before changing code.
2. Create an implementation plan. State assumptions, ask clarifying questions,
   and if possible simplify the approach. Reuse existing components before adding new code. Iterate on the plan as needed.
3. Review the draft implementation against [docs/coding_guidelines.md](docs/coding_guidelines.md). Use that guide as the source of truth for code and
   documentation patterns. Fix identified anti-patterns.
4. Run validation. Start narrowly, then finish with repo-level checks.
   Add or update unit tests for core logic. Run `pre-commit run --files ...` on touched files. Ask the user to run task-level validation when needed.
5. Update related docs if the change affects them.
   Keep `README.md`, `CONTRIBUTING.md`, `environment.yaml`,
   `environment-dev.yaml`, and `pyproject.toml` aligned and updated.
6. Log important architectural or workflow decisions in
   [docs/decision_log.md](docs/decision_log.md).


## Code reviews

1. Inspect the real entrypoints first. Check all relevant DAGs, tasks, configs,
   and affected input/output paths before doing a detailed review.
2. Check that the code does what the documentation says, and meets any specific
   requirements mentioned by the user in the review request.
3. Review the code for logical inconsistencies, potential bugs, and other
   functional problems, including config/I/O contract mismatches and missing or
   weak validation/tests.
4. Identify opportunities for simplifying the code without losing functionality.
5. Review the code against [docs/coding_guidelines.md](docs/coding_guidelines.md).
6. Summarize all findings in a structured report. Start with violation of
   requirements and bugs, ranked by severity. Then list possible simplifications. Finally list coding guideline violations. Every point should be numbered.


## Safety constraints

- Do not move large datasets into the repository.
- Avoid destructive git commands unless explicitly requested.
- Suggest to make commits and pull requests when appropriate, but always wait
for user approval.


## Validation and testing

- Start with the smallest check that directly tests what changed.
- Run linting and checks using `pre-commit run --files <paths>` on the files that were changed.
- Add a unit test for each core method that implements logic or dataframe
  processing.
- If task-level or end-to-end validation is needed, ask the user to run it.
- Follow `docs/testing.md` for shared test structure and `src/tests/AGENTS.md`
  for test-local rules.
