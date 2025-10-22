# Project Overview

This repository contains a project for building LEGO instructions.

## Git Workflow

To ensure a clean and isolated development environment for each task, it is highly recommended to use `git worktree`. This practice prevents conflicts and allows for concurrent work on different features or fixes.

Before starting a new task, create a dedicated worktree and branch:

```bash
git worktree add -b <new-branch-name> ../<worktree-directory-name>
```

All work for the task should be performed within the newly created `<worktree-directory-name>`. When the task is complete and the branch is merged, the worktree can be removed.


## Build Tool

This project uses `pants` as a build tool. For more information, see the [Pants documentation summary](docs/pants-AGENTS.md).

## Coding & Testing Conventions

- **Modern Python**: We adhere to modern Python conventions, utilizing linters and formatters to maintain code quality and readability.
- **Type Hinting**: All new Python code should include comprehensive type hints to improve code clarity, maintainability, and enable static analysis.
- **Dataclasses**: Prefer using dataclasses for data structures to enhance readability and maintainability.
- **Testable Code**: Always strive to write code that is easily testable, and ensure new features and bug fixes are accompanied by appropriate tests.
- **Dependency Management**: When introducing new libraries or packages, always verify their established usage within the project (e.g., `requirements.txt`, `pyproject.toml`) or confirm with the user before adding them.
- **Error Handling**: All new or modified code should include robust error handling mechanisms to ensure application stability and provide clear feedback in case of issues.
- **Documentation**: For significant code changes or new features, ensure that relevant documentation (e.g., docstrings, inline comments, `README.md` updates) is created or updated to reflect the changes.
- **Formatting and Linting**: Before committing any changes, always run `pants fmt` and `pants check` to ensure code is clean and conforms to the project style. The pre-commit hooks will also run these checks.
- **Unit Tests**: Test files should be placed next to the source file they are testing (e.g., `crud.py` and `crud_test.py` in the same directory). We prefer `pytest` style tests. Always write tests for new features and bug fixes.
- **Integration Tests**: A dedicated `tests/` directory within each component (or at the root for broader integration tests) should be used for integration tests.

## Agent-Specific Guidelines

These guidelines are for the AI agent interacting with this repository:

- **Commit Workflow**: After each logical and stable set of changes, the agent should prompt the user to commit.
- **Git Add Specificity**: When staging changes, the agent must use `git add <file>` for specific files and avoid `git add .`.
- **Pre-commit Hooks**: Before committing, the agent must run `pre-commit run --all-files` and then stage any resulting changes from the hooks.
- **Process Management**: The agent must avoid running interactive or long-running server processes in the foreground (e.g., `npm start`). For UI testing, rely on non-interactive commands like `npm run build` for verification.
- **File Deprecation**: When deprecating files, the agent must first read their content and migrate any necessary code or functionality before deleting them to prevent data loss.
- **Commit Frequency**: The agent should aim for small, regular git commits.
