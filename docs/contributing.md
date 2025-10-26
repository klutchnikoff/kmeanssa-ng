---
title: Contributing to kmeanssa-ng
---


We’re thrilled that you’re interested in contributing to `kmeanssa-ng`!
Your help is valuable for making this project better. This document
provides guidelines to ensure a smooth and effective contribution
process.

## How to Contribute

Contributions are welcome in many forms, including but not limited to:

- Reporting bugs
- Suggesting enhancements
- Improving documentation
- Submitting code changes

## Reporting Bugs

If you encounter a bug, please open an issue on our GitLab repository. A
great bug report includes:

1.  **A clear title and description:** Summarize the issue concisely.
2.  **Steps to reproduce:** Provide a minimal code example that
    consistently reproduces the problem.
3.  **Expected behavior:** What you expected to happen.
4.  **Actual behavior:** What actually happened, including any error
    messages and tracebacks.
5.  **Your environment:** The version of `kmeanssa-ng`, Python, and your
    operating system.

## Suggesting Enhancements

If you have an idea for a new feature or an improvement to an existing
one, please open an issue to discuss it. This allows us to coordinate
efforts and ensure the proposal aligns with the project’s goals.

## Your First Code Contribution

Ready to contribute code? Here’s how to get started:

1.  **Fork the repository** on GitLab.
2.  **Clone your fork** locally:
    `git clone https://plmlab.math.cnrs.fr/nicolas.klutchnikoff/kmeanssa-ng.git`
3.  **Set up the development environment** (see below).
4.  **Create a new branch** for your changes:
    `git checkout -b feature/my-new-feature` or `fix/a-specific-bug`.
5.  **Make your changes**, ensuring you add or update tests and
    documentation as needed.
6.  **Verify your changes** by running the test suite and code style
    checks.
7.  **Commit your changes** with a clear and descriptive commit message.
8.  **Push your branch** to your fork:
    `git push origin feature/my-new-feature`.
9.  **Open a Merge Request** from your fork to the main `kmeanssa-ng`
    repository.

## Setting Up the Development Environment

This project uses [PDM](https://pdm-project.org/) for dependency
management.

1.  After cloning the repository, navigate to the project root.
2.  Install all dependencies, including development and optional groups:
    `bash     pdm install -d`

This command creates a virtual environment in the `.venv` directory and
installs all necessary packages for development, testing, and
documentation.

## Running Tests

To ensure your changes haven’t introduced any regressions, run the test
suite using `pytest`:

``` bash
pdm run pytest
```

This will execute all tests. You can also run specific tests by
providing a path to the test file.

## Code Formatting and Quality

We use [Ruff](https://github.com/astral-sh/ruff) for code formatting and
linting. Before committing your code, please run the following commands:

- To format your code automatically: `bash     pdm run format`
- To check for any linting errors: `bash     pdm run check`

## Submitting Changes

Once your changes are ready and verified, push them to your fork and
open a Merge Request. Provide a clear description of the changes you’ve
made and link to any relevant issues. We will review your contribution
as soon as possible.

Thank you for contributing!

## Release Process

This checklist outlines the steps to publish a new version of
`kmeanssa-ng`.

#### 1. Preparation

- [ ] **Sync the main branch:**

  ``` bash
  git checkout main
  git pull
  ```

- [ ] **Create a release branch:**

  ``` bash
  git checkout -b release/vX.Y.Z  # e.g., release/v0.5.0
  ```

#### 2. Verification and Cleanup

- [ ] **Format the code:**

  ``` bash
  pdm run format
  ```

- [ ] **Lint the code:**

  ``` bash
  pdm run check
  ```

- [ ] **Run the test suite:**

  ``` bash
  pdm run test
  ```

#### 3. Version Bump

- [ ] **Update `CHANGELOG.md`:** Add a new section for the version with
  a list of changes.

- [ ] **Bump the version:** Use `pdm bump` to increment the version
  number.

  ``` bash
  pdm bump minor  # or patch, major
  ```

- [ ] **Update the lock file:**

  ``` bash
  pdm install
  ```

#### 4. Documentation

- [ ] **Render the documentation:**

  ``` bash
  pdm run makedoc
  ```

#### 5. Commit and Pull Request

- [ ] **Add all modified files:**

  ``` bash
  git add .
  ```

- [ ] **Create the release commit:**

  ``` bash
  git commit -m "chore(release): version X.Y.Z"
  ```

- [ ] **Push the branch and create a Pull/Merge Request** to `main`.

  ``` bash
  git push -u origin release/vX.Y.Z
  ```

#### 6. Publication (after PR merge)

- [ ] **Check out and sync the main branch:**

  ``` bash
  git checkout main
  git pull
  ```

- [ ] **Create a Git tag:**

  ``` bash
  git tag vX.Y.Z
  ```

- [ ] **Push the tag to the remote repository:**

  ``` bash
  git push origin vX.Y.Z
  ```

- [ ] **Publish to PyPI:**

  ``` bash
  pdm publish
  ```
