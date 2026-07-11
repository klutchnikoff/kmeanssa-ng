# Contributing to kmeanssa-ng

Thanks for your interest in `kmeanssa-ng`! Contributions, bug reports and
questions are all welcome. This project is hosted on GitHub at
<https://github.com/klutchnikoff/kmeanssa-ng>, where issues and pull requests are
welcome. (The maintainers also develop on a GitLab instance, from which the
GitHub repository is mirrored and where the CI and releases run.)

## Reporting bugs and problems

Please open an **issue** on the GitHub tracker:
<https://github.com/klutchnikoff/kmeanssa-ng/issues>.

A helpful report includes:

- what you did and what you expected to happen;
- the actual behaviour (error message / traceback, or the wrong result);
- a minimal, self-contained snippet that reproduces the problem;
- your `kmeanssa-ng`, Python and OS versions.

## Seeking support

For usage questions, open an issue with the *question* label, or contact the
maintainer at <nicolas.klutchnikoff@univ-rennes2.fr>. The
[documentation](https://kmeanssa-ng.readthedocs.io/) (API reference and
tutorials) is the best first stop.

## Contributing code

1. **Fork** the repository on GitHub and create a topic branch off `main`
   (e.g. `feature/my-change` or `fix/some-bug`).
2. Make your change with tests and, when relevant, documentation.
3. Open a **pull request** targeting `main`. A maintainer integrates accepted
   pull requests through the GitLab mirror, where the CI (lint + tests on
   Python 3.10–3.12) must pass; merges go through review.

Small, focused pull requests are easiest to review. If you plan a larger
change, please open an issue first to discuss the design.

## Development setup

The project is managed with [PDM](https://pdm-project.org/):

```bash
git clone https://github.com/klutchnikoff/kmeanssa-ng.git
cd kmeanssa-ng
pdm install                 # installs the package and the dev dependency groups
pdm run pre-commit install                     # ruff check + format on commit
pdm run pre-commit install --hook-type pre-push  # pytest on push
```

Then, before opening a pull request:

```bash
pdm run pytest              # run the test suite
pdm run ruff check src      # lint
pdm run ruff format src     # format
```

Please keep new code covered by tests and consistent with the surrounding style
(the pre-commit hooks enforce `ruff` formatting and linting; the pre-push hook
runs the tests).

## Releasing (maintainers)

Releases are automated by the CI. To cut version `X.Y.Z`:

1. Bump the version in `pyproject.toml` **and** `CITATION.cff` (and set
   `date-released`), on `main`.
2. Create and push an annotated tag `vX.Y.Z`.

Pushing the tag triggers the `deploy` CI job, which publishes the package to
**PyPI** (`pdm publish`), and **Read the Docs** rebuilds the documentation for
the new version. No manual upload is needed.

## Code of conduct

Please be respectful and constructive in all interactions. We follow the spirit
of the [Contributor Covenant](https://www.contributor-covenant.org/).
