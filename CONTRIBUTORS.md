# Developer Instructions

## uv Installation

We use [uv](https://docs.astral.sh/uv/) to manage the package. and its dependencies.

Getting started:

- [install uv](https://docs.astral.sh/uv/getting-started/installation/)
- `git clone` the repo and `cd` into it
- run `uv sync` to install all development dependencies in an editable environment

## Pre-commit

We use [pre-commit](https://pre-commit.com/) and [ruff](https://github.com/astral-sh/ruff) for linting and formatting.

Getting started:

- `uv run pre-commit install`

Once installed, the `.pre-commit-config.yaml` hooks will be run automatically when attempting to `git commit`.
Errors that cannot be auto-fixed  will be reported and must be resolved before committing.

In many cases, formatting or auto-fixes will be applied automatically.
You can stage these changes with `git add -u` and retry commiting.

## Running the tests

Run the unit tests with `uv run pytest tests/`.

## Dependency management

- use `uv add package_name` to add dependencies
- use `uv remove package_name` to remove dependencies
- use `uv add dev_package_name --dev` to add a dev dependency, i.e. that devs need but you don't want shipped to users
- use `uv pip install testing_package_name` to ephemerally install a package you think you might need, but don't want to add to dependencies just yet

## Running python/scripts:

- use `uv run python`, `uv run jupyterlab` etc. to automatically activate the environment and run your command
- alternatively use `source .venv/bin/activate` to explicitly activate environment and use `python`, `jupyterlab` etc. as usual
  - note that if you're using an IDE like VS Code, it probably activates the environment automatically

## Branch protection and usage

The `dev` and `main` branches are protected and do not allow direct commits.

All development must happen on feature branches (or forks of the repository).
Contributions should be submitted as pull requests (PRs) targeting the `dev` branch.
These PRs trigger the standard automated checks and, once approved, can be merged into `dev`.

Releases are made by opening a PR from `dev` into `main`. 
This PR runs additional checks, including verification that:
- the `uv.lock` file is up to date,
- expanded test coverage passes across supported operating systems, and
- the version in `dev` is higher than the current version in `main`.

Once these checks pass and the PR is merged, the release process is automated:
the package is deployed to PyPI and a corresponding GitHub tag and release are
created.

## Merge strategy

FIXME is that the case? The commit history of `dev` and `main`
is spotty.

This project uses merge commits, so select "Create a merge commit" when merging
a pull request.

## Release checklist

### Regular release

```
- [ ] Confirm that issues and pull request are addressed as desired.
- [ ] Check out current dev branch.
- [ ] Create release branch.
- [ ] Handle deprecations.
- [ ] Review documentation, see if everything is consistent and up-to-date.
- [ ] Try examples if not run automatically in CI.
- [ ] Check `uv.lock` file with `uv lock --check`, upgrade with `uv lock --upgrade` if necessary.
- [ ] Bump version as appropriate with `uv version --bump <BUMP>`, `BUMP` in `{'major', 'minor', 'patch'}`.
- [ ] Create PR against current main branch to see change set. Copy this checklist into the PR description.
- [ ] Review change set compared to previous release and update changelog in `README.md`.
- [ ] Fix CI issues as necessary.
- [ ] Change target of PR from `main` to `dev`.
- [ ] Merge after CI clears.
- [ ] Create PR from `dev` to `main`.
- [ ] Merge after CI clears.
- [ ] Confirm upload to PyPI.
- [ ] Install newly released version from PyPI into clean environment, confirm
      version and confirm correct operation.
```

### Point release to backport hotfixes

This might be necessary if the dev branch has progressed, but is not ready for a
full release yet, and an emergency fix to issues in the current stable release
is required. A typical case might be new upstream releases that prevent a clean
installation or correct operation, and e.g. require updated version pins as a
temporary work-around.

```
- [ ] Check out version of current release, e.g. by checking out `main`.
- [ ] Create hotfix branch.
- [ ] Apply and test hotfix.
- [ ] Bump version with `uv version --bump patch`.
- [ ] Update changelog in `README.md`.
- [ ] Create pull request into `main` branch. Copy this checklist into the PR description.
- [ ] Fix CI issues as necessary, merge when tests pass.
- [ ] Confirm updated package versions on PyPI.
- [ ] Install into a clean environment from PyPI and confirm fix for the issue
      that required a point release.
```
