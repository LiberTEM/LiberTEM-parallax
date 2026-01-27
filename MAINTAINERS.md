# Maintainer documentation

Releases are made by opening a PR from `dev` into `main`. 
This PR runs additional checks, including verification that:
- the `uv.lock` file is up to date,
- expanded test coverage passes across supported operating systems, and
- the version in `dev` is higher than the current version in `main`.

Once these checks pass and the PR is merged, the release process is automated:
the package is deployed to PyPI and a corresponding GitHub tag and release are
created.

## Merge strategy

This project rebases commits for merging, so select "Rebase and merge" when merging
a pull request.

Hint: If the commit history of a PR is too wild, one can [clean it
up](https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History) with an
interactive rebase, e.g. `git rebase -i dev`.

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