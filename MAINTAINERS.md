# Maintainer documentation

This document describes workflows and responsibilities specific to maintainers
of the LiberTEM project.

## Release process

Releases are made automatically by opening a pull request (PR) from `dev` into `main`. 

This PR runs additional checks, including verification that:
- the `uv.lock` file is up to date,
- expanded test coverage passes across supported operating systems, and
- the version in `dev` is higher than the current version in `main`.

Once these checks pass and the PR is merged, the release process is fully automated:
the package is published to PyPI and a corresponding GitHub tag and release are
created.

## Merge strategy

This project uses a linear commit history. When merging pull requests, select
**“Rebase and merge”**.

If the commit history of a PR needs cleanup, use an interactive rebase before
merging, for example:

```
git rebase -i dev
```

See the [Git documentation on rewriting history](https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History)
for details.

## Release checklist

### Regular release

```
- [ ] Confirm that relevant issues and pull request are addressed.
- [ ] Check out current `dev` branch.
- [ ] Handle any pending deprecations.
- [ ] Review documentation for consistency and completeness.
- [ ] Run examples manually if they are not covered by CI.
- [ ] Verify the lockfile with `uv lock --check`; update it with `uv lock` if necessary.
- [ ] Bump version as appropriate with `uv version --bump <BUMP>`, `BUMP` in `{'major', 'minor', 'patch'}`.
- [ ] Open a PR from `dev` to `main` and copy this checklist into the PR description.
- [ ] Review change set compared to previous release and update changelog in `README.md`.
- [ ] Address any CI failures.
- [ ] Merge once CI passes.
- [ ] Confirm that the release was uploaded to PyPI.
- [ ] Install newly released version from PyPI into clean environment, confirm version and correct operation.
```

### Point release to backport hotfixes

Ideally, all features merged into `dev` are release-ready and simply awaiting a
release cycle.

Occasionally, however, an urgent fix may be required for the current stable
release (e.g. upstream changes breaking installation or operation), and we do
*not* want this fix to include the new features currently in `dev`.

In this workflow, `dev` remains the **only** branch from which releases are made,
but it is temporarily brought into a release-ready state via pull requests.
Direct pushes to `dev` are never required.

#### 0. Initial state

```
main:  A ── B ── C                  (v1.2.0)
dev:   A ── B ── C ── D ── E ── F   (new features, not ready)
```

---

#### 1. Create and test the hotfix from `main`

```
git checkout main
git checkout -b hotfix/critical-bug
# apply fix
git commit -m "Fix critical bug"
```

Current state:

```
hotfix: A ── B ── C ── H
```

---

#### 2. Open PR: `hotfix/critical-bug` → `dev`

The hotfix is merged into `dev` via a pull request (e.g. using a rebase or
cherry-pick strategy in the PR).

Current state:

```
dev: A ── B ── C ── D ── E ── F ── H
```

The fix is now present in `dev`, but `dev` is still not release-ready.

---

#### 3. Temporarily prepare `dev` for a patch release

Create a **release-prep branch** from `dev` that removes unreleased features:

```
git checkout dev
git checkout -b release-prep/v1.2.1
git revert D E F
git commit -m "Temporarily revert unreleased features for v1.2.1"
```

> **Hint:** If many commits need to be excluded, it can be simpler to reset
> `dev` to the current `main` HEAD and re-apply only the hotfix:
>
> ```bash
> git reset --hard main
> git cherry-pick H
> ```

Current state:

```
release-prep/v1.2.1:
A ── B ── C ── H
```

Open a PR: `release-prep/v1.2.1` → `dev`, and merge it.

Now `dev` is release-ready:

```
dev: A ── B ── C ── H
```

---

#### 4. Perform the release

- Bump the patch version on `dev`
- Open PR: `dev` → `main`
- CI runs, the release is tagged and published

```
main: A ── B ── C ── H   (v1.2.1)
```

---

#### 5. Restore `dev` to its original development state

Re-apply the reverted development commits via a PR:

```
git checkout dev
git checkout -b restore-dev-post-release
git revert <revert-commit>
git commit -m "Restore deferred development commits after v1.2.1 release"
```

Open PR: `restore-dev-post-release` → `dev`, and merge.

Final state:

```
main: A ── B ── C ── H
dev:  A ── B ── C ── D ── E ── F ── H
```

The hotfix remains, development continues uninterrupted, and no branch
protection rules were violated.

---

**Key principle:**  
`dev` is always the release staging branch, but its contents are shaped via pull
requests to match the intended release scope.
