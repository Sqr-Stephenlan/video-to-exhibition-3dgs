# Contributing

Thank you for contributing to this repository. These guidelines apply to every
contributor, including AI coding agents.

## Before you begin

- Read `AGENTS.md` and follow all repository-specific instructions.
- Inspect the relevant files and the current Git status before editing.
- Keep each change focused on the requested task; do not mix unrelated cleanup
  into the same change.

## Git safety

- Do not commit directly to `main`; work on the assigned feature branch.
- Do not create additional long-lived branches. If a temporary branch is
  necessary, ask first.
- Do not delete local or remote branches unless explicitly instructed.
- Never rewrite shared history. Do not use `git push --force`,
  `git push --force-with-lease`, `git rebase`, or `git reset --hard` on a
  shared branch. If rewriting history is required, stop and ask the user.
- Never force-push, delete tags, modify GitHub Actions, change repository
  settings, change the license, or otherwise rewrite Git history without
  explicit user approval.

## Making changes

- Preserve existing behavior unless the task explicitly changes it.
- Modify only the files needed for the assigned task. Do not refactor unrelated
  code, format the entire repository, or rename directories unless requested.
- Avoid changing third-party code unless the task specifically requires it.
- Keep changes minimal, fix root causes rather than adding workarounds, and do
  not introduce unnecessary dependencies.
- Follow the existing project structure and coding style.
- Use the project's documented commands and local environment for development
  and validation.
- Add or update tests and documentation when they are needed to keep the
  change correct and understandable.
- Avoid committing generated outputs, large datasets, secrets, or credentials.

## Verification and handoff

- Run the relevant checks when practical, and report checks that could not be
  run.
- Review the final diff to confirm it contains only intended changes.
- Remove debug code, temporary files, and commented-out code before handoff.
- Use clear, concise commit messages that describe the change.

## Communication and decisions

- If an ambiguity requires a material scope, product, or architecture decision,
  stop, explain the uncertainty, and ask for clarification.
- Do not make independent architectural decisions. For routine details that can
  be resolved safely and reversibly, use the least-risk assumption and state it
  in the handoff.
- When uncertain, choose the safest operation; never optimize at the cost of
  repository safety.
