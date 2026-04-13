# Atoms Scratch Worker Workflow

Use this workflow when the real provider repos live outside the writable root and worker write tasks are stalling on sandbox boundaries.

## Goal

Let workers edit isolated scratch repos under `/tmp`, then copy reviewed changes back into the real sibling repos with explicit escalated commands.

## Why this works

- `/tmp` is writable in the current environment.
- A plain scratch clone does not mutate the source repo's `.git` metadata.
- `git worktree add` is less useful here because it still writes to the source repo's git metadata and therefore still crosses the protected boundary.

## Setup

Prepare scratch repos from the real siblings:

```bash
python scripts/prepare_scratch_atoms_repos.py --refresh \
  /Users/conrad/personal/sciona-atoms \
  /Users/conrad/personal/sciona-atoms-bio \
  /Users/conrad/personal/sciona-atoms-fintech \
  /Users/conrad/personal/sciona-atoms-ml \
  /Users/conrad/personal/sciona-atoms-physics \
  /Users/conrad/personal/sciona-atoms-robotics \
  /Users/conrad/personal/sciona-atoms-signal
```

Default scratch root:

```text
/tmp/sciona-atoms-worker-scratch
```

The helper clones each repo into `/tmp` and mirrors any local dirty working-tree paths from the source repo into the scratch clone.

## Worker usage

- Point each worker at one scratch repo only.
- Let workers make edits, run tests, and produce a clean diff in `/tmp`.
- Keep one worker per provider repo to avoid merge conflicts inside the scratch clone.

## Writeback

Recommended writeback flow:

1. Inspect the worker's changed files in the scratch repo.
2. Generate a diff or file list from the scratch repo.
3. Copy only the reviewed changed files back into the real sibling repo with escalated commands.
4. Run verification against the real repo.
5. Commit and push from the real repo, not from the scratch clone.

## Practical rule

- Use workers for read and write work inside `/tmp`.
- Use direct escalated commands only for final writeback into the real sibling repos.
