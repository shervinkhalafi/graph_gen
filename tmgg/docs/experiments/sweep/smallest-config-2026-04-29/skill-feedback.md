# Skill feedback log

Append-only. One entry per round noting any deviation from
`smallest-config-search/SKILL.md`, brittle rules, new failure modes,
sharper diagnostic-interpretation heuristics. Drives Phase 4
skill-refinement edits.

## Round 1 — wrapper-misconfig kill + relaunch (new failure mode)

**What happened.** Initial round-1 launches went out via
`launch_round.py` with `--async-eval`. The 2 SBM trainer pods landed
in W&B project `discrete-diffusion` instead of the canonical
`tmgg-smallest-config-sweep`, because `run-upstream-digress-sbm-modal.zsh`
defaulted `WANDB_PROJECT=discrete-diffusion` (legacy from before the
sweep). Meanwhile `launch_round.py` hardcoded the canonical project
into the rounds.jsonl row, and `fetch_outcomes` / `watch_runs` both
default to `--project tmgg-smallest-config-sweep` and ignore the row's
`wandb_project` field. Net result: the SBM runs would have been
silently invisible to the entire downstream pipeline.

**How it was caught.** Discovered post-launch by reading the
wrapper's echoed `tmgg-modal run ...` command in the launch log
before scheduling the watch wakeups. Both SBM pods cancelled via
`scripts.sweep.kill_call --yes` within minutes of spawn (no training
or W&B run had started yet, so no orphaned runs in either project).

**Fix shipped.**
1. `run-upstream-digress-sbm-modal.zsh:20` default flipped to
   `tmgg-smallest-config-sweep`.
2. `launch_round.py` now scans the wrapper's stdout for
   `wandb_project=<value>` and raises before appending a launched row
   if the value disagrees with `CANONICAL_WANDB_PROJECT`. Two
   regression tests pin both the mismatch and missing-token failure
   modes (`tests/sweep/test_launch_round.py`).
3. `find_pending_launches` in `fetch_outcomes.py` now pairs by
   `(run_uid, latest-outcome-ts)` instead of `run_uid` alone, so the
   relaunch-after-kill flow (where original + cancel-outcome + relaunch
   share a `run_uid` because the config hash is deterministic) doesn't
   silently drop the relaunch as "already paired". Two regression
   tests in `tests/sweep/test_fetch_outcomes.py`.

**Bookkeeping.** Cancel-outcomes for the two killed launches
(`failure_kind=cancelled_wrapper_misconfig`) appended to rounds.jsonl
with the kill rationale in `gate_reason`. Relaunch artefact lives at
`round-1-sbm-relaunch.yaml` (separate from `round-1.yaml` so the
audit trail of "original 3-launch round + 2-launch SBM relaunch"
stays unambiguous).

**Suggested skill edit (Phase 4).** Add a "verify wrapper output"
checklist item to Step 4 of the loop body: after invoking
`launch_round`, scan the launch log for `wandb_project=<canonical>`
on every spawned pod before scheduling watch wakeups. The new
in-launcher guard now does this automatically and crashes loudly,
so this is belt-and-braces — but a sweep operator who manually
edits a wrapper for a one-off would benefit from the explicit check.
Also: the kill_call workflow's "After kill, append a watches.jsonl
row" rule (SKILL.md Step 4.5) is wrong for non-watcher kills; the
canonical-project failure was a manual operator kill triggered by
reading launch output, not a watcher decision. Document
`failure_kind=cancelled_*` outcome rows as a first-class kill type
with no watches.jsonl entry.

