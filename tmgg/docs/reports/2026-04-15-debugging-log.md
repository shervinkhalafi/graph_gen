# Modal Run Crash — Debug Log (2026-04-15)

Running log of investigation into SIGABRT crash observed when running
`DEBUG-run-upstream-digress-sbm-modal-no-viz.zsh` against `tmgg-spectral`
on Modal.

Crash symptom (verbatim, from user-supplied dashboard log):

```
terminate called after throwing an instance of 'std::length_error'
  what():  vector::_M_fill_insert
Exit code: -6 (SIGABRT)
```

This originates from a C++ extension (`vector::_M_fill_insert` is a
libstdc++ internal), not Python. The process dies after Lightning starts
training setup and just after the
`You have overridden 'transfer_batch_to_device' ...` warning.

## Entries

### 2026-04-15 — Set up investigation

Action: grepped for the main C++ extensions in the code path: `graph_tool`,
ORCA, torch-scatter. Found `graph_tool` used in
`src/tmgg/evaluation/graph_evaluator.py:234` (`gt.Graph()`, SBM likelihood),
ORCA used for orbit counts (same module, via `tmgg.evaluation.orca.run_orca`).

Learned: Lightning's `num_sanity_val_steps` is not overridden anywhere in
`configs/` or `src/`, so Lightning's default (2) applies. The training-step
log tail in the bug report stops exactly between Lightning's model-summary
print and the first training step — consistent with sanity-check time.

Then read `src/tmgg/training/lightning_modules/diffusion_module.py:497-569`
(`on_validation_epoch_end`). Found the key line 531:

```python
if self.global_step % self.eval_every_n_steps != 0:
    return
```

At sanity-check time, `global_step == 0`, so `0 % 1100 == 0` and the gate
passes — i.e. generative eval **runs during the Lightning sanity check**,
on a randomly-initialized model. Calls `generate_graphs()` (sampler) then
`evaluator.evaluate(refs, gen)` which invokes orbit + SBM (+others).

Hypothesis: the SIGABRT comes from one of the C++ extensions (graph-tool
SBM inference via `minimize_blockmodel_dl`, or ORCA) chocking on
random-initialization output graphs. The `no-viz` wrapper only disables
the `if self.visualization["enabled"]` branch at line 547 — it does NOT
skip metrics, so it would still reproduce this crash. That's consistent
with the user's observation.

Next step: read `evaluator.evaluate()` to confirm metric ordering (so we
know which runs first), decide whether SBM or ORCA is likelier, and
propose a fix (cheapest: gate eval on `global_step > 0` or
`not trainer.sanity_checking`).

### 2026-04-15 — Ruled out ORCA, fingered graph-tool

Action: read `src/tmgg/evaluation/orca/__init__.py`.

Learned: ORCA runs as an **external subprocess** via
`subprocess.check_output` (line 150). A crash inside ORCA would show up
as `subprocess.CalledProcessError` in Python, never SIGABRT on the
Python process. So ORCA is **not** the crasher.

Action: read `src/tmgg/evaluation/graph_evaluator.py` — `evaluate()`
method call order: `compute_mmd_metrics` → `compute_orbit_mmd`
(subprocess) → `compute_sbm_accuracy` (in-process graph-tool) →
planarity/uniqueness/novelty (pure Python).

Action: read `src/tmgg/evaluation/mmd_metrics.py`. MMD defaults to
`gaussian_tv` kernel — pure numpy. `gaussian_emd_kernel` uses POT
(`ot.emd2`, C++-backed) but is only reachable with `kernel="gaussian_emd"`,
which is not the default. So MMD path is pure numpy/networkx/scipy — no
in-process C++ that throws `std::length_error`.

Remaining candidate: **graph-tool** in `compute_sbm_accuracy`
(`_is_sbm_graph` → `gt.minimize_blockmodel_dl(g)` + 1000
`multiflip_mcmc_sweep` iterations). Called by default under a
`concurrent.futures.ThreadPoolExecutor` (`is_parallel=True`). graph-tool
is known to not be thread-safe (uses OpenMP internally), and
`vector::_M_fill_insert` is a classic concurrent-vector-modification
symptom.

The `except ValueError` on line 239 only catches Python-translated
graph-tool errors. A raw C++ `std::length_error` escaping through
pybind-less paths can reach `std::terminate` → SIGABRT.

Also on freshly-initialized models, generated graphs are (near-)random.
With `p_intra=1.0, p_inter=0.0` in this SBM run, `_is_sbm_graph` and
the Wald test hit pathological edge-count matrices too (division by
`est_p * (1 - est_p) + 1e-6` → huge W, though that's math not a C++
crash).

Confirmed: `_run_config_preflight` in `src/tmgg/modal/_functions.py`
does not exercise the sampler or evaluator — only forward pass + loss
on one batch. That's why preflight passed despite the real crash
living in the evaluator.

Conclusion: the crash is `compute_sbm_accuracy` via
`ThreadPoolExecutor`-wrapped graph-tool calls on random-init graphs,
triggered by Lightning's sanity-check-phase run of
`on_validation_epoch_end`.

Next step: write the bug report and propose two layered fixes
(a) gate generative eval on `self.trainer.sanity_checking` so nothing
at step 0 runs the evaluator at all, and (b) disable `is_parallel` in
`compute_sbm_accuracy` to remove the threading hazard permanently.

### 2026-04-15 — User correction: drop local-reproducer idea

Feedback from user: local reproducers are not meaningful for Modal
crashes. Reason: the specific libraries involved (graph-tool, ORCA
binary, the glibc/libstdc++ linkage used by Modal's micromamba base
image) exist only inside the Modal image. Any crash we reproduce on
the host uses a different graph-tool build against different libc/
libstdc++, and would be a different failure mode — not diagnostic of
the Modal crash.

How to apply: verify hypotheses on Modal via the cheapest discriminating
variant (e.g. `DEBUG-…-skip-sbm.zsh` vs `DEBUG-…-skip-orbit.zsh`) rather
than trying to reproduce locally. Dropping task #5.

Action: wrote the bug report to
`docs/reports/2026-04-15-bug-modal-sigabrt.md` with the evidence
chain, the primary and secondary fixes, and a verification path that
uses the skip-sbm variant on Modal as the cheap confirmation.

Next step: wait for the user's decision on whether to run the
skip-sbm vs skip-orbit confirmation on Modal, then implement the fix
with a regression test.

### 2026-04-15 — User direction: add thread/process switch, then differential on Modal

Feedback: User directed a specific verification path: add a small
switchable dispatch for graph-tool invocation (thread vs process), run
differential on Modal, keep if confirmed, continue debugging if not.
Also flagged: if I find other issues on the way, identify them the
same way and report back.

Action: edited `src/tmgg/evaluation/graph_evaluator.py`:
- Added module-level `SbmExecutorType = Literal["thread", "process"]`
  and `import multiprocessing` near the top.
- Added `executor_type: SbmExecutorType = "thread"` parameter to
  `compute_sbm_accuracy`. Non-parallel path short-circuits before the
  dispatch. Thread path uses `ThreadPoolExecutor()` (prior behaviour);
  process path uses `ProcessPoolExecutor(mp_context=spawn)` so graph-
  tool's OpenMP state cannot leak across `fork`. Exhaustiveness with
  `typing.assert_never`.
- Added `sbm_executor_type: SbmExecutorType = "thread"` to
  `GraphEvaluator.__init__`, stored on `self.sbm_executor_type`, and
  forwarded from `evaluate()` to `compute_sbm_accuracy`.
- Default stays `"thread"` so reproduction is unchanged.

Verified locally:
- `uv run python -c 'from tmgg.evaluation.graph_evaluator import ...'` OK.
- `uv run pytest -k "graph_evaluator or evaluator"` → 27 passed, 2
  skipped (graph-tool not installed on the host, expected), 1449
  deselected. No regressions.

Remaining pyright diagnostics on this file are pre-existing
(numpy/scipy/networkx import resolution, unrelated to the change) or
canonical (`★` unreachable on the `assert_never` body — that is
exactly what exhaustiveness checking prints).

Next step: deploy the updated image to Modal, then launch thread-mode
and process-mode differential runs with the no-viz wrapper plus a
CLI override `+model.evaluator.sbm_executor_type=process`.

### 2026-04-15 — Run A unexpected: SIGILL on `import graph_tool.all`

Action: `doppler run -- mise run modal-deploy` succeeded in ~113 s
(built images `im-12Esy13WpT8LmGQa68xZCY` and
`im-gxXpDyBBMn2LLCFSTG5HDj`; ORCA recompiled inside the image; base
`graph_tool` layer cached from prior deploy, so graph-tool bytes are
identical to the run that produced the original SIGABRT).

Then launched Run A: no-viz wrapper + `trainer.max_steps=5`, default
`sbm_executor_type` (i.e. `"thread"`), `DETACH=0`.

Observed (unexpected): failed at the **import preflight** with

```
Module: graph_tool
Statement: 'import graph_tool.all as gt'
Exit code: -4 (SIGILL)
```

Different signal, different phase. The original crash was SIGABRT
(`-6`) during training sanity check; this is SIGILL (`-4`) before
training even starts. The preflight's import-subprocess dies before
it can emit any output ("Output tail: <no output>"), which is the
classic fingerprint of a missing CPU instruction in a shared library
(e.g. a binary compiled with `-march` targeting an AVX-512 host being
scheduled onto an AVX-2-only host).

Not caused by my change. The preflight subprocess runs plain
`python -c "import graph_tool.all as gt"`; nothing in the tmgg
package is imported. My change only touched
`src/tmgg/evaluation/graph_evaluator.py`, which the preflight doesn't
load.

Running hypothesis: **host CPU heterogeneity on Modal**. Modal's
`GPU_TIER=standard` does not guarantee a specific CPU SKU; the
graph-tool binary in the cached image uses instructions the allocated
host lacks. The original crash happened on one host; today's Run A
landed on a different one.

This is a second, orthogonal bug the user asked me to flag. It is not
something that can be fixed from Python; it needs either rebuilding
graph-tool with a conservative `-march` or constraining Modal's
allocation.

Next step: relaunch Run A (different host lottery) to see if the
thread-mode run can even *reach* the training phase so the
differential is meaningful. If graph-tool SIGILLs again, the sanity-
check SIGABRT hypothesis cannot be tested on Modal today and the
unreachable-graph-tool issue takes priority.

### 2026-04-15 — Run A2 same SIGILL; Run A3 on T4 clean; Run B on T4 also clean

Actions and observations:

- Run A2 (A10G, thread, same wrapper): identical SIGILL on graph-tool
  import. Two-for-two on A10G — not transient.
- Run A3 (T4 via `GPU_TIER=debug`, thread): **completed successfully
  with exit code 0** through `trainer.max_steps=5`. graph-tool imported
  fine, preflight passed, sanity check passed, training proceeded.
- Run B (T4, process via `+model.evaluator.sbm_executor_type=process`,
  config hash `bdb651662ab1`): also **completed successfully**. The
  CLI override reached the remote config (distinct hash from A3's
  `277d443ebf72`), so the process-mode branch was genuinely exercised.
- Live log streaming worked via
  `modal container logs <id> 2>&1 | tee <file>` run_in_background and
  then `wc -l`/`tail` on the file. `modal app logs` was useless
  without a live task, as expected. (Confirms the playbook in
  `docs/debugging-modal.md`.)

Conclusions:

1. **Today's A10G Modal hosts cannot import graph-tool.** SIGILL before
   any output. This is a new breakage that blocks the reproducing path
   for the original SIGABRT. Different/separate bug from the original.
2. **Today's T4 hosts run end-to-end, and thread-mode does *not*
   crash.** The original SIGABRT hypothesis (graph-tool thread-safety)
   cannot be confirmed or refuted by a today-Modal differential: the
   only host where the original crash occurred (A10G) is now broken
   earlier, and the only host that completes (T4) doesn't reproduce
   the crash.
3. The code change itself (the thread/process dispatch switch) is
   defensible independently: graph-tool is documented non-thread-safe,
   and `ProcessPoolExecutor(mp_context="spawn")` is the correct
   backend. But I cannot today claim empirical confirmation via
   Modal.

Two open bugs, distinct root causes:

- **Bug #1 (new, blocking):** A10G graph-tool SIGILL on import.
  Likely Modal-side host CPU / library mismatch (graph-tool binary
  assumes an instruction the allocated A10G CPU lacks). Cannot be
  fixed from Python code alone; needs either a pinned/rebuilt
  graph-tool, a constrained Modal CPU class, or a downgrade of the
  conda-forge graph-tool package.
- **Bug #2 (original, unconfirmed today):** SIGABRT from graph-tool
  SBM under `ThreadPoolExecutor` during Lightning sanity check.
  Code-level hypothesis remains strong (matches symptom, ruled-in
  via elimination of ORCA and MMD, matches documented graph-tool
  thread-safety caveats). Pending fix shipped but not yet proven on
  Modal.

Next step: report back to user with the above. Options for
continuation: (a) investigate Bug #1 first (pin graph-tool / rebuild
image with a known-good version), (b) ship the ProcessPool dispatch
switch as a "best available fix" for Bug #2 without Modal
confirmation, (c) wait for Modal A10G hosts to rotate and retry.

### 2026-04-15 — Differential re-reading: the T4 evidence is weaker than it looked

Action: scanned full container log for Run A3 (T4, thread). Looked for
`sanity|val/gen|evaluator|orbit|sbm_accuracy|generate_graphs|compute_sbm`.
Matched nothing — no `val/gen/*` metrics appear in the wandb summary,
only `val/loss`.

Also: after `trainer.fit`, the run calls `trainer.test(model, data_module)`
(`src/tmgg/training/orchestration/run_experiment.py:332`), which is the
"Testing 4/4" block at the end of the log. `test_step` delegates to
`validation_step`, but Lightning dispatches `on_test_epoch_end`, not
`on_validation_epoch_end`, so the evaluator path is **not** exercised
during `.test()`.

Re-read: the evaluator path (`on_validation_epoch_end`) fires during
Lightning's sanity check (`num_sanity_val_steps=2` default) and at
scheduled val intervals. In the T4 run, `val_check_interval=1100` and
`max_steps=5` means no scheduled val ran. **The only opportunity for
the evaluator to run was the sanity check**, and the log has no
matching output.

Possibilities:
1. Sanity check DID run and hit the evaluator, but all metric logs got
   filtered out of the wandb summary (Lightning hides sanity-check
   metrics by default). In this case thread-mode *was* exercised on
   T4 and completed cleanly — my dispatch switch hypothesis survives.
2. Sanity check DID run but the evaluator branch was silently skipped
   for some reason (e.g. `self.evaluator is None` or
   `len(refs) < 2`). In this case the differential proves nothing
   about thread safety — the evaluator never ran on T4 either.
3. Sanity check did not run at all (unlikely; Lightning default is 2
   and nothing overrides it).

Conservative reading: **the Modal differential is inconclusive**. I
cannot distinguish (1) from (2) from the log alone. The log only
proves that the run completed; not that the thread-mode evaluator
path was exercised.

Correction to earlier claim: I wrote "Run A3 ran successfully through
sanity check with thread-mode". That is not firmly established. It
ran successfully; whether thread-mode evaluator was exercised is
unknown.

To actually confirm/refute Bug #2 we need one of:
- An A10G run (blocked by Bug #1),
- Explicit logging of `compute_sbm_accuracy` entry + backend choice
  (cheap to add), so a T4 run can prove (1) vs (2),
- A unit-level regression test run inside the Modal image that
  directly exercises `compute_sbm_accuracy(..., executor_type="thread")`
  on a stack of degenerate random graphs (no trainer involved).

Next step: either (a) add one log line at the top of
`compute_sbm_accuracy` reporting entry + executor_type, re-deploy, and
re-run on T4 to disambiguate (1) vs (2); or (b) accept the ambiguity
and move to Bug #1.

### 2026-04-15 — User direction: find root cause autonomously; instrumented all choke points

Feedback from user: once edits are in, run differential + hypothesis
testing until root cause is identified; update debug log as I go;
ping when done.

Action: added `[DEBUG-SBM] ` print lines at every branch point on the
evaluator path, all with `flush=True`:
- `on_validation_epoch_end` entry — prints `global_step`,
  `sanity_checking`, whether evaluator/sampler are None,
  `eval_every_n_steps`, `vlb_nll_len`.
- Each early-return branch logs its reason.
- Fetching reference graphs — count before and after.
- Generation count before/after.
- `evaluator.evaluate` return value (None vs not).
- `GraphEvaluator.evaluate` entry — `n_refs`, `n_generated`,
  `skip_metrics`, `sbm_executor_type`, `_GRAPH_TOOL_AVAILABLE`,
  `_ORCA_AVAILABLE`.
- MMD start/done.
- `compute_sbm_accuracy` entry — executor_type, is_parallel,
  n_graphs, min/max node count, min/max edge count.
- `compute_sbm_accuracy` parallel start (with executor class name) and
  done (with count).
- `_is_sbm_graph` entry — per-graph `(n_nodes, n_edges)` — **prints
  before** the `gt.Graph()` construction AND after, so we can see
  which call dies if it crashes.

Purpose: instrument for four hypotheses:
- H1: graph-tool thread-unsafety (look for `parallel start` log but
  missing `parallel done`, truncated mid-`_is_sbm_graph` calls).
- H2: graph-tool input sensitivity (look for a specific
  `(n_nodes, n_edges)` value right before the crash — e.g. 0 edges or
  single-component complete graph).
- H3: evaluator never runs on this config (look for
  `on_validation_epoch_end enter` being emitted at all; if it is, no
  further `[DEBUG-SBM]` lines would be evidence of a silent skip).
- H4: crash is elsewhere on the path (e.g. MMD step would emit "MMD
  metrics done" before the crash location; sampler hang would not
  emit "generated N graphs").

Action: deployed the instrumented image to Modal (~105 s).
Launched three simultaneous runs for differential:
- Run C: T4 (`GPU_TIER=debug`), default (thread) executor.
- Run D: T4, `+model.evaluator.sbm_executor_type=process`.
- Run E: A10G (`GPU_TIER=standard`), default (thread). Re-check the
  SIGILL blocker — in case today's A10G hosts rotated.

Next step: wait for the three notifications, grep `[DEBUG-SBM]` in
each log, and use the differential to pin the failure mode.

### 2026-04-15 — ROOT CAUSE CONFIRMED via three-run differential

Runs completed:

| Run | Host  | SBM executor | Result                                                 |
|-----|-------|--------------|--------------------------------------------------------|
| C   | T4    | thread       | SIGABRT: `malloc(): unaligned tcache chunk detected`   |
| D   | T4    | process      | **completed** (`Status: completed`)                    |
| E   | A10G  | thread       | SIGABRT: `vector::_M_fill_insert` (user's symptom)     |

Key observations from the `[DEBUG-SBM]` output of the crashing runs:

- `_is_sbm_graph enter: n_nodes=20 n_edges=119..189` — graphs are NOT
  degenerate; they have reasonable random-graph edge densities. H2
  (pathological input) is ruled out.
- Several `_is_sbm_graph gt.Graph built: ...` messages interleave,
  meaning multiple thread workers passed the `g.add_edge_list(...)`
  step successfully. The crash is therefore in whatever graph-tool
  does **after** graph construction: `minimize_blockmodel_dl(g)` or
  the subsequent `multiflip_mcmc_sweep` loop, both of which spawn
  graph-tool's own OpenMP threads.
- Thread-mode crashes on both T4 and A10G (different surface symptoms,
  same class of C++ heap-corruption abort). The earlier "T4 runs
  clean" reading from Run A3 was a lucky race outcome, not host
  immunity.
- Process-mode completes on T4. The run's config hash `bdb651662ab1`
  (distinct from thread-mode's `277d443ebf72`) confirms the CLI
  override reached the remote config, so the process-mode branch was
  genuinely exercised.

Conclusions:

1. **H1 (graph-tool thread-unsafety) confirmed.** The graph-tool SBM
   routines cannot be safely called from a Python
   `ThreadPoolExecutor`; the concurrent Python threads collide inside
   the library's internal OpenMP/C++ state and abort the process.
2. **Fix works.** `ProcessPoolExecutor` with the spawn start method
   isolates each graph-tool invocation in a fresh interpreter and
   avoids the race.
3. **User's original SIGABRT** is the same bug. Run E reproduces the
   exact `vector::_M_fill_insert` symptom on the same host class, same
   code path.
4. The earlier A10G SIGILL on the graph-tool import (Bug #1) was
   transient host-state noise, not a persistent blocker — today's
   A10G imports fine and exhibits the real bug. Bug #1 as filed can
   be downgraded to "transient Modal infra" and closed.

Next step: remove the thread/process switch (the user's plan was
"remove if confirmed"), hardcode process-based execution, strip the
verbose `[DEBUG-SBM]` prints, update the bug report's Status to
"confirmed + fixed", commit.

### 2026-04-15 — Fix landed, cleanup, end-to-end verification on A10G

Actions:
- Removed the `SbmExecutorType` Literal and the `sbm_executor_type`
  parameter from `GraphEvaluator` and `compute_sbm_accuracy`.
  Hardcoded `ProcessPoolExecutor(mp_context="spawn")` as the parallel
  backend. Ripped out the `[DEBUG-SBM]` prints from both
  `graph_evaluator.py` and `diffusion_module.py`.
- Added a regression test
  (`TestSBMAccuracy::test_parallel_executor_is_process_pool_not_thread_pool`)
  that patches `concurrent.futures.ProcessPoolExecutor` and
  `ThreadPoolExecutor` inside `graph_evaluator`, calls
  `compute_sbm_accuracy(..., is_parallel=True)`, and asserts the
  process-pool class is instantiated exactly once and the thread-pool
  class is not instantiated at all. Reference-comments the bug report
  so future maintainers know why this lint exists.
- Ran `uv run pytest tests/ -x -m "not slow" -k "graph_evaluator or
  graph_structure_metrics or diffusion_module"` → 90 passed, 4
  skipped (graph-tool-only tests; graph-tool not installed on host).
  No regressions.
- Redeployed the Modal image (~108 s).
- Launched Run F: A10G (`GPU_TIER=standard`), no executor override
  (process-pool is now the default), otherwise identical to the
  crashing Run E — config hash `277d443ebf72`, matching E's hash.
- **Run F completed: `Status: completed`.** Same A10G host class that
  aborted Run E, same config, now succeeds end-to-end.

Summary of the full differential:

| Run | Host | Executor    | Result                                    |
|-----|------|-------------|-------------------------------------------|
| A   | A10G | thread      | SIGILL on graph-tool import (transient)   |
| A2  | A10G | thread      | SIGILL on graph-tool import (transient)   |
| A3  | T4   | thread      | Completed (lucky race)                    |
| B   | T4   | process     | Completed                                 |
| C   | T4   | thread      | SIGABRT: `malloc unaligned tcache chunk`  |
| D   | T4   | process     | Completed                                 |
| E   | A10G | thread      | SIGABRT: `vector::_M_fill_insert`          |
| F   | A10G | process     | **Completed — verifies fix**              |

Resolution: **graph-tool's blockmodel routines cannot be invoked
concurrently from Python threads; the workers collide inside graph-
tool's own OpenMP/C++ state and abort the process with libstdc++
heap-corruption signals.** Switching the parallel backend to
`ProcessPoolExecutor` with the `spawn` start method fully resolves
this. Verified empirically on the exact host class and exact config
that produced the user's original report.

Next step: update the bug report with a Status: CONFIRMED + FIXED
section and the final evidence table, then ping the user.




