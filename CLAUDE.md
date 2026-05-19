# CLAUDE.md

Agent guidance for the cuTAMP repo.

## Project

cuTAMP is a GPU-parallelized Task and Motion Planning solver. The package lives in `cutamp/` and is installable via `pip install -e .`. The CLI entry point is `cutamp-demo` (see `cutamp/scripts/run_cutamp.py` for flags).

## Before opening a PR

Run the unit test suite and paste the result summary into the PR body:

```bash
pytest tests/ -v
```

- Most tests require a CUDA GPU; they use `pytest.mark.skipif(not torch.cuda.is_available())`. Without a GPU they skip silently and the run does not exercise the planner or motion solver.
- Paste the trailing `=== N passed, M skipped in T s ===` line into the PR's "Test plan" section, plus anything notable (new failures, new skips, perf regressions in test runtime).
- If you can't run the suite (no GPU, env not installable), say so explicitly in the PR — don't claim tests passed.

If running inside a parent workspace that vendors cuTAMP under a pixi env (e.g. tiptop), prefix with `pixi run` from the workspace root and point pytest at this subdir: `pixi run pytest cutamp/tests/ -v`.
