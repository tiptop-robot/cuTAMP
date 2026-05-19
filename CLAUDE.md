# CLAUDE.md

Agent guidance for the cuTAMP repo. The parent tiptop `CLAUDE.md` still applies; this file adds repo-specific workflow rules.

## Before opening a PR

Run the unit test suite and include the result summary in the PR body:

```bash
# From the tiptop parent dir (pixi env lives there)
pixi run pytest cutamp/tests/ -v
```

- The suite needs a GPU. CUDA-required tests use `pytest.mark.skipif(not torch.cuda.is_available())`; without a GPU most tests will skip and the run does not exercise the code.
- Paste the trailing `=== N passed, M skipped in T s ===` line into the PR's "Test plan" section, along with anything notable (new failures, new skips, perf regressions in test runtime).
- If you cannot run the suite (no GPU, env not installable), say so explicitly in the PR — do not claim tests passed.
