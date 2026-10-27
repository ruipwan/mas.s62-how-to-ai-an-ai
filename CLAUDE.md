# Claude Code notes for this repository

Read `docs/HANDOFF.md` first: it is the full record of the design discussion, the papers behind it, the experiments run so far, and what is still open.

## What this is

Ruipeng Wang's copy of Baiyu Zhu's `bobyhaha/Team-Economy` (MAS.S62 group project, MIT Media Lab, Fall 2026): a playground where coding agents form teams, work in private workspaces, share a scratchpad and integrate contributions into one public repository. Additions live on the `swarm` branch and on `claude/new-session-jku7c5`:

- Opt in flags `regression_gate` and `memory_trace` (commit `82e1ec9`).
- `condition: "swarm"`: task board from failing visible tests, one claim call per woken agent, verified 3 way merge queue, bounties only for test flips (commit `b49dc18`, `codetest/swarm.py`).
- Verified market, the main line now: opt in flags `merge_slots`, `rent` and `task_board` on the classic engine, each one ablation (see README "Verified market"). Configs `configs/k10-gpt6-luna-scratchpad-{gate,merge,rent,board,market}.json`, demo `configs/demo-market.json`, tests `tests/test_market.py`.
- Per test ids (`failing`) are internal: `engine.visible()` strips them from everything agents see and from logs; only `task_board` shows them, grouped.

## Working rules

- Keep the classic conditions (`solo`, `random_teams`, `voluntary_fixed`, `voluntary_pooled`, baselines) behaviorally unchanged; new behavior goes behind a flag or a new condition so A/B runs on the same task, model and seed stay valid.
- Prefer the simple, efficient implementation. Reuse the vendored `hayekmas` code where it already exists (rent, bankruptcy, births, coordination fee, wealth objective are there but not wired into `codetest/`).
- Run `python -m unittest discover -s tests -v` before any paid run. Paid runs need Docker, prepared task data and `OPENROUTER_API_KEY`, or `OPENAI_API_KEY` with the configs in `configs/openai/` (`"provider": "openai"`); the ledger caps spend (see README).
- `main` stays equal to upstream `main`; put work on `swarm` or a new branch and open PRs against `bobyhaha/Team-Economy`.
- Prose, commit messages and docs: no dashes as punctuation; commas and periods instead. Replies to Ruipeng in Chinese with English technical terms, direct, few bullets.

## Quick commands

```bash
python run.py demo --config configs/demo-swarm.json --out runs/demo-swarm
python run.py serve runs/demo-swarm --port 8795
python run.py run --config configs/k10-gpt6-luna-swarm.json --task datasets/commit0-tinydb/task.json --out runs/tinydb-swarm
```
