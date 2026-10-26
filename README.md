# Agent Team: economy + shared scratchpad

A Python playground for coding agents that form teams, work in private workspaces, share notes and code, and bid to integrate a contribution into a shared repository. This package contains source and tests, without historical logs, credentials, datasets or campaign ledgers. Extracted from `code-testing-v1`; current source can differ from historical experiment snapshots.

## This copy: MAS.S62 How to AI an AI for (Almost) Anything

Copied with full history from [bobyhaha/Team-Economy](https://github.com/bobyhaha/Team-Economy) (commit `58ebbf2`, Baiyu Zhu). Two opt-in changes are added on top. Both default to off, so existing configs, manifests and prompt revisions are unchanged.

| Flag | What it does |
| --- | --- |
| `"regression_gate": true` | On commit0, a code change whose visible test score is below the public high water mark cannot enter the auction. Equal scores pass. Bids still choose among the remaining candidates. Blocked candidates emit `regression_blocked` events, and the system prompt states the rule. The prompt revision gets a `+regression-gate` suffix. |
| `"memory_trace": true` | Each agent keeps a bounded trace (2000 characters) of its own actions across slots and rounds, instead of having its memory replaced by its latest message. |

A/B run for the gate, same task, model and seed:

```bash
python run.py run --config configs/k5-gpt6-luna-scratchpad.json --task datasets/commit0-tinydb/task.json --out runs/tinydb-base
python run.py run --config configs/k5-gpt6-luna-scratchpad-gate.json --task datasets/commit0-tinydb/task.json --out runs/tinydb-gate
```

Compare final hidden score, accepted regressions, rounds without a publication, and cost per point. Test `memory_trace` as a separate run so the two effects stay separable. The new unit tests are in `tests/test_v1.py`; run the full suite before paid runs.

## Free quick start

Requires Python 3.11+ and Git. No Docker or API key is needed for the scripted demo.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'requests>=2.32,<3'
python run.py demo --config configs/demo-scratchpad.json --team-evolution --out runs/demo
python run.py serve runs/demo --port 8795
```

Open http://localhost:8795/replay.html. Decisions are scripted: this demonstrates the engine, not model intelligence or benchmark performance. The scratchpad is enabled; the tests below exercise note revisions and code sharing/reuse. Use a new output directory for each run.

```bash
python -m unittest discover -s tests -v
```

## Mechanism

1. Agents can form and leave teams, with maximum size `max(1, floor(k / 2))`.
2. Teams work privately from the current public repository: inspect, edit and test.
3. A shared scratchpad holds task claims, findings and blockers. Notes have owners and revisions. Outdated updates are rejected to protect newer information.
4. Teams share code candidates; others can fetch them. Fetching does not automatically apply or merge code.
5. Members vote and pledge simulated money. Eligibility requires majority consent, a positive pooled bid, supporting participation, and a buildable change or qualifying probe contribution.
6. An eligible highest-bid candidate wins integration into the shared repository; ties use the configured RNG. The auction does not rank candidates by test accuracy.

Internal wealth is simulated. API requests cost real money, tracked separately. Agent count denotes population, not simultaneous API requests.

## Real coding runs

Requires Docker, prepared task data and an OpenRouter credential. The included five- and ten-agent configs are historical examples: review model identifier, provider price limits, reasoning and budget before use. They allow extensions from ten to fifty rounds; set `max_rounds` equal to `rounds` for a fixed horizon. For a standalone experiment, `parallel_experiments` can be false.

```bash
python run.py prepare commit0 tinydb --out datasets/commit0-tinydb
python run.py build --task datasets/commit0-tinydb/task.json
python run.py plan --config configs/k5-gpt6-luna-scratchpad.json
```

Set `OPENROUTER_API_KEY` in the process environment using your secret manager. Never commit credentials. After reviewing the plan:

```bash
python run.py preflight --task datasets/commit0-tinydb/task.json --paid
python run.py run --config configs/k5-gpt6-luna-scratchpad.json --task datasets/commit0-tinydb/task.json --out runs/my-tinydb
python run.py evaluate runs/my-tinydb
python run.py serve runs/my-tinydb --port 8795
```

TinyDB uses adapted visible development tests, not a hidden generalization score. Preparation downloads data; `run` makes paid calls. The scripted demo and offline unit tests do neither.

## Budget and stopping

The ledger has a $100 cumulative campaign ceiling plus the configured per-run ceiling. These defaults are not authorization to spend. Uncertain requests retain their reservations. Do not delete reservations or the ledger to bypass billing errors.

```bash
python run.py budget
# Stop one run before its next request:
touch runs/my-tinydb/STOP
# Or stop the local campaign:
touch artifacts/STOP
```

An in-flight request may still complete and be charged. Final evaluation is separate from inference.

## Code map

| File | Purpose |
| --- | --- |
| `configs/demo-scratchpad.json` | Small scripted example |
| `codetest/config.py` | Settings and validation |
| `codetest/engine.py` | Work, votes, auctions and integration |
| `codetest/policy.py` | Prompts and observations |
| `codetest/scratchpad.py` | Versioned notes and shared artifacts |
| `codetest/actions.py` | Structured action schemas |
| `codetest/budget.py` | API accounting and caps |
| `vendor/hayekmas/adapters/teams/team.py` | Team manager |
| `tests/test_scratchpad.py` | Note, artifact and reuse tests |

Core dependencies and UI assets are included. Campaign-specific recovery scripts and unrelated experimental variants are omitted. This package does not resume an existing paid campaign.
