# Agent Team: economy + shared scratchpad

A Python playground for coding agents that form teams, work in private workspaces, share notes and code, and bid to integrate a contribution into a shared repository. This package contains source and tests, without historical logs, credentials, datasets or campaign ledgers. Extracted from `code-testing-v1`; current source can differ from historical experiment snapshots.

## This copy: MAS.S62 How to AI an AI for (Almost) Anything

Copied with full history from [bobyhaha/Team-Economy](https://github.com/bobyhaha/Team-Economy) (commit `58ebbf2`, Baiyu Zhu). Everything added on top is opt in and defaults to off, so existing configs, manifests and prompt revisions are unchanged.

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

### Swarm condition

`"condition": "swarm"` replaces team formation, discussion, ballots and the auction with a stigmergic loop. The classic conditions are untouched, so the two can be compared on the same task, model and seed.

1. **Task board.** The harness turns the public repository's failing visible tests into open tasks, one per test module (`junit` now keeps per-test ids). Each task carries a virtual bounty, `bounty_base` per test, growing by `bounty_growth` for every round the test stays failing. Before the package builds and scores there is a single `bootstrap` task.
2. **Claims instead of teams.** Each round the harness wakes at most two agents per open group slot (richest first, random ties); the rest sleep and cost nothing. A woken agent makes one `claim` call: open its own group on a task, `join` the leader of a group on that task (sharing one private workspace, up to `max_team_size`), or rest. A task admits `crowding_cap` independent groups (twice that during bootstrap). Groups last one round.
3. **Equal work, as before.** Every active agent keeps the same work allowance; `work` observations carry the claimed task and the board.
4. **Verified merge queue.** All candidates are checked, ranked by expected gain (newly passing minus newly failing tests), and merged one at a time: the first via the classic tree replacement, later ones by `git apply --3way` of their scratchpad patch onto the moved head, re-running the visible tests after each merge. A merge must flip at least one test to passing and may not break a passing test; otherwise the head is reverted and the candidate stays available as a scratchpad artifact. Merges per round therefore scale with the population.
5. **Bounties instead of bids.** Money is issued only when tests flip in the public repository: the flipped tests' bounties go to the candidate's workers in equal shares, after `royalty_rate` of the sum is paid to authors of artifacts the group fetched. No pledge, no bucket brigade, no burn. `assert_accounting` still holds. Each agent sees `your_record`, the tests it has flipped per task, so specialization can form in context.

```bash
python run.py demo --config configs/demo-swarm.json --out runs/demo-swarm
python run.py run --config configs/k10-gpt6-luna-swarm.json --task datasets/commit0-tinydb/task.json --out runs/tinydb-swarm
```

`metrics.json` gains a `swarm` block: merges per round, 3-way merges, rejected merges by reason, duplicate task claims, resting agent rounds, bounties issued and the royalty share. Suggested comparisons: swarm against `voluntary_pooled` plus scratchpad at k = 5, 10 and 20 (scaling curve), a same-dollar k = 1 baseline, and the three ablations that the swarm bundles (merge queue only, bounties only, board only). Swarm currently supports commit0 and the scripted fixture; ProgramBench would need a fixed reference probe pool to define flips. Unit tests: `tests/test_swarm.py`.

### Verified market (main line)

The swarm removes the market. This variant keeps the Economy of Minds skeleton of the classic conditions, where teams bid for the right to publish, each publication pays the previous one (bucket brigade) and progress earns a dividend, and adds three opt-in flags that bring in what the swarm does better. Each flag is one ablation, and all default to off.

| Flag | Swarm advantage | Change on the classic engine |
| --- | --- | --- |
| `"merge_slots": n` (n > 1) | Merges per round scale with the population | Eligible candidates merge in bid order, highest first. The first replaces the tree; later ones are applied as 3-way patches onto the moved head. Every merge is re-checked and kept only if at least one more visible test passes and no passing test fails; a rejected merge is reverted and not charged. Each merge pays the one before it. Needs `shared_scratchpad` and commit0 (or the fixture). |
| `"rent": r` | Idle agents cost nothing | Working a round costs `r` wealth, burned. An agent below the rent rests: it leaves its team and makes no calls. Payments still reach it, and it works again once it can pay. Wealth now has consequences, as in EOM. |
| `"task_board": true` | Stigmergy | Agents see the failing visible tests grouped by module (at most 40 modules, 5 sample names each). Evidence for choosing work, not an assignment. |

The market orders, verification decides. Prompt revisions get `+merge-slots-n`, `+rent` and `+task-board` suffixes. Per-test ids stay internal otherwise: classic agents see counts only, as upstream.

```bash
python run.py demo --config configs/demo-market.json --out runs/demo-market
```

Paid ablations on TinyDB (k = 10, same model and seed as `k10-gpt6-luna-scratchpad.json`, which is the baseline):

```bash
for v in "" -gate -merge -rent -board -market; do
  python run.py run --config configs/k10-gpt6-luna-scratchpad$v.json --task datasets/commit0-tinydb/task.json --out runs/tinydb-k10$v
done
```

`-market` turns on all four (gate, `merge_slots` 10, `rent` 0.1, task board). Rent 0.1 lets an agent with no income work about ten rounds; one TinyDB test is worth a dividend of about 0.05, so producers cover it easily. `metrics.json` gains a `market` block for every classic run, the baseline included:

- `bid_gain_spearman` (and `_early`, `_late`): rank correlation between each eligible candidate's bid and its own verified gain, the visible pass count minus the public count at the start of its round. A rise from early to late means the market learns to price quality.
- `top_bid_was_top_gain_share`: in rounds with competing candidates, how often the highest bid also had the highest gain.
- `merges_per_round`, `three_way_merges`, `rejected_merges`, `skipped_for_slots`, `rent_paid`, `resting_agent_rounds`.

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

### OpenAI directly

`"provider": "openai"` sends the same requests to `api.openai.com` with `OPENAI_API_KEY` instead of OpenRouter; `openai/gpt-6-luna` becomes `gpt-6-luna`. OpenAI returns token usage, not cost, so the ledger charges usage at `input_price`, `cached_input_price` and `output_price` (USD per million tokens), which may not exceed the price caps used for reservations. `cached_input_price` defaults to `input_price`, so recorded spend is an upper bound until the cached price is confirmed. `configs/openai/` holds the six k = 10 configs with this route at $0.10 input and $0.50 output, the GPT-6 Luna list price reported at launch; check it on OpenAI's pricing page before paid runs.

```bash
export OPENAI_API_KEY=...   # from your secret manager
python run.py run --config configs/openai/k10-gpt6-luna-scratchpad.json --task datasets/commit0-tinydb/task.json --out runs/tinydb-k10
python run.py run --config configs/openai/k10-gpt6-luna-scratchpad-market.json --task datasets/commit0-tinydb/task.json --out runs/tinydb-k10-market
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
