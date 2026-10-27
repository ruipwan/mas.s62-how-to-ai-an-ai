# Handoff: MAS.S62 team economy project (Ruipeng's copy)

Written 2026-10-27 as a complete record of the chat in which this fork and the swarm condition were designed and built. Read this first when continuing the work in Claude Code. Chinese and English are mixed on purpose; that is how the discussion ran.

## 1. Context

- Class: MAS.S62 "How to AI an AI for (Almost) Anything" (Paul Liang, MIT Media Lab, Fall 2026). Group project: a decentralized multi agent team economy for coding agents.
- Team split (from the shared Google Doc "分工"): Baiyu Zhu and Ruipeng Wang do **better designs + mechanisms**; Yifan Kang and Yanchen Liu **run benchmarks and analyze team behaviors**.
- Upstream code: https://github.com/bobyhaha/Team-Economy (Baiyu). This repository is a full history copy of it (commits `1bfcfc0`, `58ebbf2`) with Ruipeng's commits on top. Target GitHub location: https://github.com/ruipwan/mas.s62-how-to-ai-an-ai (created, may still be empty; see section 7).
- Doc notes that framed the work: "Decentralized system is necessary for a scalable agentic system"; findings so far: communication overhead; scalable only if results are verifiable (coding); more agents != better; agents need enough budget to overcome coordination cost; scratchpad is useful for delegating work. Goals: scalable agentic system + better benchmark performance; an emergent world built by a multi agent system evaluated through an economic valuation framework (reference machineplaying.org). Problem noted: "Bid is not always optimal". Better designs: evolution at team / agent / society level. Questions: when and how to evolve; keep only the evolution that makes money each round or overall. "Money ~ tokens". "Env: get question right -> reward". Benches: Commit-0, ProgramBench, NL2Repo, OpenRSI Index / Autoresearch Bench.

## 2. Material read and what each contributed

1. **ExplorationBench** (Fudan, Tencent Hunyuan, Tsinghua; arXiv 2609.30199; 机器之心 2026-10-19). Two "alien worlds" (AlienCode, 31 changed rules; AlienLogic, 24 changed rules), 10 frontier systems, four exploration rounds, Best@3. Key numbers: reading examples only, best 15.7% on AlienCode; after four rounds best trajectory 89.0%; thinking without environment feedback at most 11.0%; autonomous exploration beats replaying one's own best probes by a median 17.4 points; GPT-5.6 Sol 88.1% by exploring versus 69.5% when told the rules up front; one well designed probe moved a rule report from 8 to 28 correct rules and accuracy from 11.0% to 81.9%; 30 trajectories had terminal variance from 5.7% to 79.0% for the same system; 6 of 30 AlienCode trajectories ended below an earlier milestone; the largest single step contributed 47% to 92% of total gain. Lesson used here: real feedback beats more thinking, self designed experiments beat replay, discoveries are jumpy, and regressions happen, so integration needs a verified gate and consolidation matters.
2. **Economy of Minds (EOM)** (Harvard, MIT et al.; arXiv 2606.02859). Agents share one frozen LLM, differ by prompt, bid for control each step, pay the previous winner (bucket brigade), keep environment reward; rent, bankruptcy, births with prompt mutation. Results: MATH partial agents 15.9% to 57.0% vs 51.9% complete agent; Finance 60.0% vs 50.0% debate; exploration ablation drops the mean from 52.5 to 26.0 (more than removing exploitation, 33.5). Observed: specialists beat a generalist; topology evolves from long audit loops to compact pipelines. No token accounting in the paper. Lesson: prices replace conversation; credit assignment by payment chains; specialization emerges.
3. **Team-Economy repo** (Baiyu). Vendored `hayekmas` is EOM code; `codetest/` adds teams, private workspaces, scratchpad, pooled bids, one publication per round, bucket brigade to the previous publication's participants, progress dividends only on commit0. Structural problems found: no income on ProgramBench or NL2Repo during runs; money has no consequences (vendor rent, bankruptcy, births, coordination fee and wealth objective are not wired into the coding engine); bids scale with team size; one write slot per round; cooperation is prompted, not emergent; agent memory overwritten by the latest message.
4. **Multi agent cooperation through in context co player inference** (arXiv 2602.16301). Iterated prisoner's dilemma; cooperation emerges only when agents must infer opponents in context and the opponent pool is diverse (50% learners, 50% tabular). Telling agents the opponent identity, or removing the mixed pool, gives mutual defection. Lesson: keep diversity, consider anonymous histories, add a self interested condition to test whether cooperation is emergent.
5. **Anthropic, "How we built our multi agent research system"**: orchestrator plus parallel subagents; 90.2% over single agent on their research eval; token usage explains 80% of variance; agents use about 4x tokens, multi agent about 15x; coding is a poorer fit because of shared context dependencies; scale effort to complexity (1 agent with 3 to 10 calls for simple tasks, 2 to 4 subagents for comparisons, 10+ for complex).
6. **Claude, "When to fold subagents into your main agent loop"**: subagents are worth it only for self contained pieces with clean interfaces, parallel work with disposable context, black box verification; handoffs are a lossy "telephone game"; one customer saw a single loop about 2x faster.
7. **machineplaying.org** ("Is this machine playing?", MIT): single agent Eko with memory files; play experience improved later tasks (about 5.0 vs 2.5 five block towers). No economy there; the economic valuation framework would be this project's own contribution.
8. **OpenRSI Index**: single runs cost 5,800 to 20,900 H100 hours; only its Judge rules (tampering, leakage, budget overrun scored zero) are worth borrowing.

## 3. Experimental evidence available before the redesign

Results table (TinyDB, 201 tests, GPT-6-Luna low reasoning):

| Setting | Final tests | Accuracy | Confirmed cost |
| --- | --- | --- | --- |
| Economy k=1 | 174/201 | 86.57% | $0.1005 |
| Economy k=5 | 157/201 | 78.11% | $0.5744 |
| Economy k=10 | 188/201 | 93.53% | $1.1171 |
| Economy k=20 | 59/201 | 29.35% | $2.3985 |
| Economy + scratchpad k=5 | 201/201 | 100% | $0.7455 |
| Economy + scratchpad k=10 (original) | interrupted round 5 | no score | $0.7457 |
| Independent 10 + synthesis | 49/201 | 24.38% | $0.7706 |
| Decentralized 10 | 104/201 | 51.74% | $1.0139 |
| Centralized 9 + orchestrator | 24/201 | 11.94% | $0.6755 |

Replay video of a k=10 scratchpad run (seed 7): round 2 a 69 pass candidate lost to a 48 pass candidate because bids were 0.05 vs 0.075; Team 1 grew to 5 (the cap) and published almost every round with bids rising 0.05 to 0.325; 35 artifacts saved, 11 fetched; 472 note revisions; final 170/201 visible at $1.71.

A separate earlier discussion concluded: tightly coupled tasks need sequential single point integration; search like (discovery) tasks suit async + scale; async designs must handle staleness, diversity collapse, evaluator bottleneck and continuous credit; suggested experiments: same compute k=1 baseline, scaling curve, cross task negative control, and an ablation allowing multiple publishes per round on Commit-0.

## 4. Design conclusions reached in the chat

- Why agents spend tokens to communicate: an LLM agent's only state is its context, so communication is moving state between contexts; it is the same cost as memory for a single agent. Copies of one model mostly produce echoes; what is worth transmitting is environment evidence, commitments and interfaces, and results of divergent exploration. Messages also train the receiver in context (so verification matters). Cost shapes signals (sparse coding, iterated learning, costly signalling), but uniform token pricing only filters chatter; honesty needs stakes. Coase: team size is set by internal coordination cost versus market transaction cost.
- Simple, effective first change: a **regression gate** (a candidate scoring below the public high water mark cannot enter the auction), plus appending agent memory instead of overwriting it. Both implemented as opt in flags.
- Swarm redesign (implemented): divide the task, not the agents. Task board from failing visible tests (stigmergy), one claim call per woken agent, ephemeral groups, verified merge queue with 3 way patches so merges per round scale with k, bounties only for verified test flips, royalties for fetched artifacts, demand driven activation so idle agents cost nothing.
- Not yet implemented, discussed as next steps: money that buys compute (token priced actions and messages, with a base allowance to avoid deadlock); information market (bounties for discoveries and probes); second price auction if bids are ever kept; adversarial fixed strategy agents (free riders, claim and not deliver, spammers, tit for tat); wealth objective condition versus society objective; evolution of agents (vendor `evolution.py` is ready), of team charters, and of institution parameters in a slow outer loop; cross task consolidation (notes and skills that pay rent and earn when cited); message value measured by receiver surprisal; ProgramBench support via a fixed reference probe pool; NL2Repo via cross team test bounties.

## 5. What is implemented in this repository

Commit `82e1ec9` "Add opt-in regression gate and agent memory trace": `regression_gate` and `memory_trace` config flags, default off; `Engine.regressed()`; `regression_blocked` events; prompt suffix `+regression-gate`; config `configs/k5-gpt6-luna-scratchpad-gate.json`; tests in `tests/test_v1.py`.

Commit `b49dc18` "Add the swarm condition": `condition: "swarm"` (requires `shared_scratchpad: true`, commit0 or the scripted fixture only).

- `codetest/benchmarks.py`: `junit()` keeps per test ids as `failing`.
- `codetest/config.py`: `crowding_cap` (2), `bounty_base` (0.05), `bounty_growth` (0.25), `royalty_rate` (0.2); swarm control call estimate is k per round.
- `codetest/engine.py`: `candidate()` split into `work()` and `assess()`; `working_teams()` hook; `fetched_from` recorded for royalties.
- `codetest/swarm.py` (`SwarmEngine`): `board()` groups failing tests by module with growing bounties and a single `bootstrap` task until the package scores; `form()` wakes at most two agents per open group slot (richest first, random ties), one `claim` call each (open a group, join a leader, or rest), `crowding_cap` groups per task (double during bootstrap), groups last one round; `candidate()` with no discussion or ballot; `apply()` ranks by expected gain and `merge()` applies the first candidate by tree replacement and later ones with `git apply --3way` of the scratchpad `patch.diff`, re checks after each merge, requires at least one flipped test and no broken test, reverts otherwise; `pay()` mints bounties only for flips, royalty to fetched artifact authors, accounting assertion kept; `specialty` record per agent shown as `your_record`.
- `codetest/policy.py`: `SWARM_SYSTEM` prompt, `claim` schema, group rule, prompt revision `swarm-1`.
- `codetest/actions.py`: `claim` structured schema. `codetest/fixture.py`: fixture tests grouped `fixture.group_{i // 5}::module_{i:02}`, scripted claims. `codetest/cli.py`: engine selection. `codetest/report.py`: `swarm` metrics block (merges per round, 3 way merges, rejected merges by reason, duplicate claims, resting agent rounds, bounties, royalty share).
- Configs: `configs/demo-swarm.json` (k=8, 3 rounds, scripted), `configs/k10-gpt6-luna-swarm.json` (paid, mirrors `k10-gpt6-luna-scratchpad.json`).
- Tests: `tests/test_swarm.py` (8 tests). Full suite at this commit: 71 tests pass, 3 skipped (Docker or ProgramBench). Scripted demo: round 0 merged 3 candidates (1 direct, 2 three way), 8 tests flipped, 0.4 virtual bounty issued; later rounds correctly rejected 8 zero gain candidates. Replay page renders without JS errors.

## 6. Experiment plan agreed in the chat

1. Log diagnostics on existing runs (cheap, for the benchmark pair): winner team size, tie rate, rounds without merge, wasted candidates, fetch frequency, coordination share of cost (`phase_costs` exists in vendor code).
2. A/B: `configs/k10-gpt6-luna-swarm.json` versus `configs/k10-gpt6-luna-scratchpad.json`, same task, model and seed. Metrics: final hidden score, cost per passing test, `metrics.json` swarm block.
3. Scaling curve k in {1, 5, 10, 20} for swarm and for economy + scratchpad; same dollar k=1 baseline (k=1 at $0.75 is roughly 70 rounds).
4. Ablations: merge queue only, bounties only, board only.
5. Negative control on a discovery style task (ProgramBench probes) to test "task structure decides architecture".
6. Separate run for `memory_trace` and for `regression_gate` on the classic condition.
7. Later: wealth objective versus society objective; fixed strategy adversaries; evolution between episodes; report Gini, royalty share, verified progress per token.

## 7. Repository and delivery state

- GitHub `ruipwan/mas.s62-how-to-ai-an-ai` (pushed 2026-10-27 from Claude Code on the web): `main` = Baiyu's `58ebbf2` (the bootstrap workflow commit was replaced by a force push, as planned); `swarm` = the handoff state `4dd6520`; `claude/new-session-jku7c5` = `swarm` plus the session 2 commits in section 9. Upstream: https://github.com/bobyhaha/Team-Economy.
- Planned PRs to Baiyu's repo: first only the regression gate and memory trace (small, answers "bid is not always optimal"); the verified market after the k = 10 ablations have data. The swarm stays here as a comparison condition.
- Before paid runs: `python -m unittest discover -s tests -v`. Paid runs need Docker, prepared commit0 data and `OPENROUTER_API_KEY`, as in the README.

## 8. Working agreements with Ruipeng

- Principle for implementation: choose the simple, efficient option; keep the classic conditions untouched so A/B comparisons stay valid; new behavior behind opt in flags or a new condition.
- Replies in Chinese with English technical terms; direct and action first; few bullet points; never use dashes as punctuation in prose or commit messages (use commas and periods).
- Verify numbers against sources; say plainly what was not run or not verified.

## 9. Session 2 (Claude Code on the web, 2026-10-27)

- Restored the bundle and pushed the branches (section 7). Free checks: full unit suite, scripted demos, `plan` for the paid configs. The cloud environment could not run paid experiments: no `OPENROUTER_API_KEY`, and its network policy denied `openrouter.ai` and `datasets-server.huggingface.co` (needed by `prepare commit0`). Docker works there after starting `dockerd`.
- Review of the swarm commit found a leak into the classic conditions: `junit()` started returning per test ids (`failing`), which reached every agent through `public_feedback` and the ballot's `candidate_check`, up to 201 ids per observation on TinyDB, under an unchanged prompt revision. Fixed in `190dedb` with `engine.visible()`; the classic scripted demo again matches upstream `main` event for event apart from two added log fields, and the exported submission is identical.
- Found and fixed a latent race in `Repository.replace` (`d22f690`): `copy_source` keeps timestamps, so a same size rewrite could match git's cached stat data and be dropped. It made the swarm full run test fail for some hash seeds, and it fails on `4dd6520` too.
- Design discussion: the classic engine is the faithful Economy of Minds / Hayek machine skeleton (bid for control, bucket brigade, environment dividend) but without the half that gives money consequences (rent, bankruptcy, births are in `vendor/` but not wired). Its concrete failures are an auction blind to verified quality (69 vs 48 passing example), one write slot per round (k = 20 at 29%), and costless bids. The swarm fixes these by dropping the market, which also bundles five mechanisms into one A/B. Decision: keep the EOM skeleton and add the swarm's advantages as separate flags, so each is one ablation.
- Implemented the verified market: `merge_slots`, `rent`, `task_board` (README "Verified market"). Swarm's merge code was moved into `Engine.verified_merge()` and is shared. New `market` metrics measure whether bids track each candidate's verified gain, early versus late, for every classic run including the baseline.
- Revised experiment plan, TinyDB k = 10, same model and seed: baseline `k10-gpt6-luna-scratchpad`, then `-gate`, `-merge`, `-rent`, `-board`, `-market` (all). About $1 to $2 each. Then the k scaling curve for the best variant against the baseline and the swarm. Open ended phase after that: wire `vendor/hayekmas/adapters/teams/evolution.py` (bankruptcy removal, births by mutating the richest agent's strategy) across several commit0 repositories and evaluate on a held out repository, so evolved behavior must still cash out in verified test flips.
