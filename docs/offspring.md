# Offspring as a controlled population experiment

The offspring option tests whether retaining and varying an agent's strategy can improve a coding society under a limited budget. It adds a population lifecycle to the existing market. It does not establish that reproduction improves coding performance, and it does not train or modify the underlying model weights.

The option is disabled by default. It is an explicit extension of `voluntary_pooled`; existing configurations retain their original behavior. Enable it with `"offspring": true` in an otherwise valid configuration. This document describes the experimental mechanism and proposed evaluation, not a recommendation to start additional paid runs.

## Mechanism

After every two completed rounds, the lifecycle may replace one mature active agent with low virtual wealth. A wealthy eligible parent transfers 0.25 units of its own virtual wealth to a child with a new, unique identity. Replacement requires an eligible retiree and sufficient parental funds. The active population stays at the configured size, `k`.

The child inherits a bounded strategy, the parent's bounded own note and a bounded copy of public repository notes. It does not inherit the parent's private inbox, team membership or historical contribution credit. A strategy contains at most four instructions drawn from eight fixed heuristics. Seeded local mutation adds, replaces or removes one instruction within that bound. It does not use LLM reflection or make an additional model call. The child then participates through the ordinary membership, work and publication mechanisms.

Retired agents remain as financial accounts. Payments owed to their earlier accepted contributions still reach those accounts. A child does not acquire those claims by copying its parent's strategy. The birth transfer creates no new virtual money, and accounting includes both active and retired balances, together with the existing reward issuance and rent or bid burns.

This conservation rule has a cost: positive balances held by retired accounts become unavailable to active agents, and later payments to retired contributors can reduce active capital further. Keeping total wealth conserved does not guarantee that the living population has enough wealth to remain productive. Active and retired balances must therefore be reported separately.

## What this changes

Swarm coordination and offspring address different questions. The existing swarm mechanism organizes work around visible failures, shared evidence and verified integration. A fixed population market allocates publication opportunities and transfers credit among persistent agents. Offspring adds inheritance, variation and replacement over time. It can be combined with verified integration, but reproduction itself does not verify a patch or resolve a dependency.

The working hypothesis is that a useful strategy can survive beyond its original holder, while replacement can redirect effort away from persistently weak participants. Possible countereffects include copying an unsuccessful habit, removing a useful specialist too early, reducing diversity, losing active capital and spending more context tokens on inherited material. Wealth can reflect payment-chain position, team membership and opportunity to act as well as useful work. A wealthy parent is a selection heuristic, not a verified label of coding ability.

The existing shared scratchpad already lets agents exchange notes and patches. Offspring changes which strategies occupy future work slots, rather than introducing information sharing for the first time. A useful extension should therefore beat simpler sharing controls, not merely benefit from copying another agent's notes.

Because mutation uses a fixed repertoire, this first version tests a constrained lifecycle rather than open-ended discovery of new strategies. An inherited note can also contain knowledge about the current task. Improvement on that task alone would not demonstrate transferable learning.

## Relationship to prior work

[Economy of Minds, version 2](https://arxiv.org/html/2606.02859v2) separates within-episode auctions from population adaptation across episodes. Its agents share a frozen model but differ in prompts. Wealthy agents produce variants, bankrupt agents are replaced, and new agents receive initial wealth. Evaluation freezes the learned population and disables economic updates. The present experiment instead replaces agents within one coding task, preserves a fixed active population and funds a child through a parent transfer. It is an adaptation inspired by EoM, not a reproduction of its protocol or evidence that its reported gains transfer to this setting.

The [official EoM implementation](https://github.com/zhentingqi/EoM) also includes domain-specific roles and wake-up rules. Its CloudCast implementation has roles such as planner, implementer and evaluator. Those constraints differ from the identical general instructions used by the current coding population.

[GEPA](https://arxiv.org/abs/2507.19457) proposes prompt updates by reflecting on execution traces and retaining complementary candidates. It supports studying strategy variation and selection, but does not establish that agent births outperform a fixed population under this experiment's budget.

The [Darwin Gödel Machine](https://arxiv.org/html/2505.22954v3) evolves an archive of coding-agent implementations, including their tools and workflows. Its search space and evaluation expense are substantially broader than a small prompt mutation. It provides a precedent for preserving useful ancestors and testing descendants, not a cost-matched justification for this lifecycle.

## Future comparisons

These are proposed controls, not additional experiments launched by enabling the feature:

| Condition | Purpose |
| --- | --- |
| Original fixed population market | Compare the whole extension, including its additional strategy and lineage context, with the existing market. |
| Frozen offspring framework | Use the same offspring prompt and initial strategies, but set `offspring_every`, `rounds` and `max_rounds` to 10 so no birth occurs within the ten-round run. Isolate the lifecycle from the added initial context. |
| Wealth-selected parent, clone without mutation | Isolate replacement and inheritance from strategy variation. |
| Wealth-selected parent, predefined mutation | Test the selected offspring mechanism. |
| Random eligible parent, same mutation | Test whether wealth-based parental selection adds value. |

Keep replacement timing, maturity requirements, retirement rules, endowment, model, task information, test protection and merge rules fixed wherever the comparison permits. Record the mutation identifier, parent and child identities, retirement reason and strategy hashes. Seeded lifecycle choices are reproducible; provider-generated responses are not guaranteed to be.

Each active agent retains the existing `work_calls` allocation. A fixed `k` therefore preserves scheduled work opportunities, but it does not guarantee equal tokens or dollars. Inherited text changes prompt length, team changes can alter control calls, and generated response lengths vary. Report quality against actual cumulative model spending and token use, including all control and evaluation work. Match the available strategy-context allowance in controls when testing inheritance, so extra memory capacity is not mistaken for an evolutionary benefit.

Evaluate several task instances and seeds with the same initial conditions. Measure final held-out quality, verified progress per dollar, regressions, accepted contributions, active capital, lineage diversity and the fraction of children that actually receive work. If claiming transferable adaptation, evolve on training tasks, freeze the resulting strategies and evaluate on separate tasks without further selection.

## Low-budget interpretation

A short run provides few replacement opportunities. A late child may receive little or no work, and a single successful lineage may reflect favorable task allocation or stochastic output. Visible tests are development feedback, not independent evidence of generalization. Synthetic fixtures can validate population size, inheritance boundaries, deterministic lifecycle choices and financial conservation; they cannot establish a coding-performance advantage.

A useful result would be a repeatable quality or efficiency improvement over the simpler controls at a matched total budget. Until that comparison exists, offspring should be described as an experimental population mechanism with explicit tradeoffs.

For offspring runs, membership persistence compares only identities present in both consecutive snapshots. A new identity is not counted as an agent switching teams. The standard wealth Gini refers to active agents; retired wealth is reported separately.
