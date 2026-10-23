"""Episode-boundary births/removals; historical team credit is already settled.

Shares upstream mutation prompts, probabilities and strict-negative bankruptcy.
Unlike HayekMAS, this team lifecycle does not replay a task or preserve specialist
roles. A completed team episode stays completed.
"""

from copy import deepcopy
import math

from hayekmas.base.prompts import (
    clean_trainable_prompt_output, format_mutate_agent_prompt, format_spawn_from_bankruptcy_prompt,
)
from .agent import TeamAgent


LIFECYCLE_RULES = (
    " Training population evolution occurs after the episode reward and reflection: wealth below zero "
    "causes removal (zero alone does not). New agents receive fresh initial wealth and start solo; "
    "they inherit no past team credit. Scheduled births may increase the population up to the stated cap. "
    "Only the editable strategy changes during mutation; immutable rules and role-free identity remain. "
)


def configure_agent(engine, agent):
    cfg = engine.config
    agent.frozen_system_prompt = (agent.COMPACT_ROUND_SYSTEM_PROMPT if cfg.round_schedule == "compact"
                                 else agent.ROUND_SYSTEM_PROMPT)
    agent.frozen_system_prompt += LIFECYCLE_RULES + (
        f" Population cap={cfg.num_agents * cfg.population_cap_multiplier}; initial wealth={cfg.initial_wealth}; "
        f"birth interval={cfg.birth_interval} tasks; births/event={cfg.num_births_per_interval}; "
        f"bankruptcy probabilities p_a={cfg.p_a}, p_b={cfg.p_b}; periodic good probability={cfg.periodical_good_p}; "
        f"rent={cfg.rent} every {cfg.rent_interval} training tasks (zero disables)."
    )
    agent.configure_objective(cfg)
    if cfg.token_profile == "solve_first":
        from .token_allocation import PHASE_REASONING
        agent.phase_reasoning_efforts = dict(PHASE_REASONING)


def birth(engine, source, kind, reason):
    from .mas import BudgetExceeded
    from .openrouter import SpendLimitExceeded

    cap = engine.config.num_agents * engine.config.population_cap_multiplier
    if len(engine.agents) >= cap:
        engine.emit("birth_skipped", reason="population cap", population_size=len(engine.agents), cap=cap)
        return False
    if source is None:
        engine.emit("birth_skipped", reason="no available parent", birth_kind=kind)
        return False
    if engine.remaining_calls() < 1:
        engine.emit("birth_skipped", reason="decision-call budget exhausted", birth_kind=kind)
        return False
    if kind == "good":
        prompt = format_mutate_agent_prompt(source.frozen_system_prompt, source.trainable_system_prompt)
    else:
        prompt = format_spawn_from_bankruptcy_prompt(
            source.frozen_system_prompt, source.trainable_system_prompt,
            agent_trace=source.recent_failure_trace,
            task_description=source.recent_failure_task, correct_answer=source.recent_failure_answer,
        )
    try:
        decision = engine.ask(f"birth_{kind}", source, {"mutation_specification": prompt}, engine.config.update_tokens)
    except (BudgetExceeded, SpendLimitExceeded):
        raise
    except Exception as exc:
        # A provider may raise a transport/parsing error after reserving money.
        # Its blocked state means unresolved spending, not a recoverable mutation.
        client = getattr(engine.policy, "client", None)
        if getattr(client, "blocked", False) or getattr(getattr(client, "native", None), "blocked", False):
            raise
        engine.emit("birth_failed", parent=source.name, birth_kind=kind, error_type=type(exc).__name__)
        return False
    text = decision.get("strategy")
    strategy = clean_trainable_prompt_output(text, source.frozen_system_prompt) if isinstance(text, str) else ""
    fallback = len(strategy) < 10
    strategy = source.trainable_system_prompt if fallback else engine.tokens.clip(strategy, engine.config.update_tokens)
    name = f"agent-{engine.next_agent_number}"
    engine.next_agent_number += 1
    child = TeamAgent(name, engine.config.initial_wealth)
    configure_agent(engine, child)
    child.trainable_system_prompt = strategy
    child.parent_agent_id, child.parent_agent_name = source.id, source.name
    child.father_agent_id, child.father_agent_name = source.father_agent_id, source.father_agent_name
    child.root_ancestor_class = source.root_ancestor_class
    child.spawn_method = f"{kind}_birth"
    child.public_summary = f"New independent agent; {kind} birth from {source.name}; no completed tasks."
    engine.population.add_agent(child)
    engine.inboxes[name] = []
    engine.birth_endowment_total += child.wealth
    state = next(a for a in engine.state()["agents"] if a["name"] == name)
    engine.emit("agent_born", agent=name, parent=source.name, birth_kind=kind, reason=reason,
                initial_wealth=child.wealth, strategy_fallback=fallback, state=state,
                population_size=len(engine.agents), cap=cap)
    return True


def evolve(engine, env, event_start):
    cfg = engine.config
    completed = engine.round + 1
    starting = list(engine.agents)
    for agent in starting:
        agent.tasks_lived += 1
        if (env.get_terminal_score() or 0) < 1:
            # Own decisions plus public submissions; never expose another team's private chat.
            evidence = [e for e in engine.events[event_start:]
                        if e["event"] == "submission" or
                        (e.get("agent") == agent.name and e["event"] in {"decision_response", "team_message"})]
            from .mas import dumps
            agent.recent_failure_trace = engine.tokens.clip(dumps(evidence), cfg.evidence_tokens, tail=True)
            agent.recent_failure_task = engine.tokens.clip(env.get_task_description(), cfg.environment_tokens)
            answer = env.get_correct_answer() if hasattr(env, "get_correct_answer") else ""
            agent.recent_failure_answer = engine.tokens.clip(str(answer or ""), cfg.evidence_tokens)
    if cfg.rent > 0 and cfg.rent_interval > 0 and completed % cfg.rent_interval == 0:
        for agent in starting:
            agent.lose_money(cfg.rent)
        engine.rent_burn_total += cfg.rent * len(starting)
        engine.emit("population_rent", cost_per_agent=cfg.rent, costs={a.name: cfg.rent for a in starting},
                    wealth={a.name: a.wealth for a in starting})
    bankrupt = [a for a in starting if a.check_bankruptcy()]
    for agent in bankrupt:
        state = next(a for a in engine.state()["agents"] if a["name"] == agent.name)
        agent.bankruptcy_episode = completed
        engine.retired_agents.append(deepcopy({**state, "bankruptcy_episode": completed}))
        old, dissolved = engine.team_manager.leave_team(agent)
        if old:
            engine.emit("left", agent=agent.name, team=old, reason="bankruptcy")
        if dissolved:
            engine.emit("team_dissolved", team=old, remaining=dissolved)
        # Signed balance: writing off debt removes a negative amount from the living ledger.
        engine.removed_wealth_total += agent.wealth
        engine.population.remove_agent(agent)
        engine.inboxes.pop(agent.name, None)
        engine.emit("agent_removed", agent=agent.name, reason="bankruptcy", state=state,
                    removed_wealth=agent.wealth, population_size=len(engine.agents))
    # No invitations or payment-chain references survive the task boundary.
    engine.invitations.clear()
    engine.previous_winner, engine.previous_members = None, ()
    engine.scratchpads.clear()
    for source in bankrupt:
        draw = engine.rng.random()
        if draw < cfg.p_a:
            birth(engine, max(engine.agents, key=lambda a: a.wealth, default=None), "good", "bankruptcy")
        elif draw < cfg.p_a + cfg.p_b:
            birth(engine, source, "bad", "bankruptcy")
    if cfg.birth_interval > 0 and completed % cfg.birth_interval == 0:
        for _ in range(cfg.num_births_per_interval):
            if len(engine.agents) >= cfg.num_agents * cfg.population_cap_multiplier:
                engine.emit("birth_skipped", reason="population cap", population_size=len(engine.agents))
                break
            good = engine.rng.random() < cfg.periodical_good_p
            source = ((max if good else min)(engine.agents, key=lambda a: a.wealth, default=None))
            birth(engine, source, "good" if good else "bad", "periodic")
    engine.public_summaries = {
        t.team_id: {"members": list(t.member_ids), "summary": t.summary}
        for t in engine.team_manager.active_teams(engine.round)
    }
    engine.assert_accounting()
    engine.emit("population_evolved", population_size=len(engine.agents),
                cap=cfg.num_agents * cfg.population_cap_multiplier,
                wealth=math.fsum(a.wealth for a in engine.agents),
                agents=engine.state()["agents"], accounting=engine.state()["accounting"])
