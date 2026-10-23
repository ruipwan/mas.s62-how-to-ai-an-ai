"""Round-level voluntary teaming with per-member, per-action path rewards.

The legacy engine uses `round` for an episode/task and `step` for a decision
round. Keep those storage keys for existing replays; expose both in observations.
"""

from collections import Counter
from dataclasses import asdict
import math

from .agent import TeamAction
from .mas import BudgetExceeded, dumps


def balances(engine):
    return {agent.name: agent.wealth for agent in engine.agents}


def membership(engine):
    return {agent.name: agent.team_tag for agent in engine.agents}


def reserve(engine):
    return engine.finalization_reserve(min(len(engine.agents), engine.config.max_team_size))


def auction_calls(engine):
    return sum(len(members) * engine.bidding_turns(len(members)) for members in engine.groups().values())


def formation_window(engine, task, enabled, *, keep_calls):
    """A public short conversation followed by consent-based membership actions."""
    if not enabled or not engine.formation_enabled():
        return
    if engine.compact_rounds:
        from .compact_rounds import form

        return form(engine, task, keep_calls=keep_calls)
    cost = len(engine.agents) * (engine.config.formation_turns + 1)
    if engine.remaining_calls() < cost + keep_calls:
        engine.emit("phase_skipped", phase="round_membership", reason="reserved answer calls")
        return
    discussion = []
    context = {
        "episode": engine.round, "step": engine.step, "task": task,
        "environment_state": engine.environment_state,
        "stage": "round_start", "previous_step": engine.step - 1 if engine.step else None,
        "coordination_fee_lambda": engine.config.coordination_fee_lambda,
    }
    engine.emit("membership_window", stage=context["stage"], membership=membership(engine))
    order = list(engine.agents)
    engine.rng.shuffle(order)
    for agent in order:
        observation = {**context, "roster": engine.roster(),
                       "discussion": engine.tokens.clip(dumps(discussion), engine.config.evidence_tokens, tail=True)}
        reply = engine.ask("round_recap", agent, observation)
        message = reply.get("message", "")
        if not isinstance(message, str):
            engine.invalid(agent, "round_recap", "message must be text")
            continue
        discussion.append({"author": agent.name, "text": message})
        engine.broadcast(agent, engine.agents, message, "regroup")
    engine.formation_context = {
        **context, "public_discussion": engine.tokens.clip(dumps(discussion), engine.config.evidence_tokens, tail=True)
    }
    engine.form_teams()
    engine.emit("membership_committed", membership=membership(engine))


def coordinate(engine, task):
    """Consent to a maximum fee, allow departures, then charge surviving groups."""
    rate = engine.config.coordination_fee_lambda
    fees = {agent.name: 0.0 for agent in engine.agents}
    departures = []
    for group, members in engine.groups().items():
        if len(members) < 2:
            continue
        maximum_fee = rate * (len(members) - 1)
        for agent in members:
            if agent.wealth < maximum_fee:
                participate, reason = False, "Cannot afford the announced maximum fee"
            else:
                reply = engine.ask("coordinate", agent, {
                    "task": task, "team": group, "members": [a.name for a in members],
                    "maximum_fee": maximum_fee, "lambda": rate,
                    "fee_rule": "lambda * (remaining team size - 1), once per round, including losing teams",
                })
                participate = reply.get("participate") is True
                reason = reply.get("reason", "")
                if type(reply.get("participate")) is not bool or not isinstance(reason, str) or not reason.strip():
                    engine.invalid(agent, "coordinate", "invalid consent; leave without a charge")
                    participate, reason = False, "No valid consent"
            engine.emit("coordination_decision", agent=agent.name, group=group,
                        participate=participate, maximum_fee=maximum_fee, reason=reason)
            if not participate:
                departures.append(agent)
    for agent in departures:
        if agent.team_tag:
            engine.apply_formation(agent, {"action": "leave"})
    for group, members in engine.groups().items():
        fee = rate * (len(members) - 1)
        if not fee:
            continue
        for agent in members:
            agent.lose_money(fee)
            fees[agent.name] = fee
            engine.remember(agent, f"Paid coordination fee {fee} in decision round {engine.step}.")
        burned = fee * len(members)
        engine.coordination_burn_total += burned
        engine.emit("coordination_fee", group=group, members=[a.name for a in members],
                    lambda_rate=rate, per_member=fee, costs={a.name: fee for a in members},
                    burned=burned, wealth=balances(engine))
    engine.assert_accounting()
    return fees


def discuss_before_bidding(engine, task, keep_calls):
    fees = {agent.name: 0.0 for agent in engine.agents}
    if engine.config.coordination_fee_lambda > 0:
        # Never charge unless the complete consent window AND one discussion turn fit.
        if engine.remaining_calls() < 2 * len(engine.agents) + keep_calls:
            engine.emit("phase_skipped", phase="pre_bid_discussion", reason="reserved answer calls; no fee charged")
            return fees
        fees = coordinate(engine, task)
    turns = 1 if engine.compact_rounds else engine.config.discussion_turns
    for turn in range(turns):
        if engine.remaining_calls() < len(engine.agents) + keep_calls:
            engine.emit("phase_skipped", phase="pre_bid_discussion", reason="reserved answer calls")
            break
        for group, members in engine.groups().items():
            transcript = engine.scratchpads.setdefault(group, [])
            order = list(members)
            engine.rng.shuffle(order)
            for agent in order:
                reply = engine.ask("round_chat", agent, engine.scratchpad(
                    task, members, transcript, phase="pre_bid_discussion", turn=turn,
                    **({"next_decision": "One simultaneous act vote and personal pledge per member; "
                        "strict majority activates the team. Discuss readiness and proposed contributions now."}
                       if engine.compact_rounds else {})))
                message = reply.get("message", "")
                if not isinstance(message, str):
                    engine.invalid(agent, "round_chat", "message must be text")
                    continue
                transcript.append({"author": agent.name, "text": message, "step": engine.step})
                engine.broadcast(agent, members, message, "pre_bid")
    return fees


def select_action(engine, task, group, members, final, *, closing_public_work=False):
    # Historical finalizers may have changed teams. Do not expose the current
    # occupants' private scratchpad to former members.
    transcript = [] if closing_public_work else engine.scratchpads.setdefault(group, [])
    candidates = []
    if engine.config.shared_draft_passes:
        for turn in range(engine.config.shared_draft_passes):
            keep = engine.finalization_reserve(len(members)) if final else len(members) + reserve(engine)
            if engine.remaining_calls() < len(members) + keep:
                engine.emit("phase_skipped", phase="shared_work", reason="reserved finalization calls")
                break
            order = list(members)
            engine.rng.shuffle(order)
            for agent in order:
                current = candidates[-1]["answer"] if candidates else None
                reply = engine.ask("shared_work", agent, engine.scratchpad(
                    task, members, transcript, phase="shared_drafting", turn=turn,
                    shared_draft=current, preparing_final=final, must_finalize=False),
                    engine.config.solution_tokens)
                message = reply.get("message", "")
                if isinstance(message, str) and message:
                    transcript.append({"author": agent.name, "text": message})
                    engine.broadcast(agent, members, message, "team")
                answer = reply.get("candidate")
                if isinstance(answer, str) and answer.strip():
                    engine.record_candidate(agent, members, candidates, answer, False)
                elif answer is not None:
                    engine.invalid(agent, "shared_work", "invalid draft; previous draft retained")
        # Only the current version is eligible; superseded drafts remain in logs.
        candidates = candidates[-1:]

    if final:
        candidates = engine.finalize(members, task, transcript, candidates, "end of episode")
    elif not engine.config.shared_draft_passes:
        for turn in range(engine.config.discussion_turns):
            order = list(members)
            engine.rng.shuffle(order)
            for agent in order:
                reply = engine.ask("round_work", agent, engine.scratchpad(
                    task, members, transcript, phase="round_action", turn=turn,
                    candidates=candidates, must_finalize=False), engine.config.solution_tokens)
                message = reply.get("message", "")
                if not isinstance(message, str):
                    engine.invalid(agent, "round_work", "message must be text")
                    message = ""
                if message:
                    transcript.append({"author": agent.name, "text": message})
                    engine.broadcast(agent, members, message, "team")
                answer = reply.get("candidate")
                if isinstance(answer, str) and answer.strip():
                    engine.record_candidate(agent, members, candidates, answer, False)
    if not candidates:
        engine.emit("no_submission", reason="no valid final candidate" if final else "no intermediate candidate")
        return None
    votes = Counter()
    ids = {candidate["id"] for candidate in candidates}
    for agent in members:
        reply = engine.ask("vote", agent, {
            "task": task, "candidates": candidates,
            "discussion": engine.tokens.clip(dumps(transcript), engine.config.evidence_tokens, tail=True),
        })
        selected = reply.get("candidate_id")
        if isinstance(selected, str) and selected in ids:
            votes[selected] += 1
            engine.emit("vote", agent=agent.name, candidate_id=selected)
        else:
            engine.invalid(agent, "vote", "invalid candidate ID; abstention")
    if votes:
        best = max(votes.values())
        selected = engine.rng.choice(sorted(key for key, count in votes.items() if count == best))
    elif final:
        # Keep a real complete proposal when only the selection interface fails.
        selected = engine.rng.choice(sorted(ids))
        engine.emit("selection_fallback", reason="no valid final votes", candidate_id=selected)
    else:
        engine.emit("no_submission", reason="no valid intermediate votes")
        return None
    candidate = next(value for value in candidates if value["id"] == selected)
    engine.emit("submission", candidate_id=selected, answer=candidate["answer"], final=final,
                votes=dict(votes), members=[agent.name for agent in members], group=group)
    return TeamAction(candidate["answer"], group, final)


def pay_path(engine, path, reward):
    """Issue a full R/N to every saved member for each accepted action."""
    if not math.isfinite(reward) or reward < 0:
        raise ValueError("Round-team protocol requires a finite nonnegative terminal reward")
    income = {agent.name: 0.0 for agent in engine.agents}
    share = reward / len(path) if path else 0.0
    payouts = []
    for contribution in path:
        credits = {name: share for name in contribution["members"]}
        for name, amount in credits.items():
            engine.lookup(name).gain_money(amount)
            income[name] += amount
        payouts.append({**contribution, "credits": credits})
    issued = math.fsum(income.values())
    engine.reward_total += issued
    engine.emit("path_reward", environment_reward=reward, contributing_rounds=len(path),
                per_member_per_round=share, credits=income, payouts=payouts, issued=issued,
                wealth=balances(engine))
    return income, payouts, issued


def run_episode(engine, env, *, formation=True, reflection=True, training=True):
    if not engine.agents:
        raise RuntimeError("Population is empty; no agents can act")
    episode_event_start = len(engine.events)
    env.initialize()
    task = env.task.public()
    engine.previous_winner, engine.previous_members = None, ()
    engine.scratchpads = {}
    engine.step = 0
    engine.environment_state = engine.tokens.clip(env.get_state_description(), engine.config.environment_tokens, tail=True)
    paid = {agent.name: 0.0 for agent in engine.agents}
    coordination_paid = dict(paid)
    income, bid_income = dict(paid), dict(paid)
    contributions, opening = dict(paid), balances(engine)
    path, steps, payouts = [], [], []
    reward = issued = 0.0
    winner, members = None, ()
    for step in range(engine.config.max_steps):
        engine.step = step
        engine.environment_state = engine.tokens.clip(env.get_state_description(), engine.config.environment_tokens, tail=True)
        engine.emit("decision_round_started", episode=engine.round, decision_round=step,
                    membership=membership(engine), wealth=balances(engine))
        if engine.remaining_calls() < reserve(engine):
            raise BudgetExceeded("Insufficient calls for a complete final-answer window")
        # Every round starts with public membership discussion, then private team work.
        discussion_reserve = (len(engine.agents) * (2 if engine.config.coordination_fee_lambda > 0 else 1)
                              if engine.compact_rounds else 0)
        final = step == engine.config.max_steps - 1
        closing_public_work = engine.config.terminal_policy == "public_work" and final and bool(path)
        formation_window(engine, task, formation and not closing_public_work,
                         keep_calls=len(engine.agents) * engine.bidding_turns(engine.config.max_team_size)
                         + reserve(engine) + discussion_reserve)
        final = step == engine.config.max_steps - 1
        if closing_public_work:
            winner = path[-1]["team"]
            members = tuple(engine.lookup(name) for name in path[-1]["members"])
            contributions, opening = dict.fromkeys(paid, 0.0), balances(engine)
        elif engine.remaining_calls() >= auction_calls(engine) + reserve(engine) + discussion_reserve:
            fees = discuss_before_bidding(engine, task, auction_calls(engine) + reserve(engine))
            for name, fee in fees.items():
                coordination_paid[name] += fee
            winner, members, contributions, opening = engine.auction(task)
            for agent in members:
                paid[agent.name] += contributions[agent.name] * engine.config.bid_cost_rate
            if members and engine.previous_members:
                payment = math.fsum(contributions[agent.name] for agent in members) * engine.config.bid_cost_rate
                for agent in engine.previous_members:
                    bid_income[agent.name] += payment / len(engine.previous_members)
            work_calls = len(members) * (engine.config.discussion_turns + 1)
            if engine.remaining_calls() < work_calls + auction_calls(engine) + reserve(engine) + discussion_reserve:
                final = True
                engine.emit("phase_skipped", phase="later_rounds", reason="reserved answer calls")
        else:
            if engine.compact_rounds:
                engine.emit("phase_skipped", phase="auction",
                            reason="insufficient calls for discussion, ballots and final answer reserve")
            winner, members, contributions, opening = None, (), dict(paid), balances(engine)
            contributions = {name: 0.0 for name in contributions}
            final = True
        if not members and final and engine.config.terminal_policy == "public_work" and path:
            closing_public_work = True
            winner = path[-1]["team"]
            members = tuple(engine.lookup(name) for name in path[-1]["members"])
        if closing_public_work:
            engine.emit("public_work_finalization", winner=winner,
                        members=[a.name for a in members], source_step=path[-1]["step"],
                        bid_charged=0.0, adds_path_credit=False)
        if not members and final and not engine.compact_rounds and engine.config.terminal_policy == "funded":
            groups = engine.groups()
            winner = engine.rng.choice(sorted(groups))
            members = tuple(groups[winner])
            engine.emit("finalization_recovery", reason="no funded final-round team", winner=winner,
                        members=[agent.name for agent in members], bid_charged=0.0)
        elif not members and final:
            engine.emit("no_submission", reason="no funded, majority-approved team at deadline; abstention respected")
        action = select_action(engine, task, winner, members, final, closing_public_work=closing_public_work) if members else None
        if action is not None:
            result = env.apply(action)
            if not closing_public_work:
                path.append({"step": step, "team": winner, "members": [agent.name for agent in members],
                             "final": action.final})
            if action.final:
                reward = result
            elif result != 0:
                raise ValueError("Round-team protocol expects reward only for the final answer")
        if final:
            income, payouts, issued = pay_path(engine, path, reward)
        record = {
            "step": step, "task_id": task["id"], "winner": winner,
            "closing_public_work": closing_public_work,
            "members": [agent.name for agent in members], "accepted": action is not None,
            "final": bool(action and action.final), "score": env.get_terminal_score(),
            "reward": reward if final else 0.0, "share": reward / len(path) if final and path else 0.0,
            "contributions": dict(contributions), "wealth": balances(engine), "membership": membership(engine),
        }
        engine.emit("settlement", **record, reward_distribution="per_member_path",
                    credits=income if final else {}, issued=issued if final else 0.0)
        steps.append(record)
        engine.emit("step_complete", metrics=record)
        engine.assert_accounting()
        if members:
            engine.previous_winner, engine.previous_members = winner, tuple(members)
        engine.environment_state = engine.tokens.clip(env.get_state_description(), engine.config.environment_tokens, tail=True)
        if final:
            break
    engine.step = None
    for agent in engine.agents:
        summary = {"episode": engine.round, "team": agent.team_tag, "wealth": agent.wealth,
                   "paid": paid[agent.name], "bid_income": bid_income[agent.name],
                   "coordination_paid": coordination_paid[agent.name],
                   "reward_received": income[agent.name],
                   "contributing_rounds": sum(agent.name in item["members"] for item in path)}
        engine.remember(agent, dumps(summary))
        agent.public_summary = engine.tokens.clip(dumps(summary), engine.config.summary_tokens)
    for team in engine.team_manager.active_teams(engine.round):
        record = {"episode": engine.round, "members": list(team.member_ids),
                  "current_members_reward": math.fsum(income[name] for name in team.member_ids),
                  "accepted_rounds": [item["step"] for item in path if item["team"] == team.team_id]}
        team.trajectory = (team.trajectory + [record])[-8:]
        team.summary = engine.tokens.clip(dumps(team.trajectory), engine.config.summary_tokens, tail=True)
    engine.public_summaries = {team.team_id: {"members": list(team.member_ids), "summary": team.summary}
                               for team in engine.team_manager.active_teams(engine.round)}
    if reflection:
        if engine.remaining_calls() >= 3 * len(engine.agents):
            engine.reflect()
        else:
            engine.emit("phase_skipped", phase="reflection", reason="insufficient remaining calls")
    engine.assert_accounting()
    metric = {
        "round": engine.round, "task_id": task["id"], "protocol": "rounds", "score": env.get_terminal_score(),
        "reward": reward, "reward_issued": issued, "reward_distribution": "per_member_path",
        "contributing_rounds": len(path), "credit_path": payouts, "winner": winner,
        "steps": steps, "step_count": len(steps), "paid": paid, "bid_income": bid_income,
        "reward_income": income, "team_sizes": [len(group) for group in engine.groups().values()],
        "coordination_paid": coordination_paid, "coordination_burn_total": engine.coordination_burn_total,
        "wealth": balances(engine), "membership": membership(engine), "contributions": contributions,
        "contribution_fraction": {name: contributions[name] / opening[name] if opening[name] else 0 for name in opening},
        "bid_paid_total": engine.bid_paid_total, "bid_transfer_total": engine.bid_transfer_total,
        "bid_burn_total": engine.bid_burn_total, "reflection_burn_total": engine.reflection_burn_total,
        "reward_total": engine.reward_total, "invalid_actions": engine.invalid_actions,
    }
    if training and engine.config.evolution_enabled:
        from .evolution import evolve
        metric["population_before_evolution"] = len(engine.agents)
        evolve(engine, env, episode_event_start)
        metric["population_size"] = len(engine.agents)
        metric["wealth"] = balances(engine)
        metric["membership"] = membership(engine)
        metric["team_sizes"] = [len(group) for group in engine.groups().values()]
        metric["evolution_accounting"] = engine.state()["accounting"]
    engine.metrics.append(metric)
    engine.emit("round_complete", metrics=metric, agents=engine.state()["agents"],
                teams=[asdict(team) for team in engine.team_manager.active_teams(engine.round)])
    engine.round += 1
    return metric
