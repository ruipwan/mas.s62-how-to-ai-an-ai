"""Bounded membership proposals and one team activation/pledge ballot per member.

All LLM calls still represent individual agents. Collective decisions use an
explicit voting rule, never a model acting as a privileged team representative.
"""

import math

from .mas import dumps


def form(engine, task, *, keep_calls):
    # Reserve the worst case: k proposals and at most k targeted invitation replies.
    if engine.remaining_calls() < 2 * len(engine.agents) + keep_calls:
        engine.emit("phase_skipped", phase="round_membership", reason="reserved answer calls")
        return
    engine.invitations.clear()
    engine.emit("membership_window", stage="round_start", schedule="compact",
                membership={a.name: a.team_tag for a in engine.agents})
    context = {"episode": engine.round, "step": engine.step, "task": task,
               "environment_state": engine.environment_state, "roster": engine.roster(),
               "coordination_fee_lambda": engine.config.coordination_fee_lambda}
    proposals, public = [], []
    order = list(engine.agents)
    engine.rng.shuffle(order)
    # Freeze membership during proposals; a late departure can make an early
    # invitation valid. Agents see earlier short public messages, not mutations.
    for agent in order:
        reply = engine.ask("round_membership", agent, {
            **context, "summary": agent.summary,
            "discussion": engine.tokens.clip(dumps(public), engine.config.evidence_tokens, tail=True),
        })
        leave, target = reply.get("leave"), reply.get("invite")
        reason, message = reply.get("reason"), reply.get("message", "")
        if (type(leave) is not bool or not isinstance(reason, str) or not reason.strip()
                or not isinstance(message, str)
                or (target is not None and (not isinstance(target, str) or engine.lookup(target) is None
                                           or target == agent.name))):
            engine.invalid(agent, "round_membership", "invalid proposal; membership unchanged")
            continue
        proposals.append((agent, leave, target, message, reason))
        public.append({"author": agent.name, "leave": leave, "invite": target,
                       "text": message, "reason": reason})
        engine.broadcast(agent, engine.agents, message or reason, "regroup")
        engine.emit("membership_decision", agent=agent.name, action="propose", leave=leave,
                    target=target, reason=reason, team=agent.team_tag)
    for agent, leave, _, _, _ in proposals:
        if leave and agent.team_tag:
            engine.apply_formation(agent, {"action": "leave"})
    for agent, _, target, message, reason in proposals:
        if target is not None:
            recipient = engine.lookup(target)
            size = sum(a.team_tag == agent.team_tag for a in engine.agents) if agent.team_tag else 1
            if recipient.team_tag is not None or size >= engine.config.max_team_size:
                engine.emit("invitation_skipped", agent=agent.name, target=target,
                            reason="recipient stayed in a team or inviter's team is full")
                continue
            engine.apply_formation(agent, {"action": "invite", "target": target, "text": message or reason})
    replies = 0
    engine.rng.shuffle(order)
    for agent in order:
        if agent.team_tag:
            continue
        invitations = []
        for invitation in engine.invitations.values():
            inviter = engine.lookup(invitation["from"])
            if invitation["to"] != agent.name or inviter.team_tag != invitation["team"]:
                continue
            size = sum(a.team_tag == inviter.team_tag for a in engine.agents) if inviter.team_tag else 1
            if size < engine.config.max_team_size:
                invitations.append(invitation)
        if not invitations:
            continue
        replies += 1
        reply = engine.ask("round_join", agent, {
            **context, "roster": engine.roster(), "invitations": invitations,
            "discussion": engine.tokens.clip(dumps(public), engine.config.evidence_tokens, tail=True),
        })
        selected, reason = reply.get("invitation"), reply.get("reason")
        if ("invitation" not in reply or not isinstance(reason, str) or not reason.strip()
                or (selected is not None and (not isinstance(selected, str)
                                              or selected not in {i["id"] for i in invitations}))):
            engine.invalid(agent, "round_join", "invalid acceptance; remain solo")
            continue
        engine.emit("membership_decision", agent=agent.name, action="accept" if selected else "pass",
                    reason=reason, invitation=selected, team=agent.team_tag)
        engine.broadcast(agent, engine.agents, reason, "regroup")
        if selected is not None:
            engine.apply_formation(agent, {"action": "accept", "invitation": selected})
    engine.invitations.clear()
    engine.emit("membership_committed", schedule="compact", proposals=len(order), invitation_replies=replies,
                membership={a.name: a.team_tag for a in engine.agents})


def commit(engine, task, groups, opening_wealth):
    contributions = {a.name: 0.0 for a in engine.agents}
    votes = {a.name: False for a in engine.agents}
    team_activation = {}
    for group, members in groups.items():
        transcript = engine.scratchpads.setdefault(group, [])
        observation = engine.scratchpad(
            task, members, transcript, phase="team_commit",
            wealth={a.name: opening_wealth[a.name] for a in members},
            activation_rule="strict majority of all members; ties/invalid votes mean no; no forced final action",
            bid_cost_rate=engine.config.bid_cost_rate,
            **({"bidding_rule": "fixed", "team_base_bid": engine.config.team_base_bid}
               if engine.config.team_bid_rule == "fixed" else {}),
        )
        order, messages = list(members), []
        engine.rng.shuffle(order)
        # Everyone sees the same completed discussion. No current ballot or
        # pledge is visible until all members have independently authorized it.
        for agent in order:
            reply = engine.ask("round_commit", agent, observation, engine.config.bid_tokens)
            act, amount, reason = reply.get("act"), reply.get("contribution"), reply.get("reason")
            fixed = engine.config.team_bid_rule == "fixed"
            if fixed:
                consent = reply.get("authorize_base_bid")
                valid = (type(act) is bool and type(consent) is bool
                         and isinstance(reason, str) and bool(reason.strip()))
                act = act is True and consent is True and opening_wealth[agent.name] >= (
                    engine.config.team_base_bid * engine.config.bid_cost_rate)
                amount = 0.0
            else:
                valid = (type(act) is bool and type(amount) in (int, float) and math.isfinite(amount)
                     and 0 <= amount <= opening_wealth[agent.name]
                     and isinstance(reason, str) and bool(reason.strip()))
            if valid:
                votes[agent.name] = act
                contributions[agent.name] = float(amount) if act else 0.0
                messages.append((agent, reason))
            else:
                engine.invalid(agent, "round_commit", "invalid ballot; no act vote or pledge")
            engine.emit("activation", agent=agent.name, group=group, active=votes[agent.name],
                        valid=valid, turn=0, reason=reason if isinstance(reason, str) else "Invalid ballot")
        yes = sum(votes[a.name] for a in members)
        active = yes > len(members) / 2
        team_activation[group] = active
        if engine.config.team_bid_rule == "fixed" and active:
            for member in members:
                contributions[member.name] = engine.config.team_base_bid / yes if votes[member.name] else 0.0
        offered = {a.name: contributions[a.name] for a in members}
        if not active:
            for agent in members:
                contributions[agent.name] = 0.0
        engine.emit("team_activation", group=group, members=[a.name for a in members], active=active,
                    yes_votes=yes, required_votes=len(members) // 2 + 1,
                    votes={a.name: votes[a.name] for a in members}, offered=offered,
                    bid=math.fsum(contributions[a.name] for a in members), rule="strict_majority")
        for agent, reason in messages:
            transcript.append({"author": agent.name, "text": reason, "step": engine.step,
                               "act": votes[agent.name], "pledge": contributions[agent.name]})
            engine.broadcast(agent, members, reason, "bidding")
        for agent in members:
            engine.emit("pledge", group=group, agent=agent.name, amount=contributions[agent.name], turn=0)
            engine.remember(agent, f"Team {'entered' if active else 'abstained'}; "
                            f"binding pledge {contributions[agent.name]} in round {engine.step}.")
        engine.emit("contributions_committed", group=group,
                    contributions={a.name: contributions[a.name] for a in members})
    return contributions, votes, team_activation
