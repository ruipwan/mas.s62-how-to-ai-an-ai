"""Team auction execution using EoM's BaseAgent, Population and BaseEnv.

The original HayekMAS individual auction remains available unchanged. This
engine extends the within-task payment chain to voluntary teams.
"""

from collections import Counter
from dataclasses import asdict
import json
import math
import random

from hayekmas.base.population import Population
from .agent import TeamAction, TeamAgent
from .config import TokenBudget
from .policy import INSTRUCTIONS
from .team import TeamManager


class BudgetExceeded(RuntimeError):
    pass


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


class TeamMAS:
    def __init__(self, config, policy, event_sink=None):
        self.config = config
        self.policy = policy
        self.tokens = TokenBudget()
        self.rng = random.Random(config.seed)
        self.population = Population()
        for i in range(config.num_agents):
            agent = TeamAgent(f"agent-{i}", config.initial_wealth)
            if config.interaction_protocol == "rounds":
                agent.frozen_system_prompt = (agent.COMPACT_ROUND_SYSTEM_PROMPT if config.round_schedule == "compact"
                                              else agent.ROUND_SYSTEM_PROMPT)
            if config.token_profile == "solve_first":
                from .token_allocation import PHASE_REASONING

                agent.phase_reasoning_efforts = dict(PHASE_REASONING)
            agent.configure_objective(config)
            self.population.add_agent(agent)
        self.round = 0
        self.step = None
        self.previous_winner = None
        self.previous_members = ()
        self.team_manager = TeamManager(self.population, config.max_team_size)
        self.invitations = self.team_manager.invitations
        self.scratchpads = {}
        self.public_summaries = {}
        self.inboxes = {a.name: [] for a in self.agents}
        self.events = []
        self.metrics = []
        self.event_sink = event_sink
        self.calls = 0
        self.requested_output_tokens = 0
        self.reference_input_tokens = 0
        self.invalid_actions = 0
        self.reward_total = 0.0
        self.bid_burn_total = 0.0
        self.bid_paid_total = 0.0
        self.bid_transfer_total = 0.0
        self.reflection_burn_total = 0.0
        self.coordination_burn_total = 0.0
        self.initial_total = config.num_agents * config.initial_wealth
        self.birth_endowment_total = 0.0
        self.removed_wealth_total = 0.0
        self.rent_burn_total = 0.0
        self.next_agent_number = config.num_agents
        self.retired_agents = []
        self.training = True
        if config.evolution_enabled:
            from .evolution import configure_agent
            for agent in self.agents:
                configure_agent(self, agent)
        if config.condition == "random_fixed":
            shuffled = list(self.agents)
            self.rng.shuffle(shuffled)
            for offset in range(0, len(shuffled), config.fixed_team_size):
                group = shuffled[offset : offset + config.fixed_team_size]
                if len(group) > 1:
                    self.team_manager.create_team(group, 0)
        self.emit(
            "initialized", condition=config.condition, backend=policy.label, config=asdict(config), roster=self.roster()
        )

    @property
    def compact_rounds(self):
        return self.config.interaction_protocol == "rounds" and self.config.round_schedule == "compact"

    @property
    def agents(self):
        return self.population.get_all()

    def lookup(self, name):
        return next((a for a in self.agents if a.name == name), None)

    def emit(self, kind, **data):
        event = {"event": kind, "round": self.round, "step": self.step, **data}
        self.events.append(event)
        if self.event_sink:
            self.event_sink(event)

    def invalid(self, agent, phase, reason):
        self.invalid_actions += 1
        self.emit("invalid_action", agent=agent.name, phase=phase, reason=reason)

    def roster(self):
        return [
            {"name": a.name, "team": a.team_tag, "wealth": a.wealth, "public_summary": a.public_summary}
            for a in self.agents
        ]

    def groups(self):
        groups = {}
        for agent in self.agents:
            groups.setdefault(agent.team_tag or f"solo:{agent.name}", []).append(agent)
        return groups

    def remember(self, agent, entry):
        agent.remember(entry, self.tokens, self.config.summary_tokens)

    def remaining_calls(self):
        remaining = self.config.max_calls - self.calls
        policy_remaining = getattr(self.policy, "remaining_calls", None)
        if callable(policy_remaining):
            remaining = min(remaining, policy_remaining())
        return max(0, remaining)

    @staticmethod
    def finalization_reserve(member_count):
        # One final proposal, at most one correction, and one vote per member;
        # reserve one judge call even for environments that do not need it.
        return 3 * member_count + 1

    def auction_reserve(self):
        groups = self.groups().values()
        bidding = sum(len(g) * self.bidding_turns(len(g)) for g in groups)
        if self.config.bidding_mode == "sealed":
            bidding += len(self.agents)  # One optional all-zero funding repair.
        largest = max(len(g) for g in groups)
        return bidding + self.collaboration_reserve(largest)

    def bidding_turns(self, member_count):
        if self.compact_rounds:
            return 1
        return self.config.bidding_turns if member_count > 1 and self.config.bidding_mode == "negotiated" else 1

    def collaboration_reserve(self, member_count):
        if self.config.collaboration_mode == "reviewed":
            from .reviewed import call_reserve

            return call_reserve(member_count)
        return self.finalization_reserve(member_count)

    def ask(self, phase, agent, observation, max_tokens=None):
        cap = max_tokens or self.config.action_tokens
        observation = {**observation, "you": {"name": agent.name, "wealth": agent.wealth, "team": agent.team_tag}}
        if self.config.token_profile == "solve_first":
            if phase in {"round_membership", "round_join", "round_commit"}:
                from .token_allocation import control_observation

                observation = control_observation(self, phase, agent, observation)
            if phase == "round_chat":
                cap = self.config.discussion_tokens
            elif phase == "round_commit":
                cap = min(cap, 96)
            elif phase == "vote":
                cap = min(cap, 64)
        if self.calls >= self.config.max_calls:
            raise BudgetExceeded(f"Decision call budget {self.config.max_calls} exhausted")
        prompt = dumps({"instruction": INSTRUCTIONS[phase], "observation": observation})
        input_tokens = self.tokens.count(agent.get_system_prompt()) + self.tokens.count(prompt)
        if input_tokens > self.config.context_tokens:
            raise BudgetExceeded(f"{phase} observation exceeds context_tokens; increase the configured bound")
        self.calls += 1
        self.requested_output_tokens += cap
        self.reference_input_tokens += input_tokens
        # The audit log is omniscient; it is never passed back as observation.
        self.emit(
            "decision_request",
            agent=agent.name,
            phase=phase,
            observation=observation,
            strategy=agent.trainable_system_prompt,
            input_tokens=input_tokens,
            output_limit=cap,
            **({"reasoning_effort": agent.phase_reasoning_efforts.get(phase)}
               if self.config.token_profile == "solve_first" else {}),
        )
        response = self.policy.respond(phase, agent, observation, cap)
        if not isinstance(response, str):
            raise TypeError("Policy must return JSON text")
        self.emit("decision_response", agent=agent.name, phase=phase, response=response)
        if self.tokens.count(response) > cap:
            self.invalid(agent, phase, "response exceeds reference token limit")
            return {}
        try:
            result = json.loads(response, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        except (ValueError, TypeError):
            self.invalid(agent, phase, "malformed JSON")
            return {}
        if not isinstance(result, dict):
            self.invalid(agent, phase, "response must be an object")
            return {}
        return result

    def reflection_observation(self, agent):
        # Resolution: detailed own bounded trace > teammate summary > public team outcome.
        peers = [a for a in self.agents if a is not agent and agent.team_tag and a.team_tag == agent.team_tag]
        others = {key: value for key, value in self.public_summaries.items() if agent.name not in value["members"]}
        total = self.config.evidence_tokens
        evidence = {
            "self": self.tokens.clip("\n".join(agent.trajectory), total // 2 - 32, tail=True),
            "teammates": self.tokens.clip(dumps({a.name: a.summary for a in peers}), total // 4 - 32),
            "other_teams": self.tokens.clip(dumps(others), total // 4 - 32),
        }
        # JSON escaping can add tokens; enforce the bound on the serialized evidence.
        while self.tokens.count(dumps(evidence)) > total:
            largest = max(evidence, key=lambda k: self.tokens.count(evidence[k]))
            evidence[largest] = self.tokens.clip(evidence[largest], max(0, self.tokens.count(evidence[largest]) - 32))
        return evidence

    def judge(self, prompt):
        """Environment-only model call; never enters any agent observation."""
        if self.calls >= self.config.max_calls:
            raise BudgetExceeded("Decision/judge call budget exhausted")
        count = self.tokens.count(prompt)
        if count > self.config.context_tokens:
            raise BudgetExceeded("Rubric judge context exceeds context_tokens")
        self.calls += 1
        self.reference_input_tokens += count
        self.requested_output_tokens += self.config.judge_tokens
        self.emit("judge_request", input_tokens=count, output_limit=self.config.judge_tokens)
        options = {}
        if getattr(self.policy.client, "api_name", None) == "openrouter":
            options["kind"] = "judge"
            if self.config.token_profile == "solve_first":
                options["reasoning_effort"] = "medium"
        response = self.policy.client.generate(prompt, max_tokens=self.config.judge_tokens, **options)
        self.emit("judge_response", response=response)
        return response

    def reflect(self):
        for agent in self.agents:
            choice = self.ask(
                "reflect",
                agent,
                {
                    "round": self.round,
                    "wealth": agent.wealth,
                    "reflection_cost": self.config.reflection_cost,
                    "summary": agent.summary,
                },
            )
            if choice.get("reflect") is not True:
                continue
            if agent.wealth < self.config.reflection_cost:
                self.invalid(agent, "reflect", "insufficient wealth")
                continue
            if self.calls + 2 > self.config.max_calls:
                raise BudgetExceeded("Insufficient decision-call budget to complete reflection")
            agent.lose_money(self.config.reflection_cost)
            self.reflection_burn_total += self.config.reflection_cost
            self.remember(agent, f"Paid {self.config.reflection_cost} for reflection after round {self.round}.")
            self.emit("reflection_paid", agent=agent.name, cost=self.config.reflection_cost)
            evidence = self.reflection_observation(agent)
            inspection = self.ask("inspect", agent, evidence, self.config.inspect_tokens)
            analysis = inspection.get("analysis")
            if not isinstance(analysis, str) or not analysis.strip():
                self.invalid(agent, "inspect", "missing analysis; fee remains spent")
                continue
            improvement = self.ask("improve", agent, {"analysis": analysis}, self.config.update_tokens)
            strategy = improvement.get("strategy")
            if isinstance(strategy, str) and strategy.strip():
                agent.trainable_system_prompt = self.tokens.clip(strategy, self.config.update_tokens)
                self.remember(agent, "Updated my strategy: " + agent.trainable_system_prompt)
                self.emit("strategy_updated", agent=agent.name, strategy=agent.trainable_system_prompt)
            else:
                self.invalid(agent, "improve", "missing strategy; fee remains spent")

    def formation_enabled(self):
        if self.config.interaction_protocol == "rounds" and self.config.condition == "dynamic":
            return True
        return (self.config.condition == "dynamic" and self.round % self.config.negotiation_interval == 0) or (
            self.config.condition == "self_selected_fixed" and self.round == 0
        )

    def deliver(self, sender, recipient, text, channel):
        text = self.tokens.clip(text, self.config.action_tokens)
        message = {"from": sender.name, "text": text, "channel": channel}
        self.inboxes[recipient.name].append(message)
        self.inboxes[recipient.name] = self.inboxes[recipient.name][-8:]
        self.remember(sender, f"To {recipient.name}: {text}")
        self.remember(recipient, f"From {sender.name}: {text}")
        self.emit("message", sender=sender.name, recipient=recipient.name, text=text, channel=channel)

    def apply_formation(self, agent, decision):
        if not self.formation_enabled():
            self.invalid(agent, "formation", "membership is fixed in this condition")
            return
        action = decision.get("action", "pass")
        if not isinstance(action, str):
            self.invalid(agent, "formation", "action must be text")
            return
        if action == "pass":
            return
        if action == "leave":
            old, dissolved = self.team_manager.leave_team(agent)
            self.remember(agent, f"Left {old}.")
            self.emit("left", agent=agent.name, team=old)
            if dissolved:
                self.emit("team_dissolved", team=old, remaining=dissolved)
            return
        if action in {"invite", "message"}:
            recipient = self.lookup(decision.get("target"))
            text = decision.get("text", "")
            if recipient is None or recipient is agent or not isinstance(text, str):
                self.invalid(agent, "formation", "invalid recipient or text")
                return
            if action == "message":
                self.deliver(agent, recipient, text, "formation")
                return
            try:
                invitation = self.team_manager.invite(agent, recipient, self.round)
            except ValueError as error:
                self.invalid(agent, "formation", str(error))
                return
            self.deliver(agent, recipient, text, "invitation")
            self.emit("invited", **invitation)
            return
        if action == "accept":
            try:
                tag, inviter = self.team_manager.accept_invite(agent, decision.get("invitation"), self.round)
            except ValueError as error:
                self.invalid(agent, "formation", str(error))
                return
            self.emit("joined", agent=agent.name, inviter=inviter, team=tag)
            self.remember(agent, f"Joined {tag} after accepting {inviter}'s invitation.")
            return
        self.invalid(agent, "formation", "unknown action (no kick or merge action exists)")

    def form_teams(self):
        self.invitations.clear()
        if not self.formation_enabled():
            return
        for turn in range(self.config.formation_turns):
            order = list(self.agents)
            self.rng.shuffle(order)
            for agent in order:
                observation = {
                    "round": self.round,
                    "turn": turn,
                    "roster": self.roster(),
                    "summary": agent.summary,
                    "inbox": self.inboxes[agent.name],
                    "invitations": [v for v in self.invitations.values() if v["to"] == agent.name],
                }
                if self.config.interaction_protocol == "rounds":
                    observation.update(getattr(self, "formation_context", {}))
                    observation["decision_format"] = (
                        'Include "reason": "one short explanation" with your membership action. '
                        'Explain why you invite, accept, leave, message, or stay. No roles are assigned.'
                    )
                decision = self.ask("formation", agent, observation)
                if self.config.interaction_protocol == "rounds":
                    reason = decision.get("reason")
                    if not isinstance(reason, str) or not reason.strip():
                        self.invalid(agent, "formation", "membership decision requires a short reason")
                        continue
                    self.emit("membership_decision", agent=agent.name, action=decision.get("action"),
                              reason=reason, target=decision.get("target"), team=agent.team_tag)
                self.apply_formation(agent, decision)
        self.invitations.clear()

    def broadcast(self, sender, members, text, channel):
        if not text:
            return
        self.emit("team_message", agent=sender.name, members=[a.name for a in members], text=text, channel=channel)
        for recipient in members:
            self.remember(recipient, f"[{channel}] {sender.name}: {text}")
            if recipient is not sender:
                self.emit("message", sender=sender.name, recipient=recipient.name, text=text, channel=channel)

    def scratchpad(self, task, members, transcript, **extra):
        return {
            "task": task,
            "environment_state": {
                "phase": extra.get("phase", "collaboration"),
                "task_id": task["id"],
                "state": getattr(self, "environment_state", ""),
            },
            "step": self.step,
            "steps_remaining": self.config.max_steps - (self.step or 0),
            "must_finalize": (self.step or 0) == self.config.max_steps - 1,
            "previous_winner": self.previous_winner,
            "previous_members": [a.name for a in self.previous_members],
            "members": [{"name": a.name, "public_summary": a.public_summary} for a in members],
            "discussion": self.tokens.clip(dumps(transcript), self.config.evidence_tokens, tail=True),
            "objective": ("Maximize your own long-term wealth. Collaborate however you think is useful."
                          if self.config.interaction_protocol == "legacy" else
                          TeamAgent.SOCIETY_OBJECTIVE if self.config.objective_mode == "society" else TeamAgent.WEALTH_OBJECTIVE),
            **({"reward_rule": "At episode end, each member of each accepted contributing round receives "
                 "R/N. No division by team size; repeat contributors receive repeat shares. "
                 "Membership is frozen at submission; zero-money members are included."}
               if self.config.interaction_protocol == "rounds" else {}),
            **extra,
        }

    def auction(self, task):
        # Membership and wealth are frozen throughout negotiation, bidding and settlement.
        groups = self.groups()
        opening_wealth = {a.name: a.wealth for a in self.agents}
        if self.compact_rounds:
            from .compact_rounds import commit

            contributions, activation, team_activation = commit(self, task, groups, opening_wealth)
            return self.settle_auction(groups, opening_wealth, contributions, activation, team_activation)
        contributions = {a.name: 0.0 for a in self.agents}
        activation = {a.name: False for a in self.agents}
        for group in groups:
            self.scratchpads.setdefault(group, [])
        for group, members in groups.items():
            transcript = self.scratchpads[group]
            turns = self.bidding_turns(len(members))
            for turn in range(turns):
                order = list(members)
                self.rng.shuffle(order)
                for agent in order:
                    phase = "negotiate" if len(members) > 1 else "contribute"
                    observation = self.scratchpad(
                        task,
                        members,
                        transcript,
                        phase="bidding",
                        turn=turn,
                        total_turns=turns,
                        wealth={a.name: opening_wealth[a.name] for a in members},
                        pledges={a.name: contributions[a.name] for a in members},
                        reward_if_correct=self.config.reward,
                        bid_cost_rate=self.config.bid_cost_rate,
                    )
                    if self.config.bidding_mode == "sealed":
                        phase = "assess_bid"
                        # No transcript, evolving pledge, or peer private assessment
                        # is exposed before this single binding decision.
                        observation = {
                            "task": task, "environment_state": getattr(self, "environment_state", ""),
                            "members": [a.name for a in members],
                            "reward_if_correct": self.config.reward,
                            "equal_reward_share_if_correct": self.config.reward / len(members),
                            "bid_cost_rate": self.config.bid_cost_rate,
                        }
                    if self.config.interaction_protocol == "rounds":
                        phase = "round_bid"
                        observation["activation"] = {a.name: activation[a.name] for a in members}
                    decision = self.ask(
                        phase, agent, observation,
                        self.config.bid_tokens if phase == "assess_bid" else None,
                    )
                    value = decision.get("contribution")
                    if self.config.interaction_protocol == "rounds":
                        act = decision.get("act")
                        valid = (type(act) is bool and type(value) in (int, float)
                                 and math.isfinite(value) and 0 <= value <= opening_wealth[agent.name])
                        if valid:
                            activation[agent.name] = act
                            value = value if act else 0.0
                        else:
                            value = None
                        self.emit("activation", agent=agent.name, group=group,
                                  active=activation[agent.name], valid=valid, turn=turn)
                    if (
                        type(value) not in (int, float)
                        or not math.isfinite(value)
                        or value < 0
                        or value > opening_wealth[agent.name]
                    ):
                        self.invalid(agent, phase, "invalid pledge; previous valid pledge retained")
                    else:
                        contributions[agent.name] = float(value)
                    message = decision.get("message", "")
                    if not isinstance(message, str):
                        self.invalid(agent, phase, "message must be text")
                        message = ""
                    entry = {"author": agent.name, "text": message, "pledge": contributions[agent.name], "turn": turn}
                    transcript.append(entry)
                    self.broadcast(agent, members, message, "bidding")
                    self.emit("pledge", group=group, agent=agent.name, amount=contributions[agent.name], turn=turn)
            self.emit(
                "contributions_committed", group=group, contributions={a.name: contributions[a.name] for a in members}
            )
            for agent in members:
                self.remember(agent, f"Committed {contributions[agent.name]} to {group}'s bid after discussion.")
        if self.config.bidding_mode == "sealed" and not any(contributions.values()):
            self.repair_funding(task, groups, opening_wealth, contributions)
        return self.settle_auction(groups, opening_wealth, contributions, activation)

    def settle_auction(self, groups, opening_wealth, contributions, activation, team_activation=None):
        """Shared original payment chain; compact abstentions arrive with zero bids."""
        for agent in self.agents:
            agent.set_bid(contributions[agent.name])
        for team in self.team_manager.active_teams(self.round):
            team.contributions = {name: contributions[name] for name in team.member_ids}
        bids = {group: math.fsum(contributions[a.name] for a in members) for group, members in groups.items()}
        maximum = max(bids.values())
        winner = (
            self.rng.choice(sorted(group for group, bid in bids.items() if bid == maximum)) if maximum > 0 else None
        )
        members = tuple(groups[winner]) if winner else ()
        for agent in members:
            agent.lose_money(contributions[agent.name] * self.config.bid_cost_rate)
        paid = bids[winner] * self.config.bid_cost_rate if winner else 0.0
        self.bid_paid_total += paid
        credits = {}
        if self.previous_members and winner:
            share = paid / len(self.previous_members)
            for recipient in self.previous_members:
                recipient.gain_money(share)
                credits[recipient.name] = share
                self.remember(recipient, f"Received {share} from {winner}'s next bid.")
            self.bid_transfer_total += paid
        else:
            self.bid_burn_total += paid
        self.emit(
            "bid_transfer",
            payer=winner,
            members=[a.name for a in members],
            receiver=self.previous_winner if winner else None,
            credits=credits,
            total=paid if credits else 0.0,
            burned=paid if not credits else 0.0,
        )
        self.emit(
            "auction",
            bids=bids,
            contributions=contributions,
            opening_wealth=opening_wealth,
            winner=winner,
            members=[a.name for a in members],
            paid=paid,
            credits=credits,
            burned=paid if not credits else 0.0,
            **({"activation": activation} if self.config.interaction_protocol == "rounds" else {}),
            **({"team_activation": team_activation} if team_activation is not None else {}),
        )
        return winner, members, contributions, opening_wealth

    def repair_funding(self, task, groups, opening_wealth, contributions):
        """One voluntary visible-pledge round when sealed bids all abstained."""
        self.emit("funding_repair_started", reason="all initial bids were zero")
        for group, members in groups.items():
            transcript = self.scratchpads[group]
            order = list(members)
            self.rng.shuffle(order)
            for agent in order:
                observation = self.scratchpad(
                    task, members, transcript, phase="funding_repair", turn=0, total_turns=1,
                    wealth={a.name: opening_wealth[a.name] for a in members},
                    pledges={a.name: contributions[a.name] for a in members},
                    reward_if_correct=self.config.reward, bid_cost_rate=self.config.bid_cost_rate,
                    funding_repair={
                        "reason": "All initial bids were zero, so no team can act or earn a task reward. "
                        "This is one final chance to voluntarily fund a positive team bid. "
                        "Observe your teammates' actual pledges; no one is forced to contribute.",
                        "turns_remaining_after_this": 0,
                    },
                )
                decision = self.ask("negotiate", agent, observation, self.config.bid_tokens)
                value = decision.get("contribution")
                if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= opening_wealth[agent.name]:
                    contributions[agent.name] = float(value)
                else:
                    self.invalid(agent, "negotiate", "invalid pledge during funding repair; zero retained")
                message = decision.get("message", "")
                if not isinstance(message, str):
                    self.invalid(agent, "negotiate", "funding repair message must be text")
                    message = ""
                transcript.append({"author": agent.name, "text": message,
                                   "pledge": contributions[agent.name], "turn": 1})
                self.broadcast(agent, members, message, "bidding")
                self.emit("pledge", group=group, agent=agent.name, amount=contributions[agent.name],
                          turn=1, funding_repair=True)
            self.emit("contributions_committed", group=group,
                      contributions={a.name: contributions[a.name] for a in members}, funding_repair=True)

    def record_candidate(self, agent, members, candidates, answer, final):
        candidate = {
            "id": f"candidate-{max((int(c['id'].split('-')[-1]) for c in candidates), default=-1) + 1}",
            "author": agent.name,
            "answer": str(answer),
            "final": final,
        }
        candidates.append(candidate)
        self.emit("candidate", **candidate)
        for recipient in members:
            self.remember(recipient, f"{agent.name} proposed {candidate['id']}: {answer}")

    def finalize(self, members, task, transcript, candidates, reason):
        self.emit("finalization_started", reason=reason, members=[a.name for a in members])
        order = list(members)
        self.rng.shuffle(order)
        for agent in order:
            error = None
            for attempt in range(2):
                observation = self.scratchpad(
                    task, members, transcript, phase="finalization", must_finalize=True,
                    candidates=candidates, attempt=attempt + 1, correction=error,
                )
                action = self.ask("finalize", agent, observation, self.config.solution_tokens)
                answer = action.get("candidate")
                if action.get("abstain") is True and answer is None:
                    self.emit("finalization_abstained", agent=agent.name)
                    break
                if (
                    isinstance(answer, str) and answer.strip()
                    and action.get("final") is True
                    and action.get("abstain", False) is False
                ):
                    self.record_candidate(agent, members, candidates, answer, True)
                    # Make proposed answers available as shared notes as well.
                    transcript.append({"author": agent.name, "text": answer})
                    break
                error = 'Return a nonempty string candidate with final=true, or {"abstain":true}.'
                self.invalid(agent, "finalize", error)
        return [candidate for candidate in candidates if candidate["final"]]

    def collaborate(self, members, task):
        if self.config.collaboration_mode == "reviewed":
            from .reviewed import collaborate

            return collaborate(self, members, task)
        group = members[0].team_tag or f"solo:{members[0].name}" if members else None
        transcript, candidates = self.scratchpads.get(group, []), []
        if not members:
            return None
        enabled = self.config.finalization_enabled
        reserve = self.finalization_reserve(len(members))
        if enabled and self.remaining_calls() < reserve:
            raise BudgetExceeded("Insufficient calls for final proposals, corrections, votes and grading")
        available_discussion_turns = self.config.discussion_turns
        if enabled:
            available_discussion_turns = min(
                available_discussion_turns, (self.remaining_calls() - reserve) // len(members)
            )
        discussion_shortened = False
        for turn in range(self.config.discussion_turns):
            if enabled and self.remaining_calls() < reserve + len(members):
                discussion_shortened = True
                self.emit("discussion_shortened", reason="reserved final-answer calls", turn=turn)
                break
            order = list(members)
            self.rng.shuffle(order)
            for agent in order:
                observation = self.scratchpad(task, members, transcript, turn=turn, candidates=candidates)
                if enabled:
                    observation["discussion_window"] = {
                        "total_turns_per_member": available_discussion_turns,
                        "your_turns_remaining_after_this": available_discussion_turns - turn - 1,
                        "submission_rule": "Text in message alone is discussion, not a candidate. "
                        "If no eligible candidate is available, a final-answer window opens before voting.",
                    }
                action = self.ask("discuss", agent, observation, self.config.solution_tokens)
                message = action.get("message", "")
                if not isinstance(message, str):
                    self.invalid(agent, "discuss", "message must be text")
                    message = ""
                if message:
                    transcript.append({"author": agent.name, "text": message})
                    self.broadcast(agent, members, message, "team")
                answer = action.get("candidate")
                if type(answer) in (str, int, float) and str(answer).strip():
                    final = action.get("final", True)
                    if type(final) is not bool or ((self.step or 0) == self.config.max_steps - 1 and not final):
                        self.invalid(agent, "discuss", "invalid final flag or intermediate step at task limit")
                        continue
                    self.record_candidate(agent, members, candidates, answer, final)
        if enabled:
            # Do not submit an intermediate action if the remaining call budget
            # cannot fund the next auction and its complete final-answer window.
            must_finish = (
                (self.step or 0) == self.config.max_steps - 1
                or discussion_shortened
                or self.remaining_calls() - len(members) < self.auction_reserve()
            )
            eligible = [c for c in candidates if c["final"]] if must_finish else candidates
            if not eligible:
                reason = "final answer required" if must_finish else "no valid candidate"
                candidates = self.finalize(members, task, transcript, candidates, reason)
            else:
                candidates = eligible
        if not candidates:
            self.emit("no_submission", reason="no valid candidate")
            return None
        ids = {candidate["id"] for candidate in candidates}
        votes = Counter()
        for agent in members:
            decision = self.ask(
                "vote",
                agent,
                {
                    "task": task,
                    "candidates": candidates,
                    "discussion": self.tokens.clip(dumps(transcript), self.config.evidence_tokens, tail=True),
                },
            )
            candidate_id = decision.get("candidate_id")
            if isinstance(candidate_id, str) and candidate_id in ids:
                votes[candidate_id] += 1
                self.emit("vote", agent=agent.name, candidate_id=candidate_id)
            else:
                self.invalid(agent, "vote", "invalid candidate ID; abstention")
        # All-abstain is a failed submission, not an arbitrary fallback answer.
        if not votes:
            self.emit("no_submission", reason="no valid votes")
            return None
        best = max(votes.values())
        selected = self.rng.choice(sorted(key for key, count in votes.items() if count == best))
        candidate = next(c for c in candidates if c["id"] == selected)
        self.emit(
            "submission", candidate_id=selected, answer=candidate["answer"], final=candidate["final"], votes=dict(votes)
        )
        return TeamAction(candidate["answer"], group, candidate["final"])

    def assert_accounting(self):
        total = math.fsum(a.wealth for a in self.agents)
        expected = (self.initial_total + self.reward_total - self.bid_burn_total
                    - self.reflection_burn_total - self.coordination_burn_total
                    + self.birth_endowment_total - self.removed_wealth_total - self.rent_burn_total)
        if any(not math.isfinite(a.wealth) or a.wealth < -1e-9 for a in self.agents):
            raise AssertionError("Negative or nonfinite personal wealth")
        if not math.isclose(total, expected, rel_tol=1e-10, abs_tol=1e-8):
            raise AssertionError(f"Ledger mismatch: {total} != {expected}")

    def run_one_episode(self, env, *, formation=True, reflection=True, training=None):
        training = self.training if training is None else training
        if self.config.interaction_protocol == "rounds":
            from .round_protocol import run_episode

            return run_episode(self, env, formation=formation, reflection=reflection and training, training=training)
        env.initialize()
        self.step = None
        self.previous_winner, self.previous_members = None, ()
        self.scratchpads = {}
        if formation:
            # Formation can change group sizes. Use a conservative bound before
            # spending on it so an admitted auction can still produce an answer.
            formation_reserve = (
                len(self.agents) * (self.config.formation_turns +
                                   (2 if self.config.bidding_mode == "sealed" else self.config.bidding_turns))
                + self.collaboration_reserve(min(len(self.agents), self.config.max_team_size))
            )
            if (
                self.config.finalization_enabled and self.formation_enabled()
                and self.remaining_calls() < formation_reserve
            ):
                self.invitations.clear()
                self.emit("phase_skipped", phase="formation", reason="reserved final-answer calls")
            else:
                self.form_teams()
        task = env.task.public()
        total_paid = {a.name: 0.0 for a in self.agents}
        total_credits = dict(total_paid)
        total_rewards = dict(total_paid)
        steps = []
        reward = 0.0
        for step in range(self.config.max_steps):
            self.step = step
            self.environment_state = self.tokens.clip(
                env.get_state_description(), self.config.environment_tokens, tail=True
            )
            if self.config.finalization_enabled and self.remaining_calls() < self.auction_reserve():
                raise BudgetExceeded("Insufficient calls to start an auction and complete finalization")
            winner, members, contributions, opening = self.auction(task)
            for agent in members:
                total_paid[agent.name] += contributions[agent.name] * self.config.bid_cost_rate
            if members and self.previous_members:
                payment = math.fsum(contributions[a.name] for a in members) * self.config.bid_cost_rate
                for agent in self.previous_members:
                    total_credits[agent.name] += payment / len(self.previous_members)
            action = self.collaborate(members, task)
            step_reward = env.apply(action) if action is not None else 0.0
            share = step_reward / len(members) if members else 0.0
            for agent in members:
                agent.gain_money(share)
                total_rewards[agent.name] += share
            reward += step_reward
            self.reward_total += step_reward
            public = {
                "task_id": task["id"],
                "winner": winner,
                "members": [a.name for a in members],
                "reward": step_reward,
                "share": share,
                "score": env.get_terminal_score(),
                "final": action.final if action is not None else False,
            }
            self.emit("settlement", **public, wealth={a.name: a.wealth for a in self.agents})
            self.assert_accounting()
            steps.append(
                {
                    "step": step,
                    **public,
                    "contributions": contributions,
                    "wealth": {a.name: a.wealth for a in self.agents},
                }
            )
            self.emit("step_complete", metrics=steps[-1])
            if action is None or env.reach_termination():
                break
            self.previous_winner, self.previous_members = winner, tuple(members)
        self.step = None
        for agent in self.agents:
            self.remember(
                agent,
                dumps(
                    {
                        "round": self.round,
                        "team": agent.team_tag,
                        "contribution": contributions[agent.name],
                        "paid": total_paid[agent.name],
                        "bid_income": total_credits[agent.name],
                        "received": total_rewards[agent.name],
                        "wealth": agent.wealth,
                        "public": public,
                    }
                ),
            )
            agent.public_summary = self.tokens.clip(
                dumps(
                    {
                        "round": self.round,
                        "team": agent.team_tag,
                        "won": agent in members,
                        "reward_received": total_rewards[agent.name],
                        "bid_income": total_credits[agent.name],
                        "wealth": agent.wealth,
                    }
                ),
                self.config.summary_tokens,
            )
        for team in self.team_manager.active_teams(self.round):
            record = {
                "round": self.round,
                "won": team.team_id == winner,
                "reward": math.fsum(total_rewards[n] for n in team.member_ids),
                "bid_income": math.fsum(total_credits[n] for n in team.member_ids),
                "contributions": team.contributions,
                "discussion": self.tokens.clip(
                    dumps(self.scratchpads.get(team.team_id, [])), self.config.summary_tokens
                ),
            }
            team.trajectory.append(record)
            team.trajectory = team.trajectory[-8:]
            team.summary = self.tokens.clip(dumps(team.trajectory), self.config.summary_tokens, tail=True)
        # A bounded published extract, never another team's complete transcript.
        self.public_summaries = {
            team.team_id: {"members": list(team.member_ids), "summary": team.summary}
            for team in self.team_manager.active_teams(self.round)
        }
        if reflection:
            if self.config.finalization_enabled and self.remaining_calls() < 3 * len(self.agents):
                self.emit("phase_skipped", phase="reflection", reason="insufficient remaining calls")
            else:
                self.reflect()
        self.assert_accounting()
        metric = {
            "round": self.round,
            "task_id": task["id"],
            "score": env.get_terminal_score(),
            "reward": reward,
            "winner": winner,
            "steps": steps,
            "step_count": len(steps),
            "paid": total_paid,
            "bid_income": total_credits,
            "reward_income": total_rewards,
            "team_sizes": [len(g) for g in self.groups().values()],
            "wealth": {a.name: a.wealth for a in self.agents},
            "membership": {a.name: a.team_tag for a in self.agents},
            "contributions": contributions,
            "contribution_fraction": {
                a.name: contributions[a.name] / opening[a.name] if opening[a.name] else 0 for a in self.agents
            },
            "bid_paid_total": self.bid_paid_total,
            "bid_transfer_total": self.bid_transfer_total,
            "bid_burn_total": self.bid_burn_total,
            "reflection_burn_total": self.reflection_burn_total,
            "reward_total": self.reward_total,
            "invalid_actions": self.invalid_actions,
        }
        self.metrics.append(metric)
        self.emit(
            "round_complete",
            metrics=metric,
            agents=self.state()["agents"],
            teams=[asdict(team) for team in self.team_manager.active_teams(self.round)],
        )
        self.round += 1
        return metric

    def state(self):
        return {
            "round": self.round,
            "config": asdict(self.config),
            "backend": self.policy.label,
            "agents": [
                {
                    "name": a.name,
                    "wealth": a.wealth,
                    "team": a.team_tag,
                    "strategy": a.trainable_system_prompt,
                    "summary": a.summary,
                    **({"parent": a.parent_agent_name, "spawn_method": a.spawn_method,
                        "tasks_lived": a.tasks_lived} if self.config.evolution_enabled else {}),
                }
                for a in self.agents
            ],
            "accounting": {
                "initial": self.initial_total,
                **({"birth_endowments": self.birth_endowment_total, "removed_wealth": self.removed_wealth_total,
                    "rent_burned": self.rent_burn_total} if self.config.evolution_enabled else {}),
                "rewards": self.reward_total,
                "bids_spent": self.bid_paid_total,
                "bids_transferred": self.bid_transfer_total,
                "bids_burned": self.bid_burn_total,
                "reflection_spent": self.reflection_burn_total,
                **({"coordination_spent": self.coordination_burn_total}
                   if self.config.interaction_protocol == "rounds" else {}),
            },
            **({"retired_agents": self.retired_agents} if self.config.evolution_enabled else {}),
            "usage": {
                "decision_calls": self.calls,
                "requested_output_token_ceiling": self.requested_output_tokens,
                "reference_input_tokens": self.reference_input_tokens,
                "provider": self.policy.client.usage()
                if hasattr(getattr(self.policy, "client", None), "usage")
                else None,
                "note": "Reference tokenizer o200k_base. OpenRouter native usage is recorded separately; other clients are not dollar-metered.",
            },
        }
