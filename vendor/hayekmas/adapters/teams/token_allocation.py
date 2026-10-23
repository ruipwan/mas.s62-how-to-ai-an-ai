"""Spend bounded control calls on decisions, with larger windows for solving."""

from copy import deepcopy


PHASE_REASONING = {
    "round_membership": "none", "round_join": "none", "round_commit": "none",
    "coordinate": "none", "vote": "none", "reflect": "none",
    "round_chat": "medium", "round_work": "high", "shared_work": "high", "finalize": "high",
    "inspect": "medium", "improve": "medium", "birth_good": "medium", "birth_bad": "medium",
}
SOLVING_PHASES = {"shared_work", "round_chat", "round_work", "finalize", "discuss", "draft", "independent_check", "review", "revise", "solve"}


def control_observation(engine, phase, agent, observation):
    """Clip explanatory text, never identity, membership, invitation IDs or money.

    The full source events stay in the audit log. Clipping is deterministic and
    requires no summarizer call. Count JSON escaping, instructions and the full
    agent system/strategy prompt in the input limit before making a paid call.
    """
    from .mas import BudgetExceeded, dumps
    from .policy import INSTRUCTIONS

    obs = deepcopy(observation)
    text_fields = []

    def field(parent, key, cap, tail=False):
        if key in parent and isinstance(parent[key], str):
            original = parent[key]
            parent[key] = engine.tokens.clip(original, cap, tail=tail)
            if original != parent[key]:
                obs["context_truncated"] = True
            text_fields.append((parent, key, tail))

    if phase == "round_commit":
        # The preceding private conversation already contained the full task and
        # public work. This call only commits a vote and the member's own money.
        obs = {key: obs[key] for key in ("step", "steps_remaining", "must_finalize", "members", "wealth",
               "activation_rule", "bid_cost_rate", "bidding_rule", "team_base_bid", "you") if key in obs}
        obs["task_id"] = observation["task"]["id"]
        obs["members"] = [{"name": m["name"]} for m in observation["members"]]
        # Include every current member's latest discussion message, with a fair
        # per-member allowance, instead of the trailing slice of a long history.
        group = agent.team_tag or f"solo:{agent.name}"
        recent = {}
        names = {m["name"] for m in obs["members"]}
        for entry in engine.scratchpads.get(group, []):
            if entry.get("step") == engine.step and entry.get("author") in names:
                recent[entry["author"]] = entry.get("text", "")
        obs["discussion"] = [{"author": name, "text": recent[name]} for name in sorted(recent)]
        for entry in obs["discussion"]:
            field(entry, "text", 384, tail=True)
        limit = min(engine.config.pledge_context_tokens, engine.config.context_tokens)
    else:
        field(obs.get("task", {}), "problem", 768)
        field(obs, "environment_state", 1536, tail=True)
        field(obs, "summary", 192, tail=True)
        field(obs, "discussion", 768, tail=True)
        for member in obs.get("roster", []):
            field(member, "public_summary", 96)
        limit = min(engine.config.formation_context_tokens, engine.config.context_tokens)

    obs["context_note"] = "Bounded excerpts; missing details are not evidence of absence."

    def size():
        return engine.tokens.count(agent.get_system_prompt()) + engine.tokens.count(
            dumps({"instruction": INSTRUCTIONS[phase], "observation": obs}))

    while size() > limit:
        parent, key, tail = max(text_fields, key=lambda f: engine.tokens.count(f[0][f[1]]), default=({}, "", False))
        count = engine.tokens.count(parent.get(key, ""))
        if not count:
            raise BudgetExceeded(f"{phase} identities, balances and system prompt exceed its control context budget")
        parent[key] = engine.tokens.clip(parent[key], max(0, count - max(32, count // 4)), tail=tail)
        obs["context_truncated"] = True
    return obs


def phase_costs(records):
    """Aggregate confirmed charges, keeping uncertain reservations separate."""
    phases = {}
    for row in records:
        phase = row.get("kind", "unknown")
        item = phases.setdefault(phase, {"calls": 0, "cost_usd": 0.0, "reserved_usd": 0.0,
                                        "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0})
        item["calls"] += 1
        item["cost_usd"] += row.get("cost_usd", 0)
        item["reserved_usd"] += row.get("reserved_usd", 0) if "cost_usd" not in row else 0
        item["input_tokens"] += row.get("prompt_tokens", 0)
        item["output_tokens"] += row.get("completion_tokens", 0)
        if "reasoning_tokens" not in row:
            item["reasoning_tokens"] = None
        elif item["reasoning_tokens"] is not None:
            item["reasoning_tokens"] += row["reasoning_tokens"]
    total = sum(p["cost_usd"] for p in phases.values())
    solving = sum(v["cost_usd"] for p, v in phases.items() if p in SOLVING_PHASES)
    return {"phases": phases, "solving_cost_usd": solving,
            "solving_share_of_confirmed_cost": solving / total if total else None}
