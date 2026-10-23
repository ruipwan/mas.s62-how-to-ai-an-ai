"""All agents receive identical interfaces and choose behavior for themselves."""

import ast
import json
import random

from hayekmas.utils.llm import LLMConfig, get_llm_client


INSTRUCTIONS = {
    "birth_good": 'Create a concise transferable variation of the source strategy. Follow the mutation specification in the observation. Return {"strategy":"new editable strategy"}. Never modify immutable rules or assign a fixed role.',
    "birth_bad": 'Repair a general weakness using the supplied failure context and mutation specification. Return {"strategy":"new editable strategy"}. Do not memorize task-specific answers, modify immutable rules, or assign a fixed role.',
    "round_membership": (
        "Briefly discuss next-round membership with the population and make one proposal. Return "
        '{"leave":false, "invite":"agent name or null", "message":"short public message", "reason":"short reason"}. '
        "Use JSON null for no invitation. leave=true exits your current team. You may leave AND invite one "
        "agent in this call. All departures are applied before invitations. An invite can succeed only if its "
        "recipient is then solo and explicitly accepts. Keeping membership is the default; there are no "
        "automatic joins, assigned roles, kicks or unilateral merges."
    ),
    "round_join": (
        "You have valid invitations after this round's membership proposals. Accept at most one or decline all. "
        'Return {"invitation":"listed invitation id or null", "reason":"short explanation"}. '
        "Use JSON null to stay solo. You cannot be added without your acceptance. No further invitation pass follows."
    ),
    "round_commit": (
        "If observation.bidding_rule is fixed: return {\"act\":true, \"authorize_base_bid\":true, "
        "\"reason\":\"at most 12 words\"}. Both booleans must be true to consent. This authorizes up to "
        "team_base_bid*bid_cost_rate from your own wealth if your team wins; the actual charge is split "
        "equally among eligible consenting yes voters. False means no consent and no charge. "
        "If bidding_rule is not fixed, use the voluntary-pledge instructions that follow. "
        "Make a brief decision from the completed team discussion; do not solve the problem again here. "
        "The private discussion is complete. Cast ONE binding vote on whether your TEAM should act this round, "
        "and authorize only your own contribution. Other current ballots are hidden. Return "
        '{"act":true, "contribution":number, "reason":"at most 12 words"}. '
        "A strict majority of all members must vote yes; ties and invalid ballots do not activate the team. "
        "act=false sets your own contribution to zero. You may vote yes with zero money. If the team abstains, "
        "all pledges are canceled. Otherwise its bid is the sum of members' valid personal pledges; only a "
        "winning positive bid is paid. You cannot commit another member's wealth. No forced deadline action. "
        "All members still receive R/N for accepted team work, irrespective of pledge or vote."
    ),
    "coordinate": (
        "Team formation has ended. Decide whether to consent to the announced maximum coordination fee "
        'for this round. Return {"participate":true, "reason":"short explanation"} or false to leave. '
        "The fee is burned from your wealth before discussion, even if the team loses the auction. "
        "Other members may leave; your actual fee can only decrease. No response means no consent."
    ),
    "round_chat": (
        "Discuss the task and current shared solution freely with your team before deciding whether "
        'to act or pledge money. Return {"message":"your contribution to the conversation"}.'
    ),
    "round_bid": (
        "Use the team conversation to decide whether to act and negotiate your personal pledge. "
        'Return {"act":true, "contribution":number, "message":"text"}. '
        "act=false sets your pledge to zero. Only you can commit your money, between zero and your "
        "available wealth. You may revise your decision and amount; the last valid decision binds. "
        "The greatest positive team sum wins. Membership, not payment size, determines path credit: "
        "every member of a team that submits accepted work receives R/N for that round at episode end."
    ),
    "shared_work": (
        "Work with your teammates on the latest shared draft. Spend your effort solving the problem. "
        "You may extend, correct, reorganize, or retain the draft as useful; no roles are assigned. "
        "Return {\"message\":\"brief explanation of your contribution\", \"candidate\":\"complete updated draft or null to retain it\"}. "
        "Read the current draft and teammates’ messages before updating it. This is preparation; "
        "a separate finalization step submits the final answer when required."
    ),
    "round_work": (
        "Use the shared team discussion freely to propose the next public contribution to the solution. "
        'Return {"message":"text", "candidate":"public work or null"}. '
        "This is an intermediate round, so the chosen candidate is added to the shared solution; "
        "it is not the final answer. Each member can propose work and vote. No roles are assigned."
    ),
    "round_recap": (
        "A new decision round is starting. You may briefly talk to the population about the shared work "
        'or future team membership. Return {"message":"text"}. '
        "A voluntary invitation/acceptance/leave window follows. Joining cannot change past credit."
    ),
    "assess_bid": (
        "Make one sealed personal pledge. Assess relevant expertise and the main uncertainty in at most "
        "two short sentences; do not solve the entire problem or negotiate. Other current pledges are hidden. "
        'Return {"message":"brief assessment", "contribution":number}. Only your money may be pledged, '
        "between zero and your available wealth. Winning members pay their own pledges times bid_cost_rate; "
        "the eventual reward is split equally among winning members."
    ),
    "draft": (
        "Independently produce your best complete answer. Derive a checklist of requested deliverables "
        "from the public question, check units and assumptions, and distinguish facts from uncertainty. "
        "No teammate draft is visible yet. Your answer must stand alone; it may be submitted if revision fails. "
        'Return {"candidate":"complete answer", "final":true, "checklist":["requested deliverable"]}, '
        'or {"abstain":true}. Do not invent missing experimental details or sources.'
    ),
    "independent_check": (
        "Independently solve and check the public question before seeing any teammate draft. "
        "Prioritize calculations, units, assumptions and easily missed requested deliverables. "
        "Write a complete alternative answer that can stand alone, stating uncertainty explicitly. "
        'Return {"candidate":"complete answer", "final":true, "checklist":["requested deliverable"]}, '
        'or {"abstain":true}. Do not invent missing experimental details or sources.'
    ),
    "review": (
        "Compare the primary draft with the independently generated alternatives and the public question. "
        "For each requested deliverable, check coverage and correctness. Recompute disputed quantities. "
        "Identify concrete errors, disagreements and missing evidence; agreement is not verification. "
        "If an input is truncated, state what you could not verify. Use only supplied information; "
        "do not invent sources or claim to have run tools. "
        'Return {"checks":[{"requirement":"text", "status":"covered|missing|uncertain", '
        '"evidence":"calculation or reason"}], "issues":["specific correction or disagreement"]}.'
    ),
    "revise": (
        "Produce one complete final answer using the public question, independent drafts and reviews. "
        "Address each checklist item and review issue; resolve disagreements by calculation or evidence, "
        "not majority vote. Reject unsupported corrections and state unresolved uncertainty in the answer. "
        "Check truncated-input warnings. Do not just summarize the discussion or invent missing facts. "
        'Return {"candidate":"complete revised answer", "final":true, '
        '"checklist":["requested deliverable"], "resolutions":["how an issue was resolved or remains uncertain"]}, '
        'or {"abstain":true}. The candidate field is the entire submitted answer.'
    ),
    "reflect": 'Choose whether to pay for reflection now. Return {"reflect": true} or {"reflect": false}.',
    "inspect": 'Inspect only the supplied experience. Return {"analysis": "your observations"}.',
    "improve": 'Use your inspection to update your future strategy. Return {"strategy": "new strategy"}.',
    "formation": (
        'Choose one action: {"action":"invite", "target":"agent name", "text":"message"}, '
        '{"action":"accept", "invitation":"invitation id"}, {"action":"leave"}, '
        '{"action":"message", "target":"agent name", "text":"message"}, or {"action":"pass"}. '
        "Only unteamed agents may accept. An invitation from a teammate admits a new member "
        "to that team. Invites expire after this formation phase. No kicking or unilateral merging."
    ),
    "contribute": 'As a singleton choose your personal contribution. Return {"contribution": number}.',
    "negotiate": (
        "Discuss how much each teammate should contribute using the visible pledges and conversation. "
        "You may ask others to change their amounts, explain a proposed allocation, and revise your own pledge. "
        'Return {"message":"text", "contribution":number}. The contribution sets ONLY your pledge. '
        "Your last valid pledge at the end of this fixed discussion window is binding if your team wins. "
        "You may contribute zero. No agent can pledge money on behalf of another."
    ),
    "discuss": (
        'Use the team channel freely. Return {"message":"text", "candidate":"text or null", "final":true}. '
        "Set final=false for an intermediate public step; final=true submits an answer and ends the task. "
        "You can discuss without a candidate. On must_finalize=true only final answers are eligible. "
        "Everyone has the same actions. "
        "After the fixed discussion window, every member votes for a candidate."
    ),
    "finalize": (
        "Discussion has ended. Propose your best complete final answer using the task and shared notes. "
        'Return {"candidate":"your complete answer", "final":true}, or explicitly abstain with '
        '{"abstain":true}. State unresolved assumptions or uncertainty in the answer. '
        "Text in message alone is not submitted. Intermediate steps and empty candidates are not eligible. "
        "Every member has the same opportunity; the team votes on eligible final candidates. "
        "A malformed response receives at most one correction attempt. Abstention is not retried."
    ),
    "vote": 'Choose a candidate ID from the supplied list. Return {"candidate_id":"id"}. Ties are broken by seeded lottery.',
}


class ModelPolicy:
    label = "llm"

    def __init__(self, model, budget=None):
        if model.get("api") == "openrouter":
            from .openrouter import OpenRouterClient

            if not model.get("name") or model.get("api_base"):
                raise ValueError("OpenRouter requires an explicit model name and uses its fixed official endpoint")
            self.client = OpenRouterClient(
                model["name"], budget=budget or {}, api_key=model.get("api_key"), **model.get("extra_kwargs", {})
            )
            return
        cfg = LLMConfig(**model)
        if cfg.api not in {"localhost", "litellm", "together"}:
            raise ValueError("Team prototype supports localhost, litellm or together model APIs")
        if not cfg.name:
            raise ValueError("Set an explicit model name for reproducible runs")
        kwargs = dict(cfg.extra_kwargs)
        kwargs.update(api_key=cfg.api_key, api_base=cfg.api_base)
        if cfg.api == "localhost":
            kwargs.update(max_retries=1, timeout=60)
        self.client = get_llm_client(cfg.api, cfg.name, **kwargs)
        self.client.MAX_EMPTY_RETRIES = 1

    def respond(self, phase, agent, observation, max_tokens):
        options = {}
        if self.client.api_name == "openrouter":
            options["kind"] = phase
            effort = getattr(agent, "phase_reasoning_efforts", {}).get(phase)
            if effort is not None:
                options["reasoning_effort"] = effort
        return self.client.generate(
            json.dumps(
                {"instruction": INSTRUCTIONS[phase], "observation": observation},
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            ),
            system_prompt=agent.get_system_prompt(),
            max_tokens=max_tokens,
            **options,
        )


class DemoPolicy:
    """Scripted mechanism smoke test; never evidence of emergent behavior."""

    label = "scripted_demo"

    def __init__(self, seed):
        self.rng = random.Random(seed)

    def respond(self, phase, agent, observation, max_tokens):
        if phase in {"birth_good", "birth_bad"}:
            result = {"strategy": "Check assumptions and verify intermediate results before committing to a conclusion."}
        elif phase == "coordinate":
            result = {"participate": True, "reason": "The planned collaboration is worth the announced fee."}
        elif phase == "reflect":
            result = {"reflect": observation["round"] > 0 and observation["round"] % 4 == 0}
        elif phase == "inspect":
            result = {"analysis": "Compare money spent with equal reward share in the supplied history."}
        elif phase == "improve":
            result = {
                "strategy": "Preserve capital by contributing a small fraction of wealth. Check arithmetic before voting."
            }
        elif phase == "round_membership":
            size = len(observation["roster"])
            own = int(agent.name.split("-")[-1])
            offset = (observation["episode"] + observation["step"]) % size
            peer = f"agent-{(offset + (((own - offset) % size) ^ 1)) % size}"
            result = {"leave": agent.team_tag is not None, "invite": peer if peer != agent.name else None,
                      "message": "Would you like to try this partnership?", "reason": "Try a new pairing this round."}
        elif phase == "round_join":
            result = {"invitation": observation["invitations"][0]["id"], "reason": "I accept this partnership."}
        elif phase == "round_commit":
            # Scripted fixture: show some collective abstentions in the replay.
            # This is not a learned strategy or evidence about model behavior.
            first = min(int(m["name"].split("-")[-1]) for m in observation["members"])
            active = (first + observation["step"]) % 3 != 2
            result = {"act": active, "contribution": round(agent.wealth * self.rng.uniform(.03, .12), 6) if active else 0,
                      "reason": "Scripted demo vote: contribute this round." if active else
                                "Scripted demo vote: let another team contribute this round."}
            if observation.get("bidding_rule") == "fixed":
                result.pop("contribution")
                result["authorize_base_bid"] = active
        elif phase == "formation":
            invitations = observation["invitations"]
            if (
                observation["round"] > 0 or (observation.get("stage") == "round_start" and observation["step"] > 0)
            ) and observation["turn"] == 0 and agent.team_tag is not None:
                result = {"action": "leave"}
            elif agent.team_tag is None and invitations:
                result = {"action": "accept", "invitation": invitations[0]["id"]}
            elif agent.team_tag is None:
                own = int(agent.name.split("-")[-1])
                peer = f"agent-{own ^ (2 if (observation['round'] // 5) % 2 else 1)}"
                if observation.get("stage") == "round_start":
                    size = len(observation["roster"])
                    offset = (observation["round"] + observation["step"]) % size
                    peer = f"agent-{(offset + (((own - offset) % size) ^ 1)) % size}"
                names = {p["name"] for p in observation["roster"] if p["team"] is None}
                result = (
                    {"action": "invite", "target": peer, "text": "Would you like to work together?"}
                    if peer in names
                    else {"action": "pass"}
                )
            else:
                result = {"action": "pass"}
            if observation.get("stage") == "round_start":
                result["reason"] = "Try this partnership for the next public contribution."
        elif phase in {"round_chat", "round_recap"}:
            result = {"message": "Let us use the shared progress and decide together how to contribute."}
        elif phase in {"contribute", "negotiate", "assess_bid", "round_bid"}:
            result = {
                **({"act": True} if phase == "round_bid" else {}),
                "contribution": round(agent.wealth * self.rng.uniform(0.03, 0.12), 6),
                "message": ("I can attempt this arithmetic. This is my one binding pledge."
                            if phase == "assess_bid" else
                            "Here is my current pledge. Please consider contributing an affordable amount too."),
            }
        elif phase in {"discuss", "finalize", "draft", "independent_check", "revise", "round_work", "shared_work"}:
            # Uses only the public problem, never the environment's hidden answer.
            problem = observation["task"]["problem"]
            try:
                values = ast.literal_eval(problem.split(":", 1)[1].strip())
                answer = str(sum(values))
            except (SyntaxError, ValueError, TypeError, IndexError):
                answer = None
            result = {
                "message": "I independently checked the public arithmetic.",
                "candidate": answer,
                "final": observation["must_finalize"],
                **({"checklist": ["Compute the sum"], "resolutions": ["Checked the arithmetic"]}
                   if phase in {"draft", "independent_check", "revise"} else {}),
            }
            if phase in {"finalize", "draft", "independent_check", "revise"} and answer is None:
                result = {"abstain": True}
        elif phase == "review":
            result = {"checks": [{"requirement": "Compute the sum", "status": "covered",
                                  "evidence": "Recomputed the public arithmetic"}], "issues": []}
        else:
            candidates = observation["candidates"]
            result = {"candidate_id": candidates[0]["id"] if candidates else None}
        return json.dumps(result)
