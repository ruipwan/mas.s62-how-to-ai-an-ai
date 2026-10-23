"""Opt-in independent drafts, evidence-based review, and a revised final answer.

Roles are temporary and seeded. They confer no extra money or voting power.
Only explicitly final, complete candidates can be used as submission fallbacks.
"""

from .agent import TeamAction


def call_reserve(member_count):
    # One independent proposal and at most one format repair per member,
    # one review per peer (self-review for a singleton), one revision with at
    # most one repair, and one grader. Reserve before taking anyone's bid.
    return 2 * member_count + max(1, member_count - 1) + 2 + 1


def _strings(value):
    return isinstance(value, list) and all(isinstance(x, str) and x.strip() for x in value)


def _answer(engine, phase, agent, observation):
    error = None
    for attempt in range(2):
        action = engine.ask(phase, agent, {
            **observation, "attempt": attempt + 1, "correction": error,
        }, engine.config.solution_tokens)
        if action.get("abstain") is True and action.get("candidate") is None:
            engine.emit("review_abstained", agent=agent.name, phase=phase)
            return None, True
        answer = action.get("candidate")
        checklist = action.get("checklist")
        if (isinstance(answer, str) and answer.strip() and action.get("final") is True
                and action.get("abstain", False) is False and _strings(checklist) and checklist
                and (phase != "revise" or _strings(action.get("resolutions")))):
            return {"author": agent.name, "answer": answer, "checklist": checklist,
                    "resolutions": action.get("resolutions", [])}, False
        error = ('Return a nonempty candidate string, final=true, and a nonempty checklist of strings, '
                 'or {"abstain":true}. Revision also requires a resolutions list of strings.')
        engine.invalid(agent, phase, error)
    return None, False


def _bounded(engine, records):
    """Bound shared evidence, and explicitly report truncation to the reviewer."""
    import json

    limit = engine.config.evidence_tokens // max(1, len(records))
    result = []
    for record in records:
        text = json.dumps(record, ensure_ascii=False)
        clipped = engine.tokens.clip(text, limit)
        result.append({"author": record["author"], "text": clipped, "truncated": clipped != text})
    return result


def collaborate(engine, members, task):
    from .mas import BudgetExceeded

    if not members:
        return None
    if engine.remaining_calls() < call_reserve(len(members)):
        raise BudgetExceeded("Insufficient calls for independent drafts, review, revision and grading")
    group = members[0].team_tag or f"solo:{members[0].name}"
    transcript = engine.scratchpads.setdefault(group, [])
    order = list(members)
    engine.rng.shuffle(order)
    engine.emit("review_started", members=[a.name for a in order], drafter=order[0].name,
                reviewers=[a.name for a in order[1:]] or [order[0].name])
    base = engine.scratchpad(task, members, transcript, phase="independent_draft", must_finalize=True)
    proposals = []
    # Do not broadcast or remember any proposal until all independent calls
    # have finished. Even sequential HTTP requests see the same pre-draft state.
    for index, agent in enumerate(order):
        phase = "draft" if index == 0 else "independent_check"
        proposal, _ = _answer(engine, phase, agent, base)
        if proposal:
            proposals.append(proposal)
            engine.emit("review_draft", phase=phase, **proposal)
    if not proposals:
        engine.emit("no_submission", reason="no valid independently proposed final answer")
        return None
    primary = proposals[0]
    author = engine.lookup(primary["author"])
    reviewers = [a for a in order if a is not author] or [author]
    shared = {**base, "environment_state": {**base["environment_state"], "phase": "review"},
              "primary_author": author.name,
              "drafts": _bounded(engine, proposals)}
    reviews = []
    for reviewer in reviewers:
        review = engine.ask("review", reviewer, shared, engine.config.solution_tokens)
        checks, issues = review.get("checks"), review.get("issues")
        valid = (isinstance(checks, list) and bool(checks) and _strings(issues)
                 and all(isinstance(c, dict) and isinstance(c.get("requirement"), str)
                         and bool(c["requirement"].strip())
                         and c.get("status") in {"covered", "missing", "uncertain"}
                         and isinstance(c.get("evidence"), str) and bool(c["evidence"].strip())
                         for c in checks))
        if not valid:
            engine.invalid(reviewer, "review", "missing or invalid evidence-based checklist")
            review = {"checks": [], "issues": ["Review unavailable: verify the full answer yourself."],
                      "unavailable": True}
        review = {**review, "author": reviewer.name}
        reviews.append(review)
        engine.emit("review_feedback", **review)
    revision, abstained = _answer(engine, "revise", author, {
        **shared, "environment_state": {**base["environment_state"], "phase": "revision"},
        "reviews": _bounded(engine, reviews),
    })
    if revision:
        selected, selection = revision, "revision"
    else:
        # A malformed or empty response must not erase a valid final
        # proposal. An explicit abstention withdraws this author's proposal.
        eligible = [p for p in proposals if not abstained or p["author"] != author.name]
        if not eligible:
            engine.emit("no_submission", reason="revision abstained; no other valid final proposal")
            return None
        selected, selection = eligible[0], "draft_fallback"
        engine.emit("review_fallback", author=selected["author"],
                    reason="revision abstained" if abstained else "revision invalid after one repair")
    engine.emit("review_revision", selection=selection, **selected)
    for record in proposals + reviews + [selected]:
        transcript.append({"author": record["author"], "text": str(record)})
    candidates = []
    engine.record_candidate(engine.lookup(selected["author"]), members, candidates, selected["answer"], True)
    engine.emit("submission", candidate_id=candidates[0]["id"], answer=selected["answer"], final=True,
                votes={}, selection=selection, protocol="reviewed")
    return TeamAction(selected["answer"], group, True)
