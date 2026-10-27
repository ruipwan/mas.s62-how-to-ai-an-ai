"""Audit metrics and an offline, portable interaction replay."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from .io import write_json, sha
from .config import BASELINES


def metrics(state, events):
    snapshots = state.get('snapshots', [])
    grouped, memberships = [], []
    for row in snapshots:
        sizes = Counter(a['team'] for a in row['agents'])
        grouped += [sizes[a['team']] for a in row['agents']]
        teams = {a['id']: tuple(sorted(b['id'] for b in row['agents'] if b['team'] == a['team'])) for a in row['agents']}
        memberships.append(teams)
    switches = sum(prev.get(a) != team for prev, now in zip(memberships, memberships[1:]) for a, team in now.items())
    comparisons = sum(len(now) for now in memberships[1:])
    wealth = sorted(a['wealth'] for a in state.get('agents', []))
    gini = sum((2 * i - len(wealth) - 1) * w for i, w in enumerate(wealth, 1)) / (len(wealth) * sum(wealth)) if sum(wealth) else 0
    activity = Counter((e['agent'], e.get('tool', 'unknown')) for e in events if e['event'] == 'tool')
    by_agent, by_tool = Counter(), Counter()
    for (a, tool), n in activity.items():
        by_agent[a] += n; by_tool[tool] += n
    total = sum(activity.values())
    mi = sum(n / total * math.log2(n * total / (by_agent[a] * by_tool[t])) for (a, t), n in activity.items()) if total else 0
    pairs = [(r['bid'], r['after']['score'] - r['before']['score']) for r in state.get('history', [])
             if r['after'].get('score') is not None and r['before'].get('score') is not None
             and r['after'].get('total') == r['before'].get('total')]
    correlation = None
    if len(pairs) >= 3:
        mx, my = [sum(p[i] for p in pairs) / len(pairs) for i in (0, 1)]
        xx = sum((x - mx) ** 2 for x, _ in pairs); yy = sum((y - my) ** 2 for _, y in pairs)
        if xx * yy > 0:
            correlation = sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(xx * yy)
    costs, tokens = defaultdict(float), Counter()
    for e in events:
        if e['event'] == 'api_charged':
            costs[e['phase']] += e['cost_usd']
            for key in ('prompt_tokens', 'completion_tokens'):
                tokens[key] += (e.get('usage') or {}).get(key, 0)
    return {'agent_round_team_fraction': sum(n > 1 for n in grouped) / len(grouped) if grouped else 0,
            'agent_weighted_mean_team_size': sum(grouped) / len(grouped) if grouped else 1,
            'membership_persistence': 1 - switches / comparisons if comparisons else None,
            'agent_membership_changes': switches, 'wealth_gini': gini,
            'agent_tool_mutual_information_bits': mi,
            'specialization_caveat': 'Tool categories are chosen by agents; descriptive, not causal evidence.',
            'bid_progress_correlation': correlation, 'comparable_bid_score_pairs': len(pairs),
            'phase_cost_usd': dict(costs), 'tokens': dict(tokens),
            'work_calls_by_agent': dict(by_agent), 'tool_counts': dict(by_tool),
            'invalid_decisions': sum(e['event'] in {'invalid_action', 'invalid_response'} for e in events),
            'no_publication_rounds': sum(e['event'] == 'no_publication' for e in events),
            'regressions_blocked': sum(e['event'] == 'regression_blocked' for e in events),
            'market': market_metrics(state, events),
            'swarm': swarm_metrics(state, events)}


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out, i = [0.0] * len(values), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2
        i = j + 1
    return out


def spearman(pairs):
    if len(pairs) < 3:
        return None
    x, y = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    mx, my = sum(x) / len(x), sum(y) / len(y)
    xx, yy = sum((a - mx) ** 2 for a in x), sum((b - my) ** 2 for b in y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(xx * yy) if xx * yy > 0 else None


def market_metrics(state, events):
    """Version 2, descriptive bid and publication diagnostics for classic conditions.

    Raw gain is the candidate's visible pass count minus the round's starting count.
    Selection comparisons require comparable counts for every eligible candidate in
    the round. They do not infer unrecorded attempt order or tie outcomes.
    """
    completed = {e['round'] for e in events if e['event'] == 'snapshot' and e.get('round', -1) >= 0}
    start = {e['round'] + 1: e.get('feedback') or {} for e in events if e['event'] == 'snapshot'}
    by_round, eligible_counts = defaultdict(list), Counter()
    for e in events:
        if e['event'] != 'candidate' or 'task' in e or not e.get('eligible'):
            continue
        eligible_counts[e['round']] += 1
        base, check, bid = start.get(e['round'], {}), e.get('check') or {}, e.get('bid')
        counts = (base.get('passed'), check.get('passed'), base.get('total'), check.get('total'))
        if (type(bid) in (int, float) and math.isfinite(bid)
                and all(type(n) is int for n in counts)
                and all(type(n) in (int, float) and math.isfinite(n) and 0 <= n <= 1
                        for n in (base.get('score'), check.get('score')))
                and base['total'] == check['total'] and base['total'] > 0
                and 0 <= base['passed'] <= base['total'] and 0 <= check['passed'] <= check['total']):
            by_round[e['round']].append({'team': e.get('team'), 'bid': bid,
                                       'gain': check['passed'] - base['passed']})
    ordered = sorted(by_round)
    half = len(ordered) // 2
    pairs = lambda rows: [(c['bid'], c['gain']) for c in rows]
    pooled = lambda rs: [p for r in rs for p in pairs(by_round[r])]
    comparable = {r: by_round[r] for r, count in eligible_counts.items() if len(by_round[r]) == count}
    multiple = {r: rows for r, rows in comparable.items() if len(rows) > 1}
    contested = {r: rows for r, rows in multiple.items() if len({c['gain'] for c in rows}) > 1}
    exact_ties = state.get('config', {}).get('merge_slots', 1) > 1

    def top_bids(rows):
        best = max(c['bid'] for c in rows)
        # Match the actual selector: classic uses isclose, the merge queue sorts raw bids.
        return [c for c in rows if (c['bid'] == best if exact_ties
                                   else math.isclose(c['bid'], best, abs_tol=1e-10))]

    expected_hits = []
    for rows in contested.values():
        tied = top_bids(rows)
        best_gain = max(c['gain'] for c in rows)
        expected_hits.append(sum(c['gain'] == best_gain for c in tied) / len(tied))
    top_share = sum(expected_hits) / len(expected_hits) if expected_hits else None
    within_round = [rho for rows in comparable.values() if (rho := spearman(pairs(rows))) is not None]
    merges = [e for e in events if e['event'] == 'publication' and 'task' not in e]
    first_publication = {}
    for publication in merges:
        first_publication.setdefault(publication['round'], publication)
    actual_hits, regrets, absent, unmatched = [], [], 0, 0
    for round_id, rows in contested.items():
        publication = first_publication.get(round_id)
        if publication is None:
            absent += 1
            continue
        matches = [c for c in rows if c['team'] is not None and c['team'] == publication.get('team')]
        if len(matches) != 1:
            unmatched += 1
            continue
        best_gain, chosen_gain = max(c['gain'] for c in rows), matches[0]['gain']
        actual_hits.append(chosen_gain == best_gain)
        regrets.append(best_gain - chosen_gain)
    completed_merges = sum(e['round'] in completed for e in merges)
    return {'metrics_version': 2,
            'gain_definition': 'Candidate visible passed count minus the round starting count, with equal positive test totals and finite scores in [0, 1] for both checks.',
            'bid_tie_policy': ('exact equality, as in merge queue ordering' if exact_ties else
                               'math.isclose to maximum bid, rel_tol=1e-9 and abs_tol=1e-10, as in classic selection'),
            'descriptive_caveat': ('Bidders already receive candidate checks. Pooled bids are confounded by team size, wealth and round. '
                                   'Pooled and within-round correlations are descriptive, not evidence of price learning. '
                                   'Tie-averaged shares are uniform expectations, not observed first-attempt outcomes. '
                                   'First publication is the first recorded successful acceptance, possibly after rejected attempts. '
                                   'Raw gain regret compares candidate checks, not marginal gains after merging or an achievable oracle.'),
            'eligible_candidates_scored': len(pooled(ordered)),
            'selection_comparable_rounds': len(comparable),
            'selection_excluded_incomplete_rounds': len(eligible_counts) - len(comparable),
            'bid_gain_spearman': spearman(pooled(ordered)),
            'bid_gain_spearman_early': spearman(pooled(ordered[:half])),
            'bid_gain_spearman_late': spearman(pooled(ordered[half:])),
            'bid_gain_spearman_within_round_mean': sum(within_round) / len(within_round) if within_round else None,
            'bid_gain_spearman_within_round_valid_rounds': len(within_round),
            'within_round_spearman_definition': 'Equal-weight mean over complete comparable rounds with at least three candidates and nonconstant bids and gains.',
            # Compatibility alias, corrected from the optimistic any-tied-candidate definition in v1.
            'top_bid_was_top_gain_share': top_share,
            'top_bid_tie_averaged_top_gain_share': top_share,
            'contested_rounds': len(contested),
            'top_bid_tie_rounds': sum(len(top_bids(rows)) > 1 for rows in multiple.values()),
            'top_bid_tie_rounds_denominator': len(multiple),
            'first_publication_top_gain_share': sum(actual_hits) / len(actual_hits) if actual_hits else None,
            'first_publication_top_gain_hits': sum(actual_hits),
            'first_publication_comparable_contested_rounds': len(actual_hits),
            'first_publication_raw_gain_regret_mean': sum(regrets) / len(regrets) if regrets else None,
            'first_publication_raw_gain_regret_total': sum(regrets),
            'first_publication_raw_gain_regret_rounds': len(regrets),
            'contested_rounds_without_publication': absent,
            'contested_rounds_unmatched_first_publication': unmatched,
            'merges': len(merges), 'completed_rounds': len(completed),
            'completed_round_merges': completed_merges,
            'merges_per_round': completed_merges / len(completed) if completed else None,
            'three_way_merges': sum(e.get('mode') == '3way' for e in merges),
            'rejected_merges': dict(Counter(e['reason'].split(':')[0] for e in events
                                            if e['event'] == 'merge_rejected' and 'task' not in e)),
            'skipped_for_slots': sum(e['event'] == 'merge_skipped' for e in events),
            'rent_paid': state.get('rent_paid'),
            'resting_agent_rounds': sum(len(e['agents']) for e in events
                                        if e['event'] == 'rest' and e.get('reason') == 'wealth below rent')}


def swarm_metrics(state, events):
    """Merge throughput, wasted candidates, duplicate claims and rest, for the swarm condition."""
    rounds = max(1, sum(e['event'] == 'snapshot' and e.get('round', -1) >= 0 for e in events))
    candidates = [e for e in events if e['event'] == 'candidate' and 'task' in e]
    per_task_round = Counter((e['round'], e['task']) for e in candidates)
    merges = [e for e in events if e['event'] == 'publication' and 'task' in e]
    return {'merges': len(merges), 'merges_per_round': len(merges) / rounds,
            'three_way_merges': sum(e.get('mode') == '3way' for e in merges),
            'rejected_merges': dict(Counter(e['reason'] for e in events if e['event'] == 'merge_rejected' and 'task' in e)),
            'eligible_candidates': sum(e['eligible'] for e in candidates),
            'duplicate_task_claims': sum(n - 1 for n in per_task_round.values() if n > 1),
            'resting_agent_rounds': sum(len(e['agents']) for e in events if e['event'] == 'rest' and 'open_tasks' in e),
            'invalid_claims': sum(e['event'] == 'invalid_action' and e.get('phase') == 'claim' for e in events),
            'bounties_issued': state.get('bounties_issued'),
            'royalty_share': (sum(e.get('royalty', 0) for e in merges) / state['bounties_issued']
                             if state.get('bounties_issued') else None)}


def replay(directory, *, filename='replay.html'):
    root = Path(directory)
    if Path(filename).name != filename or not filename.endswith('.html'):
        raise ValueError('Replay filename must be an HTML file directly in the run directory')
    state = json.loads((root / 'state.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    events = []
    for line in (root / 'events.jsonl').read_text().splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if manifest['status'] != 'running':
                raise
    summary = metrics(state, events)
    write_json(root / 'metrics.json', summary)
    patches = {}
    for row in state.get('history', []):
        p = root / row['patch']
        if p.is_file():
            patches[row['patch']] = p.read_text(errors='replace')[:24000]
    # Full decisions remain in events.jsonl; avoid repeating their large model payloads.
    shown = [e for e in events if e['event'] not in {'decision', 'api_reserved'}]
    data = {'manifest': manifest, 'state': state, 'events': shown, 'metrics': summary, 'patches': patches}
    handoff = root / 'scratchpad-reuse-audit.json'
    if handoff.exists():
        data['handoff'] = json.loads(handoff.read_text())
    encoded = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    viewer = Path(__file__).with_name('baseline-viewer.html' if state['config']['condition'] in BASELINES else 'viewer.html')
    template = viewer.read_text()
    # The optional guided story is offline presentation, separate from inference.
    story_path = Path(__file__).with_name('handoff-view.html')
    story = story_path.read_text() if data.get('handoff') and story_path.exists() else ''
    page = template.replace('/*__DATA__*/null', encoded).replace('<!--__HANDOFF_VIEW__-->', story)
    (root / filename).write_text(page)
    write_json(root / (Path(filename).stem + '-renderer.json'), {'viewer_sha256': sha(viewer),
               'renderer_sha256': sha(Path(__file__)),
               'handoff_view_sha256': sha(story_path) if story else None,
               'note': 'Rendering may be refreshed independently of the frozen inference source snapshot.'})
    return root / filename
