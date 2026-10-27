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
            'swarm': swarm_metrics(state, events)}


def swarm_metrics(state, events):
    """Merge throughput, wasted candidates, duplicate claims and rest, for the swarm condition."""
    rounds = max(1, sum(e['event'] == 'snapshot' and e.get('round', -1) >= 0 for e in events))
    candidates = [e for e in events if e['event'] == 'candidate' and 'task' in e]
    per_task_round = Counter((e['round'], e['task']) for e in candidates)
    merges = [e for e in events if e['event'] == 'publication' and 'task' in e]
    return {'merges': len(merges), 'merges_per_round': len(merges) / rounds,
            'three_way_merges': sum(e.get('mode') == '3way' for e in merges),
            'rejected_merges': dict(Counter(e['reason'] for e in events if e['event'] == 'merge_rejected')),
            'eligible_candidates': sum(e['eligible'] for e in candidates),
            'duplicate_task_claims': sum(n - 1 for n in per_task_round.values() if n > 1),
            'resting_agent_rounds': sum(len(e['agents']) for e in events if e['event'] == 'rest'),
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
