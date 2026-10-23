"""Evidence-based review after every execution, including interrupted runs.

No additional model calls or hidden-test access. Human interpretation can be
recorded in review.json; measurements are always recomputed from saved evidence.
"""
from collections import Counter, defaultdict
from html import escape
import json
from pathlib import Path
import statistics

from .io import write_json
from .config import BASELINES


def analyze(directory):
    root = Path(directory)
    manifest = json.loads((root / 'manifest.json').read_text())
    state = json.loads((root / 'state.json').read_text()) if (root / 'state.json').exists() else {}
    baseline = manifest['config']['condition'] in BASELINES
    events = []
    for line in (root / 'events.jsonl').read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if manifest['status'] != 'running':
                raise
    charged = [e for e in events if e['event'] == 'api_charged']
    tools = [e for e in events if e['event'] == 'tool']
    candidates = [e for e in events if e['event'] == 'candidate']
    if baseline:
        candidates = [{**e, 'changed': bool(e.get('round_changed_paths', e['changed_paths'])), 'eligible': False}
                      for e in events if e['event'] == 'worker_checkpoint']
    reserved = {e['call_id']: e['elapsed'] for e in events if e['event'] == 'api_reserved'}
    latencies = [e['elapsed'] - reserved[e['call_id']] for e in charged if e['call_id'] in reserved]
    memberships = [e for e in events if e['event'] == 'membership']
    groups = Counter(a['team'] for a in memberships[-1]['agents']) if memberships else Counter()
    evaluation = json.loads((root / 'evaluation.json').read_text()) if (root / 'evaluation.json').exists() else None
    checkpoints = json.loads((root / 'checkpoints.json').read_text()) if (root / 'checkpoints.json').exists() else []
    costs = defaultdict(float)
    for e in charged:
        costs[e['phase']] += e['cost_usd']
    completions = sorted(e.get('usage', {}).get('completion_tokens', 0) for e in charged)
    reasoning = [(e.get('usage', {}).get('completion_tokens_details') or {}).get('reasoning_tokens', 0) or 0 for e in charged]
    agent_usage, team_usage = {}, {}
    rosters = {e['round']: {a['id']: a['team'] for a in e['agents']} for e in memberships}
    def account(table, key, event):
        row = table.setdefault(key, {'calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0, 'reported_usd': 0})
        usage = event.get('usage') or {}
        row['calls'] += 1
        row['input_tokens'] += usage.get('prompt_tokens', 0)
        row['output_tokens'] += usage.get('completion_tokens', 0)
        row['reasoning_tokens'] += (usage.get('completion_tokens_details') or {}).get('reasoning_tokens', 0) or 0
        row['reported_usd'] += event['cost_usd']
    for e in charged:
        account(agent_usage, e['agent'], e)
        team = 'formation' if e['phase'] in {'membership', 'join'} else rosters.get(e.get('round'), {}).get(e['agent'], 'unknown')
        account(team_usage, (e.get('round', -1), team), e)
    data = {
        'directory': str(root), 'model': manifest['config']['model'],
        'backend': manifest['backend'],
        'continuation_of': manifest.get('continuation_of'),
        'recovery': manifest.get('recovery'),
        'prompt_revision': manifest.get('prompt_revision', 'original'),
        'reasoning_enabled': manifest['config'].get('reasoning_enabled'),
        'reasoning_effort': manifest['config'].get('reasoning_effort'),
        'benchmark': manifest['task']['benchmark'], 'task': manifest['task']['id'],
        'status': manifest['status'], 'error': manifest.get('error'),
        'k': manifest['config']['k'], 'team_cap': manifest['config']['max_team_size'],
        'rounds_allocated': manifest['config']['rounds'],
        'max_rounds': manifest['config'].get('max_rounds', manifest['config']['rounds']),
        'completed_rounds': sum(e['event'] == 'snapshot' and e.get('round', -1) >= 0 for e in events),
        'answer_checkpoints': checkpoints,
        'usable_answer_at_last_checkpoint': checkpoints[-1]['answer_ready'] if checkpoints else None,
        'messages_by_channel': dict(Counter(e.get('channel', 'unknown') for e in events if e['event'] == 'message')),
        'message_deliveries': sum(len(e['message_ids']) for e in events if e['event'] == 'message_delivered'),
        'scratchpad': {
            'enabled': manifest['config'].get('shared_scratchpad', False),
            'writes': sum(e['event'] == 'scratchpad_write' for e in events),
            'rejected_writes': sum(e['event'] == 'scratchpad_rejected' for e in events),
            'shared_artifacts': sum(e['event'] == 'scratchpad_artifact' for e in events),
            'fetches': sum(e['event'] == 'scratchpad_fetch' for e in events),
            'cross_agent_fetches': sum(e['event'] == 'scratchpad_fetch' and e['agent'] not in e['authors'] for e in events),
            'stale_base_fetches': sum(e['event'] == 'scratchpad_fetch' and not e['same_base'] for e in events),
            'writes_by_agent': dict(Counter(e['agent'] for e in events if e['event'] == 'scratchpad_write')),
            'fetches_by_agent': dict(Counter(e['agent'] for e in events if e['event'] == 'scratchpad_fetch')),
            'note': 'Fetch records access, not proof of successful integration or correctness. Notes are peer claims.'},
        'work_calls_per_agent_per_round': manifest['config']['work_calls'],
        'condition': manifest['config']['condition'], 'seed': manifest['config']['seed'],
        'baseline': {'enabled': baseline, 'worker_count': manifest.get('worker_count'),
                     'orchestration_rounds': sum(e['event'] == 'orchestration' for e in events),
                     'peer_messages': sum(e['event'] == 'peer_message' for e in events),
                     'votes': sum(e['event'] == 'vote' for e in events),
                     'vote_results': [e for e in events if e['event'] == 'vote_result'],
                     'output_checkpoints': [e for e in events if e['event'] in {'integration_result', 'direct_result'}],
                     'synthesis_usd': costs.get('synthesis', 0),
                     'coordination_usd': sum(costs.get(p, 0) for p in ('peer', 'vote', 'orchestrate')),
                     'worker_usd': costs.get('independent', 0)},
        'reported_usd': sum(e['cost_usd'] for e in charged), 'paid_calls': len(charged),
        'unconfirmed_usd_at_stop': manifest.get('usage', {}).get('unconfirmed_usd', 0),
        'calls_by_phase': dict(Counter(e['phase'] for e in charged)),
        'cost_by_phase': dict(costs),
        'providers': dict(Counter(e.get('provider', 'unknown') for e in charged)),
        'finish_reasons': dict(Counter(e.get('finish_reason', 'not_recorded') for e in charged)),
        'response_token_policy': {
            'adaptive': manifest['config'].get('adaptive_tokens', False),
            'initial_work': manifest['config']['work_tokens'],
            'initial_control': manifest['config']['control_tokens'],
            'maximum': manifest['config'].get('max_response_tokens') if manifest['config'].get('adaptive_tokens') else None,
            'requested_limits': dict(Counter(str(e['max_tokens']) for e in charged if 'max_tokens' in e)),
            'extensions': [e for e in events if e['event'] == 'response_tokens_extended'],
            'length_limited_invalid_responses': sum(e['event'] == 'invalid_response' and e.get('finish_reason') == 'length' for e in events),
        },
        'completion_tokens': {'median': statistics.median(completions) if completions else 0,
                              'p95': completions[int((len(completions) - 1) * .95)] if completions else 0,
                              'max': max(completions, default=0), 'total': sum(completions)},
        'reasoning_tokens': {'max': max(reasoning, default=0), 'total': sum(reasoning)},
        'reasoning_report_inconsistencies': sum((e.get('usage', {}).get('completion_tokens_details') or {}).get('reasoning_tokens', 0) > e.get('usage', {}).get('completion_tokens', 0)
                                               for e in charged if (e.get('usage', {}).get('completion_tokens_details') or {}).get('reasoning_tokens') is not None),
        'usage_by_agent': agent_usage,
        'usage_by_team_round': [{'round': r, 'team': team, **usage} for (r, team), usage in sorted(team_usage.items())],
        'token_accounting_note': 'Output tokens include provider-reported reasoning; do not add reasoning twice. Formation calls are separate from the later team roster. A solo trial does not bound ten-member team usage.',
        'work_tools': dict(Counter(e['tool'] for e in tools)), 'actual_work_calls': len(tools),
        'invalid_responses': sum(e['event'] == 'invalid_response' for e in events),
        'normalized_responses': sum(e['event'] == 'response_normalized' for e in events),
        'invalid_actions': sum(e['event'] == 'invalid_action' for e in events)
            + sum(e['result'].get('status') == 'invalid_action' for e in tools),
        'candidates': len(candidates), 'eligible_candidates': sum(e.get('eligible', False) for e in candidates),
        'changed_candidates': sum(e.get('changed', False) for e in candidates),
        'candidate_check_statuses': dict(Counter(e['check'].get('status', 'unknown') for e in candidates)),
        'scorable_candidates': sum(e['check'].get('score') is not None for e in candidates),
        'candidate_changed_paths': dict(Counter(p for e in candidates for p in e.get('changed_paths', []))),
        'publications': sum(e['event'] == 'publication' for e in events),
        'contribution_contracts': {
            'enabled': manifest['config'].get('contribution_contracts', False),
            'offers': sum(e['event'] == 'contract_offer' for e in events),
            'funded': sum(e['event'] == 'contract_funded' for e in events),
            'accepted': sum(e['event'] == 'contract_accepted' for e in events),
            'paid': sum(e['event'] == 'contract_paid' for e in events),
            'paid_wealth': sum(e.get('amount', 0) for e in events if e['event'] == 'contract_paid'),
            'refunds': dict(Counter(e['reason'] for e in events if e['event'] == 'contract_refunded')),
            'open_escrow': (state.get('contribution_market') or {}).get('escrow', 0),
            'note': 'Virtual transfers, not API dollars. A settled contract verifies agreed examples and copied paths, not final native correctness.'},
        'last_team_sizes': sorted(groups.values(), reverse=True),
        'membership_snapshots': len(memberships),
        'median_api_seconds': round(statistics.median(latencies), 2) if latencies else None,
        'elapsed_seconds': events[-1]['elapsed'] if events else 0,
        'wall_seconds': manifest.get('finished_at', manifest['started_at']) - manifest['started_at'],
        'evaluation': evaluation,
        'source_hashes': manifest['source_hashes'],
        'caveat': 'Development run, not a full-suite accuracy estimate. Tool labels are agent-declared.',
    }
    worked, failed, limits = [], [], []
    if data['status'] == 'completed':
        worked.append('The allocated inference trajectory completed and the public source was exported.')
    elif (root / 'submission').exists():
        worked.append('Interruption-safe finalization exported the last public source without another paid call.')
    if data['changed_candidates']:
        worked.append(f"{data['changed_candidates']} candidate workspaces contained recorded file changes.")
    if data['publications']:
        worked.append(f"{data['publications']} contributions were accepted into the public repository.")
    if baseline and data['baseline']['output_checkpoints']:
        worked.append('The configured direct export, plurality vote or synthesis stage produced a public repository without market actions.')
        limits.append('This user-authorized SAS repair adds a separate ten-round block after the original baseline. It is extra compute and must not replace the frozen ten-round baseline.' if manifest.get('repair') else
                      'All baselines have a fixed ten-round comparison horizon. Worker and coordination compute differ by architecture and are reported separately.')
    if baseline:
        latest = {}
        for row in candidates:
            latest[row['agent']] = row
        valid = [row for row in latest.values() if row['check'].get('score') is not None]
        best = max(valid, key=lambda row: row['check']['score']) if valid else None
        data['baseline']['latest_workers_scorable'] = len(valid)
        data['baseline']['best_latest_worker'] = ({'agent': best['agent'], 'round': best['round'], 'check': best['check']} if best else None)
        if evaluation and evaluation.get('score') is not None:
            if best and best['check'].get('total') == evaluation.get('total') and best['check']['score'] > evaluation['score']:
                failed.append(f"Final output scored {100*evaluation['score']:.2f}% versus {100*best['check']['score']:.2f}% for {best['agent']}'s latest visible check. Selection or integration lost observed performance; the private candidate is not the final result.")
        if data['status'] == 'completed' and data['condition'] == 'decentralized_vote' and not any(e.get('winner') for e in data['baseline']['vote_results']):
            failed.append('No valid final vote selected a worker repository.')
    if evaluation and evaluation.get('score') is not None and evaluation['score'] < 1:
        failed.append(f"Final evaluation leaves {evaluation.get('total', 0)-evaluation.get('passed', 0)} tests unresolved; inspect submission/final-check diagnostics.")
    if data['scratchpad']['enabled']:
        pad = data['scratchpad']
        worked.append(f"Shared memory retained {pad['writes']} note revisions and {pad['shared_artifacts']} candidate artifacts; {pad['cross_agent_fetches']} fetches accessed another agent's candidate.")
        if not pad['fetches'] and data['status'] == 'completed':
            failed.append('No retained candidate was fetched; artifact retention was available but unused.')
        if pad['rejected_writes']:
            failed.append(f"{pad['rejected_writes']} scratchpad updates were rejected; inspect ownership, size or revision conflicts.")
        limits.append('The scratchpad treatment combines persistent notes and candidate retrieval, and adds input context; one seed cannot isolate their individual effects.')
    if charged and not data['invalid_responses'] and not data['invalid_actions']:
        worked.append('All returned model decisions were parsed without a recorded invalid action.')
    if data['normalized_responses']:
        worked.append(f"{data['normalized_responses']} single JSON fences were normalized without retries or extra calls.")
    if data['status'] != 'completed':
        failed.append('The trajectory is incomplete: ' + (data['error'] or data['status']))
    if data['invalid_responses'] or data['invalid_actions']:
        failed.append(f"Recorded {data['invalid_responses']} malformed responses and {data['invalid_actions']} invalid actions; counts may overlap.")
    if not data['changed_candidates'] and tools:
        failed.append('Work calls did not yield a recorded changed candidate.')
    if data['benchmark'] == 'commit0' and data['candidates'] and not data['scorable_candidates']:
        failed.append('No candidate had a numeric visible test score. Inspect checker diagnostics before attributing this to the model or environment.')
    if not data['publications'] and not baseline:
        failed.append('No contribution reached the public repository in this trajectory.')
    if data['status'] == 'completed' and checkpoints and not checkpoints[-1]['answer_ready']:
        failed.append('Reached the configured round cap without a usable source submission. Completion of the run is not completion of the task.')
    if data['unconfirmed_usd_at_stop']:
        failed.append(f"An interrupted request retains ${data['unconfirmed_usd_at_stop']:.8f} of uncertain cost; this is not a confirmed charge.")
    if data['membership_snapshots'] < 2 and not baseline:
        limits.append('One formation snapshot cannot establish team persistence or growth over time.')
    if data['reasoning_report_inconsistencies']:
        limits.append(f"{data['reasoning_report_inconsistencies']} receipts report more reasoning than completion tokens. Preserve the provider values, but do not treat that breakdown as internally consistent compute accounting.")
    if data['status'] != 'completed':
        limits.append('Do not compare this partial allocation as a completed benchmark or impute a zero score.')
    limits += [('An installable/buildable repository does not establish correctness.' if baseline else
                'Auction eligibility and an installable/buildable repository do not establish correctness.'),
               'One task and seed do not establish model ranking or a benefit from teams.',
               'Agents in a single population are not independent experimental replicates.']
    if manifest.get('continuation_of'):
        limits.append('This is a continuation, not an independent replicate. Counts and costs include the parent episode; do not sum both rows. Uncertain sent requests consumed their original slots without retries.')
    review_path = root / 'review.json'
    review = json.loads(review_path.read_text()) if review_path.exists() else {}
    data['analysis'] = {'what_worked': worked, 'failure_modes': failed, 'limits': limits,
                        'review': review,
                        'next_gate': 'Review diagnostics, completion, publication and native evaluation before expanding the paid campaign.'}
    write_json(root / 'analysis.json', data)
    population_note = (f"{data['baseline']['worker_count']} workers; no market or team-size rule" if baseline else f"team cap={data['team_cap']}")
    lines = [f"# Execution review: {root.name}", '',
             f"Model: `{data['model']}`. Prompt: `{data['prompt_revision']}`. Status: **{data['status']}**.", '',
             f"k={data['k']}; {population_note}; {data['paid_calls']} charged calls; ${data['reported_usd']:.8f} reported.", '',
             f"Work slots used: {data['actual_work_calls']}; changed candidates: {data['changed_candidates']}; eligible: {data['eligible_candidates']}; publications: {data['publications']}.", '']
    for title, items in [('What worked', worked), ('Failure modes', failed), ('Limits on conclusions', limits)]:
        lines += ['## ' + title, ''] + ['- ' + item for item in items] + ['']
    if review:
        lines += ['## Inspected interpretation and next decision', '']
        for key, value in review.items():
            lines += [f"**{key.replace('_', ' ').capitalize()}:** {value}", '']
    lines += ['## Evidence', '', 'See `events.jsonl`, `manifest.json`, `metrics.json`, candidate checker artifacts, and `evaluation.json` when present. Measurements are in `analysis.json`.']
    (root / 'analysis.md').write_text('\n'.join(lines) + '\n')
    blocks = []
    for title, items in [('What worked', worked), ('Failure modes', failed), ('Limits on conclusions', limits)]:
        blocks.append('<section><h2>' + escape(title) + '</h2><ul>' +
                      ''.join('<li>' + escape(item) + '</li>' for item in items) + '</ul></section>')
    if review:
        blocks.append('<section><h2>Inspected interpretation and next decision</h2>' +
                      ''.join('<h3>' + escape(k.replace('_', ' ').capitalize()) + '</h3><p>' + escape(str(v)) + '</p>'
                              for k, v in review.items()) + '</section>')
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Execution review</title><style>body{margin:0;background:#f5f8f8;color:#193039;font:16px/1.65 -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif}main{max-width:900px;margin:45px auto;padding:0 25px 60px}a{color:#087f82}h1{font-size:32px;line-height:1.2;overflow-wrap:anywhere}h2{font-size:22px}h3{font-size:16px}section{background:white;border:1px solid #dce5e8;border-radius:12px;padding:15px 28px;margin:20px 0}li{margin:8px 0}.meta{color:#60747c}code{overflow-wrap:anywhere}</style><main>'''
    html += '<a href="../../artifacts/index.html">← Experiment notebook</a><h1>' + escape(root.name) + '</h1>'
    html += '<p class="meta">' + escape(data['model'] + ' / ' + data['prompt_revision'] + ' / ' + data['status']) + '</p>'
    html += f"<p>k={data['k']}; {escape(population_note)}. {data['paid_calls']} charged calls. ${data['reported_usd']:.8f} reported.</p>"
    html += ''.join(blocks)
    if (root / 'replay.html').exists():
        html += '<p><a href="replay.html">Explore agent interactions →</a></p>'
    html += '<p class="meta">Evidence: analysis.json, events.jsonl, manifest.json, candidate checks, and evaluation.json when present.</p></main></html>'
    (root / 'analysis.html').write_text(html)
    return data
