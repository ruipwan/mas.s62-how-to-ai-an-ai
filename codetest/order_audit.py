"""Offline, one-round merge-order counterfactuals. No model or campaign-ledger imports.

Every policy starts from an observed round base and receives the same historical
eligible candidates. Later-round inputs are never regenerated. This does not
estimate complete counterfactual trajectories or causal market effects.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
import time

from .benchmarks import Task
from .fixture import FixtureTask
from .io import sha, write_json
from .report import metrics
from .sandbox import Repository, run

SCOPE = ('One-round counterfactual on frozen historical eligible candidates, not a '
         'complete alternative trajectory. Visible tests only. No new model calls. '
         'Gain uses standalone same-base candidate performance, not an oracle final ordering.')


def _implementation_hashes():
    names = ('order_audit.py', 'report.py', 'benchmarks.py', 'sandbox.py', 'fixture.py', 'io.py', 'config.py')
    return {'codetest/' + name: sha(Path(__file__).with_name(name)) for name in names}


def _read(path):
    return json.loads(Path(path).read_text())


def _tree_digest(root):
    # Same input fingerprint as the frozen experiment, without importing Engine.
    digest = hashlib.sha256()
    for p in sorted(Path(root).rglob('*')):
        rel = p.relative_to(root)
        if any(n in {'.git', '.venv', '__pycache__', '.pytest_cache', 'build', 'executable'} for n in rel.parts):
            continue
        if p.is_symlink():
            raise ValueError('Symlink in audit input')
        if p.is_file():
            digest.update(str(rel).encode()); digest.update(p.read_bytes())
            digest.update(str(p.stat().st_mode & 0o111).encode())
    return digest.hexdigest()


def _files(root):
    root = Path(root)
    result = {}
    for p in sorted(root.rglob('*')):
        if p.is_symlink():
            raise ValueError('Symlink in candidate artifact')
        if p.is_file():
            result[str(p.relative_to(root))] = sha(p)
    return result


def _repo(path):
    # Opening an existing repository must not initialize or change its source.
    repo = Repository.__new__(Repository)
    repo.path = Path(path)
    return repo


def _git_state(root):
    repo = _repo(root / 'public')
    return {'head': repo.version, 'refs': repo.git('show-ref'),
            'working_tree': repo.git('--no-optional-locks', 'status', '--porcelain'),
            'tree': repo.git('rev-parse', 'HEAD^{tree}')}


def _source_guard(root):
    return {'files': _files_excluding_git(root), 'git': _git_state(root)}


def _files_excluding_git(root):
    result = {}
    for p in sorted(root.rglob('*')):
        if '.git' in p.relative_to(root).parts:
            continue
        if p.is_symlink():
            raise ValueError('Symlink in source run')
        if p.is_file():
            result[str(p.relative_to(root))] = sha(p)
    return result


def _clone(root, destination, base):
    # Separate object storage, no worktrees or metadata writes in the original run.
    _repo(root / 'public').git('clone', '--no-hardlinks', '--quiet', str(root / 'public'), str(destination))
    repo = _repo(destination)
    repo.git('config', 'user.name', 'Offline order audit')
    repo.git('config', 'user.email', 'offline@code-testing.invalid')
    repo.git('checkout', '--quiet', '--detach', base)
    repo.git('reset', '--hard', '--quiet', base)
    return repo


def _load_run(root):
    manifest, state = _read(root / 'manifest.json'), _read(root / 'state.json')
    if manifest.get('status') != 'completed':
        raise ValueError('All source runs must have status completed')
    if state.get('config') != manifest.get('config'):
        raise ValueError('State and manifest configs disagree')
    n = state.get('completed_rounds')
    snapshots = state.get('snapshots', [])
    if type(n) is not int or n < 1 or n != state.get('round_limit') or [s.get('round') for s in snapshots] != list(range(-1, n)):
        raise ValueError('Source run has incomplete round snapshots')
    events = [_read_line(line) for line in (root / 'events.jsonl').read_text().splitlines() if line.strip()]
    if [e.get('seq') for e in events] != list(range(len(events))):
        raise ValueError('Source event sequence is incomplete')
    event_snaps = [{k: v for k, v in e.items() if k not in {'event', 'seq', 'elapsed'}}
                   for e in events if e['event'] == 'snapshot']
    if event_snaps != snapshots:
        raise ValueError('State and event snapshots disagree')
    finals = [e for e in events if e['event'] == 'finalized']
    if not finals or finals[-1].get('commit') != snapshots[-1]['commit'] or _repo(root / 'public').version != snapshots[-1]['commit']:
        raise ValueError('Finalized source commit is missing or inconsistent')
    task_path = Path(manifest['task_path']).resolve()
    task = Task.load(task_path)
    if task.data['benchmark'] not in {'commit0', 'fixture'}:
        raise ValueError('Order audit supports Commit0 and offline fixtures only')
    if task.data['benchmark'] == 'fixture':
        task = FixtureTask(task.root, task.data)
    if task.data != manifest['task'] or sha(task_path) != manifest['task_manifest_sha256'] or _tree_digest(task.root) != manifest['task_input_sha256']:
        raise ValueError('Task inputs differ from the recorded run')
    for rel, expected in manifest.get('source_hashes', {}).items():
        p = (root / 'source-snapshot' / rel).resolve()
        if not p.is_relative_to((root / 'source-snapshot').resolve()) or not p.is_file() or sha(p) != expected:
            raise ValueError('Frozen inference source hash mismatch')
    frozen_hashes = manifest.get('source_hashes', {})
    # Historical candidate checks must use the same verifier as fresh replay checks.
    # Only handbuilt synthetic fixtures without any inference snapshot are exempt.
    if task.data['benchmark'] != 'fixture' or frozen_hashes:
        verifier_names = ['benchmarks.py', 'sandbox.py']
        if task.data['benchmark'] == 'fixture':
            verifier_names.append('fixture.py')
        for name in verifier_names:
            expected = frozen_hashes.get('codetest/' + name)
            if expected is None or sha(Path(__file__).with_name(name)) != expected:
                raise ValueError('Current verifier implementation differs from the recorded run: ' + name)
    if task.data['benchmark'] == 'commit0':
        image = task.data.get('image', '')
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', image):
            raise ValueError('Audit requires a pinned local Docker image ID')
        present = run(['docker', 'image', 'inspect', image], check=False)
        if present.returncode or json.loads(present.stdout)[0]['Id'] != image:
            raise ValueError('Recorded Docker image must already exist locally; audit never builds or pulls images')
    rounds = []
    for r in range(n):
        base = snapshots[r]['commit']
        candidate_events = [e for e in events if e['event'] == 'candidate' and e.get('round') == r]
        if any('task' in e for e in candidate_events):
            raise ValueError('Swarm candidates are outside this classic-market audit')
        paths = list((root / 'candidates' / f'round-{r:02}').glob('*/candidate.json'))
        saved = {}
        for path in paths:
            c = _read(path)
            if c['team'] in saved:
                raise ValueError('Duplicate candidate team in one round')
            saved[c['team']] = (c, path)
        if len(candidate_events) != len(saved) or {e['team'] for e in candidate_events} != set(saved):
            raise ValueError('Candidate records are incomplete or duplicated')
        eligible = []
        for e in candidate_events:
            c, path = saved[e['team']]
            for key in ('team', 'base_commit', 'bid', 'eligible', 'changed', 'shared_artifact'):
                if c.get(key) != e.get(key):
                    raise ValueError('Candidate file and event disagree')
            if c.get('base_commit') != base:
                raise ValueError('Round candidates do not share the recorded base')
            if not c.get('eligible'):
                continue
            bid = c.get('bid')
            if type(bid) not in (int, float) or not math.isfinite(bid) or bid <= 0:
                raise ValueError('Invalid eligible candidate bid')
            ident = c.get('shared_artifact', '')
            if not re.fullmatch(r'[A-Za-z0-9_.-]+', ident):
                raise ValueError('Eligible candidate is missing its immutable artifact')
            artifact = root / 'scratchpad' / 'artifacts' / ident
            meta = _read(artifact / 'metadata.json')
            source, patch = artifact / 'source', artifact / 'patch.diff'
            if any(meta.get(k) != v for k, v in {'id': ident, 'round': r, 'team': c['team'], 'base_commit': base}.items()):
                raise ValueError('Candidate artifact identity mismatch')
            if _files(source) != meta['source_sha256'] or sha(patch) != meta['patch_sha256']:
                raise ValueError('Candidate artifact hash mismatch')
            tracked = path.parent / 'tracked-source'
            if _files(tracked) != meta['source_sha256'] or _tree_digest(tracked) != _tree_digest(source):
                raise ValueError('Candidate source differs from its immutable artifact')
            if {k: v for k, v in c['check'].items() if k != 'failing'} != e.get('check'):
                raise ValueError('Candidate check and event disagree')
            eligible.append({**c, 'source': str(source), '_patch': str(patch), '_event_seq': e['seq'],
                             '_input_hashes': {'candidate': sha(path), 'metadata': sha(artifact / 'metadata.json'),
                                               'patch': sha(patch), 'source_tree': _tree_digest(source)}})
        attempts = [e['team'] for e in events if e.get('round') == r and e['event'] in
                    {'publication', 'merge_rejected', 'merge_skipped'}]
        observed = (manifest['config'].get('merge_slots', 1) > 1 and len(attempts) == len(eligible)
                    and len(set(attempts)) == len(attempts) and set(attempts) == {c['team'] for c in eligible})
        if observed:
            bids = {c['team']: c['bid'] for c in eligible}
            observed = all(bids[a] >= bids[b] for a, b in zip(attempts, attempts[1:]))
        rounds.append({'round': r, 'base_commit': base, 'base_observed': snapshots[r]['feedback'],
                       'candidates': eligible, 'observed_bid_order': attempts if observed else None})
    return {'root': root, 'manifest': manifest, 'state': state, 'events': events, 'task': task, 'rounds': rounds}


def _read_line(line):
    return json.loads(line)


def strict_decision(before, after):
    """Same test-set gate as Engine.verified_merge, including bootstrap semantics."""
    if not after.get('build') or after.get('score') is None or after.get('failing') is None:
        return False, 'merged tree does not build or score', []
    if before.get('score') is None or before.get('failing') is None:
        return True, 'bootstrap scored tree', []
    if after.get('total') != before.get('total'):
        return False, 'visible test denominator changed', []
    prior, current = set(before['failing']), set(after['failing'])
    if current - prior:
        return False, 'breaks passing tests', []
    flipped = sorted(prior - current)
    return (True, 'verified improvement', flipped) if flipped else (False, 'no additional test passes', [])


def _orders(candidates, base_check, observed, seeds):
    names = [c['team'] for c in candidates]
    by_name = {c['team']: c for c in candidates}
    rng = random.Random(seeds[0])
    tie = {name: rng.random() for name in names}
    bid = observed if observed is not None else sorted(names, key=lambda n: (-by_name[n]['bid'], tie[n]))
    yield 'bid', None if observed is not None else seeds[0], bid, 'observed_attempt_order' if observed is not None else 'seeded_not_observed'
    def gain(name):
        c = by_name[name]['check']
        if c.get('build') and c.get('score') is not None and c.get('passed') is not None:
            if base_check.get('score') is None:
                return None  # No scored base means no defined marginal gain.
            if c.get('total') == base_check.get('total'):
                return c['passed'] - base_check['passed']
        return None
    gains = {name: gain(name) for name in names}
    ordered = sorted(names, key=lambda n: (gains[n] is None, -(gains[n] or 0), tie[n]))
    yield 'gain', seeds[0], ordered, 'same_base_gain' if base_check.get('score') is not None else 'bootstrap_gain_undefined_seeded_ties'
    for seed in seeds:
        ordered = list(names)
        random.Random(seed).shuffle(ordered)
        yield 'random', seed, ordered, 'seeded_uniform_permutation'


def _verify(data, repo, report_dir):
    tree = repo.git('rev-parse', 'HEAD^{tree}')
    check = data['task'].check(repo.path, report_dir)
    signature = tuple(check.get(k) for k in ('build', 'score', 'passed', 'total')) + (
        tuple(sorted(check['failing'])) if check.get('failing') is not None else None,)
    seen = data.setdefault('_verification_signatures', {})
    if tree in seen and seen[tree] != signature:
        raise ValueError('Non-deterministic verification for identical Git tree: ' + tree)
    seen[tree] = signature
    return check


def _apply_policy(data, round_data, base_check, policy, seed, order, order_source, out, slots):
    started = time.monotonic()
    candidates = {c['team']: c for c in round_data['candidates']}
    rows, accepted, calls = [], 0, 0
    feedback = copy.deepcopy(base_check)
    with tempfile.TemporaryDirectory(prefix='codetest-order-audit-') as temp:
        repo = _clone(data['root'], Path(temp) / 'repo', round_data['base_commit'])
        for index, name in enumerate(order):
            c = candidates[name]
            row = {'team': name, 'bid': c['bid'], 'order_index': index}
            if accepted >= slots:
                rows.append({**row, 'accepted': False, 'reason': 'no merge slot left this round', 'skipped': True})
                continue
            head = repo.version
            mode = 'direct' if head == c['base_commit'] else '3way'
            row['mode'] = mode
            try:
                if mode == 'direct':
                    patch = repo.replace(Path(c['source']), head)
                    if not patch:
                        raise RuntimeError('no tracked change')
                else:
                    repo.git('apply', '--3way', c['_patch'])
                    repo.git('add', '--all')
                    patch = repo.git('diff', '--cached', '--binary')
                    if not patch:
                        raise RuntimeError('no tracked change')
                    repo.git('commit', '-qm', 'Offline candidate merge')
            except (RuntimeError, ValueError):
                repo.git('reset', '--hard', '--quiet', head); repo.git('clean', '-qfd')
                rows.append({**row, 'accepted': False, 'reason': 'conflict or no tracked change'})
                continue
            # Fresh verification even for a direct merge. Cached candidate checks only rank gain.
            check = _verify(data, repo, out / 'checks' / f'{index:02}')
            calls += 1
            good, reason, flipped = strict_decision(feedback, check)
            rows.append({**row, 'accepted': good, 'reason': reason, 'flipped': flipped, 'check': check,
                         'tree': repo.git('rev-parse', 'HEAD^{tree}')})
            if good:
                accepted += 1; feedback = check
            else:
                repo.git('reset', '--hard', '--quiet', head); repo.git('clean', '-qfd')
        result = {'policy': policy, 'seed': seed, 'order_source': order_source, 'order': order,
                  'accepted': accepted, 'attempts': rows, 'final_check': feedback,
                  'final_tree': repo.git('rev-parse', 'HEAD^{tree}'), 'verification_calls': calls,
                  'seconds': time.monotonic() - started}
    write_json(out / 'result.json', result)
    return result


def run_order_audit(run_dirs, out, random_seeds=(7, 17, 27), merge_slots=10):
    roots = [Path(p).resolve() for p in run_dirs]
    out = Path(out).resolve()
    seeds = list(random_seeds)
    if not roots or len(set(roots)) != len(roots):
        raise ValueError('Provide distinct source runs')
    if len(seeds) < 3 or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError('Provide at least three distinct integer random seeds')
    if type(merge_slots) is not int or merge_slots < 1:
        raise ValueError('merge_slots must be a positive integer')
    if out.exists():
        raise ValueError('Audit output already exists; no overwrite or retry')
    if any(out.is_relative_to(r) or r.is_relative_to(out) for r in roots):
        raise ValueError('Audit output must be separate from source runs')
    # No output claim or local evaluation while either experiment is still running.
    manifests = [_read(r / 'manifest.json') for r in roots]
    if any(m.get('status') != 'completed' for m in manifests):
        raise ValueError('All source runs must have status completed')
    if any(out.is_relative_to(Path(m['task_path']).resolve().parent) for m in manifests):
        raise ValueError('Audit output must be separate from task inputs')
    out.mkdir(parents=True, exist_ok=False)
    status = {'status': 'running', 'pid': os.getpid(), 'started_at': time.time(), 'runs': [str(r) for r in roots]}
    write_json(out / 'status.json', status)
    try:
        guards = [_source_guard(r) for r in roots]
        datasets = [_load_run(r) for r in roots]
        result = {'version': 1, 'scope': SCOPE, 'merge_slots': merge_slots, 'random_seeds': seeds,
                  'verifier': 'fresh Task.check for every applied tree; strict nonregression and positive flips; bootstrap exception',
                  'implementation_sha256': sha(Path(__file__)),
                  'implementation_source_hashes': _implementation_hashes(), 'runs': []}
        for index, data in enumerate(datasets):
            root = data['root']
            destination = out / f'run-{index:02}-{root.name}'
            write_json(destination / 'metrics-corrected.json', metrics(data['state'], data['events']))
            item = {'source_run': str(root), 'source_guard': guards[index],
                    'task_manifest_sha256': data['manifest']['task_manifest_sha256'],
                    'task_input_sha256': data['manifest']['task_input_sha256'],
                    'inference_source_hashes': data['manifest'].get('source_hashes', {}),
                    'task': data['task'].data, 'rounds': []}
            for rd in data['rounds']:
                rdout = destination / f'round-{rd["round"]:02}'
                base_started = time.monotonic()
                with tempfile.TemporaryDirectory(prefix='codetest-order-base-') as temp:
                    repo = _clone(root, Path(temp) / 'repo', rd['base_commit'])
                    base_check = _verify(data, repo, rdout / 'base-check')
                    base_tree = repo.git('rev-parse', 'HEAD^{tree}')
                observed = rd['base_observed']
                if any(base_check.get(k) != observed.get(k) for k in ('build', 'score', 'passed', 'total')):
                    raise ValueError('Fresh base verification differs from recorded feedback')
                summary = {'round': rd['round'], 'base_commit': rd['base_commit'], 'base_tree': base_tree,
                           'base_check': base_check, 'base_verification_calls': 1,
                           'base_verification_seconds': time.monotonic() - base_started,
                           'eligible': [{'team': c['team'], 'bid': c['bid'], 'check': c['check'],
                                         'input_hashes': c['_input_hashes'], 'event_seq': c['_event_seq']}
                                        for c in rd['candidates']], 'policies': []}
                for policy, seed, order, origin in _orders(rd['candidates'], base_check, rd['observed_bid_order'], seeds):
                    summary['policies'].append(_apply_policy(data, rd, base_check, policy, seed, order, origin,
                                                             rdout / f'{policy}-{seed if seed is not None else "observed"}', merge_slots))
                random_scores = [p['final_check'].get('score') for p in summary['policies'] if p['policy'] == 'random']
                known = [s for s in random_scores if s is not None]
                summary['random_summary'] = {'repeats': len(random_scores), 'scored': len(known),
                    'mean': sum(known) / len(known) if known else None,
                    'min': min(known) if known else None, 'max': max(known) if known else None,
                    'note': 'Descriptive range and mean over fixed seeds, not a confidence interval.'}
                write_json(rdout / 'summary.json', summary)
                item['rounds'].append(summary)
            if _tree_digest(data['task'].root) != data['manifest']['task_input_sha256']:
                raise ValueError('Task inputs changed during audit')
            result['runs'].append(item)
        if any(_source_guard(r) != guard for r, guard in zip(roots, guards)):
            raise ValueError('Source run files or Git references changed during audit')
        if _implementation_hashes() != result['implementation_source_hashes']:
            raise ValueError('Audit implementation changed during evaluation')
        write_json(out / 'results.json', result)
        write_json(out / 'status.json', {**status, 'status': 'completed', 'finished_at': time.time()})
        return result
    except BaseException as error:
        write_json(out / 'status.json', {**status, 'status': 'failed', 'finished_at': time.time(),
                                        'error_type': type(error).__name__, 'reason': str(error)[:1500]})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=SCOPE)
    parser.add_argument('--run', action='append', required=True, dest='runs')
    parser.add_argument('--out', required=True)
    parser.add_argument('--random-seeds', type=int, nargs='+', default=[7, 17, 27])
    parser.add_argument('--merge-slots', type=int, default=10)
    args = parser.parse_args(argv)
    result = run_order_audit(args.runs, args.out, args.random_seeds, args.merge_slots)
    print(json.dumps({'status': 'completed', 'out': args.out, 'runs': len(result['runs']), 'scope': SCOPE}))


if __name__ == '__main__':
    main()
