import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from codetest.fixture import create_task, FixtureTask
from codetest.io import Events, sha, write_json
from codetest.order_audit import run_order_audit, strict_decision, _tree_digest, _source_guard, _orders
from codetest.sandbox import Repository, copy_source
from codetest.scratchpad import Scratchpad


def saved_run(root, observed=False):
    root.mkdir()
    task = create_task(root / 'fixture-task')
    repo = Repository(root / 'public', task.seed)
    base = repo.version
    events = Events(root)
    initial = task.check(repo.path, root / 'initial-check')
    visible = lambda c: {k: v for k, v in c.items() if k != 'failing'}
    snapshots = []
    def snapshot(r):
        row = {'round': r, 'agents': [], 'commit': base, 'feedback': visible(initial), 'accepted': 0, 'probes': 0}
        snapshots.append(row)
        events.emit('snapshot', **row)
    snapshot(-1)
    scratch = Scratchpad(root / 'scratchpad', events.emit)
    # A and B conflict on module_00. A gives two tests, B one; C is independent.
    for name, bid, files in [('A', .05, {0: 1, 1: 1}), ('B', .1, {0: 2}), ('C', .025, {2: 1})]:
        directory = root / 'candidates' / 'round-00' / name
        source = directory / 'tracked-source'
        copy_source(task.seed, source)
        for number, value in files.items():
            (source / f'module_{number:02}.py').write_text(f'def compute(value):\n    return value + {value}\n')
        c = {'team': name, 'members': [name], 'base_commit': base, 'source': str(source),
             'check': task.check(source, directory / 'check'), 'changed': True,
             'changed_paths': [f'module_{i:02}.py' for i in files], 'code_changed': True,
             'new_probes': [], 'workers': [name], 'fetched_from': [], 'votes': {name: True},
             'pledges': {name: bid}, 'bid': bid, 'participants': [name], 'eligible': True}
        c['shared_artifact'] = scratch.share(c, repo.path, 0)
        events.emit('candidate', round=0, **{k: v for k, v in c.items() if k not in ('source', 'check')}, check=visible(c['check']))
        write_json(directory / 'candidate.json', c)
    if observed:
        for name in ['B', 'A', 'C']:
            events.emit('merge_rejected', round=0, team=name, reason='historical fixture rejection')
    snapshot(0)
    events.emit('finalized', round=0, commit=base)
    config = {'merge_slots': 3 if observed else 1}
    state = {'config': config, 'completed_rounds': 1, 'round_limit': 1, 'snapshots': snapshots,
             'history': [], 'agents': []}
    write_json(root / 'state.json', state)
    write_json(root / 'manifest.json', {'status': 'completed', 'config': config, 'task': task.data,
               'task_path': str(task.root / 'task.json'), 'task_manifest_sha256': sha(task.root / 'task.json'),
               'task_input_sha256': _tree_digest(task.root), 'source_hashes': {}})
    return root


class OrderAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_frozen_candidates_conflicts_and_rejections_change_order_outcome(self):
        source = saved_run(self.root / 'source')
        before = _source_guard(source)
        result = run_order_audit([source], self.root / 'audit', merge_slots=2)
        r = result['runs'][0]['rounds'][0]
        bid, gain = r['policies'][:2]
        self.assertEqual(bid['order'], ['B', 'A', 'C'])
        self.assertEqual(bid['order_source'], 'seeded_not_observed')
        self.assertEqual(bid['final_check']['passed'], 2)
        self.assertEqual(gain['final_check']['passed'], 3)
        self.assertEqual(bid['attempts'][1]['reason'], 'conflict or no tracked change')
        self.assertTrue(bid['attempts'][2]['accepted'])  # A rejection consumes no merge slot.
        self.assertEqual(len({tuple(sorted(p['order'])) for p in r['policies']}), 1)
        self.assertEqual(r['random_summary']['repeats'], 3)
        self.assertEqual(r['base_verification_calls'], 1)
        self.assertEqual(before, _source_guard(source))
        self.assertFalse((source / 'metrics.json').exists())
        self.assertTrue((self.root / 'audit/run-00-source/metrics-corrected.json').is_file())
        self.assertEqual(json.loads((self.root / 'audit/status.json').read_text())['status'], 'completed')
        second = run_order_audit([source], self.root / 'repeat', merge_slots=2)
        self.assertEqual([(p['order'], p['final_tree']) for p in r['policies']],
                         [(p['order'], p['final_tree']) for p in second['runs'][0]['rounds'][0]['policies']])

    def test_observed_bid_order_requires_complete_multi_merge_attempt_log(self):
        source = saved_run(self.root / 'source', observed=True)
        result = run_order_audit([source], self.root / 'audit')
        bid = result['runs'][0]['rounds'][0]['policies'][0]
        self.assertEqual(bid['order_source'], 'observed_attempt_order')
        self.assertIsNone(bid['seed'])
        self.assertEqual(bid['order'], ['B', 'A', 'C'])

    def test_running_second_run_blocks_all_evaluation_and_output(self):
        first = saved_run(self.root / 'first')
        second = saved_run(self.root / 'second')
        m = json.loads((second / 'manifest.json').read_text()); m['status'] = 'running'
        write_json(second / 'manifest.json', m)
        with patch('codetest.order_audit.FixtureTask.check', side_effect=AssertionError('must not verify')):
            with self.assertRaisesRegex(ValueError, 'completed'):
                run_order_audit([first, second], self.root / 'audit')
        self.assertFalse((self.root / 'audit').exists())

    def test_tamper_fails_before_verification_and_preserves_failed_marker(self):
        source = saved_run(self.root / 'source')
        patch_path = source / 'scratchpad/artifacts/r01-A/patch.diff'
        patch_path.write_text(patch_path.read_text() + 'tampered\n')
        with patch('codetest.order_audit.FixtureTask.check', side_effect=AssertionError('must not verify')):
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                run_order_audit([source], self.root / 'audit')
        status = json.loads((self.root / 'audit/status.json').read_text())
        self.assertEqual(status['status'], 'failed')
        self.assertIn('hash mismatch', status['reason'])
        with self.assertRaisesRegex(ValueError, 'already exists'):
            run_order_audit([source], self.root / 'audit')

    def test_consistent_old_snapshot_cannot_use_a_changed_current_verifier(self):
        source = saved_run(self.root / 'source')
        manifest = json.loads((source / 'manifest.json').read_text())
        hashes = {}
        for name in ('benchmarks.py', 'sandbox.py', 'fixture.py'):
            target = source / 'source-snapshot/codetest' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / 'codetest' / name, target)
            if name == 'benchmarks.py':
                target.write_text(target.read_text() + '\n# internally consistent older verifier snapshot\n')
            hashes['codetest/' + name] = sha(target)
        manifest['source_hashes'] = hashes
        write_json(source / 'manifest.json', manifest)
        # Manifest and saved snapshots agree, but the loaded verifier does not.
        with patch('codetest.order_audit.FixtureTask.check', side_effect=AssertionError('must not verify')):
            with self.assertRaisesRegex(ValueError, 'Current verifier implementation.*benchmarks.py'):
                run_order_audit([source], self.root / 'audit')
        status = json.loads((self.root / 'audit/status.json').read_text())
        self.assertEqual(status['status'], 'failed')
        self.assertIn('verifier implementation', status['reason'])

    def test_identical_tree_with_different_fresh_results_fails_without_retry(self):
        source = saved_run(self.root / 'source')
        check = FixtureTask.check
        seen = set()
        inconsistent = []
        def fluctuating(task, path, report_dir):
            result = check(task, path, report_dir)
            digest = _tree_digest(path)
            if digest in seen:
                inconsistent.append(digest)
                result = {**result, 'passed': result['passed'] + 1,
                          'score': (result['passed'] + 1) / result['total'],
                          'failing': result['failing'][1:]}
            seen.add(digest)
            return result
        with patch.object(FixtureTask, 'check', fluctuating):
            with self.assertRaisesRegex(ValueError, 'Non-deterministic verification'):
                run_order_audit([source], self.root / 'audit')
        self.assertEqual(len(inconsistent), 1)  # Stops on the first mismatch, no retry.
        status = json.loads((self.root / 'audit/status.json').read_text())
        self.assertEqual(status['status'], 'failed')
        self.assertIn('Non-deterministic verification', status['reason'])
        self.assertFalse((self.root / 'audit/results.json').exists())

    def test_missing_snapshot_or_candidate_fails_closed(self):
        for broken in ['snapshot', 'candidate']:
            source = saved_run(self.root / broken)
            if broken == 'snapshot':
                s = json.loads((source / 'state.json').read_text()); s['completed_rounds'] = 2
                write_json(source / 'state.json', s)
            else:
                (source / 'candidates/round-00/A/candidate.json').unlink()
            with self.assertRaises(ValueError):
                run_order_audit([source], self.root / ('audit-' + broken))

    def test_output_cannot_modify_source_run(self):
        source = saved_run(self.root / 'source')
        before = _source_guard(source)
        with self.assertRaisesRegex(ValueError, 'separate'):
            run_order_audit([source], source / 'audit')
        self.assertEqual(before, _source_guard(source))

    def test_strict_gate_bootstrap_test_identity_and_denominator(self):
        before = {'build': True, 'score': .5, 'passed': 1, 'total': 2, 'failing': ['a']}
        self.assertEqual(strict_decision(before, before)[1], 'no additional test passes')
        replacement = {**before, 'failing': ['b']}
        self.assertEqual(strict_decision(before, replacement)[1], 'breaks passing tests')
        changed = {**before, 'total': 3, 'passed': 3, 'score': 1, 'failing': []}
        self.assertEqual(strict_decision(before, changed)[1], 'visible test denominator changed')
        self.assertFalse(strict_decision(before, {**before, 'score': None})[0])
        success = {**before, 'passed': 2, 'score': 1, 'failing': []}
        self.assertEqual(strict_decision(before, success), (True, 'verified improvement', ['a']))
        self.assertEqual(strict_decision({'score': None}, before), (True, 'bootstrap scored tree', []))

    def test_ties_reproducible_and_unscored_gain_is_not_zero(self):
        cs = [{'team': 'a', 'bid': .1, 'check': {'build': True, 'score': None}},
              {'team': 'b', 'bid': .1, 'check': {'build': True, 'score': .1, 'passed': 2, 'total': 20}}]
        base = {'score': None}
        one = list(_orders(cs, base, None, [7, 17, 27]))
        self.assertEqual(one, list(_orders(cs, base, None, [7, 17, 27])))
        self.assertEqual(one[1][2], ['b', 'a'])
        self.assertEqual(one[1][3], 'bootstrap_gain_undefined_seeded_ties')
        self.assertEqual([p[1] for p in one if p[0] == 'random'], [7, 17, 27])

    def test_real_cli_demo_export_integrates_with_order_audit(self):
        source = self.root / 'cli-demo'
        completed = subprocess.run(
            [sys.executable, str(ROOT / 'run.py'), 'demo', '--config',
             str(ROOT / 'configs/demo-market.json'), '--out', str(source)],
            cwd=ROOT, check=True, capture_output=True, text=True, timeout=120)
        self.assertIn('"status": "completed"', completed.stdout)
        manifest = json.loads((source / 'manifest.json').read_text())
        self.assertEqual(manifest['backend'], 'scripted_fixture')
        self.assertEqual(manifest['usage']['accounted_usd'], 0)
        self.assertEqual(manifest['usage']['calls'], 0)
        before = _source_guard(source)
        result = run_order_audit([source], self.root / 'cli-audit')
        self.assertEqual(before, _source_guard(source))
        self.assertEqual(len(result['runs'][0]['rounds']), manifest['config']['rounds'])
        first = result['runs'][0]['rounds'][0]
        self.assertGreater(len(first['eligible']), 1)
        self.assertEqual(first['policies'][0]['order_source'], 'observed_attempt_order')
        self.assertGreater(first['policies'][0]['accepted'], 1)
        hashes = result['implementation_source_hashes']
        for name in ('order_audit.py', 'report.py', 'benchmarks.py', 'sandbox.py', 'fixture.py'):
            self.assertEqual(hashes['codetest/' + name], sha(ROOT / 'codetest' / name))
        saved = json.loads((self.root / 'cli-audit/results.json').read_text())
        self.assertEqual(saved['implementation_source_hashes'], hashes)
        self.assertEqual(json.loads((self.root / 'cli-audit/status.json').read_text())['status'], 'completed')

    def test_module_import_has_no_model_ledger_or_engine_capability(self):
        script = ('import sys; import codetest.order_audit; '
                  'assert not any(x in sys.modules for x in '
                  '("codetest.engine", "codetest.budget", "codetest.policy", "codetest.cli", "requests"))')
        subprocess.run([sys.executable, '-c', script], cwd=ROOT, check=True, capture_output=True)


if __name__ == '__main__':
    unittest.main()
