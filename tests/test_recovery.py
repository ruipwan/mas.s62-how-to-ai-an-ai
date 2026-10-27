import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor')]
from codetest.config import Config
from codetest.engine import Engine, tree_digest
from codetest.fixture import FixtureClient, create_task
from codetest.io import Events, write_json, sha
from codetest.policy import Policy, prompt_revision
from codetest.recovery import RecoveryDivergence, run_recovery, validate_recovery, _fingerprint


class MeteredFixture(FixtureClient):
    def __init__(self, events, fail_at):
        self.events, self.fail_at, self.calls = events, fail_at, 0
        self.last_completion = {}

    def generate(self, messages, max_tokens, phase, agent):
        ident = f'fixture-call-{self.calls:03}'
        self.events.emit('api_reserved', call_id=ident, phase=phase, agent=agent,
                         reserved_usd=.01, max_tokens=max_tokens)
        if self.calls == self.fail_at:
            self.events.emit('api_stopped', call_id=ident, phase=phase, agent=agent)
            raise TimeoutError('scripted interrupted request')
        self.calls += 1
        receipt = {'call_id': ident, 'phase': phase, 'agent': agent, 'cost_usd': 0,
                   'max_tokens': max_tokens, 'finish_reason': 'stop', 'usage': {},
                   'id': ident, 'model': 'scripted', 'provider': 'scripted'}
        self.events.emit('api_charged', **receipt)
        self.last_completion = receipt
        return super().generate(messages, max_tokens, phase, agent)


def interrupted_run(root, fail_at=42):
    root.mkdir()
    task = create_task(root / 'fixture-task')
    cfg = Config(k=4, rounds=3, max_rounds=3, work_calls=4, shared_scratchpad=True, seed=7)
    events = Events(root)
    client = MeteredFixture(events, fail_at)
    engine = Engine(cfg, task, Policy(client, cfg, events), events, root)
    manifest = {'version': 1, 'backend': 'scripted_fixture', 'status': 'interrupted',
                'prompt_revision': prompt_revision(cfg), 'config': cfg.as_dict(), 'task': task.data,
                'task_path': str(task.root / 'task.json'), 'task_manifest_sha256': sha(task.root / 'task.json'),
                'task_input_sha256': tree_digest(task.root),
                'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in (ROOT / 'codetest').glob('*') if p.is_file()},
                'team_manager_sha256': sha(ROOT / 'vendor/hayekmas/adapters/teams/team.py'),
                'usage': {'accounted_usd': .01, 'calls': fail_at + 1}}
    for rel in manifest['source_hashes']:
        target = root / 'source-snapshot' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    write_json(root / 'manifest.json', manifest)
    try:
        engine.run()
    except TimeoutError:
        events.emit('stopped', reason='scripted interrupted request')
    else:
        raise AssertionError('Fixture did not reach its interruption')
    engine.finalize()
    write_json(root / 'manifest.json', manifest)
    return cfg


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.cfg = interrupted_run(self.source)

    def test_offline_validation_reaches_exact_failed_slot_without_live_or_ledger(self):
        before = _fingerprint(self.source)
        with patch('codetest.budget.Ledger.register', side_effect=AssertionError('no ledger')):
            with patch('requests.post', side_effect=AssertionError('no network')):
                result = validate_recovery(self.source, self.root / 'validated')
        self.assertEqual(result['status'], 'validated')
        report = json.loads((self.root / 'validated/recovery-report.json').read_text())
        self.assertEqual(report['replayed_successful_calls'], 42)
        self.assertGreater(report['verified_candidates'], 0)
        self.assertGreater(report['active_commands_reexecuted'], 0)
        self.assertEqual(before, _fingerprint(self.source))

    def test_live_factory_only_after_verified_replay_and_finishes_original_horizon(self):
        calls = []
        def live(context):
            report = context['replay_report']
            self.assertEqual(report['replayed_successful_calls'], 42)
            self.assertEqual(context['cfg'].rounds, 3)
            self.assertTrue(callable(context['event']))
            calls.append(context)
            return FixtureClient()
        result = run_recovery(self.source, self.root / 'resumed', live,
                              usage_callback=lambda: {'accounted_usd': .01, 'prior_retained_usd': .01})
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(len(calls), 1)
        state = json.loads((self.root / 'resumed/state.json').read_text())
        self.assertEqual(state['completed_rounds'], 3)
        events = [json.loads(x) for x in (self.root / 'resumed/events.jsonl').read_text().splitlines()]
        charged = [e for e in events if e['event'] == 'api_charged']
        self.assertEqual(len(charged), 42)
        self.assertTrue(all(e['historical_reference'] for e in charged))
        self.assertEqual(len({e['call_id'] for e in charged}), 42)
        self.assertEqual(result['usage']['accounted_usd'], .01)

    def test_altered_partial_tool_output_blocks_factory(self):
        path = self.source / 'events.jsonl'
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        target = next(e for e in reversed(rows) if e['event'] == 'tool' and 'output' in e['result'])
        target['result']['output'] += 'changed observation'
        path.write_text(''.join(json.dumps(e) + '\n' for e in rows))
        def forbidden(context):
            raise AssertionError('Live factory must remain blocked')
        with self.assertRaises(RecoveryDivergence):
            run_recovery(self.source, self.root / 'failed', forbidden)
        self.assertEqual(json.loads((self.root / 'failed/manifest.json').read_text())['status'], 'recovery_failed')

    def test_saved_candidate_tamper_and_existing_output_are_rejected(self):
        candidate = next((self.source / 'candidates').glob('round-*/*/tracked-source/module_*.py'))
        candidate.write_text(candidate.read_text() + '# changed\n')
        with self.assertRaises(RecoveryDivergence):
            run_recovery(self.source, self.root / 'failed', lambda _: self.fail('live factory called'))
        with self.assertRaises(FileExistsError):
            validate_recovery(self.source, self.root / 'failed')


if __name__ == '__main__':
    unittest.main()
