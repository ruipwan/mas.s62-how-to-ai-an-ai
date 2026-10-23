import io
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor')]
from codetest.budget import Ledger, BudgetStop, BudgetBusy, OpenRouter
from codetest.config import Config
from codetest.engine import Agent, Engine, terminal_credits
from hayekmas.adapters.teams.team import TeamManager
from codetest.fixture import create_task, FixtureClient
from codetest.io import Events, write_json
from codetest.policy import Policy
from codetest.probes import validate_probe, discover
from codetest.sandbox import safe_extract
from codetest.benchmarks import junit, Task, overlay_source
from codetest.cli import execute, evaluate


class OverlayAuditTests(unittest.TestCase):
    def test_protected_submission_does_not_hide_retained_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, source, dest = (root / p for p in ['reference', 'source', 'overlay'])
            reference.mkdir(); source.mkdir()
            (reference / 'pyproject.toml').write_text('original metadata')
            (source / 'pyproject.toml').write_text('submitted metadata')
            (reference / 'implementation.py').write_text('reference implementation')
            (source / 'implementation.py').write_text('generated implementation')
            (reference / 'retained.py').write_text('unreplaced reference implementation')
            audit = overlay_source(reference, source, dest, {'pyproject.toml'})
            self.assertEqual(audit['retained_reference_files'], ['pyproject.toml', 'retained.py'])
            self.assertEqual(audit['ignored_protected_submissions'], ['pyproject.toml'])
            self.assertEqual(audit['applied_source_files'], ['implementation.py'])
            self.assertEqual((dest / 'pyproject.toml').read_text(), 'original metadata')
            self.assertEqual((dest / 'implementation.py').read_text(), 'generated implementation')


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ledger = Ledger(Path(self.temp.name) / 'budget.sqlite')
        self.ledger.register('first', 100)
        self.ledger.register('second', 100)

    def test_campaign_cap_is_shared_and_durable(self):
        call = self.ledger.reserve('first', 90, 'work', 'a', 'model')
        self.ledger.settle(call, 90, {'id': 'provider'})
        recreated = Ledger(self.ledger.path)
        with self.assertRaises(BudgetStop):
            recreated.reserve('second', 11, 'work', 'b', 'model')
        self.assertEqual(recreated.summary()['accounted_usd'], 90)

    def test_uncertain_call_blocks_new_spend(self):
        call = self.ledger.reserve('first', 2, 'work', 'a', 'model')
        self.ledger.uncertain(call)
        with self.assertRaises(BudgetStop):
            self.ledger.reserve('second', 1, 'work', 'b', 'model')
        self.assertEqual(self.ledger.summary()['unconfirmed_usd'], 2)

    def test_only_one_concurrent_reservation(self):
        def attempt(_):
            try:
                return self.ledger.reserve('first', 1, 'work', 'a', 'model')
            except BudgetStop:
                return None
        with ThreadPoolExecutor(max_workers=2) as pool:
            values = list(pool.map(attempt, range(2)))
        self.assertEqual(sum(x is not None for x in values), 1)

    def test_reviewed_error_keeps_full_budget_reservation(self):
        call = self.ledger.reserve('first', 90, 'work', 'a', 'model')
        with self.assertRaises(ValueError):
            self.ledger.retain_failed_reservation(call, 'Cannot acknowledge a live pending request')
        self.ledger.uncertain(call)
        self.ledger.retain_failed_reservation(call, 'HTTP error inspected; process exited; no replay; keep full allowance')
        summary = Ledger(self.ledger.path).summary()
        self.assertFalse(summary['blocked'])
        self.assertEqual(summary['reported_usd'], 0)
        self.assertEqual(summary['unconfirmed_usd'], 90)
        self.assertEqual(summary['accounted_usd'], 90)
        with self.assertRaises(BudgetStop):
            self.ledger.reserve('second', 11, 'work', 'b', 'model')
        self.ledger.reserve('second', 10, 'work', 'b', 'model')

    def test_overrun_blocks_campaign(self):
        call = self.ledger.reserve('first', 1, 'work', 'a', 'model')
        with self.assertRaises(BudgetStop):
            self.ledger.settle(call, 2, {})
        self.assertTrue(self.ledger.summary()['blocked'])
        self.assertEqual(self.ledger.summary()['reported_usd'], 2)

    def test_http_failure_retains_reservation_and_does_not_retry(self):
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'fixture-key'}):
            client = OpenRouter('test/model', self.ledger, 'first', {'prompt': 1, 'completion': 1}, lambda *a, **k: None, Path(self.temp.name) / 'STOP')
        with patch('codetest.budget.requests.post') as post:
            post.return_value.status_code = 503
            post.return_value.text = 'temporarily unavailable'
            post.return_value.headers = {}
            with self.assertRaises(RuntimeError):
                client.generate([{'role': 'user', 'content': 'test'}], 200, 'work', 'agent-00')
            self.assertEqual(post.call_count, 1)
        self.assertTrue(self.ledger.summary()['blocked'])
        self.assertGreater(self.ledger.summary()['unconfirmed_usd'], 0)

    def test_stop_file_blocks_before_send(self):
        (self.ledger.path.parent / 'STOP').touch()
        with self.assertRaises(BudgetStop):
            self.ledger.reserve('first', 1, 'work', 'a', 'model')
        self.assertEqual(self.ledger.summary()['calls'], 0)

    def test_explicit_reasoning_off_is_sent_without_changing_token_cap(self):
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'fixture-key'}):
            client = OpenRouter('test/model', self.ledger, 'first', {'prompt': 1, 'completion': 1},
                                lambda *a, **k: None, Path(self.temp.name) / 'STOP', reasoning_enabled=False)
        with patch('codetest.budget.requests.post') as post:
            post.return_value.status_code = 200
            post.return_value.json.return_value = {'id': 'fixture', 'usage': {'cost': .0001},
                'choices': [{'finish_reason': 'stop', 'message': {'content': '{}'}}]}
            self.assertEqual(client.generate([{'role': 'user', 'content': 'fixture'}], 2048, 'membership', 'a'), '{}')
            payload = post.call_args.kwargs['json']
            self.assertEqual(payload['reasoning'], {'enabled': False})
            self.assertEqual(payload['max_tokens'], 2048)
            self.assertEqual(post.call_count, 1)

    def test_luna_reasoning_budget_and_standard_route_are_preserved(self):
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'fixture-key'}):
            client = OpenRouter('openai/gpt-6-luna', self.ledger, 'first', {'prompt': .2, 'completion': 1},
                                lambda *a, **k: None, Path(self.temp.name) / 'STOP',
                                reasoning_effort='low', provider_ignore=['openai/flex'], structured_actions=True)
        with patch('codetest.budget.requests.post') as post:
            post.return_value.status_code = 200
            post.return_value.json.return_value = {'id': 'fixture', 'usage': {'cost': .0001},
                'choices': [{'finish_reason': 'stop', 'message': {'content': '{}'}}]}
            client.generate([{'role': 'user', 'content': 'fixture'}], 32768, 'work', 'a')
            payload = post.call_args.kwargs['json']
            self.assertEqual(payload['model'], 'openai/gpt-6-luna')
            self.assertEqual(payload['reasoning'], {'effort': 'low'})
            self.assertEqual(payload['max_tokens'], 32768)
            self.assertEqual(payload['provider']['ignore'], ['openai/flex'])
            self.assertFalse(payload['provider']['allow_fallbacks'])
            self.assertEqual(payload['response_format']['type'], 'json_schema')
            self.assertTrue(payload['response_format']['json_schema']['strict'])
            self.assertFalse(payload['response_format']['json_schema']['schema']['additionalProperties'])


class ParallelBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.first = self.make_ledger('first')
        self.second = self.make_ledger('second')

    def make_ledger(self, run, cap=100):
        ledger = Ledger(self.root / 'budget.sqlite', parallel=True)
        ledger.register(run, cap)
        self.addCleanup(ledger.close)
        return ledger

    def test_parallel_live_reservations_and_settlements_share_the_cap(self):
        a = self.first.reserve('first', 40, 'work', 'a', 'model')
        b = self.second.reserve('second', 50, 'work', 'b', 'model')
        self.assertEqual(self.first.summary()['pending'], 2)
        self.assertEqual(self.first.summary()['accounted_usd'], 90)
        self.first.settle(a, 39, {})
        self.second.settle(b, 49, {})
        self.assertEqual(self.first.summary()['reported_usd'], 88)
        with self.assertRaises(BudgetStop):
            self.first.reserve('first', 13, 'work', 'a', 'model')

    def test_simultaneous_parallel_reservations_cannot_overspend(self):
        def reserve(item):
            ledger, name = item
            try:
                return ledger.reserve(name, 60, 'work', name, 'model')
            except BudgetStop:
                return None
        with ThreadPoolExecutor(max_workers=2) as pool:
            attempts = list(pool.map(reserve, [(self.first, 'first'), (self.second, 'second')]))
        self.assertEqual(sum(x is not None for x in attempts), 1)
        self.assertEqual(self.first.summary()['accounted_usd'], 60)

    def test_parallel_request_slot_limit_and_run_cap(self):
        small = self.make_ledger('small', 2)
        with self.assertRaises(BudgetStop):
            small.reserve('small', 3, 'work', 'c', 'model')
        self.first.reserve('first', 1, 'work', 'a', 'model')
        self.second.reserve('second', 1, 'work', 'b', 'model')
        with self.assertRaises(BudgetBusy):
            small.reserve('small', 1, 'work', 'c', 'model')
        self.assertEqual(self.first.summary()['calls'], 2)

    def test_abandoned_parallel_reservation_is_retained_and_blocks(self):
        self.first.reserve('first', 20, 'work', 'a', 'model')
        # Simulate a process exiting without settlement: its OS lock vanishes.
        for lease in self.first._leases.values():
            lease.close()
        self.first._leases.clear()
        with self.assertRaises(BudgetStop):
            self.second.reserve('second', 1, 'work', 'b', 'model')
        self.assertTrue(self.second.summary()['blocked'])
        self.assertEqual(self.second.summary()['unconfirmed_usd'], 20)
        self.assertEqual(self.second.summary()['pending'], 0)

    def test_parallel_http_uncertainty_blocks_every_new_run(self):
        call = self.first.reserve('first', 5, 'work', 'a', 'model')
        self.first.uncertain(call)
        with self.assertRaises(BudgetStop):
            self.second.reserve('second', 1, 'work', 'b', 'model')
        self.assertEqual(self.second.summary()['accounted_usd'], 5)

    def test_legacy_worker_counts_parallel_reservations_without_interrupting(self):
        self.first.reserve('first', 60, 'work', 'a', 'model')
        # Execute the exact frozen legacy method against the same SQLite file.
        source = ROOT / 'runs/tinydb-luna-teams-k20-v5-s7/source-snapshot/codetest/budget.py'
        if not source.exists():
            self.skipTest('Historical legacy worker snapshot is absent')
        spec = importlib.util.spec_from_file_location('legacy_budget', source)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        legacy = module.Ledger(self.first.path)
        legacy.register('legacy', 100)
        with self.assertRaises(module.BudgetStop):
            legacy.reserve('legacy', 41, 'work', 'c', 'model')
        call = legacy.reserve('legacy', 40, 'work', 'c', 'model')
        self.assertEqual(legacy.summary()['accounted_usd'], 100)
        with self.assertRaises(BudgetBusy):
            self.second.reserve('second', 1, 'work', 'b', 'model')
        legacy.settle(call, .5, {})
        self.second.reserve('second', 1, 'work', 'b', 'model')

    def test_waiting_for_a_slot_does_not_send_or_retry_http(self):
        serial = Ledger(self.first.path); serial.register('serial', 100)
        busy = serial.reserve('serial', 1, 'work', 's', 'model')
        events = []
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'fixture-key'}):
            client = OpenRouter('test/model', self.first, 'first', {'prompt': 1, 'completion': 1},
                                lambda kind, **v: events.append(kind), self.root / 'STOP')
        with patch('codetest.budget.requests.post') as post, \
                patch('codetest.budget.time.sleep', side_effect=lambda _: serial.settle(busy, .1, {})):
            post.return_value.status_code = 200
            post.return_value.json.return_value = {'usage': {'cost': .001},
                'choices': [{'finish_reason': 'stop', 'message': {'content': '{}'}}]}
            self.assertEqual(client.generate([], 100, 'work', 'a'), '{}')
            self.assertEqual(post.call_count, 1)
        self.assertIn('api_queue_wait', events)
        self.assertIn('api_queue_ready', events)
        self.assertEqual(self.first.summary()['pending'], 0)

    def test_parallel_report_writers_publish_complete_json(self):
        path = self.root / 'report.json'
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda n: write_json(path, {'n': n, 'payload': 'x' * 10000}), range(40)))
        result = json.loads(path.read_text())
        self.assertIn(result['n'], range(40))
        self.assertEqual(result['payload'], 'x' * 10000)
        self.assertEqual(list(self.root.glob('report.json.*.tmp')), [])


class AdaptiveTokenTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.events = Events(self.root)
        self.cfg = Config(model='openai/gpt-6-luna', control_tokens=32768, work_tokens=32768,
                          adaptive_tokens=True, max_response_tokens=128000, structured_actions=True)
        self.agent = Agent('agent-00', 1)

    def client(self, content='', finish='length'):
        class Client:
            def __init__(self):
                self.content, self.finish = content, finish
                self.limits, self.prompts = [], []
                self.last_completion = {}

            def generate(self, messages, limit, phase, agent):
                self.limits.append(limit)
                self.prompts.append(messages)
                self.last_completion = {'finish_reason': self.finish, 'call_id': 'fixture'}
                return self.content
        return Client()

    def router(self, cap=1):
        ledger = Ledger(self.root / 'ledger.sqlite')
        ledger.register('run', cap)
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'fixture-key'}):
            client = OpenRouter(self.cfg.model, ledger, 'run', {'prompt': 1, 'completion': 1},
                                self.events.emit, self.root / 'STOP', structured_actions=True)
        return client, ledger

    def test_truncation_grows_only_that_agents_response_kind_without_extra_calls(self):
        client = self.client(); policy = Policy(client, self.cfg, self.events)
        for _ in range(4):
            self.assertEqual(policy.ask('work', self.agent, {}), {})
        self.assertEqual(client.limits, [32768, 65536, 128000, 128000])
        self.assertEqual(policy.calls, 4)
        self.assertEqual(policy.response_limit('discuss', self.agent.name), 32768)
        self.assertEqual(policy.response_limit('work', 'agent-01'), 32768)
        self.assertIn('Up to 65536 combined reasoning/output tokens', client.prompts[1][0]['content'])
        self.assertIn('Maximum team size = 10', client.prompts[1][0]['content'])
        extensions = [e for e in self.events.rows if e['event'] == 'response_tokens_extended']
        self.assertEqual([e['new_limit'] for e in extensions], [65536, 128000])
        policy.extend_for_no_answer(10, 'No accepted source')
        self.assertEqual(policy.response_limit('work', self.agent.name), 128000)
        self.assertEqual(policy.response_limit('work', 'agent-01'), 65536)

    def test_invalid_normal_reply_and_valid_length_reply_do_not_expand(self):
        client = self.client('not JSON', 'stop'); policy = Policy(client, self.cfg, self.events)
        self.assertEqual(policy.ask('work', self.agent, {}), {})
        client.content = json.dumps({'tool': 'inspect', 'command': 'pwd', 'message': '',
                                    'channel': 'team', 'recipients': [], 'args': [],
                                    'files': [], 'capture': [], 'stdin': ''})
        client.finish = 'length'
        self.assertEqual(policy.ask('work', self.agent, {})['command'], 'pwd')
        self.assertEqual(client.limits, [32768, 32768])
        self.assertFalse(any(e['event'] == 'response_tokens_extended' for e in self.events.rows))

    def test_empty_json_length_reply_expands_and_fixed_policy_does_not(self):
        client = self.client('{}'); policy = Policy(client, self.cfg, self.events)
        self.assertEqual(policy.ask('work', self.agent, {}), {})
        self.assertEqual(policy.response_limit('work', self.agent.name), 65536)
        self.cfg.adaptive_tokens = False
        fixed = Policy(client, self.cfg, self.events)
        fixed.ask('work', self.agent, {})
        fixed.extend_for_no_answer(10, 'No accepted source')
        self.assertEqual(fixed.response_limit('work', self.agent.name), 32768)

    def test_reasoning_only_length_response_stays_charged_without_replay(self):
        client, ledger = self.router(); policy = Policy(client, self.cfg, self.events)
        with patch('codetest.budget.requests.post') as post:
            post.return_value.status_code = 200
            post.return_value.json.return_value = {'usage': {'cost': .02, 'completion_tokens': 32768},
                'choices': [{'finish_reason': 'length', 'message': {'content': None}}]}
            self.assertEqual(policy.ask('work', self.agent, {}), {})
            self.assertEqual(post.call_count, 1)
        self.assertEqual(policy.response_limit('work', self.agent.name), 65536)
        self.assertEqual(ledger.summary()['reported_usd'], .02)
        self.assertEqual(ledger.summary()['unconfirmed_usd'], 0)
        self.assertFalse(ledger.summary()['blocked'])
        self.assertEqual(ledger.summary()['pending'], 0)
        charge = next(e for e in self.events.rows if e['event'] == 'api_charged')
        self.assertEqual(charge['max_tokens'], 32768)
        self.assertEqual(charge['finish_reason'], 'length')

    def test_enlarged_response_must_fit_budget_before_another_http_send(self):
        client, ledger = self.router(cap=.06); policy = Policy(client, self.cfg, self.events)
        with patch('codetest.budget.requests.post') as post:
            post.return_value.status_code = 200
            post.return_value.json.return_value = {'usage': {'cost': .001},
                'choices': [{'finish_reason': 'length', 'message': {'content': ''}}]}
            policy.ask('work', self.agent, {})
            with self.assertRaises(BudgetStop):
                policy.ask('work', self.agent, {})
            self.assertEqual(post.call_count, 1)
        self.assertEqual(ledger.summary()['calls'], 1)
        self.assertEqual(ledger.summary()['pending'], 0)

    def test_empty_non_length_reply_stops_without_adaptation_or_retry(self):
        client, ledger = self.router(); policy = Policy(client, self.cfg, self.events)
        with patch('codetest.budget.requests.post') as post:
            post.return_value.status_code = 200
            post.return_value.json.return_value = {'usage': {'cost': .001},
                'choices': [{'finish_reason': 'stop', 'message': {'content': None}}]}
            with self.assertRaises(RuntimeError):
                policy.ask('work', self.agent, {})
            self.assertEqual(post.call_count, 1)
        self.assertEqual(ledger.summary()['reported_usd'], .001)
        self.assertEqual(policy.response_limit('work', self.agent.name), 32768)

    def test_http_error_is_not_a_token_extension(self):
        client, ledger = self.router(); policy = Policy(client, self.cfg, self.events)
        with patch('codetest.budget.requests.post') as post:
            post.return_value.status_code = 503
            post.return_value.text, post.return_value.headers = 'unavailable', {}
            with self.assertRaises(RuntimeError):
                policy.ask('work', self.agent, {})
            self.assertEqual(post.call_count, 1)
        self.assertTrue(ledger.summary()['blocked'])
        self.assertEqual(policy.response_limit('work', self.agent.name), 32768)

    def test_config_and_estimate_cover_maximum_allocation(self):
        for value in (32767, 128001, True):
            with self.assertRaises(ValueError):
                Config(adaptive_tokens=True, work_tokens=32768, max_response_tokens=value)
        estimate = self.cfg.estimate()
        self.assertEqual(estimate['maximum_response_limits'], {'work': 128000, 'control': 128000})
        self.assertEqual(estimate['work_output_ceiling_per_max_team_round'], 10 * 8 * 128000)


class MechanismTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task = create_task(self.root / 'task')
        self.out = self.root / 'run'; self.out.mkdir()
        self.events = Events(self.out)
        self.cfg = Config(k=8, rounds=2, work_calls=3)
        self.policy = Policy(FixtureClient(), self.cfg, self.events)
        self.engine = Engine(self.cfg, self.task, self.policy, self.events, self.out)

    def test_equal_work_and_wealth_conservation(self):
        self.engine.run()
        self.engine.finalize()
        work = [e for e in self.events.rows if e['event'] == 'tool']
        self.assertEqual(len(work), 8 * 2 * 3)
        for agent in self.engine.agents:
            self.assertEqual(sum(e['agent'] == agent.name for e in work), 6)
        self.engine.assert_accounting()
        self.assertEqual(len(self.engine.history), 2)
        self.assertTrue(any(e['event'] == 'transfer' for e in self.events.rows))
        self.assertTrue((self.out / 'submission/scripted-demo/submission.tar.gz').is_file())
        self.assertFalse((self.out / 'submission/scripted-demo/source/.git').exists())

    def test_cross_team_messages_persist_and_direct_messages_stay_private(self):
        a, b, c = self.engine.agents[:3]
        self.engine.route_message(a, [a.name], {'message': 'Claiming parser implementation', 'channel': 'public'})
        self.engine.route_message(a, [a.name], {'message': 'Please test edge cases', 'channel': 'direct', 'recipients': [b.name]})
        self.assertEqual(self.engine.context(c)['public_status_board'][0]['sender'], a.name)
        self.assertEqual(self.engine.context(b)['direct_messages'][0]['text'], 'Please test edge cases')
        self.assertEqual(self.engine.context(c)['direct_messages'], [])
        self.engine.step = 3
        self.assertEqual(len(self.engine.context(b)['public_status_board']), 1)
        self.engine.ask('discuss', b)
        deliveries = [e for e in self.events.rows if e['event'] == 'message_delivered']
        self.assertEqual(len(deliveries[-1]['message_ids']), 2)
        self.engine.ask('discuss', b)
        self.assertEqual(len([e for e in self.events.rows if e['event'] == 'message_delivered']), 1)

    def test_missing_answer_extends_same_population_in_tens_and_stops_at_fifty(self):
        self.engine.cfg.rounds = self.engine.round_limit = 10
        self.engine.cfg.max_rounds = 50
        self.cfg.adaptive_tokens = True
        self.policy.token_floors = {'work': 32768, 'control': 32768}
        original_agents = list(self.engine.agents)
        with patch.object(self.engine, 'form'), patch.object(self.engine, 'candidate', return_value={'eligible': False}):
            self.engine.run()
        self.assertEqual([c['completed_rounds'] for c in self.engine.checkpoints], [10, 20, 30, 40, 50])
        self.assertEqual([c['next_limit'] for c in self.engine.checkpoints], [20, 30, 40, 50, 50])
        self.assertEqual(self.engine.state()['completed_rounds'], 50)
        self.assertTrue(all(before is after for before, after in zip(original_agents, self.engine.agents)))
        self.assertEqual(self.engine.step, 49)
        extensions = [e for e in self.events.rows if e['event'] == 'response_tokens_extended']
        self.assertEqual([e['completed_rounds'] for e in extensions], [10, 20])
        self.assertEqual([e['new_limits']['work'] for e in extensions], [65536, 128000])
        self.assertEqual(self.engine.state()['response_tokens']['population_floors']['control'], 128000)

    def test_ready_answer_stops_at_next_checkpoint_without_extra_rounds(self):
        self.engine.cfg.rounds = self.engine.round_limit = 10
        self.cfg.adaptive_tokens = True
        self.policy.token_floors = {'work': 32768, 'control': 32768}
        with patch.object(self.engine, 'form'), patch.object(self.engine, 'candidate', return_value={'eligible': False}), \
                patch.object(self.engine, 'answer_ready', side_effect=[(False, 'no answer'), (True, 'visible gate passed')]):
            self.engine.run()
        self.assertEqual(self.engine.state()['completed_rounds'], 20)
        self.assertEqual([c['completed_rounds'] for c in self.engine.checkpoints], [10, 20])
        self.assertEqual(self.policy.response_limit('work', 'agent-00'), 65536)
        self.assertEqual(len([e for e in self.events.rows if e['event'] == 'response_tokens_extended']), 1)

    def test_zero_score_is_still_an_answer_and_does_not_trigger_extension(self):
        self.task.data['benchmark'] = 'commit0'
        (self.engine.repo.path / 'answer.py').write_text('def answer(): return 0\n')
        self.engine.history = [{'round': 0}]
        self.engine.feedback = {'build': True, 'status': 'scored', 'score': 0, 'passed': 0, 'total': 10}
        self.assertTrue(self.engine.answer_ready()[0])
        self.engine.feedback.update(status='collection_error', score=None)
        self.assertFalse(self.engine.answer_ready()[0])
        self.task.data['benchmark'] = 'programbench'
        self.engine.feedback = {'build': True, 'status': 'build_only', 'score': None}
        self.assertTrue(self.engine.answer_ready()[0])

    def test_large_team_agent_receives_own_previous_tool_result(self):
        self.cfg.k = 20
        self.cfg.max_rounds = self.cfg.rounds
        self.engine.agents = [Agent(f'agent-{i:02}', 1) for i in range(20)]
        self.engine.teams = TeamManager(self.engine.agents, 10)
        self.engine.inboxes = {a.name: [] for a in self.engine.agents}
        self.engine.teams.create_team(self.engine.agents[:10], 0)
        self.engine.step = 0
        observations = []
        original = self.policy.ask
        def capture(phase, agent, observation):
            if phase == 'work':
                observations.append(observation)
            return original(phase, agent, observation)
        with patch.object(self.policy, 'ask', side_effect=capture):
            self.engine.candidate(self.engine.teams.active_teams(0)[0])
        second_slots = [o for o in observations if o['slot'] == 1]
        self.assertEqual(len(second_slots), 10)
        self.assertTrue(all(o['your_recent_work'][0]['author'] == o['agent'] for o in second_slots))

    def test_free_finalization_with_zero_accepted_work(self):
        before = [a.wealth for a in self.engine.agents]
        calls = self.policy.calls
        self.engine.finalize()
        self.assertEqual(before, [a.wealth for a in self.engine.agents])
        self.assertEqual(calls, self.policy.calls)
        self.assertEqual(self.engine.history, [])
        final = self.events.rows[-1]
        self.assertFalse(final['charged']); self.assertFalse(final['credit'])

    def test_terminal_pool_is_conserved_with_variable_team_sizes(self):
        state = {'config': {'terminal_reward': 10}, 'history': [
            {'participants': ['a']}, {'participants': ['a', 'b', 'c']}]}
        credits = terminal_credits(state, .6)
        self.assertAlmostEqual(sum(credits.values()), 6)
        self.assertAlmostEqual(credits['a'], 4)
        self.assertEqual(terminal_credits(state, None), {})

    def test_stale_candidate_has_no_payment(self):
        before = [a.wealth for a in self.engine.agents]
        self.engine.apply([{'eligible': True, 'bid': .1, 'base_commit': 'stale', 'team': 'solo:agent-00'}])
        self.assertEqual(before, [a.wealth for a in self.engine.agents])
        self.assertEqual(self.engine.history, [])
        self.assertEqual(self.events.rows[-1]['reason'], 'stale_base')

    def test_regress_then_restore_cannot_farm_dividends(self):
        from codetest.sandbox import copy_source
        self.task.data['benchmark'] = 'commit0'
        self.engine.high_water, self.engine.denominator = .5, 10
        self.engine.feedback = {'score': .5, 'total': 10}
        for i, score in enumerate([.7, .4, .7]):
            self.engine.step = i
            source = self.root / f'candidate-{i}'
            copy_source(self.engine.repo.path, source)
            (source / 'module.py').write_text(f'VALUE = {i}\n')
            proposal = {'eligible': True, 'bid': .1, 'team': 'solo:agent-00',
                        'base_commit': self.engine.repo.version, 'changed': True, 'source': str(source),
                        'new_probes': [], 'participants': ['agent-00'], 'pledges': {'agent-00': .1},
                        'check': {'build': True, 'score': score, 'total': 10}}
            self.engine.apply([proposal])
        self.assertAlmostEqual(self.engine.dividends, 2)
        self.engine.assert_accounting()

    def test_archive_contains_only_tracked_source(self):
        repo = self.engine.repo
        (repo.path / '.gitignore').write_text('executable\n')
        (repo.path / 'executable').write_bytes(b'generated binary')
        repo.git('add', '--all'); repo.git('commit', '-qm', 'Fixture ignore')
        archive = self.root / 'submission.tar.gz'
        repo.package(archive)
        with tarfile.open(archive) as t:
            self.assertNotIn('executable', t.getnames())
            self.assertFalse(any('.git/' in n for n in t.getnames()))

    def test_no_vote_receives_no_historical_credit(self):
        class Client(FixtureClient):
            def generate(self, messages, max_tokens, phase, agent):
                if phase == 'ballot':
                    return json.dumps({'yes': agent != 'agent-03', 'pledge': .1, 'message': 'fixture'})
                return super().generate(messages, max_tokens, phase, agent)
        self.policy.client = Client()
        self.engine.teams.create_team(self.engine.agents[:4], 0)
        self.engine.step = 0
        proposal = self.engine.candidate(self.engine.teams.active_teams(0)[0])
        self.assertTrue(proposal['eligible'])
        self.assertNotIn('agent-03', proposal['participants'])
        self.assertEqual(proposal['pledges']['agent-03'], 0)

    def test_exact_half_consent_is_ineligible(self):
        class Client(FixtureClient):
            def generate(self, messages, max_tokens, phase, agent):
                if phase == 'ballot':
                    return json.dumps({'yes': int(agent[-2:]) < 2, 'pledge': .1, 'message': 'fixture'})
                return super().generate(messages, max_tokens, phase, agent)
        self.policy.client = Client()
        self.engine.teams.create_team(self.engine.agents[:4], 0)
        self.engine.step = 0
        proposal = self.engine.candidate(self.engine.teams.active_teams(0)[0])
        self.assertFalse(proposal['eligible'])

    def test_interruption_still_exports_repository(self):
        with patch('codetest.fixture.FixtureClient.generate', side_effect=BudgetStop('test stop')):
            with self.assertRaises(BudgetStop):
                execute(self.cfg, None, self.root / 'interrupted', demo=True)
        out = self.root / 'interrupted'
        self.assertTrue((out / 'submission/scripted-demo/submission.tar.gz').exists())
        self.assertTrue((out / 'replay.html').exists())
        self.assertEqual(json.loads((out / 'manifest.json').read_text())['status'], 'budget_stopped')
        analysis = json.loads((out / 'analysis.json').read_text())
        self.assertEqual(analysis['status'], 'budget_stopped')
        self.assertIsNone(analysis['evaluation'])
        self.assertEqual(analysis['reported_usd'], 0)
        self.assertTrue(analysis['analysis']['failure_modes'])
        self.assertTrue((out / 'analysis.md').exists())
        self.assertTrue((out / 'analysis.html').exists())

    def test_evaluator_exception_is_saved_without_a_zero_score(self):
        out = self.root / 'evaluation-failure'
        cfg = Config(k=2, rounds=1, max_rounds=1, work_calls=1)
        execute(cfg, None, out, demo=True)
        manifest = json.loads((out / 'manifest.json').read_text())
        # Simulate only the completed-run gate; the evaluator is mocked and no
        # provider/network/container is contacted by this fixture.
        manifest['backend'] = 'openrouter'
        (out / 'manifest.json').write_text(json.dumps(manifest))
        with patch('codetest.cli.Task.final', side_effect=RuntimeError('fixture grader unavailable')):
            result = evaluate(out)
        self.assertEqual(result['status'], 'evaluator_exception')
        self.assertIsNone(result['score'])
        saved = json.loads((out / 'evaluation.json').read_text())
        self.assertEqual(saved['terminal_credits'], {})
        self.assertIsNone(json.loads((out / 'analysis.json').read_text())['evaluation']['score'])


class BoundaryTests(unittest.TestCase):
    def test_structured_probe_files_are_validated_without_silent_overwrite(self):
        from codetest.actions import normalize_action
        action = {'tool': 'probe', 'command': None, 'message': 'Inspect counts', 'channel': 'team',
                  'recipients': [], 'args': ['count'], 'stdin': '', 'capture': [],
                  'files': [{'path': 'a.csv', 'content': 'a\n1\n'}]}
        self.assertEqual(normalize_action(action, 'work')['files'], {'a.csv': 'a\n1\n'})
        action['files'].append({'path': 'a.csv', 'content': 'overwritten'})
        with self.assertRaises(ValueError):
            normalize_action(action, 'work')
        with self.assertRaises(ValueError):
            normalize_action({'yes': True, 'pledge': True, 'authorize_fixed': False, 'message': ''}, 'ballot')

    def test_strict_policy_returns_one_action_and_rejects_extra_actions(self):
        class Client:
            response = json.dumps({'tool': 'inspect', 'command': 'pwd', 'message': '', 'channel': 'team',
                                   'recipients': [], 'args': [], 'stdin': '', 'files': [], 'capture': []})
            def generate(self, *args):
                return self.response
        with tempfile.TemporaryDirectory() as directory:
            events = Events(directory); client = Client()
            policy = Policy(client, Config(structured_actions=True), events)
            self.assertEqual(policy.ask('work', Agent('agent-00', 1), {'slot': 0})['command'], 'pwd')
            client.response += '\n' + client.response
            self.assertEqual(policy.ask('work', Agent('agent-00', 1), {'slot': 1}), {})
            self.assertEqual(events.rows[-1]['event'], 'invalid_response')

    def test_probe_preview_is_readable_and_exact_bytes_are_preserved(self):
        import base64
        from codetest.probes import readable_observation
        value = {'stdout': base64.b64encode(b'column,count\na,2\n').decode(),
                 'stderr': '', 'exit_code': 0, 'files': {}}
        before = json.dumps(value)
        readable = readable_observation(value, limit=7)
        self.assertEqual(readable['stdout']['text'], 'column,')
        self.assertTrue(readable['stdout']['preview_truncated'])
        self.assertEqual(json.dumps(value), before)

    def test_fenced_response_is_logged_and_returned_by_policy(self):
        class FencedClient:
            calls = 0
            def generate(self, *args):
                self.calls += 1
                return '```json\n{"tool":"inspect","command":"pwd"}\n```'
        with tempfile.TemporaryDirectory() as directory:
            events = Events(directory)
            client = FencedClient()
            policy = Policy(client, Config(), events)
            action = policy.ask('work', Agent('agent-00', 1), {'slot': 0})
            self.assertEqual(action['command'], 'pwd')
            self.assertEqual(client.calls, 1)
            self.assertEqual([e['event'] for e in events.rows], ['response_normalized', 'decision'])
            self.assertEqual(events.rows[0]['normalization'], 'single_json_fence')

    def test_response_fence_normalization_is_unambiguous(self):
        from codetest.policy import parse_action
        action = {'tool': 'edit', 'command': 'true'}
        self.assertEqual(parse_action(json.dumps(action)), (action, False))
        self.assertEqual(parse_action('```json\n' + json.dumps(action) + '\n```'), (action, True))
        for bad in ['I propose ' + json.dumps(action), '```json\n{}\n```\n{}', '[]', '{"tool":']:
            with self.assertRaises((ValueError, TypeError)):
                parse_action(bad)

    def test_population_cap_is_derived_and_pending_joins_cannot_overfill(self):
        for k, cap in [(1, 1), (2, 1), (10, 5), (20, 10), (21, 10), (30, 15)]:
            self.assertEqual(Config(k=k).max_team_size, cap)
        agents = [Agent(f'agent-{i:02}', 1) for i in range(20)]
        manager = TeamManager(agents, Config().max_team_size)
        tag = manager.create_team(agents[:9], 0)
        first = manager.invite(agents[0], agents[9], 0)
        second = manager.invite(agents[1], agents[10], 0)
        manager.accept_invite(agents[9], first['id'], 0)
        with self.assertRaisesRegex(ValueError, 'cap reached'):
            manager.accept_invite(agents[10], second['id'], 0)
        self.assertEqual(sum(a.team_tag == tag for a in agents), 10)
        self.assertIsNone(agents[10].team_tag)

    def test_agents_receive_resolved_cap_in_every_phase(self):
        from codetest.policy import SCHEMAS
        from unittest.mock import Mock
        client = Mock(); client.generate.return_value = '{}'
        cfg = Config(k=21)
        policy = Policy(client, cfg, Mock())
        for phase in SCHEMAS:
            policy.ask(phase, Agent('agent-00', 1), {})
            system = client.generate.call_args.args[0][0]['content']
            self.assertIn('Population k = 21', system)
            self.assertIn('Maximum team size = 10 agents, including yourself', system)
            self.assertIn('ceiling, not a required size', system)

    def test_task_identity_cannot_escape_submission_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            task = create_task(Path(temp) / 'task')
            task.data['id'] = '..'
            (task.root / 'task.json').write_text(json.dumps(task.data))
            with self.assertRaises(ValueError):
                Task.load(task.root / 'task.json')

    def test_tar_rejects_escape_and_symlinks(self):
        for name, link in [('../outside', False), ('/absolute', False), ('link', True)]:
            with tempfile.TemporaryDirectory() as temp:
                buf = io.BytesIO()
                with tarfile.open(fileobj=buf, mode='w') as t:
                    info = tarfile.TarInfo(name)
                    if link:
                        info.type = tarfile.SYMTYPE; info.linkname = '/tmp'
                    t.addfile(info, io.BytesIO())
                with self.assertRaises(ValueError):
                    safe_extract(buf.getvalue(), temp)

    def test_probe_rejects_parent_paths(self):
        with self.assertRaises(ValueError):
            validate_probe({'files': {'../../secret': 'x'}})
        with self.assertRaises(ValueError):
            validate_probe({'capture': ['/etc/passwd']})

    def test_nonreproducible_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            task = create_task(Path(temp) / 'task')
            with patch('codetest.probes.observation', side_effect=[{'stdout': 'A'}, {'stdout': 'B'}]):
                with self.assertRaisesRegex(ValueError, 'Nonreproducible'):
                    discover(task, {'args': ['--help']})

    def test_junit_empty_duplicate_and_errors_do_not_score(self):
        for xml in ['<testsuite/>', '<testsuite><testcase name="a"/><testcase name="a"/></testsuite>', '<testsuite><testcase name="a"><error/></testcase></testsuite>']:
            self.assertIsNone(junit(xml)['score'])
        valid = junit('<testsuite><testcase name="a"/><testcase name="b"><failure/></testcase><testcase name="c"><skipped/></testcase></testsuite>')
        self.assertEqual(valid['score'], 1/3)

    def test_config_rejects_budget_and_nan(self):
        for fields in [{'run_cap_usd': 101}, {'initial_wealth': float('nan')}, {'k': True}, {'condition': 'manager'}]:
            with self.assertRaises(ValueError):
                Config(**fields)


@unittest.skipUnless(importlib.util.find_spec('programbench'), 'Install the optional benchmarks extra')
class NativeScorerTests(unittest.TestCase):
    def test_programbench_honors_ignored_branches_and_tests(self):
        from codetest.programbench_score import summarize
        fake = {'instance_id': 'fixture', 'branches': {
            'active': {'ignored_tests': [{'name': 'ignored-case'}]},
            'ignored-branch': {'ignored': True}}}
        result = {'test_branches': ['active', 'ignored-branch'], 'test_results': [
            {'name': 'pass', 'branch': 'active', 'status': 'passed'},
            {'name': 'fail', 'branch': 'active', 'status': 'failure'},
            {'name': 'ignored-case', 'branch': 'active', 'status': 'failure'},
            {'name': 'also-ignored', 'branch': 'ignored-branch', 'status': 'failure'}]}
        for row in result['test_results']:
            row['extra'] = {}
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / 'fixture'; folder.mkdir()
            (folder / 'fixture.eval.json').write_text(json.dumps(result))
            with patch('programbench.utils.load_data.load_all_instances', return_value=[fake]):
                scored = summarize(temp, 'fixture')
            self.assertEqual(scored['score'], .5)
            self.assertEqual(scored['total'], 2)


if __name__ == '__main__':
    unittest.main()
