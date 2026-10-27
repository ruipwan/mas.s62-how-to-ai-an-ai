import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests
from urllib3.exceptions import ReadTimeoutError

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor')]
from codetest.budget import BudgetStop, Ledger, OpenRouter

CAPS = {'prompt': .2, 'completion': 1}
PRICES = {'prompt': .1, 'cached': .01, 'completion': .5}
MESSAGES = [{'role': 'user', 'content': 'fixture'}]
TOKENS = 1000
RESERVATION = ((len(json.dumps(MESSAGES, ensure_ascii=False).encode()) + 4096) * CAPS['prompt']
               + TOKENS * CAPS['completion']) / 1e6


def response(*, status=200, content='{}', usage=True):
    result = Mock(status_code=status, headers={}, text='fixture error', content=b'fixture error')
    data = {'id': 'fixture-response', 'model': 'gpt-6-luna',
            'choices': [{'finish_reason': 'stop', 'message': {'content': content}}]}
    if usage:
        data['usage'] = {'prompt_tokens': 10, 'completion_tokens': 10}
    result.json.return_value = data
    return result


class TransportRetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ledger = Ledger(self.root / 'budget.sqlite')
        self.ledger.register('run', 5)
        self.events = []

    def client(self, *, ledger=None, **fields):
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'fixture-key'}):
            return OpenRouter('openai/gpt-6-luna', ledger or self.ledger, 'run', CAPS,
                              lambda kind, **event: self.events.append((kind, event)),
                              self.root / 'run-STOP', provider='openai', prices=PRICES, **fields)

    def enabled(self, **fields):
        return self.client(**{'read_timeout': 600, 'max_retries': 2, **fields})

    def generate(self, client, **fields):
        return client.generate(MESSAGES, TOKENS, 'work', 'agent-a', **fields)

    def rows(self, ledger=None):
        with (ledger or self.ledger).connect() as db:
            return [dict(row) for row in db.execute('SELECT * FROM calls ORDER BY created')]

    def test_default_transport_still_stops_after_one_timeout(self):
        client = self.client()
        with patch('codetest.budget.requests.post', side_effect=requests.ReadTimeout('fixture')) as post:
            with self.assertRaises(requests.ReadTimeout):
                self.generate(client)
        self.assertEqual(post.call_count, 1)
        self.assertEqual(post.call_args.kwargs['timeout'], (15, 180))
        self.assertTrue(self.ledger.summary()['blocked'])
        self.assertEqual(self.rows()[0]['status'], 'uncertain')

    def test_timeout_then_success_retains_full_reservation_and_uses_fresh_request_id(self):
        client = self.enabled()
        with patch('codetest.budget.requests.post', side_effect=[requests.ReadTimeout('fixture-key'), response()]) as post, \
                patch.object(client, '_retry_wait') as wait:
            self.assertEqual(self.generate(client), '{}')
        wait.assert_called_once_with(5)
        rows = self.rows()
        self.assertEqual([row['status'] for row in rows], ['reserved_error', 'charged'])
        self.assertIsNone(rows[0]['actual'])
        self.assertAlmostEqual(rows[0]['reserved'], RESERVATION)
        self.assertAlmostEqual(self.ledger.summary()['accounted_usd'], RESERVATION + .000006)
        ids = [call.kwargs['headers']['X-Client-Request-Id'] for call in post.call_args_list]
        self.assertEqual(len(set(ids)), 2)
        self.assertEqual(ids, [row['id'] for row in rows])
        self.assertEqual(post.call_args_list[0].kwargs['json'], post.call_args_list[1].kwargs['json'])
        self.assertEqual(post.call_args.kwargs['timeout'], (15, 600))
        self.assertFalse(self.ledger.summary()['blocked'])
        self.assertNotIn('fixture-key', json.dumps(self.events))
        self.assertNotIn('fixture-key', json.dumps(rows))

    def test_two_timeouts_then_success_waits_five_and_fifteen_seconds(self):
        client = self.enabled()
        with patch('codetest.budget.requests.post', side_effect=[requests.ConnectTimeout(), requests.ReadTimeout(), response()]) as post, \
                patch.object(client, '_retry_wait') as wait:
            self.generate(client)
        self.assertEqual(post.call_count, 3)
        self.assertEqual([call.args[0] for call in wait.call_args_list], [5, 15])
        self.assertEqual([self.ledger.transport_attempt(row['id'])['attempt'] for row in self.rows()], [1, 2, 3])

    def test_exhausted_attempts_are_persistent_and_cannot_restart_with_same_request_id(self):
        client = self.enabled()
        with patch('codetest.budget.requests.post', side_effect=requests.ReadTimeout()) as post, \
                patch.object(client, '_retry_wait'):
            with self.assertRaisesRegex(BudgetStop, 'retry limit'):
                self.generate(client, request_id='logical-1')
        self.assertEqual(post.call_count, 3)
        self.assertEqual([row['status'] for row in self.rows()], ['reserved_error'] * 3)
        self.assertAlmostEqual(self.ledger.summary()['unconfirmed_usd'], 3 * RESERVATION)
        self.assertFalse(self.ledger.summary()['blocked'])
        recreated = self.enabled(ledger=Ledger(self.ledger.path))
        with patch('codetest.budget.requests.post') as post:
            with self.assertRaisesRegex(BudgetStop, 'retry limit'):
                self.generate(recreated, request_id='logical-1')
        post.assert_not_called()

    def test_body_read_timeout_wrapper_is_retryable_but_generic_connection_error_is_not(self):
        client = self.enabled()
        wrapped = requests.ConnectionError(ReadTimeoutError(None, '/fixture', 'fixture'))
        with patch('codetest.budget.requests.post', side_effect=[wrapped, response()]) as post, \
                patch.object(client, '_retry_wait'):
            self.generate(client)
        self.assertEqual(post.call_count, 2)
        with patch('codetest.budget.requests.post', side_effect=requests.ConnectionError('read timed out')) as post:
            with self.assertRaises(requests.ConnectionError):
                self.generate(client)
        self.assertEqual(post.call_count, 1)
        self.assertEqual(self.rows()[-1]['status'], 'uncertain')

    def test_http_missing_usage_empty_and_invalid_json_are_never_retried(self):
        invalid_json = response()
        invalid_json.json.side_effect = ValueError('invalid JSON')
        for number, reply in enumerate((response(status=400), response(status=401), response(usage=False),
                                        response(content=''), invalid_json)):
            with self.subTest(number=number):
                ledger = Ledger(self.root / f'failure-{number}.sqlite')
                ledger.register('run', 5)
                client = self.enabled(ledger=ledger)
                with patch('codetest.budget.requests.post', return_value=reply) as post:
                    with self.assertRaises((RuntimeError, ValueError)):
                        self.generate(client)
                self.assertEqual(post.call_count, 1)
                self.assertEqual(len(self.rows(ledger)), 1)

    def test_cumulative_retry_reservation_cap_survives_client_and_ledger_recreation(self):
        client = self.enabled(retry_reserve_cap_usd=RESERVATION * 1.5)
        with patch('codetest.budget.requests.post', side_effect=[requests.ReadTimeout(), response()]), \
                patch.object(client, '_retry_wait'):
            self.generate(client)
        client = self.enabled(ledger=Ledger(self.ledger.path), retry_reserve_cap_usd=RESERVATION * 1.5)
        with patch('codetest.budget.requests.post', side_effect=requests.ReadTimeout()) as post, \
                patch.object(client, '_retry_wait'):
            with self.assertRaisesRegex(BudgetStop, 'retry reservation limit'):
                self.generate(client)
        self.assertEqual(post.call_count, 1)
        self.assertEqual(len(self.rows()), 3)

    def test_episode_budget_is_rechecked_before_a_retry_send(self):
        ledger = Ledger(self.root / 'small.sqlite')
        ledger.register('run', RESERVATION * 1.5)
        client = self.enabled(ledger=ledger)
        with patch('codetest.budget.requests.post', side_effect=requests.ReadTimeout()) as post, \
                patch.object(client, '_retry_wait'):
            with self.assertRaisesRegex(BudgetStop, 'episode budget'):
                self.generate(client)
        self.assertEqual(post.call_count, 1)
        self.assertEqual(len(self.rows(ledger)), 1)

    def test_run_and_campaign_stop_interrupt_the_backoff_without_another_send(self):
        for kind in ('run', 'campaign'):
            with self.subTest(kind=kind):
                client = self.enabled()
                stop = client.stop if kind == 'run' else self.root / 'STOP'
                with patch('codetest.budget.requests.post', side_effect=requests.ReadTimeout()) as post, \
                        patch('codetest.budget.time.sleep', side_effect=lambda _: stop.touch()):
                    with self.assertRaisesRegex(BudgetStop, 'STOP'):
                        self.generate(client)
                self.assertEqual(post.call_count, 1)
                stop.unlink()
        self.assertFalse(self.ledger.summary()['blocked'])

    def test_retry_rechecks_live_lease(self):
        ledger = Ledger(self.root / 'parallel.sqlite', parallel=True)
        self.addCleanup(ledger.close)
        ledger.register('run', 5)
        client = self.enabled(ledger=ledger)
        with patch('codetest.budget.requests.post', side_effect=requests.ReadTimeout()) as post, \
                patch.object(client, '_retry_wait', side_effect=lambda _: ledger.close()):
            with self.assertRaisesRegex(BudgetStop, 'live request lease'):
                self.generate(client)
        self.assertEqual(post.call_count, 1)

    def test_retained_timeout_never_exposes_uncertain_to_a_parallel_peer(self):
        ledger = Ledger(self.root / 'shared.sqlite', parallel=True)
        self.addCleanup(ledger.close)
        ledger.register('run', 5)
        peer = Ledger(ledger.path, parallel=True)
        self.addCleanup(peer.close)
        peer.register('peer', 5)
        client = self.enabled(ledger=ledger)
        def peer_call(_):
            self.assertFalse(peer.summary()['blocked'])
            ident = peer.reserve('peer', .01, 'work', 'peer-agent', 'fixture')
            peer.settle(ident, .001, {'fixture': True})
        with patch('codetest.budget.requests.post', side_effect=[requests.ReadTimeout(), response()]), \
                patch.object(client, '_retry_wait', side_effect=peer_call), \
                patch.object(ledger, 'uncertain', side_effect=AssertionError('Transient global lock')):
            self.generate(client)

    def test_legacy_resume_counts_original_attempt_and_applies_seed_only_once(self):
        client = self.enabled(initial_attempts_used=1)
        with patch('codetest.budget.requests.post', side_effect=[requests.ReadTimeout(), response(), response()]) as post, \
                patch.object(client, '_retry_wait') as wait:
            self.generate(client)
            self.generate(client)
        self.assertEqual(post.call_count, 3)
        wait.assert_called_once_with(15)
        self.assertEqual([self.ledger.transport_attempt(row['id'])['attempt'] for row in self.rows()], [2, 3, 1])

    def test_stable_request_id_resumes_after_interrupted_backoff_without_resetting_count(self):
        client = self.enabled()
        with patch('codetest.budget.requests.post', side_effect=requests.ReadTimeout()) as post, \
                patch.object(client, '_retry_wait', side_effect=BudgetStop('operator stop')):
            with self.assertRaisesRegex(BudgetStop, 'operator stop'):
                self.generate(client, request_id='resume-step')
        self.assertEqual(post.call_count, 1)
        recreated = self.enabled(ledger=Ledger(self.ledger.path))
        with patch('codetest.budget.requests.post', side_effect=[requests.ReadTimeout(), response()]) as post, \
                patch.object(recreated, '_retry_wait') as wait:
            self.generate(recreated, request_id='resume-step')
        self.assertEqual(post.call_count, 2)
        wait.assert_called_once_with(15)
        self.assertEqual([self.ledger.transport_attempt(row['id'])['attempt'] for row in self.rows()], [1, 2, 3])

    def test_stable_request_id_rejects_payload_change_and_replay_of_success(self):
        client = self.enabled()
        with patch('codetest.budget.requests.post', return_value=response()):
            self.generate(client, request_id='logical-1')
        with patch('codetest.budget.requests.post') as post:
            with self.assertRaisesRegex(BudgetStop, 'no replay'):
                self.generate(client, request_id='logical-1')
            with self.assertRaisesRegex(ValueError, 'different request contents'):
                client.generate([{'role': 'user', 'content': 'different'}], TOKENS, 'work', 'agent-a', request_id='logical-1')
        post.assert_not_called()

    def test_invalid_transport_settings_fail_before_reservation(self):
        for fields in ({'max_retries': 3}, {'max_retries': True}, {'read_timeout': float('nan')},
                       {'connect_timeout': 0}, {'initial_attempts_used': 3}, {'retry_backoff': [1]},
                       {'retry_backoff': [1, -1]}, {'retry_reserve_cap_usd': -1}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.enabled(**fields)
        self.assertEqual(self.rows(), [])


if __name__ == '__main__':
    unittest.main()
