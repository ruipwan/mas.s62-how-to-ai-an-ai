import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor')]
from codetest.budget import Ledger, OpenRouter
from codetest.cli import config
from codetest.config import Config

PRICES = {'prompt': .1, 'cached': .01, 'completion': .5}
CAPS = {'prompt': .2, 'completion': 1}


def openai_config(**fields):
    return Config(**{'model': 'openai/gpt-6-luna', 'provider': 'openai', 'input_price': .1,
                     'output_price': .5, 'input_price_cap': .2, 'output_price_cap': 1, **fields})


class OpenAIConfigTests(unittest.TestCase):
    def test_provider_is_opt_in_and_validated(self):
        legacy = Config().as_dict()
        for name in ('provider', 'input_price', 'cached_input_price', 'output_price'):
            self.assertNotIn(name, legacy)
        self.assertIsNone(Config().price_table())
        cfg = openai_config()
        self.assertEqual(cfg.price_table(), {'prompt': .1, 'cached': .1, 'completion': .5})  # Upper bound.
        self.assertEqual(openai_config(cached_input_price=.01).price_table()['cached'], .01)
        self.assertEqual(cfg.as_dict()['provider'], 'openai')
        bad = [{'provider': 'azure'}, {'input_price': None}, {'output_price': None},
               {'input_price': .3}, {'output_price': 2}, {'cached_input_price': .2}, {'input_price': -1},
               {'model': 'qwen/qwen3-coder-next'}, {'reasoning_enabled': True},
               {'provider': 'openrouter'}]  # Prices without provider openai.
        for fields in bad:
            with self.assertRaises(ValueError, msg=fields):
                openai_config(**fields)

    def test_openai_experiment_configs_load(self):
        names = sorted(p.name for p in (ROOT / 'configs/openai').glob('*.json'))
        self.assertEqual(len(names), 6)
        for name in names:
            cfg = config(ROOT / 'configs/openai' / name)
            twin = config(ROOT / 'configs' / name)
            self.assertEqual(cfg.provider, 'openai')
            # Only the route differs from the OpenRouter config of the same name.
            differs = {k for k in twin.as_dict().keys() | cfg.as_dict().keys()
                       if twin.as_dict().get(k) != cfg.as_dict().get(k)}
            self.assertLessEqual(differs, {'provider', 'input_price', 'cached_input_price', 'output_price',
                                           'provider_ignore'}, name)


class OpenAIClientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ledger = Ledger(Path(self.temp.name) / 'budget.sqlite')
        self.ledger.register('run', 5)
        self.events = []

    def client(self, **fields):
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'fixture-openai-key'}, clear=False):
            return OpenRouter('openai/gpt-6-luna', self.ledger, 'run', CAPS,
                              lambda kind, **v: self.events.append((kind, v)), Path(self.temp.name) / 'STOP',
                              reasoning_effort='low', provider_ignore=['openai/flex'], structured_actions=True,
                              provider='openai', prices=PRICES, **fields)

    def test_request_goes_to_openai_and_cost_comes_from_usage(self):
        client = self.client()
        usage = {'prompt_tokens': 10000, 'completion_tokens': 2000,
                 'prompt_tokens_details': {'cached_tokens': 4000},
                 'completion_tokens_details': {'reasoning_tokens': 1500}}
        with patch('codetest.budget.requests.post') as post:
            post.return_value.status_code = 200
            post.return_value.json.return_value = {'id': 'chatcmpl-1', 'model': 'gpt-6-luna', 'usage': usage,
                'choices': [{'finish_reason': 'stop', 'message': {'content': '{"tool": "test"}'}}]}
            self.assertEqual(client.generate([{'role': 'user', 'content': 'fixture'}], 32768, 'work', 'a'),
                             '{"tool": "test"}')
        self.assertEqual(post.call_args.args[0], 'https://api.openai.com/v1/chat/completions')
        self.assertEqual(post.call_args.kwargs['headers'], {'Authorization': 'Bearer fixture-openai-key'})
        payload = post.call_args.kwargs['json']
        self.assertEqual(payload['model'], 'gpt-6-luna')
        self.assertEqual(payload['max_completion_tokens'], 32768)
        self.assertEqual(payload['reasoning_effort'], 'low')
        self.assertFalse(payload['store'])
        self.assertNotIn('max_tokens', payload)
        self.assertNotIn('provider', payload)
        self.assertNotIn('reasoning', payload)
        self.assertTrue(payload['response_format']['json_schema']['strict'])
        expected = (6000 * .1 + 4000 * .01 + 2000 * .5) / 1e6
        charged = [v for kind, v in self.events if kind == 'api_charged'][0]
        self.assertAlmostEqual(charged['cost_usd'], expected)
        self.assertEqual(charged['provider'], 'openai')
        summary = self.ledger.summary()
        self.assertAlmostEqual(summary['reported_usd'], expected)
        self.assertFalse(summary['blocked'])

    def test_missing_usage_blocks_the_campaign_without_retry(self):
        client = self.client()
        with patch('codetest.budget.requests.post') as post:
            post.return_value.status_code = 200
            post.return_value.json.return_value = {'id': 'x', 'choices': [
                {'finish_reason': 'stop', 'message': {'content': '{}'}}]}
            with self.assertRaises(RuntimeError):
                client.generate([{'role': 'user', 'content': 'fixture'}], 1000, 'work', 'a')
            self.assertEqual(post.call_count, 1)
        self.assertTrue(self.ledger.summary()['blocked'])

    def test_http_error_names_openai_and_redacts_the_key(self):
        client = self.client()
        with patch('codetest.budget.requests.post') as post:
            post.return_value.status_code = 401
            post.return_value.text = 'bad key fixture-openai-key'
            post.return_value.headers = {}
            with self.assertRaisesRegex(RuntimeError, 'openai HTTP 401'):
                client.generate([{'role': 'user', 'content': 'fixture'}], 1000, 'work', 'a')
        error = [v for kind, v in self.events if kind == 'api_http_error'][0]
        self.assertNotIn('fixture-openai-key', error['detail'])

    def test_key_and_prices_are_required(self):
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'OPENAI_API_KEY'):
                OpenRouter('openai/gpt-6-luna', self.ledger, 'run', CAPS, lambda *a, **k: None,
                           Path(self.temp.name) / 'STOP', provider='openai', prices=PRICES)
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'k'}):
            with self.assertRaises(ValueError):
                OpenRouter('openai/gpt-6-luna', self.ledger, 'run', CAPS, lambda *a, **k: None,
                           Path(self.temp.name) / 'STOP', provider='openai', prices={**PRICES, 'completion': 2})


if __name__ == '__main__':
    unittest.main()
