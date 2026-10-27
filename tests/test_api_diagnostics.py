import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor')]
from codetest.api_diagnostics import diagnose_openai, response_diagnostics
from codetest.cli import main

KEY = 'fixture-openai-key'
MODEL = 'gpt-6-luna'
CONFIG = ROOT / 'configs/openai/k10-gpt6-luna-scratchpad.json'


def response(status, body='', headers=None):
    result = requests.Response()
    result.status_code = status
    result._content = body.encode('utf-8')
    result.encoding = 'utf-8'
    result.headers.update(headers or {})
    return result


class APIDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict('os.environ', {'OPENAI_API_KEY': KEY}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_missing_and_malformed_keys_make_no_http_request(self):
        for key in ('', ' bad-key', 'bad key', 'bad-key\n', 'bad-key\x00', 'bad-key\u200b',
                    'fixture-\u2603', 'fixture-\u4e2d', 'fixture-\u00e9'):
            with self.subTest(key=repr(key)), patch('codetest.api_diagnostics.os.environ.get', return_value=key):
                with patch('codetest.api_diagnostics.requests.get') as get:
                    result = diagnose_openai(MODEL)
                get.assert_not_called()
                self.assertFalse(result['ok'])
                self.assertFalse(result['credential_format_valid'])
                self.assertEqual(result['status'], 'credential_malformed' if key else 'credential_missing')
                if key:
                    self.assertNotIn(key, json.dumps(result, ensure_ascii=False))

    def test_cli_success_is_one_free_get_and_never_opens_the_ledger(self):
        body = json.dumps({'id': MODEL, 'object': 'model'})
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / 'ledger.sqlite'
            ledger.write_bytes(b'existing ledger, must remain unchanged')
            before = ledger.read_bytes()
            output = io.StringIO()
            with patch('codetest.api_diagnostics.requests.get', return_value=response(200, body)) as get, \
                 patch('codetest.api_diagnostics.requests.post') as post, \
                 patch('codetest.cli.Ledger', side_effect=AssertionError('Diagnostic opened ledger')), \
                 patch('codetest.cli.LEDGER', ledger), \
                 patch('sys.argv', ['run.py', 'diagnose-api', '--config', str(CONFIG)]), \
                 contextlib.redirect_stdout(output):
                main()
            self.assertEqual(ledger.read_bytes(), before)
            self.assertEqual(list(Path(directory).iterdir()), [ledger])
        get.assert_called_once_with('https://api.openai.com/v1/models/gpt-6-luna',
                                    headers={'Authorization': 'Bearer ' + KEY},
                                    timeout=(10, 30), allow_redirects=False)
        post.assert_not_called()
        result = json.loads(output.getvalue())
        self.assertTrue(result['ok'])
        self.assertEqual(result['status'], 'model_accessible')
        self.assertEqual(result['paid_requests'], 0)
        self.assertFalse(result['ledger_touched'])
        self.assertNotIn(KEY, output.getvalue())

    def test_http_results_classify_auth_access_and_model_failures_without_retry(self):
        cases = {400: 'request_rejected', 401: 'authentication_failed', 403: 'access_denied',
                 404: 'model_unavailable', 429: 'rate_limited', 503: 'server_error', 302: 'redirect_rejected'}
        for status, expected in cases.items():
            with self.subTest(status=status), patch('codetest.api_diagnostics.requests.get',
                    return_value=response(status, '', {'x-request-id': 'req-123'})) as get:
                result = diagnose_openai('openai/' + MODEL)
            get.assert_called_once()
            self.assertFalse(result['ok'])
            self.assertEqual(result['status'], expected)
            self.assertEqual(result['http']['body_bytes'], 0)
            self.assertEqual(result['http']['request_id'], 'req-123')

    def test_cli_failure_prints_json_and_exits_two(self):
        output = io.StringIO()
        with patch('codetest.api_diagnostics.requests.get', return_value=response(401, 'denied')), \
             patch('sys.argv', ['run.py', 'diagnose-api', '--config', str(CONFIG)]), \
             patch('codetest.cli.Ledger', side_effect=AssertionError('Diagnostic opened ledger')), \
             contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as error:
            main()
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(json.loads(output.getvalue())['status'], 'authentication_failed')

    def test_unexpected_success_body_does_not_claim_model_access(self):
        for body in ('<html>proxy</html>', '{}', json.dumps({'id': 'different', 'object': 'model'})):
            with self.subTest(body=body), patch('codetest.api_diagnostics.requests.get',
                    return_value=response(200, body)):
                result = diagnose_openai(MODEL)
            self.assertFalse(result['ok'])
            self.assertEqual(result['status'], 'unexpected_response')
            self.assertNotIn('detail', result['http'])

    def test_network_errors_do_not_echo_secret_exception_messages(self):
        for error in (requests.Timeout(KEY), requests.ConnectionError(KEY)):
            with self.subTest(error=type(error).__name__), patch('codetest.api_diagnostics.requests.get',
                    side_effect=error) as get:
                result = diagnose_openai(MODEL)
            get.assert_called_once()
            self.assertEqual(result['status'], 'timeout' if isinstance(error, requests.Timeout) else 'network_error')
            self.assertNotIn(KEY, json.dumps(result))

    def test_model_path_is_quoted_and_redirects_disabled(self):
        model = 'fixture/model?query#part'
        with patch('codetest.api_diagnostics.requests.get', return_value=response(404)) as get:
            diagnose_openai('openai/' + model)
        self.assertEqual(get.call_args.args[0],
                         'https://api.openai.com/v1/models/fixture%2Fmodel%3Fquery%23part')
        self.assertFalse(get.call_args.kwargs['allow_redirects'])

    def test_redaction_precedes_limits_for_every_logged_response_field(self):
        body = 'x' * 1990 + KEY + ' tail'
        headers = {name: 'prefix ' + KEY + '\n' + 'x' * 300 for name in
                   ('X-Request-Id', 'Content-Type', 'Server', 'Retry-After')}
        headers['Authorization'] = 'Bearer should-never-be-logged'
        detail = response_diagnostics(response(400, body, headers), KEY)
        encoded = json.dumps(detail)
        self.assertNotIn(KEY, encoded)
        self.assertNotIn(KEY[:8], encoded)
        self.assertNotIn('should-never-be-logged', encoded)
        self.assertEqual(detail['body_bytes'], len(body.encode('utf-8')))
        self.assertEqual(len(detail['detail']), 2000)
        for name in ('request_id', 'content_type', 'server', 'retry_after'):
            self.assertEqual(len(detail[name]), 256)
            self.assertIn('[REDACTED]', detail[name])
            self.assertNotIn('\n', detail[name])


if __name__ == '__main__':
    unittest.main()
