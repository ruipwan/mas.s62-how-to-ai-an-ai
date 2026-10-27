"""Explicit transport settings must preserve the selected campaign accounting."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from codetest.cli import execute
from codetest.config import Config


class ExplicitTransportTests(unittest.TestCase):
    def test_selected_ledger_and_transport_reach_real_client_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_root = root / 'task'
            task_root.mkdir()
            (task_root / 'task.json').write_text('{}')
            task = Mock(root=task_root, data={'benchmark': 'fixture'})
            out = root / 'run'
            selected = root / 'shared' / 'campaign.sqlite'
            ledger = Mock()
            ledger.summary.return_value = {'calls': 0, 'accounted_usd': 0}
            transport = {'read_timeout': 600, 'max_retries': 2, 'retry_reserve_cap_usd': 1.0}
            with patch('codetest.cli.validate_api_key'), patch('codetest.cli.preflight', return_value={'ready': True}), \
                    patch('codetest.cli.Ledger', return_value=ledger) as ledger_type, \
                    patch('codetest.cli.OpenRouter') as client_type, patch('codetest.cli.Engine'), \
                    patch('codetest.cli.analyze'):
                result = execute(Config(parallel_experiments=True), task, out,
                                 ledger_path=selected, transport=transport)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(ledger_type.call_args.args[0], selected)
            ledger.register.assert_called_once()
            client_type.assert_called_once()
            self.assertIs(client_type.call_args.args[1], ledger)
            self.assertEqual(client_type.call_args.kwargs['max_retries'], 2)
            self.assertEqual(client_type.call_args.kwargs['read_timeout'], 600)
            manifest = json.loads((out / 'manifest.json').read_text())
            self.assertEqual(manifest['campaign_ledger'], str(selected))
            self.assertEqual(manifest['transport'], transport)
            self.assertEqual(manifest['config']['rounds'], 10)

    def test_unknown_transport_rejected_before_output_or_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'run'
            with patch('codetest.cli.validate_api_key'), patch('codetest.cli.Ledger') as ledger_type:
                with self.assertRaises(ValueError):
                    execute(Config(), Mock(), out, transport={'provider': 'changed'})
            self.assertFalse(out.exists())
            ledger_type.assert_not_called()
