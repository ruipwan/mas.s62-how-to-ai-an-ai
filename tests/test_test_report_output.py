"""Display-only pytest output regression, with an optional offline Docker proof.

Set CODETEST_OUTPUT_TEST_IMAGE to an already installed image with pytest to run
that proof. It never pulls an image or uses a model API.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor')]
from codetest.benchmarks import Task
from codetest.sandbox import DockerSandbox


class SandboxOutputGuardTests(unittest.TestCase):
    def box(self):
        box = DockerSandbox.__new__(DockerSandbox)
        box.alive = True
        box.name, box.user, box.timeout = 'fixture-container', '65534:65534', 60
        return box

    def test_two_mebibyte_output_guard_still_terminates_the_container(self):
        box = self.box()
        process = Mock(returncode=-9)
        process.poll.return_value = None
        def start(argv, **kwargs):
            kwargs['stdout'].write(b'x' * (2 * 1024 * 1024 + 1))
            kwargs['stdout'].flush()
            return process
        with patch('codetest.sandbox.subprocess.Popen', side_effect=start), \
                patch('codetest.sandbox.time.monotonic', side_effect=[0, 1, 2]), \
                patch.object(box, 'close') as close:
            result = box.execute('fixture command', timeout=300)
        close.assert_called_once()
        process.kill.assert_called_once()
        self.assertEqual(result['status'], 'timeout_or_output_limit')
        self.assertTrue(result['output_truncated'])

    def test_three_hundred_second_ceiling_is_not_extended_by_requested_timeout(self):
        box = self.box()
        process = Mock(returncode=-9)
        process.poll.return_value = None
        with patch('codetest.sandbox.subprocess.Popen', return_value=process) as start, \
                patch('codetest.sandbox.time.monotonic', side_effect=[0, 311, 312]), \
                patch.object(box, 'close') as close:
            result = box.execute('fixture command', timeout=900)
        argv = start.call_args.args[0]
        index = argv.index('timeout')
        self.assertEqual(argv[index:index + 4], ['timeout', '-k', '2', '300'])
        close.assert_called_once()
        process.kill.assert_called_once()
        self.assertEqual(result['status'], 'timeout_or_output_limit')

    def test_visible_check_adds_only_display_flags_and_retains_three_hundred_seconds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = Task(root, {'visible_test_command': 'pytest tests -m selected', 'build_command': 'true'})
            box = Mock(name='sandbox')
            box.name = 'fixture-container'
            box.__enter__ = Mock(return_value=box)
            box.__exit__ = Mock(return_value=False)
            box.execute.side_effect = [
                {'exit_code': 0, 'status': 'ok', 'output': ''},
                {'exit_code': 1, 'status': 'ok', 'output': '1 failed, 1 passed'},
            ]
            def copy_report(argv, **kwargs):
                Path(argv[-1]).write_text('<testsuite><testcase classname="t" name="pass"/>'
                                        '<testcase classname="t" name="fail"><failure>assertion</failure></testcase></testsuite>')
                return Mock(returncode=0)
            with patch.object(task, 'sandbox', return_value=box), patch('codetest.benchmarks.run', side_effect=copy_report):
                result = task.check(root / 'source', root / 'check')
            call = box.execute.call_args_list[1]
            self.assertEqual(call.args[0], 'pytest tests -m selected --tb=no --show-capture=no --junitxml=/tmp/codetest-results.xml')
            self.assertEqual(call.kwargs, {'timeout': 300})
            self.assertEqual((result['passed'], result['total'], result['score']), (1, 2, .5))


@unittest.skipUnless(os.environ.get('CODETEST_OUTPUT_TEST_IMAGE'), 'Offline pytest Docker image not selected')
class PytestOutputIntegrationTests(unittest.TestCase):
    def test_large_failed_capture_no_longer_destroys_the_junit_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'seed'
            tests = source / 'tests'
            tests.mkdir(parents=True)
            (root / 'SPEC.md').write_text('Synthetic verifier output test, no benchmark claim.')
            (tests / 'test_large_output.py').write_text(
                "import pytest\n"
                "@pytest.mark.parametrize('case', range(8))\n"
                "def test_failure(case):\n"
                "    print('x' * 400000)\n"
                "    assert False\n"
                "def test_success():\n"
                "    assert True\n")
            # Keep pytest alive while the sandbox observes the oversized console
            # stream. This hook does not select, skip, or modify any test outcome.
            (tests / 'conftest.py').write_text(
                "import time\n"
                "def pytest_terminal_summary(terminalreporter):\n"
                "    time.sleep(.5)\n")
            task = Task(root, {'benchmark': 'fixture', 'image': os.environ['CODETEST_OUTPUT_TEST_IMAGE'],
                               'visible_test_command': 'pytest tests', 'build_command': 'true'},
                        sandbox_memory='512m', sandbox_cpus='1')
            with task.sandbox(source, reference=False) as original:
                result = original.execute('pytest tests --junitxml=/tmp/codetest-results.xml', timeout=300)
                self.assertEqual(result['status'], 'timeout_or_output_limit')
                self.assertFalse(original.alive)
            fixed = task.check(source, root / 'fixed-check')
            self.assertEqual(fixed['status'], 'scored')
            self.assertEqual((fixed['passed'], fixed['total']), (1, 9))
            self.assertAlmostEqual(fixed['score'], 1 / 9)
            self.assertEqual(len(fixed['failing']), 8)
            self.assertTrue((root / 'fixed-check/junit.xml').is_file())
            command = json.loads((root / 'fixed-check/test-command.json').read_text())
            self.assertEqual(command['status'], 'ok')
            self.assertEqual(command['exit_code'], 1)
            self.assertFalse(command['output_truncated'])


if __name__ == '__main__':
    unittest.main()
