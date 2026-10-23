"""Opt-in Docker regression for repeated immutable candidate fetches."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codetest.sandbox import DockerSandbox


@unittest.skipUnless(os.environ.get('CODETEST_DOCKER_TEST_IMAGE'), 'Requires a pinned local test image')
class SandboxInputTests(unittest.TestCase):
    def test_readonly_tree_can_be_replaced_without_agent_write_access(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'candidate'
            (source / 'nested').mkdir(parents=True)
            content = source / 'nested' / 'code.txt'
            content.write_text('first\n')
            with DockerSandbox(os.environ['CODETEST_DOCKER_TEST_IMAGE'], memory='256m', cpus='1') as box:
                box.put(source, '/public/candidates/repeated', readonly=True)
                content.write_text('replacement\n')
                box.put(source, '/public/candidates/repeated', readonly=True)
                result = box.execute(
                    'test "$(cat /public/candidates/repeated/nested/code.txt)" = replacement && '
                    'test ! -w /public/candidates/repeated/nested/code.txt && '
                    'test ! -w /public/candidates/repeated/nested && '
                    '! touch /public/candidates/repeated/nested/agent-write')
                self.assertEqual(result['exit_code'], 0, result['output'])


if __name__ == '__main__':
    unittest.main()
