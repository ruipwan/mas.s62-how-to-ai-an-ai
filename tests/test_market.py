import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor')]
from codetest.config import Config
from codetest.engine import Engine
from codetest.fixture import create_task, FixtureClient
from codetest.io import Events
from codetest.policy import Policy


class RecordingClient(FixtureClient):
    """Scripted fixture replies; keeps every observation an agent received."""
    def __init__(self):
        self.observations = []

    def generate(self, messages, max_tokens, phase, agent):
        self.observations.append((phase, agent, json.loads(messages[-1]['content']), messages[0]['content']))
        return super().generate(messages, max_tokens, phase, agent)


def market_config(**fields):
    return Config(**{'k': 6, 'rounds': 2, 'max_rounds': 2, 'work_calls': 4, 'shared_scratchpad': True, **fields})


class MarketTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task = create_task(self.root / 'task')

    def engine(self, cfg, name='run'):
        out = self.root / name
        out.mkdir()
        self.events = Events(out)
        self.client = RecordingClient()
        return Engine(cfg, self.task, Policy(self.client, cfg, self.events), self.events, out)

    def rows(self, kind):
        return [e for e in self.events.rows if e['event'] == kind]


class ClassicViewTests(MarketTestCase):
    def test_classic_agents_never_see_per_test_ids(self):
        engine = self.engine(market_config())
        engine.run()
        self.assertIsNotNone(engine.feedback.get('failing'))  # Kept internally for the swarm and merges.
        for phase, agent, observation, _ in self.client.observations:
            self.assertNotIn('failing', observation['public_feedback'], (phase, agent))
            self.assertNotIn('failing', observation.get('candidate_check', {}), (phase, agent))
            self.assertNotIn('task_board', observation)
        for row in engine.history:
            self.assertNotIn('failing', row['after'])
        self.assertFalse(any('failing' in e['feedback'] for e in self.rows('snapshot')))


class RepositoryTests(unittest.TestCase):
    def test_replace_sees_a_same_size_rewrite_with_a_preserved_timestamp(self):
        import os
        from codetest.sandbox import Repository
        with tempfile.TemporaryDirectory() as temp:
            root, old = Path(temp), 1_000_000_000
            for name, text in (('seed', 'X = 1\n'), ('candidate', 'X = 2\n')):
                (root / name).mkdir()
                (root / name / 'a.py').write_text(text)
                os.utime(root / name / 'a.py', (old, old))
            repo = Repository(root / 'repo', root / 'seed')
            # Make git trust size and mtime alone, as it can when an inode is reused within one second.
            repo.git('config', 'core.checkStat', 'minimal')
            repo.git('config', 'core.trustctime', 'false')
            self.assertIsNotNone(repo.replace(root / 'candidate', repo.version))
            self.assertEqual((root / 'repo' / 'a.py').read_text(), 'X = 2\n')
            self.assertEqual(repo.git('status', '--porcelain'), '')


if __name__ == '__main__':
    unittest.main()
