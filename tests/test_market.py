import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor')]
from codetest.config import Config
from codetest.engine import Engine, failing_board
from codetest.fixture import create_task, FixtureClient, FixtureTask
from codetest.io import Events
from codetest.policy import Policy, prompt_revision
from codetest.report import metrics, spearman
from codetest.sandbox import copy_source


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


class MarketConfigTests(unittest.TestCase):
    def test_flags_are_opt_in_and_validated(self):
        legacy = Config().as_dict()
        for name in ('merge_slots', 'rent', 'task_board'):
            self.assertNotIn(name, legacy)
        on = market_config(merge_slots=4, rent=0.1, task_board=True, regression_gate=True)
        self.assertEqual({k: on.as_dict()[k] for k in ('merge_slots', 'rent', 'task_board')},
                         {'merge_slots': 4, 'rent': 0.1, 'task_board': True})
        self.assertEqual(prompt_revision(on), 'shared-scratchpad-6+regression-gate+merge-slots-4+rent+task-board')
        self.assertEqual(prompt_revision(market_config()), 'shared-scratchpad-6')
        bad = [{'merge_slots': 0}, {'merge_slots': 2.0}, {'rent': -1}, {'rent': float('nan')}, {'task_board': 1},
               {'merge_slots': 2, 'shared_scratchpad': False},
               {'rent': 0.1, 'condition': 'swarm'}, {'task_board': True, 'condition': 'swarm'},
               {'merge_slots': 2, 'condition': 'single_agent', 'k': 1, 'parallel_experiments': True,
                'shared_scratchpad': False}]
        for fields in bad:
            with self.assertRaises(ValueError, msg=fields):
                market_config(**fields)

    def test_spearman_handles_ties_and_constants(self):
        self.assertAlmostEqual(spearman([(1, 10), (2, 20), (3, 30)]), 1.0)
        self.assertAlmostEqual(spearman([(1, 3), (2, 2), (3, 1)]), -1.0)
        self.assertAlmostEqual(spearman([(0.1, 1), (0.1, 1), (0.2, 2), (0.3, 2)]), 0.9428090415820635)
        self.assertIsNone(spearman([(1, 5), (2, 5), (3, 5)]))
        self.assertIsNone(spearman([(1, 2), (2, 3)]))


class MergeSlotTests(MarketTestCase):
    def test_bid_order_merges_several_verified_candidates_per_round(self):
        engine = self.engine(market_config(merge_slots=6))
        engine.run()
        engine.assert_accounting()
        publications = self.rows('publication')
        first = [p for p in publications if p['round'] == 0]
        self.assertGreater(len(first), 1)
        self.assertEqual(first[0]['mode'], 'direct')
        self.assertTrue(all(p['mode'] == '3way' for p in first[1:]))
        self.assertEqual([p['bid'] for p in first], sorted([p['bid'] for p in first], reverse=True))
        # Every merge flips tests and the flips add up to the public score.
        self.assertTrue(all(p['flipped'] for p in publications))
        self.assertEqual(sum(len(p['flipped']) for p in publications), engine.feedback['passed'])
        # Bucket brigade: the first merge burns its bid, every later merge pays the one before it.
        self.assertEqual(len(self.rows('burn')), 1)
        self.assertEqual(len(self.rows('transfer')), len(publications) - 1)
        # Round 1 rewrites existing modules: nothing new passes, so nothing merges or pays.
        self.assertEqual([p for p in publications if p['round'] == 1], [])
        self.assertTrue(all(r['reason'] == 'no additional test passes' for r in self.rows('merge_rejected')))
        summary = metrics(json.loads((engine.out / 'state.json').read_text()), self.events.rows)['market']
        self.assertEqual(summary['merges'], len(publications))
        self.assertEqual(summary['three_way_merges'], len(publications) - 1)
        self.assertEqual(summary['rejected_merges'], {'no additional test passes': len(self.rows('merge_rejected'))})

    def test_slots_cap_merges_and_skip_the_rest(self):
        engine = self.engine(market_config(merge_slots=2))
        engine.run()
        per_round = {}
        for p in self.rows('publication'):
            per_round[p['round']] = per_round.get(p['round'], 0) + 1
        self.assertEqual(per_round.get(0), 2)
        self.assertTrue(self.rows('merge_skipped'))
        engine.assert_accounting()

    def test_verified_merge_rejects_a_candidate_that_breaks_passing_tests(self):
        engine = self.engine(market_config(merge_slots=6))
        engine.run()
        head, before = engine.repo.version, dict(engine.feedback)
        source = self.root / 'breaking'
        copy_source(engine.repo.path, source)
        victim = next(source.glob('module_*.py'))
        victim.unlink()
        proposal = {'team': 'solo:agent-00', 'base_commit': head, 'source': str(source),
                    'check': FixtureTask.inspect(source)}
        self.assertIsNone(engine.verified_merge(proposal))
        self.assertEqual(engine.repo.version, head)
        self.assertTrue((engine.repo.path / victim.name).is_file())
        self.assertEqual(engine.feedback, before)
        self.assertTrue(self.rows('merge_rejected')[-1]['reason'].startswith('breaks passing tests:'))

    def test_prompt_states_the_queue_and_drops_the_single_winner_rule(self):
        engine = self.engine(market_config(merge_slots=6, regression_gate=False))
        engine.run()
        system = self.client.observations[0][3]
        self.assertIn('Up to 6 contributions per round', system)
        self.assertIn('Eligible candidates are merged in order of positive bid', system)
        self.assertNotIn('Only the winning contribution', system)
        self.assertNotIn('Score regressions can be accepted', system)
        self.assertNotIn('The highest positive eligible bid wins', system)

    def test_merge_slots_need_per_test_results(self):
        self.task.data['benchmark'] = 'programbench'
        with self.assertRaises(ValueError):
            self.engine(market_config(merge_slots=2))


class RentTests(MarketTestCase):
    def test_agents_that_cannot_pay_rest_without_calls_and_accounting_holds(self):
        engine = self.engine(market_config(rent=0.4, rounds=3, max_rounds=3))
        engine.run()
        engine.assert_accounting()
        rests = self.rows('rest')
        self.assertTrue(rests)
        for rest in rests:
            resting = set(rest['agents'])
            calls = {agent for phase, agent, o, _ in self.client.observations if o['round'] == rest['round']}
            self.assertFalse(resting & calls, rest['round'])
            self.assertTrue(all(o['resting_agents'] == sorted(resting) for _, _, o, _ in self.client.observations
                                if o['round'] == rest['round']))
        paid = sum(e['paid'] for e in self.rows('rent'))
        self.assertAlmostEqual(engine.rent_paid, 0.4 * paid)
        system = self.client.observations[0][3]
        self.assertIn('RENT: Working a round costs 0.4 wealth', system)

    def test_rest_leaves_the_team_and_income_wakes_the_agent(self):
        engine = self.engine(market_config(rent=0.3))
        a, b = engine.agents[:2]
        engine.teams.create_team([a, b], 0)
        a.wealth, b.wealth = 0.1, 1.9  # Wealth moves between agents; the total is conserved.
        engine.step = 0
        engine.charge_rent()
        self.assertEqual(engine.resting, {a.name})
        self.assertIsNone(a.team_tag)
        self.assertIsNone(b.team_tag)  # A two-agent team dissolves.
        self.assertAlmostEqual(b.wealth, 1.6)
        self.assertNotIn(a.name, {n for t in engine.working_teams() for n in t.member_ids})
        a.wealth += 0.5; b.wealth -= 0.5  # e.g. a bucket brigade payment for an earlier contribution
        engine.step = 1
        engine.charge_rent()
        self.assertEqual(engine.resting, set())
        self.assertAlmostEqual(a.wealth, 0.3)
        self.assertIn(a.name, {n for t in engine.working_teams() for n in t.member_ids})


class TaskBoardTests(MarketTestCase):
    def test_board_groups_failing_tests_by_module(self):
        engine = self.engine(market_config(task_board=True))
        engine.run()
        boards = [o['task_board'] for _, _, o, _ in self.client.observations]
        self.assertTrue(boards and all(b['available'] for b in boards))
        first = boards[0]
        self.assertEqual(first['failing_tests'], 20)
        self.assertEqual(sum(m['failing'] for m in first['modules']), 20)
        self.assertEqual(first['modules'][0], {'module': 'fixture.group_0', 'failing': 5,
                                               'sample_tests': [f'module_{i:02}' for i in range(5)]})
        self.assertLess(boards[-1]['failing_tests'], 20)
        self.assertNotIn('failing', self.client.observations[0][2]['public_feedback'])
        self.assertIn('TASK BOARD:', self.client.observations[0][3])

    def test_board_is_bounded_and_waits_for_a_score(self):
        self.assertFalse(failing_board({'score': None})['available'])
        many = {'score': 0, 'failing': [f'm{i:03}::t{j}' for i in range(50) for j in range(9)]}
        board = failing_board(many)
        self.assertEqual(len(board['modules']), 40)
        self.assertEqual(board['more_modules'], 10)
        self.assertTrue(all(len(m['sample_tests']) == 5 for m in board['modules']))


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
