import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor')]
from codetest.actions import action_schema, normalize_action
from codetest.benchmarks import junit
from codetest.config import Config
from codetest.fixture import create_task, FixtureClient, FixtureTask
from codetest.io import Events
from codetest.policy import Policy, prompt_revision
from codetest.report import metrics
from codetest.sandbox import copy_source
from codetest.swarm import SwarmEngine


def swarm_config(**fields):
    return Config(**{'k': 6, 'rounds': 2, 'max_rounds': 2, 'work_calls': 4, 'condition': 'swarm',
                     'shared_scratchpad': True, **fields})


class SwarmConfigTests(unittest.TestCase):
    def test_swarm_requires_scratchpad_and_hides_fields_elsewhere(self):
        with self.assertRaises(ValueError):
            Config(condition='swarm')
        with self.assertRaises(ValueError):
            swarm_config(royalty_rate=1.0)
        with self.assertRaises(ValueError):
            swarm_config(crowding_cap=0)
        self.assertNotIn('bounty_base', Config().as_dict())
        self.assertEqual(swarm_config().as_dict()['crowding_cap'], 2)
        self.assertEqual(prompt_revision(swarm_config()), 'swarm-1')
        estimate = swarm_config(k=5, max_rounds=10).estimate()
        self.assertEqual(estimate['control_calls_max'], 50)  # One claim per agent per round at most.

    def test_junit_keeps_failing_test_ids(self):
        xml = ('<testsuite><testcase classname="tests.test_a" name="ok"/>'
               '<testcase classname="tests.test_b" name="bad"><failure/></testcase></testsuite>')
        self.assertEqual(junit(xml)['failing'], ['tests.test_b::bad'])

    def test_claim_schema(self):
        self.assertEqual(set(action_schema('claim', True)['properties']), {'task', 'join', 'approach', 'message', 'scratchpad'})
        action = normalize_action({'task': 'tests.test_a', 'join': None, 'approach': 'x', 'message': ''}, 'claim')
        self.assertIsNone(action['join'])


class SwarmEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task = create_task(self.root / 'task')
        self.out = self.root / 'run'; self.out.mkdir()
        self.events = Events(self.out)
        self.cfg = swarm_config()
        self.policy = Policy(FixtureClient(), self.cfg, self.events)
        self.engine = SwarmEngine(self.cfg, self.task, self.policy, self.events, self.out)

    def rows(self, kind):
        return [e for e in self.events.rows if e['event'] == kind]

    def test_board_groups_failing_tests_and_bounties_grow_with_age(self):
        self.engine.step = 0
        self.engine.feedback = {'build': True, 'score': .5, 'passed': 2, 'total': 4,
                                'failing': ['tests.test_a::x', 'tests.test_a::y', 'tests.test_b::z']}
        board = self.engine.board()
        self.assertFalse(board['bootstrap'])
        self.assertEqual([t['id'] for t in board['tasks']], ['tests.test_a', 'tests.test_b'])
        self.assertEqual(board['tasks'][0]['failing'], 2)
        self.assertAlmostEqual(board['tasks'][0]['bounty'], 2 * self.cfg.bounty_base)
        self.assertEqual(board['tasks'][0]['open_slots'], self.cfg.crowding_cap)
        self.engine.step = 2
        self.assertAlmostEqual(self.engine.bounty('tests.test_b::z'), self.cfg.bounty_base * 1.5)
        self.engine.feedback = {'build': False, 'score': None}
        bootstrap = self.engine.board()
        self.assertTrue(bootstrap['bootstrap'])
        self.assertEqual(bootstrap['tasks'][0]['open_slots'], 2 * self.cfg.crowding_cap)

    def test_claims_respect_slots_and_rooms_and_put_the_rest_to_sleep(self):
        self.engine.feedback = {'build': True, 'score': 0, 'passed': 0, 'total': 20,
                                'failing': [f'fixture.group_0::module_{i:02}' for i in range(5)]}
        self.engine.cfg.crowding_cap = 1
        self.engine.step = 0
        self.engine.form()
        groups = self.engine.working_teams()
        self.assertEqual(len(groups), 1)
        self.assertLessEqual(len(groups[0].member_ids), self.cfg.max_team_size)
        resting = self.rows('rest')[0]['agents']
        self.assertEqual(len(resting) + len(groups[0].member_ids), self.cfg.k)
        # Only two agents per open slot are woken; the others never make a paid claim call.
        self.assertEqual(len(self.rows('claim')), 2)
        for name in groups[0].member_ids:
            tag = self.engine.teams.agent(name).team_tag
            self.assertEqual(tag, groups[0].team_id if len(groups[0].member_ids) > 1 else None)

    def test_full_run_merges_several_candidates_per_round_and_pays_only_for_flips(self):
        self.engine.run()
        self.engine.finalize()
        self.engine.assert_accounting()
        self.assertEqual(self.rows('ballot'), [])
        self.assertEqual(self.rows('membership')[0]['round'], 0)
        publications = self.rows('publication')
        # Round 0: every scripted agent adds its own module, so several candidates merge in one round.
        self.assertGreater(len([p for p in publications if p['round'] == 0]), 1)
        self.assertTrue(any(p['mode'] == '3way' for p in publications))
        flipped = sum(len(p['flipped']) for p in publications)
        self.assertEqual(flipped, self.engine.feedback['passed'])
        self.assertEqual(flipped, self.cfg.k)
        self.assertGreater(self.engine.bounties_issued, 0)
        self.assertAlmostEqual(sum(a.wealth for a in self.engine.agents), self.cfg.k + self.engine.bounties_issued)
        for record in self.engine.specialty.values():
            self.assertTrue(all(n > 0 for n in record.values()))
        # Round 1: the same modules are rewritten, no test flips, so nothing merges and nothing is paid.
        self.assertEqual([p for p in publications if p['round'] == 1], [])
        rejected = self.rows('merge_rejected')
        self.assertTrue(rejected and all(r['reason'] == 'no additional test passes' for r in rejected))
        self.assertEqual(self.engine.repo.version, publications[-1]['commit'])
        state = json.loads((self.out / 'state.json').read_text())
        summary = metrics(state, self.events.rows)['swarm']
        self.assertEqual(summary['merges'], len(publications))
        self.assertEqual(summary['rejected_merges'], {'no additional test passes': len(rejected)})
        self.assertFalse(any('failing' in r['after'] for r in state['history']))

    def test_merge_rejects_a_candidate_that_breaks_passing_tests(self):
        self.engine.run()
        head = self.engine.repo.version
        before = dict(self.engine.feedback)
        source = self.root / 'breaking'
        copy_source(self.engine.repo.path, source)
        victim = next(source.glob('module_*.py'))
        victim.unlink()
        check = FixtureTask.inspect(source)
        self.engine.step += 1
        proposal = {'team': 'solo:agent-00', 'task': 'fixture.group_0', 'members': ['agent-00'],
                    'participants': ['agent-00'], 'base_commit': head, 'source': str(source), 'check': check,
                    'changed_paths': [victim.name], 'fetched_from': [], 'shared_artifact': None}
        self.assertIsNone(self.engine.merge(proposal))
        self.assertEqual(self.engine.repo.version, head)
        self.assertTrue((self.engine.repo.path / victim.name).is_file())
        self.assertEqual(self.engine.feedback, before)
        self.assertEqual(self.rows('merge_rejected')[-1]['reason'][:21], 'breaks passing tests:')
        self.engine.assert_accounting()

    def test_royalty_goes_to_fetched_authors(self):
        self.engine.step = 0
        self.engine.feedback = {'build': True, 'score': 0, 'passed': 0, 'total': 20,
                                'failing': ['fixture.group_0::module_00', 'fixture.group_0::module_01']}
        self.engine.public_failing()
        proposal = {'team': 'solo:agent-00', 'task': 'fixture.group_0', 'members': ['agent-00'],
                    'participants': ['agent-00'], 'fetched_from': ['agent-05', 'missing-agent']}
        total, royalty, royalty_to = self.engine.pay(proposal, ['fixture.group_0::module_00'], False, {'passed': 1})
        self.assertAlmostEqual(total, self.cfg.bounty_base)
        self.assertEqual(royalty_to, ['agent-05'])
        self.assertAlmostEqual(royalty, self.cfg.bounty_base * self.cfg.royalty_rate)
        self.assertAlmostEqual(self.engine.teams.agent('agent-05').wealth, 1 + royalty)
        self.assertAlmostEqual(self.engine.teams.agent('agent-00').wealth, 1 + total - royalty)
        self.engine.assert_accounting()


if __name__ == '__main__':
    unittest.main()
