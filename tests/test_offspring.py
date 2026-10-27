import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor')]
from codetest.config import Config
from codetest.engine import Engine
from codetest.fixture import create_task, FixtureClient
from codetest.io import Events
from codetest.offspring import OffspringEngine, NOTE_LIMIT, MAX_STRATEGY_INSTRUCTIONS
from codetest.policy import Policy


class RecordingClient(FixtureClient):
    def __init__(self):
        self.calls = []

    def generate(self, messages, max_tokens, phase, agent):
        self.calls.append((phase, agent, json.loads(messages[-1]['content'])))
        return super().generate(messages, max_tokens, phase, agent)


def config(**changes):
    return Config(**{'k': 4, 'rounds': 10, 'max_rounds': 10, 'work_calls': 2,
                     'offspring': True, **changes})


class OffspringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task = create_task(self.root / 'task')
        self.created = 0

    def engine(self, cfg=None, cls=OffspringEngine):
        cfg = cfg or config()
        out = self.root / f'run-{self.created}'
        self.created += 1
        out.mkdir()
        events, client = Events(out), RecordingClient()
        engine = cls(cfg, self.task, Policy(client, cfg, events), events, out)
        engine.test_client = client
        return engine

    def events(self, engine, kind):
        return [row for row in engine.events.rows if row['event'] == kind]

    def ranked(self, engine):
        for agent, wealth in zip(engine.agents, (.1, 2.0, .9, 1.0)):
            agent.wealth = wealth
        engine.assert_accounting()
        return engine.agents[0], engine.agents[1]

    def evolve(self, engine, step=2):
        engine.step = step
        engine.evolve_before_round()

    def test_default_engine_has_identical_accounts_and_no_offspring_context(self):
        engine = self.engine(config(offspring=False), Engine)
        self.assertIs(engine.financial_accounts(), engine.agents)
        self.assertIs(engine.financial_account('agent-00'), engine.agents[0])
        self.assertNotIn('offspring', engine.state())
        self.assertNotIn('retired_accounts', engine.state())
        self.assertNotIn('strategy', engine.context(engine.agents[0]))
        self.assertNotIn('lineage', engine.context(engine.agents[0]))
        engine.assert_accounting()
        with self.assertRaisesRegex(ValueError, 'opt-in'):
            self.engine(config(offspring=False))

    def test_birth_replaces_weakest_with_unique_solo_child_and_preserves_all_money(self):
        engine = self.engine()
        victim, parent = self.ranked(engine)
        population = engine.agents
        rng_before = engine.rng.getstate()
        self.evolve(engine)
        child = engine.teams.agent('agent-04')
        self.assertIsNotNone(child)
        self.assertEqual(len(engine.agents), engine.cfg.k)
        self.assertIs(engine.agents, population)
        self.assertIs(engine.teams.population, population)
        self.assertIsNone(engine.teams.agent(victim.name))
        self.assertIs(engine.financial_account(victim.name), victim)
        self.assertEqual(child.wealth, .25)
        self.assertEqual(parent.wealth, 1.75)
        self.assertIsNone(child.team_tag)
        self.assertEqual(child.memory, '')
        self.assertEqual(engine.rng.getstate(), rng_before)
        self.assertAlmostEqual(math.fsum(a.wealth for a in engine.financial_accounts()), 4)
        engine.assert_accounting()
        born = self.events(engine, 'agent_born')[0]
        self.assertEqual((born['parent'], born['root'], born['generation']), (parent.name, parent.name, 1))
        self.assertNotEqual(born['parent_strategy_sha256'], born['strategy_sha256'])
        self.assertEqual(born['retired_agent'], victim.name)
        self.assertEqual(len(self.events(engine, 'offspring_investment')), 1)
        state = engine.state()
        self.assertEqual(state['offspring']['births'], 1)
        self.assertEqual(state['retired_accounts'], [{'id': victim.name, 'wealth': .1, 'retired_round': 2}])

    def test_descendant_generation_keeps_root_and_never_reuses_ids(self):
        engine = self.engine()
        _, parent = self.ranked(engine)
        self.evolve(engine)
        first_child = engine.teams.agent('agent-04')
        parent.wealth -= 1.5
        first_child.wealth += 1.5
        self.evolve(engine, 4)
        born = self.events(engine, 'agent_born')[-1]
        self.assertEqual(born['agent'], 'agent-05')
        self.assertEqual(born['parent'], 'agent-04')
        self.assertEqual(born['root'], parent.name)
        self.assertEqual(born['generation'], 2)
        for step in (6, 8):
            self.evolve(engine, step)
        ids = [row['id'] for row in engine.state()['offspring']['lineage']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(engine.agents), 4)
        self.assertTrue(all(len(strategy) <= MAX_STRATEGY_INSTRUCTIONS for strategy in engine.strategies.values()))
        engine.assert_accounting()

    def test_retired_contributor_receives_later_bucket_payment_not_its_replacement(self):
        engine = self.engine()
        victim, _ = self.ranked(engine)
        engine.feedback = self.task.inspect(engine.repo.path)
        engine.history.append({'participants': [victim.name]})
        self.evolve(engine)
        child = engine.teams.agent('agent-04')
        payer = engine.teams.agent('agent-03')
        retired_before, child_before = victim.wealth, child.wealth
        winner = {'team': 'solo:' + payer.name, 'pledges': {payer.name: .1}, 'bid': .1,
                  'participants': [payer.name], 'new_probes': []}
        engine.publish(winner, '', engine.feedback, 'late-payment')
        self.assertAlmostEqual(victim.wealth, retired_before + .1)
        self.assertEqual(child.wealth, child_before)
        self.assertEqual(self.events(engine, 'transfer')[-1]['recipients'], [victim.name])
        self.assertAlmostEqual(engine.state()['retired_accounts'][0]['wealth'], .2)
        engine.assert_accounting()

    def test_age_schedule_and_idempotence_prevent_extra_births(self):
        engine = self.engine()
        for step in (-1, 0, 1, 3):
            self.evolve(engine, step)
        self.assertEqual(engine.births, 0)
        self.evolve(engine, 2)
        self.evolve(engine, 2)
        self.assertEqual(engine.births, 1)
        child = engine.teams.agent('agent-04')
        child.wealth -= .2
        engine.agents[1].wealth += .2
        engine.lineage[child.name]['born_round'] = 3
        self.evolve(engine, 4)
        self.assertIn(child, engine.agents)  # Only one completed round old despite being poorest.
        self.evolve(engine, 10)
        self.assertEqual(engine.births, 2)  # No birth after the final round.
        engine.assert_accounting()

    def test_insufficient_funding_skips_without_retiring_or_spawning(self):
        for cfg, reason in ((config(initial_wealth=.1), 'parent cannot fund'),
                            (config(rent=.2), 'endowment cannot fund')):
            with self.subTest(reason=reason):
                engine = self.engine(cfg)
                before = [(a.name, a.wealth) for a in engine.agents]
                self.evolve(engine)
                self.assertEqual([(a.name, a.wealth) for a in engine.agents], before)
                self.assertEqual(engine.retired_accounts, {})
                self.assertEqual(engine.births, 0)
                self.assertIn(reason, self.events(engine, 'offspring_skipped')[-1]['reason'])
                engine.assert_accounting()

    def test_child_and_parent_can_pay_current_rent_without_retired_agent_paying(self):
        engine = self.engine(config(rent=.1))
        victim, parent = self.ranked(engine)
        engine.step = 2
        engine.charge_rent()
        child = engine.teams.agent('agent-04')
        self.assertAlmostEqual(child.wealth, .15)
        self.assertAlmostEqual(parent.wealth, 1.65)
        self.assertEqual(victim.wealth, .1)
        self.assertNotIn(child.name, engine.resting)
        self.assertAlmostEqual(engine.rent_paid, .4)
        engine.assert_accounting()

    def test_retirement_cleans_membership_and_invitations_preserving_public_artifacts(self):
        engine = self.engine(config(shared_scratchpad=True))
        victim, parent = self.ranked(engine)
        other = engine.agents[2]
        engine.teams.create_team([victim, other], 0)
        engine.teams.invitations = {'retiring': {'from': victim.name, 'to': parent.name, 'team': victim.team_tag, 'round': 2},
                                    'safe': {'from': parent.name, 'to': engine.agents[3].name, 'team': None, 'round': 2}}
        engine.scratchpad.write(victim.name, {'key': 'old-finding', 'revision': 0, 'kind': 'finding',
                                             'status': 'open', 'text': 'Historical evidence', 'dependencies': []}, 0)
        before = engine.scratchpad.observation()
        self.evolve(engine)
        self.assertIsNone(other.team_tag)
        leave = self.events(engine, 'leave')[-1]
        self.assertEqual(leave['agent'], victim.name)
        self.assertEqual(leave['dissolved'], other.name)
        self.assertEqual(leave['reason'], 'offspring retirement')
        self.assertNotIn('retiring', engine.teams.invitations)
        self.assertIn('safe', engine.teams.invitations)
        self.assertEqual(engine.scratchpad.observation(), before)
        self.assertNotIn(victim.name, engine.inboxes)
        self.assertEqual(engine.inboxes['agent-04'], [])

    def test_notes_are_bounded_separate_from_strategy_and_do_not_copy_private_inbox(self):
        engine = self.engine()
        _, parent = self.ranked(engine)
        parent.memory = 'own observation ' * 200
        engine.inboxes[parent.name].append({'id': 'private-1', 'sender': 'agent-02', 'text': 'PRIVATE_INBOX_CANARY'})
        (engine.repo.path / 'PROJECT_NOTES.md').write_text('public observations ' * 200)
        self.evolve(engine)
        child = engine.teams.agent('agent-04')
        context = engine.context(child)
        self.assertEqual(len(context['inherited_notes']['parent_memory']), NOTE_LIMIT)
        self.assertEqual(len(context['inherited_notes']['public_notes']), NOTE_LIMIT)
        self.assertEqual(context['direct_messages'], [])
        self.assertEqual(context['memory'], '')
        self.assertNotIn('PRIVATE_INBOX_CANARY', json.dumps(context))
        self.assertNotIn('own observation', context['strategy'])
        self.assertIn('unverified', context['offspring_rules'])

    def test_clone_control_keeps_strategy_exactly(self):
        engine = self.engine(config(offspring_mutation=False))
        self.ranked(engine)
        self.evolve(engine)
        born = self.events(engine, 'agent_born')[0]
        self.assertEqual(born['parent_strategy_sha256'], born['strategy_sha256'])
        self.assertEqual(born['mutation_operation'], 'clone')
        self.assertFalse(born['mutation'])

    def test_same_seed_reproduces_lineage_mutation_and_random_parent_choice(self):
        for selection in ('wealth', 'random'):
            with self.subTest(selection=selection):
                left = self.engine(config(offspring_selection=selection))
                right = self.engine(config(offspring_selection=selection))
                for engine in (left, right):
                    self.ranked(engine)
                    for step in (2, 4, 6, 8):
                        self.evolve(engine, step)
                self.assertEqual(left.state()['offspring'], right.state()['offspring'])
                self.assertEqual([(a.name, a.wealth) for a in left.financial_accounts()],
                                 [(a.name, a.wealth) for a in right.financial_accounts()])

    def test_random_parent_uses_fundable_pool_instead_of_skipping_on_poor_draw(self):
        for seed in range(5):
            with self.subTest(seed=seed):
                engine = self.engine(config(seed=seed, offspring_selection='random', rent=.1))
                for agent, wealth in zip(engine.agents, (.1, .2, .3, 3.4)):
                    agent.wealth = wealth
                engine.assert_accounting()
                self.evolve(engine)
                self.assertEqual(engine.births, 1)
                self.assertEqual(self.events(engine, 'offspring_skipped'), [])
                born = self.events(engine, 'agent_born')[0]
                self.assertEqual(born['parent'], 'agent-03')
                self.assertEqual(born['retired_agent'], 'agent-00')
                engine.assert_accounting()

    def test_mutation_control_cannot_change_future_selection_rng_with_same_wealth_history(self):
        for selection in ('wealth', 'random'):
            with self.subTest(selection=selection):
                clone = self.engine(config(offspring_selection=selection, offspring_mutation=False))
                mutated = self.engine(config(offspring_selection=selection, offspring_mutation=True))
                for step in (2, 4, 6, 8):
                    # Supply the same external wealth history. Strategy effects on real
                    # outcomes may legitimately differ; mutation RNG consumption may not.
                    for engine in (clone, mutated):
                        active_total = engine.cfg.k * engine.cfg.initial_wealth - math.fsum(
                            a.wealth for a in engine.retired_accounts.values())
                        for agent in engine.agents:
                            agent.wealth = active_total / engine.cfg.k
                        engine.assert_accounting()
                        self.evolve(engine, step)
                    left, right = self.events(clone, 'agent_born')[-1], self.events(mutated, 'agent_born')[-1]
                    for field in ('agent', 'parent', 'retired_agent', 'root', 'generation'):
                        self.assertEqual(left[field], right[field], (selection, step, field))
                    self.assertEqual(clone.offspring_rng.getstate(), mutated.offspring_rng.getstate())
                self.assertEqual(clone.births, 4)
                self.assertEqual(mutated.births, 4)
                self.assertNotEqual(clone.mutation_rng.getstate(), mutated.mutation_rng.getstate())

    def test_birth_uses_no_policy_or_model_call(self):
        engine = self.engine()
        self.ranked(engine)
        with patch.object(engine.policy, 'ask', side_effect=AssertionError('Birth invoked policy')), \
                patch.object(engine.test_client, 'generate', side_effect=AssertionError('Birth invoked model')):
            self.evolve(engine)
        self.assertEqual(engine.policy.calls, 0)
        self.assertEqual(engine.births, 1)

    def test_full_fixture_keeps_fixed_per_round_work_allocation_and_retires_inference(self):
        engine = self.engine(config(rounds=5, max_rounds=5, offspring_mutation=False))
        engine.run()
        self.assertEqual(engine.births, 2)
        self.assertEqual(engine.state()['completed_rounds'], 5)
        for step in range(5):
            work = [(agent, context) for phase, agent, context in engine.test_client.calls
                    if phase == 'work' and context['round'] == step]
            self.assertEqual(len(work), engine.cfg.k * engine.cfg.work_calls)
        for name, lineage in engine.lineage.items():
            if lineage['retired_round'] is not None:
                self.assertTrue(all(context['round'] < lineage['retired_round']
                                    for phase, agent, context in engine.test_client.calls if agent == name))
        self.assertFalse(any('birth' in phase for phase, _, _ in engine.test_client.calls))
        self.assertTrue(all(len(snapshot['agents']) == engine.cfg.k for snapshot in engine.snapshots))
        engine.assert_accounting()

    def test_unsupported_benchmark_is_rejected_before_engine_creation(self):
        self.task.data['benchmark'] = 'programbench'
        with self.assertRaisesRegex(ValueError, 'Commit0'):
            self.engine()


if __name__ == '__main__':
    unittest.main()
