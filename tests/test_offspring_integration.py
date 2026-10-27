import unittest
from pathlib import Path
from codetest.cli import config
from codetest.config import Config
from codetest.policy import prompt_revision
from codetest.report import metrics, offspring_metrics


class OffspringIntegrationTests(unittest.TestCase):
    def test_invalid_population_and_incompatible_modes_are_rejected(self):
        for changes in ({'k':1}, {'condition':'solo'}, {'contribution_contracts':True,'shared_scratchpad':True},
                        {'offspring_every':1}, {'offspring_endowment':float('nan')}, {'offspring_selection':'best'},
                        {'offspring_mutation':1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                Config(**{'offspring':True, **changes})

    def test_disabled_revision_and_metrics_keep_classic_shape(self):
        cfg = Config()
        self.assertEqual(prompt_revision(cfg), 'adaptive-response-tokens-5')
        self.assertNotIn('offspring', metrics({'config':cfg.as_dict()}, []))
        self.assertIn('+offspring-wealth-mutate', prompt_revision(Config(offspring=True)))
        self.assertIn('+offspring-wealth-clone', prompt_revision(Config(offspring=True,offspring_mutation=False)))

    def test_lineage_contribution_is_not_counted_per_teammate(self):
        state = {'agents':[{'id':'a','wealth':.5},{'id':'b','wealth':.25}],
                 'retired_accounts':[{'id':'old','wealth':.75}],
                 'offspring':{'lineage':[{'id':'a','root':'root-a','generation':0,'strategy_sha256':'same'},
                                          {'id':'b','root':'root-a','generation':1,'strategy_sha256':'same'}]}}
        events = [{'event':'agent_born','agent':'b','round':2}, {'event':'agent_retired'},
                  {'event':'offspring_investment','amount':.25},
                  {'event':'publication','round':3,'participants':['a','b']},
                  {'event':'publication','round':4,'participants':['b']},
                  {'event':'tool','agent':'b'}, {'event':'tool','agent':'old'},
                  {'event':'api_charged','phase':'birth_good'}]
        result = offspring_metrics(state, events)
        self.assertEqual(result['births'],1)
        self.assertEqual(result['child_coauthored_publications'],2)
        self.assertEqual(result['children_with_publication_credit'],1)
        self.assertEqual(result['rounds_to_first_publication_credit'],{'b':1})
        self.assertEqual(result['active_strategy_diversity'],1)
        self.assertEqual(result['largest_active_lineage_share'],1)
        self.assertEqual(result['active_wealth']+result['retired_wealth'],1.5)
        self.assertEqual(result['child_work_actions'],1)
        self.assertEqual(result['dedicated_birth_model_calls'],1)

    def test_offspring_configs_pin_horizon_and_isolate_controls(self):
        directory = Path(__file__).resolve().parents[1] / 'configs/openai'
        prefix = 'k10-gpt6-luna-scratchpad-offspring'
        base = config(directory / (prefix + '.json'))
        self.assertEqual((base.k, base.rounds, base.max_rounds), (10, 10, 10))
        self.assertEqual(base.provider, 'openai')
        self.assertTrue(base.offspring and base.shared_scratchpad and base.task_board and base.regression_gate)
        self.assertEqual(base.merge_slots, 10)
        self.assertEqual(base.rent, .1)
        for suffix, expected in [('-clone', 'offspring_mutation'), ('-random', 'offspring_selection'), ('-frozen', 'offspring_every')]:
            variant = config(directory / (prefix + suffix + '.json'))
            differences = {k for k in base.as_dict() if base.as_dict()[k] != variant.as_dict()[k]}
            self.assertEqual(differences, {expected})
        frozen = config(directory / (prefix + '-frozen.json'))
        self.assertEqual(frozen.offspring_every, frozen.max_rounds)

    def test_newborn_identity_is_not_a_membership_switch(self):
        snapshots = [
            {'agents':[{'id':'a','team':'team-1'},{'id':'old','team':'solo:old'}]},
            {'agents':[{'id':'a','team':'team-1'},{'id':'child','team':'solo:child'}]}]
        result = metrics({'config':{'offspring':True},'snapshots':snapshots}, [])
        self.assertEqual(result['agent_membership_changes'], 0)
        self.assertEqual(result['membership_persistence'], 1.0)
        result = metrics({'config':{'offspring':True},'snapshots':[
            {'agents':[{'id':'old','team':'solo:old'}]},
            {'agents':[{'id':'child','team':'solo:child'}]}]}, [])
        self.assertIsNone(result['membership_persistence'])
