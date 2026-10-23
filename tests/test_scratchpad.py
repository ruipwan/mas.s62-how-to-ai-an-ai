import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor')]
from codetest.actions import action_schema, normalize_action
from codetest.config import Config
from codetest.engine import Engine, tree_digest
from codetest.fixture import create_task, FixtureClient
from codetest.io import Events
from codetest.policy import Policy, prompt_revision
from codetest.scratchpad import Scratchpad


def note(key='parser', revision=0, text='Own parser work; depends on tokenizer signature.'):
    return dict(key=key, revision=revision, kind='task', status='claimed', text=text,
                dependencies=['tokenizer'])


class ScratchpadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.events = []
        self.pad = Scratchpad(self.root / 'pad', lambda event, **kw: self.events.append(dict(event=event, **kw)))

    def test_persistence_owner_revision_and_history(self):
        self.pad.write('agent-00', note(), 0)
        with self.assertRaisesRegex(ValueError, 'owner'):
            self.pad.write('agent-01', note(revision=1), 0)
        with self.assertRaisesRegex(ValueError, 'Revision'):
            self.pad.write('agent-00', note(), 1)
        self.pad.write('agent-00', {**note(revision=1), 'status': 'done', 'text': 'Local parser check passed.'}, 1)
        reloaded = Scratchpad(self.pad.root, lambda *a, **kw: None)
        self.assertEqual(reloaded.entries['parser']['revision'], 2)
        history = list(map(json.loads, (self.pad.root / 'history.jsonl').read_text().splitlines()))
        self.assertEqual([h['entry']['status'] for h in history], ['claimed', 'done'])
        self.assertEqual(reloaded.observation()['entry_index'][0]['owner'], 'agent-00')

    def test_bounds_and_disabled_contract(self):
        for bad in [note(key='../escape'), note(revision=True), note(text='x' * 1601),
                    {**note(), 'dependencies': ['x'] * 13}]:
            with self.assertRaises(ValueError):
                self.pad.write('agent-00', bad, 0)
        self.assertNotIn('scratchpad', action_schema('work')['properties'])
        self.assertNotIn('fetch', action_schema('work')['properties']['tool']['enum'])
        with self.assertRaises(ValueError):
            Config(shared_scratchpad='yes')
        self.assertEqual(prompt_revision(Config()), 'adaptive-response-tokens-5')
        self.assertEqual(prompt_revision(Config(shared_scratchpad=True)), 'shared-scratchpad-6')

    def test_strict_contract_accepts_nullable_note_and_fetch(self):
        action = dict(tool='fetch', command='r01-solo-agent-00', message='', channel='team',
                      recipients=[], args=[], stdin='', capture=[], files=[], scratchpad=None)
        self.assertEqual(normalize_action(action, 'work', True)['tool'], 'fetch')
        for phase in ('membership', 'join', 'discuss', 'ballot', 'work'):
            self.assertIn('scratchpad', action_schema(phase, True)['required'])

    def test_patch_integrity_stale_base_and_no_automatic_apply(self):
        base, source = self.root / 'base', self.root / 'source'
        base.mkdir(); source.mkdir()
        (base / 'module.py').write_text('value = 1\n')
        (source / 'module.py').write_text('value = 2\n')
        proposal = dict(team='solo:agent-00', members=['agent-00'], base_commit='abc',
                        source=str(source), changed_paths=['module.py'], check={'build': True, 'score': .5})
        before = tree_digest(base)
        ident = self.pad.share(proposal, base, 0)
        self.assertEqual(tree_digest(base), before)
        calls = []
        class Box:
            def put(self, *args, **kw):
                calls.append((args, kw))
        fetched = self.pad.fetch(ident, 'agent-01', 'newer', Box(), 2)
        self.assertFalse(fetched['same_base'])
        self.assertEqual(fetched['authors'], ['agent-00'])
        self.assertTrue(calls[0][1]['readonly'])
        self.assertEqual(tree_digest(base), before)
        patch = self.pad.root / 'artifacts' / ident / 'patch.diff'
        self.assertIn('+value = 2', patch.read_text())
        for ident_bad in ('../secret', '/tmp', None, 'unknown'):
            with self.assertRaises(ValueError):
                self.pad.fetch(ident_bad, 'agent-01', 'abc', Box(), 0)
        (source / 'module.py').write_text('value = 999\n')
        self.pad.fetch(ident, 'agent-01', 'abc', Box(), 0)
        patch.write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'integrity'):
            self.pad.fetch(ident, 'agent-01', 'abc', Box(), 0)

    def test_cross_team_retention_preserves_work_allocation_and_auction(self):
        task = create_task(self.root / 'task')
        out = self.root / 'run'; out.mkdir()
        events = Events(out)
        cfg = Config(k=5, rounds=3, work_calls=3, shared_scratchpad=True)
        class Client(FixtureClient):
            def generate(self, messages, max_tokens, phase, agent):
                obs = json.loads(messages[-1]['content'])
                reply = json.loads(super().generate(messages, max_tokens, phase, agent))
                if phase == 'membership' and obs['round'] == 0:
                    reply['scratchpad'] = note(key=agent + '-task')
                if phase == 'work' and obs['slot'] == 0:
                    choices = [a for a in obs['shared_scratchpad']['recent_artifacts'] if agent not in a['authors']]
                    if choices:
                        reply.update(tool='fetch', command=choices[0]['id'])
                return json.dumps(reply)
        engine = Engine(cfg, task, Policy(Client(), cfg, events), events, out)
        engine.run(); engine.finalize()
        self.assertEqual(len([e for e in events.rows if e['event'] == 'tool']), 5 * 3 * 3)
        self.assertEqual(len(engine.history), 3)
        self.assertEqual(len(engine.scratchpad.entries), 5)
        self.assertTrue(any(e['event'] == 'scratchpad_fetch' and e['round'] > 0 for e in events.rows))
        candidates = [e for e in events.rows if e['event'] == 'candidate']
        for accepted in engine.history:
            eligible = [c for c in candidates if c['round'] == accepted['round'] and c['eligible']]
            self.assertEqual(accepted['bid'], max(c['bid'] for c in eligible))
        accepted_teams = {(h['round'], h['team']) for h in engine.history}
        self.assertTrue(any((a['round'], a['team']) not in accepted_teams for a in engine.scratchpad.artifacts.values()))
        self.assertNotIn('scratchpad', engine.repo.git('ls-files'))
        engine.assert_accounting()


if __name__ == '__main__':
    unittest.main()
