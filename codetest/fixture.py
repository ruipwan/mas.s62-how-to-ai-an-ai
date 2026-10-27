"""Scripted fixture for zero-dollar orchestration checks. Never a benchmark score.

This box recognizes three fixed commands; it cannot execute a shell or model code.
"""
import ast
import json
from pathlib import Path
import tempfile

from .benchmarks import Task
from .sandbox import copy_source


class FixtureBox:
    def __init__(self, source):
        self.temp = tempfile.TemporaryDirectory(prefix='codetest-fixture-')
        self.path = Path(self.temp.name)
        copy_source(source, self.path)
        self.alive = True

    def execute(self, command, **kwargs):
        if command == 'fixture:inspect':
            output = 'Files: ' + ', '.join(p.name for p in self.path.glob('*.py'))
        elif command == 'fixture:test':
            output = json.dumps(FixtureTask.inspect(self.path))
        elif isinstance(command, str) and command.startswith('fixture:edit:'):
            _, _, identity, epoch = command.split(':')
            if not identity.isdigit() or not epoch.isdigit():
                raise ValueError('Fixture accepts numeric identities only')
            file = self.path / f'module_{int(identity):02}.py'
            file.write_text(f'def compute(value):\n    return value + {int(identity) + int(epoch)}\n')
            output = f'Wrote {file.name}; inspect and test before publishing.'
        else:
            raise ValueError('Fixture cannot execute arbitrary commands')
        return {'command': command, 'output': output, 'exit_code': 0, 'status': 'ok', 'seconds': 0}

    def put(self, *args, **kwargs):
        pass

    def snapshot(self, destination):
        copy_source(self.path, destination)

    def close(self):
        self.alive = False
        self.temp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class FixtureTask(Task):
    def sandbox(self, source, **kwargs):
        return FixtureBox(source)

    @staticmethod
    def inspect(source):
        files = list(Path(source).glob('module_*.py'))
        for p in files:
            ast.parse(p.read_text())
        # One synthetic "test" per expected module, in four groups of five, so the swarm demo has a task board.
        present = {p.name for p in files}
        failing = [f'fixture.group_{i // 5}::module_{i:02}' for i in range(20) if f'module_{i:02}.py' not in present]
        return {'build': True, 'status': 'scripted_fixture', 'score': len(files) / 20,
                'passed': len(files), 'total': 20, 'failing': failing,
                'note': 'Synthetic module coverage, not correctness'}

    def check(self, source, report_dir):
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        return self.inspect(source)


class FixtureClient:
    def generate(self, messages, max_tokens, phase, agent):
        o = json.loads(messages[-1]['content'])
        i, epoch = int(agent.split('-')[-1]), o['round']
        if phase == 'membership':
            # Deliberately exercises invitation, consent, exit and changing coalitions.
            target = f'agent-{i + 1:02}' if i % 2 == 0 and i < 19 else None
            reply = {'leave': epoch == 2 and i % 4 == 1, 'invite': target,
                     'message': 'Let us compare module behavior and inspect each other’s changes.'}
        elif phase == 'join':
            reply = {'invitation': o['invitations'][0]['id'], 'message': 'I accept; we can test the same branch.'}
        elif phase == 'claim':
            # Agent i can only ever flip module i, so it prefers the task group holding that module.
            # Odd agents join an existing group with room; even agents open a group when a slot is open.
            tasks = o['task_board']['tasks']
            mine = f'fixture.group_{(i // 5) % 4}'
            reply = {'task': None, 'join': None, 'approach': '', 'message': 'Resting: nothing fits this round.'}
            for t in sorted(tasks, key=lambda t: t['id'] != mine):
                room = [g['leader'] for g in t['groups'] if g['room']]
                if i % 2 and room:
                    reply = {'task': t['id'], 'join': room[0], 'approach': f'Add module {i:02} beside the leader.',
                             'message': 'Joining to share one workspace.'}
                elif t['open_slots'] > 0:
                    reply = {'task': t['id'], 'join': None, 'approach': f'Write module {i:02}.',
                             'message': 'Opening an independent group.'}
                elif room:
                    reply = {'task': t['id'], 'join': room[0], 'approach': f'Add module {i:02}.', 'message': 'Joining.'}
                else:
                    continue
                break
        elif phase == 'discuss':
            reply = {'message': f'I will inspect module {i:02}, check the interface, and record a reproducible change.'}
        elif phase == 'work':
            slot = o['slot']
            tool = ['inspect', 'edit', 'test', 'message'][slot % 4]
            command = f'fixture:edit:{i}:{epoch}' if tool == 'edit' else 'fixture:' + ('test' if tool == 'test' else 'inspect')
            reply = {'tool': tool, 'command': command,
                     'message': ['Inspecting current branch.', 'A small isolated implementation is ready.',
                                 'Checking the updated interface.', 'Review complete; ready to vote.'][slot % 4]}
        else:
            reply = {'yes': True, 'pledge': [.025, .05, .1][(i + epoch) % 3],
                     'authorize_fixed': True, 'message': 'The change is concrete and the local check passed.'}
        return json.dumps(reply)


class EvolutionFixtureClient(FixtureClient):
    """Scripted growth and departures, to demonstrate history and cap enforcement."""
    def generate(self, messages, max_tokens, phase, agent):
        o = json.loads(messages[-1]['content'])
        if phase != 'membership':
            return super().generate(messages, max_tokens, phase, agent)
        i, epoch, cap = int(agent.split('-')[-1]), o['round'], o['max_team_size']
        group, local = divmod(i, cap)
        target, leave = None, False
        # Two independent groups grow 2 -> 4 -> 8 -> 10 for k=20.
        if group < 2 and cap >= 2:
            if epoch < 4:
                size = min(2 ** epoch, cap)
                if local < size and local + size < cap:
                    target = group * cap + local + size
            elif epoch == 4:
                leave = local >= max(2, cap - 4)
            elif epoch == 5:
                first = max(2, cap - 4)
                if local >= first and (local - first) % 2 == 0 and local + 1 < cap:
                    target = i + 1
        return json.dumps({'leave': leave, 'invite': f'agent-{target:02}' if target is not None else None,
                           'message': 'Scripted team-history demonstration: grow, depart, then regroup.'})


class BaselineFixtureClient(FixtureClient):
    """Synthetic architecture demo. Never paid model or benchmark evidence."""
    def generate(self, messages, max_tokens, phase, agent):
        o = json.loads(messages[-1]['content'])
        if phase == 'orchestrate':
            return json.dumps({'message': 'Scripted complementary module assignments.', 'assignments': [
                {'agent': r['agent'], 'task': 'Implement your own numeric module.', 'inputs': []}
                for r in o['worker_reports']]})
        if phase == 'peer':
            return json.dumps({'message': 'Scripted peer report: inspect the numeric module interface.'})
        if phase == 'vote':
            return json.dumps({'choice': o['worker_results'][0]['agent'], 'message': 'Scripted equal vote.'})
        if phase == 'synthesis':
            rows = o['worker_results']
            chosen = rows[o['slot'] % len(rows)]['agent']
            command = 'fixture:edit:' + chosen.split('-')[-1] + ':' + str(o['round'])
            return json.dumps({'tool': 'edit', 'command': command, 'message': 'Scripted synthesis example.',
                               'args': [], 'stdin': '', 'files': [], 'capture': []})
        reply = json.loads(super().generate(messages, max_tokens, 'work', agent))
        return json.dumps({**reply, 'args': [], 'stdin': '', 'files': [], 'capture': []})


def create_task(root):
    root = Path(root)
    (root / 'seed').mkdir(parents=True)
    (root / 'SPEC.md').write_text('Scripted 20-agent fixture: build a collection of small numeric modules.\n')
    (root / 'seed/PROJECT_NOTES.md').write_text('# Public notes\nSynthetic fixture; no benchmark claim.\n')
    (root / 'seed/.gitignore').write_text('__pycache__/\n')
    data = {'benchmark': 'fixture', 'id': 'scripted-demo', 'test_visibility': 'synthetic fixture', 'image': None}
    (root / 'task.json').write_text(json.dumps(data, indent=2))
    return FixtureTask(root, data)
