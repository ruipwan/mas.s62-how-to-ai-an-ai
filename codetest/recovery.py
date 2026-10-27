"""Historical-action recovery for interrupted classic Commit0/fixture episodes.

Saved normalized decisions and observed tool results are replayed without model
calls. This reconstructs observed logical state, not a proof of identical original
model inputs: request/observation hashes and active-workspace snapshots were not
recorded by the original harness. A live client is constructed only after the
entire historical prefix and saved state pass validation.
"""
import copy
from dataclasses import fields
import json
from pathlib import Path
import shutil
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'vendor') not in sys.path:
    sys.path.insert(0, str(ROOT / 'vendor'))
from .benchmarks import Task, junit
from .config import Config, BASELINES
from .engine import Engine, tree_digest
from .fixture import FixtureTask
from .io import Events, sha, write_json
from .policy import Policy
from .report import replay
from .sandbox import Repository, copy_source


class RecoveryDivergence(RuntimeError):
    pass


class RecoveryReady(Exception):
    """Offline validation has reached the first missing action; do not go live."""


def _load(path):
    return json.loads(Path(path).read_text())


def _repo(path):
    repo = Repository.__new__(Repository)
    repo.path = Path(path)
    return repo


def _fingerprint(root):
    files = {}
    for path in sorted(root.rglob('*')):
        if '.git' in path.relative_to(root).parts:
            continue
        if path.is_symlink():
            raise RecoveryDivergence('Symlink in historical run')
        if path.is_file():
            files[str(path.relative_to(root))] = sha(path)
    repo = _repo(root / 'public')
    return {'files': files, 'head': repo.version, 'refs': repo.git('show-ref'),
            'status': repo.git('--no-optional-locks', 'status', '--porcelain')}


def _logical(row):
    result = {k: v for k, v in row.items() if k not in {'seq', 'elapsed'}}
    # Workers were a Python set before participants were serialized. Order has no
    # effect on allocation; publication participants are already sorted by Engine.
    if 'participants' in result:
        result['participants'] = sorted(result['participants'])
    return result


class ReplayEvents(Events):
    def __init__(self, out, expected):
        super().__init__(out)
        self.expected, self.cursor, self.replaying = expected, 0, True

    def emit(self, kind, **values):
        row = {'round': self.round, 'event': kind, **values}
        if self.replaying and not kind.startswith('api_'):
            if self.cursor >= len(self.expected):
                raise RecoveryDivergence('Extra historical event: ' + kind)
            expected = self.expected[self.cursor]
            if _logical(row) != _logical(expected):
                raise RecoveryDivergence(f'Historical event mismatch at original seq {expected["seq"]}: {kind}')
            self.cursor += 1
            row.update(historical_reference=True, original_seq=expected['seq'], original_elapsed=expected['elapsed'])
        self.append(row)


class ReplayClient:
    def __init__(self, session):
        self.session, self.cursor, self.live = session, 0, None
        self.last_completion = {}
        self.last_decision = None

    def generate(self, messages, max_tokens, phase, agent):
        session = self.session
        observation = json.loads(messages[-1]['content'])
        identity = {'round': observation['round'], 'phase': phase, 'agent': agent, 'max_tokens': max_tokens}
        if self.cursor < len(session.decisions):
            record = session.decisions[self.cursor]
            if any(record.get(k) != v for k, v in identity.items()):
                raise RecoveryDivergence('Historical call identity or response limit mismatch')
            reserved, charged = session.call_pairs[self.cursor]
            for e in (reserved, charged):
                session.events.emit(e['event'], **{k: v for k, v in e.items() if k not in {'event', 'seq', 'elapsed'}},
                                    historical_reference=True, original_seq=e['seq'], original_elapsed=e['elapsed'])
            self.last_completion = {k: charged.get(k) for k in ('call_id', 'finish_reason', 'usage', 'id', 'model', 'provider')}
            self.last_decision = record
            value = copy.deepcopy(record['response'])
            if phase == 'work' and session.config.structured_actions and isinstance(value.get('files'), dict):
                value['files'] = [{'path': k, 'content': v} for k, v in value['files'].items()]
            self.cursor += 1
            return json.dumps(value, ensure_ascii=False)
        if self.live is None:
            session.verify_boundary(identity)
            session.events.replaying = False
            session.events.emit('recovery_verified', source_run=str(session.source),
                                replayed_calls=self.cursor, next_call=identity,
                                note='Historical actions and observed state verified; original request hashes were not recorded.')
            write_json(session.out / 'recovery-status.json', {'status': 'verified', 'report': session.report})
            context = {'out': str(session.out), 'source_run': str(session.source), 'cfg': session.config,
                       'event': session.events.emit, 'next_call': identity, 'replay_report': copy.deepcopy(session.report)}
            self.live = session.factory(context)
            if self.live is None:
                raise ValueError('Live client factory returned no client')
        raw = self.live.generate(messages, max_tokens, phase, agent)
        self.last_completion = getattr(self.live, 'last_completion', {})
        return raw


class ReplayBox:
    def __init__(self, session, source):
        self.session = session
        ident = session.active_candidate
        self.saved = session.candidates.get(ident) if session.events.replaying else None
        self.box = None if self.saved else session.task.sandbox(source)
        self.alive = True

    def put(self, *args, **kwargs):
        if self.box:
            return self.box.put(*args, **kwargs)

    def execute(self, command, **kwargs):
        session = self.session
        if not session.events.replaying:
            if self.box is None:
                raise RecoveryDivergence('Completed historical sandbox unexpectedly reached live execution')
            return self.box.execute(command, **kwargs)
        decision = session.client.last_decision
        tool = session.tool_by_decision.get(decision['seq'])
        if tool is None or decision['response'].get('command') != command:
            raise RecoveryDivergence('Historical command has no matching recorded tool result')
        expected = tool['result']
        if self.box:
            actual = self.box.execute(command, **kwargs)
            # Timing alone is excluded. Different output, exit status, truncation or
            # command behavior is a hard barrier before the first live model call.
            comparable = lambda r: {k: v for k, v in r.items() if k != 'seconds'}
            if comparable(actual) != comparable(expected):
                raise RecoveryDivergence('Active workspace command output differs from historical result')
            session.active_commands_reexecuted += 1
        return copy.deepcopy(expected)

    def snapshot(self, destination):
        if self.saved:
            copy_source(self.saved['_directory'] / 'source', destination)
        else:
            self.box.snapshot(destination)

    def close(self):
        if self.box:
            self.box.close()
        self.alive = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class ReplayTask:
    def __init__(self, session):
        self.session = session

    def __getattr__(self, name):
        return getattr(self.session.task, name)

    def sandbox(self, source, **kwargs):
        return ReplayBox(self.session, source)

    def check(self, source, report_dir):
        session = self.session
        if not session.events.replaying:
            return session.task.check(source, report_dir)
        report_dir = Path(report_dir)
        if report_dir == session.out / 'initial-check':
            shutil.copytree(session.source / 'initial-check', report_dir)
            return copy.deepcopy(session.initial_check)
        candidate = session.candidates.get(session.active_candidate)
        if candidate is not None and report_dir.parent == session.out / candidate['_directory'].relative_to(session.source):
            if tree_digest(source) != tree_digest(candidate['_directory'] / 'tracked-source'):
                raise RecoveryDivergence('Completed candidate source reconstruction differs')
            shutil.copytree(candidate['_directory'] / 'check', report_dir)
            session.verified_candidates.add(session.active_candidate)
            return copy.deepcopy(candidate['check'])
        raise RecoveryDivergence('Unrecorded verification requested during historical replay')


class RecoverySession:
    def __init__(self, source, out, factory):
        self.source, self.out, self.factory = source, out, factory
        self.original = _load(source / 'manifest.json')
        self.saved_state = _load(source / 'state.json')
        if self.original.get('status') not in {'interrupted', 'budget_stopped'}:
            raise ValueError('Recovery requires a stopped, incomplete source run')
        rows = [json.loads(line) for line in (source / 'events.jsonl').read_text().splitlines() if line.strip()]
        stopped = [e for e in rows if e['event'] == 'api_stopped']
        if len(stopped) != 1:
            raise ValueError('This recovery version requires exactly one interrupted model request')
        failed_id = stopped[0]['call_id']
        failure = next(e for e in rows if e['event'] == 'api_reserved' and e['call_id'] == failed_id)
        if any(e['event'] == 'api_charged' and e['call_id'] == failed_id for e in rows):
            raise ValueError('Failed request already charged; missing-response recovery needs separate review')
        prefix = [e for e in rows if e['seq'] < failure['seq']]
        if any(e['event'] in {'invalid_response', 'response_normalized', 'response_tokens_extended', 'rounds_extended'} for e in prefix):
            raise ValueError('Recovery of invalid, normalized-fence or adaptive-extension histories is not implemented')
        self.failed_request = failure
        self.decisions = [e for e in prefix if e['event'] == 'decision']
        charged = [e for e in prefix if e['event'] == 'api_charged']
        reservations = {e['call_id']: e for e in prefix if e['event'] == 'api_reserved'}
        if len(charged) != len(self.decisions):
            raise ValueError('Historical successful requests and decisions are not one-to-one')
        self.call_pairs = []
        for c, decision in zip(charged, self.decisions):
            if any(c.get(k) != decision.get(k) for k in ('round', 'phase', 'agent', 'max_tokens')) or c['seq'] >= decision['seq']:
                raise ValueError('Historical response pairing is inconsistent')
            self.call_pairs.append((reservations[c['call_id']], c))
        names = {f.name for f in fields(Config)}
        self.config = Config(**{k: v for k, v in self.original['config'].items() if k in names})
        if self.config.condition in BASELINES or self.config.condition == 'swarm' or self.config.contribution_contracts or self.config.merge_slots != 1:
            raise ValueError('Recovery currently supports classic single-publication episodes only')
        self.task = Task.load(self.original['task_path'])
        if self.task.data['benchmark'] == 'fixture':
            self.task = FixtureTask(self.task.root, self.task.data)
        elif self.task.data['benchmark'] != 'commit0':
            raise ValueError('Recovery supports Commit0 and scripted fixtures only')
        if sha(self.task.root / 'task.json') != self.original['task_manifest_sha256'] or tree_digest(self.task.root) != self.original['task_input_sha256']:
            raise RecoveryDivergence('Task inputs changed since the source run')
        critical = ('engine.py', 'policy.py', 'actions.py', 'scratchpad.py', 'sandbox.py', 'benchmarks.py', 'config.py', 'fixture.py')
        for name in critical:
            rel = 'codetest/' + name
            expected = self.original['source_hashes'].get(rel)
            if expected != sha(ROOT / rel) or expected != sha(source / 'source-snapshot' / rel):
                raise RecoveryDivergence('Frozen inference implementation mismatch: ' + name)
        if self.original['team_manager_sha256'] != sha(ROOT / 'vendor/hayekmas/adapters/teams/team.py'):
            raise RecoveryDivergence('TeamManager implementation changed')
        initial = next(e for e in prefix if e['event'] == 'snapshot' and e['round'] == -1)
        self.initial_commit = initial['commit']
        self.initial_check = copy.deepcopy(initial['feedback'])
        if self.task.data['benchmark'] == 'fixture':
            self.initial_check = self.task.inspect(self.task.seed)
        elif self.initial_check.get('score') is not None:
            parsed = junit((source / 'initial-check/junit.xml').read_text())
            self.initial_check['failing'] = parsed['failing']
        self.candidates, self.verified_candidates = {}, set()
        for event in (e for e in prefix if e['event'] == 'candidate'):
            ident = (event['round'], event['team'])
            directory = source / 'candidates' / f'round-{event["round"]:02}' / event['team'].replace(':', '-')
            c = _load(directory / 'candidate.json')
            actual = {k: v for k, v in c.items() if k not in {'source', 'check'}}
            actual['check'] = {k: v for k, v in c['check'].items() if k != 'failing'}
            expected = {k: v for k, v in event.items() if k not in {'seq', 'elapsed', 'event', 'round'}}
            if _logical(actual) != _logical(expected):
                raise RecoveryDivergence('Saved candidate differs from its recorded event')
            self.candidates[ident] = {**c, '_directory': directory}
        self.publications = [e for e in prefix if e['event'] == 'publication']
        self.publication_cursor = 0
        self.tool_by_decision = {}
        latest = None
        for event in prefix:
            if event['event'] == 'decision':
                latest = event
            elif event['event'] == 'tool':
                if latest is None or latest['phase'] != 'work' or latest['agent'] != event['agent']:
                    raise ValueError('Tool has no preceding work decision')
                self.tool_by_decision[latest['seq']] = event
        self.guard = _fingerprint(source)
        self.events = ReplayEvents(out, [e for e in prefix if not e['event'].startswith('api_')])
        self.client = ReplayClient(self)
        self.policy = Policy(self.client, self.config, self.events)
        self.engine = Engine(self.config, ReplayTask(self), self.policy, self.events, out)
        # Replace only the new output's bootstrap checkout, preserving historical Git IDs.
        shutil.rmtree(out / 'public')
        _repo(source / 'public').git('clone', '--no-hardlinks', '--quiet', str(source / 'public'), str(out / 'public'))
        self.engine.repo = _repo(out / 'public')
        self.engine.repo.git('config', 'user.name', 'Code Testing recovery')
        self.engine.repo.git('config', 'user.email', 'local@code-testing.invalid')
        self.engine.repo.git('reset', '--hard', '--quiet', self.initial_commit)
        self.active_candidate = None
        self.active_commands_reexecuted = 0
        self.report = {}
        original_candidate = self.engine.candidate
        def candidate(engine, team):
            self.active_candidate = (engine.step, team.team_id)
            try:
                return original_candidate(team)
            finally:
                self.active_candidate = None
        self.engine.candidate = types.MethodType(candidate, self.engine)
        original_publish = self.engine.publish
        def publish(engine, winner, patch, check, patch_name, **extra):
            if self.events.replaying:
                if self.publication_cursor >= len(self.publications):
                    raise RecoveryDivergence('Unexpected historical publication')
                expected = self.publications[self.publication_cursor]
                expected_tree = _repo(source / 'public').git('rev-parse', expected['commit'] + '^{tree}')
                if engine.repo.git('rev-parse', 'HEAD^{tree}') != expected_tree:
                    raise RecoveryDivergence('Historical publication tree differs')
                if patch != (source / expected['patch']).read_text():
                    raise RecoveryDivergence('Historical publication patch differs')
                engine.repo.git('reset', '--hard', '--quiet', expected['commit'])
                self.publication_cursor += 1
            return original_publish(winner, patch, check, patch_name, **extra)
        self.engine.publish = types.MethodType(publish, self.engine)

    def verify_boundary(self, identity):
        if any(self.failed_request.get(k) != v for k, v in identity.items()):
            raise RecoveryDivergence('First missing call identity differs from the interrupted request')
        if self.events.cursor != len(self.events.expected) or self.verified_candidates != set(self.candidates) or self.publication_cursor != len(self.publications):
            raise RecoveryDivergence('Historical prefix not completely reconstructed')
        if self.engine.scratchpad and _load(self.out / 'scratchpad/index.json') != _load(self.source / 'scratchpad/index.json'):
            raise RecoveryDivergence('Scratchpad differs at recovery boundary')
        current = self.engine.state()
        for key in ('agents', 'history', 'snapshots', 'public_status_board', 'response_tokens', 'shared_scratchpad',
                    'completed_rounds', 'round_limit', 'checkpoints', 'burned', 'progress_dividends',
                    'terminal_issued', 'public_feedback', 'probe_count', 'model_calls'):
            if current.get(key) != self.saved_state.get(key):
                raise RecoveryDivergence('Saved partial state mismatch: ' + key)
        if self.engine.repo.version != self.guard['head'] or _fingerprint(self.source) != self.guard:
            raise RecoveryDivergence('Historical repository or source files changed')
        self.report = {'replayed_successful_calls': self.client.cursor,
                       'replayed_tool_results': len(self.tool_by_decision),
                       'verified_candidates': len(self.verified_candidates),
                       'verified_publications': self.publication_cursor,
                       'active_commands_reexecuted': self.active_commands_reexecuted,
                       'failed_request': self.failed_request, 'next_call': identity,
                       'source_manifest_sha256': sha(self.source / 'manifest.json'),
                       'source_events_sha256': sha(self.source / 'events.jsonl'),
                       'source_state_sha256': sha(self.source / 'state.json'),
                       'scope': 'Historical-action replay; no original request hashes or active-workspace checkpoint existed.'}
        write_json(self.out / 'recovery-report.json', self.report)


def run_recovery(source_run, out, live_client_factory, *, usage_callback=None, recovery_metadata=None):
    """Run free historical replay, validate, then invoke the caller-owned live factory.

    factory(context) receives cfg, event, out, source_run, next_call, replay_report.
    It owns credentials, transport settings and any new ledger registration. This
    module never registers, retains, settles, or resets campaign-ledger charges.
    """
    source, out = Path(source_run).resolve(), Path(out).resolve()
    if out == source or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError('Recovery output must be separate from its immutable source')
    out.mkdir(parents=True, exist_ok=False)
    session, failure, validated = None, None, False
    manifest = {'status': 'reconstructing', 'started_at': time.time(), 'recovery': {
        'source_run': str(source), 'method': 'historical-action replay', 'metadata': recovery_metadata or {}}}
    write_json(out / 'manifest.json', manifest)
    try:
        session = RecoverySession(source, out, live_client_factory)
        original = session.original
        manifest = {**original, **manifest, 'source_hashes': {
            str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'codetest').glob('*')) if p.is_file()}}
        manifest.pop('ledger_run', None)
        manifest.pop('campaign_ledger', None)
        manifest.pop('error', None)
        manifest.pop('finished_at', None)
        for rel in manifest['source_hashes']:
            target = out / 'source-snapshot' / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, target)
        team = out / 'source-snapshot/vendor/hayekmas/adapters/teams/team.py'
        team.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'vendor/hayekmas/adapters/teams/team.py', team)
        write_json(out / 'manifest.json', manifest)
        session.engine.run()
        manifest['status'] = 'completed'
    except RecoveryReady:
        validated = True
        manifest['status'] = 'validated'
    except BaseException as error:
        failure = error
        manifest['status'] = 'interrupted' if session and session.client.live else 'recovery_failed'
        manifest['error'] = type(error).__name__ + ': ' + str(error)[:1500]
    finally:
        if session:
            session.events.replaying = False
            try:
                session.engine.finalize()
            except BaseException as error:
                failure = failure or error
                manifest['status'] = 'finalization_failed'
                manifest['finalization_error'] = type(error).__name__ + ': ' + str(error)[:1000]
            manifest['recovery']['report'] = session.report
        manifest['finished_at'] = time.time()
        manifest['usage'] = usage_callback() if usage_callback else {'note': 'Ledger accounting is owned by the caller; historical calls are references only.'}
        write_json(out / 'manifest.json', manifest)
        write_json(out / 'recovery-status.json', {'status': manifest['status'], 'error': manifest.get('error'),
                                                'report': session.report if session else {}})
        if (out / 'state.json').exists():
            replay(out)
    result = {'status': manifest['status'], 'out': str(out), 'usage': manifest['usage'], 'replay': str(out / 'replay.html')}
    if failure:
        raise failure
    return result


def validate_recovery(source_run, out):
    def stop_before_live(context):
        raise RecoveryReady()
    return run_recovery(source_run, out, stop_before_live)
