"""Explicit preparation, inference and final evaluation commands."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import uuid
import copy
from contextlib import nullcontext

from .benchmarks import Task, prepare
from .budget import Ledger, OpenRouter, BudgetStop
from .config import Config, BASELINES
from .engine import Engine, terminal_credits, tree_digest
from .fixture import create_task, FixtureClient, EvolutionFixtureClient, BaselineFixtureClient
from .io import Events, write_json, sha
from .policy import Policy, prompt_revision
from .report import replay
from .sandbox import run
from .analysis import analyze

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'artifacts/campaign.sqlite'


def config(path=None):
    data = json.loads(Path(path).read_text()) if path else {}
    recorded_cap = data.pop('max_team_size', None)
    cfg = Config(**data)
    if recorded_cap is not None and recorded_cap != cfg.max_team_size:
        raise ValueError('max_team_size is derived from k as max(1, floor(k/2)); remove the stale override')
    return cfg


def preflight(task=None, paid=False):
    checks = {'python': sys.version.split()[0], 'host': platform.platform(),
              'git': bool(shutil.which('git')), 'docker_cli': bool(shutil.which('docker')),
              'api_key_present': bool(os.environ.get('OPENROUTER_API_KEY'))}
    if checks['docker_cli']:
        try:
            info = run(['docker', 'info', '--format', '{{json .}}'], timeout=15)
            data = json.loads(info.stdout)
            checks['docker_server'] = data.get('OSType', '') + '/' + data.get('Architecture', '')
        except (RuntimeError, subprocess.TimeoutExpired, ValueError) as e:
            checks['docker_server_error'] = str(e)
    if task and checks.get('docker_server'):
        image = run(['docker', 'image', 'inspect', task.data['image']], check=False)
        checks['image_present'] = image.returncode == 0
        if checks['image_present']:
            checks['image_id'] = json.loads(image.stdout)[0]['Id']
            try:
                with task.sandbox(task.seed) as box:
                    probe = box.execute('pwd && test ! -d .git && test -r /task/SPEC.md')
                    checks['sandbox_ready'] = probe['exit_code'] == 0
            except (RuntimeError, ValueError) as e:
                checks['sandbox_error'] = str(e)
    checks['ready'] = bool(checks['git'] and checks.get('docker_server') and (not task or checks.get('sandbox_ready'))
                           and (not paid or checks['api_key_present']))
    return checks


def execute(cfg, task, out, *, demo=False, team_evolution=False):
    baseline = cfg.condition in BASELINES
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    if demo:
        task = create_task(out / 'fixture-task')
        checks = {'ready': True, 'execution': 'scripted fixture, no shell or network'}
    else:
        if baseline:
            task = copy.copy(task)
            task.sandbox_memory = str(cfg.baseline_memory_mb) + 'm'
            task.sandbox_cpus = '1'
        if baseline:
            from .independent import SandboxSlots
            preflight_slot = SandboxSlots(1, ROOT / 'artifacts/baseline-sandbox-leases')
        else:
            preflight_slot = nullcontext()
        with preflight_slot:
            checks = preflight(task, paid=True)
        write_json(out / 'preflight.json', checks)
        if not checks['ready']:
            raise ValueError('Preflight failed. See ' + str(out / 'preflight.json') + '; no paid calls made.')
    events = Events(out)
    manifest = {'version': 1, 'backend': 'scripted_fixture' if demo else 'openrouter',
                'prompt_revision': prompt_revision(cfg),
                'status': 'running', 'started_at': time.time(), 'task': task.data,
                'task_path': str(task.root / 'task.json'),
                'task_manifest_sha256': sha(task.root / 'task.json'), 'config': cfg.as_dict(),
                'task_input_sha256': tree_digest(task.root),
                'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'codetest').glob('*')) if p.is_file()},
                'vendor_manifest_sha256': sha(ROOT / 'vendor/SOURCE_MANIFEST.json'),
                'team_manager_sha256': sha(ROOT / 'vendor/hayekmas/adapters/teams/team.py'),
                'preflight': checks, 'official_score': None,
                'note': 'Fresh population. Control overhead is measured separately from equal private work allocations.'}
    if baseline:
        manifest['architecture'] = cfg.condition
        manifest['worker_count'] = cfg.worker_count
        manifest['note'] = 'No market. Work, peer discussion, orchestration, voting and synthesis are recorded separately. Baseline sources are isolated from market experiments.'
    elif cfg.condition == 'swarm':
        manifest['architecture'] = 'swarm'
        manifest['note'] = ('Task board from failing visible tests, one claim call per active agent, verified merge '
                            'queue, bounties only for test flips. Control overhead is measured separately from equal '
                            'private work allocations.')
    write_json(out / 'manifest.json', manifest)
    # Keep the exact implementation alongside its hashes as the harness evolves.
    for relative, expected in manifest['source_hashes'].items():
        source = ROOT / relative
        target = out / 'source-snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha(target) != expected:
            raise RuntimeError('Source changed while creating the run snapshot')
    team_source = out / 'source-snapshot/vendor/hayekmas/adapters/teams/team.py'
    team_source.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / 'vendor/hayekmas/adapters/teams/team.py', team_source)
    ledger = None
    if demo:
        client = BaselineFixtureClient() if baseline else EvolutionFixtureClient() if team_evolution else FixtureClient()
        if team_evolution:
            manifest['demo_scenario'] = 'Scripted growth, departures, and regrouping; not model behavior'
    else:
        ledger = Ledger(LEDGER, parallel=cfg.parallel_experiments,
                        parallel_requests=cfg.independent_concurrency if baseline else None)
        ident = uuid.uuid4().hex
        ledger.register(ident, cfg.run_cap_usd)
        manifest['ledger_run'] = ident
        manifest['campaign_ledger'] = str(LEDGER)
        client = OpenRouter(cfg.model, ledger, ident,
                            {'prompt': cfg.input_price_cap, 'completion': cfg.output_price_cap}, events.emit, out / 'STOP',
                            reasoning_enabled=cfg.reasoning_enabled, reasoning_effort=cfg.reasoning_effort,
                            provider_ignore=cfg.provider_ignore, structured_actions=cfg.structured_actions,
                            shared_scratchpad=cfg.shared_scratchpad,
                            contribution_contracts=cfg.contribution_contracts)
        write_json(out / 'manifest.json', manifest)
    policy = Policy(client, cfg, events)
    engine, failure = None, None
    try:
        if baseline:
            from .independent import IndependentEngine, IndependentPolicy
            def policy_factory(name):
                own_client = BaselineFixtureClient() if demo else OpenRouter(
                    cfg.model, ledger, ident, {'prompt': cfg.input_price_cap, 'completion': cfg.output_price_cap},
                    events.emit, out / 'STOP', reasoning_enabled=cfg.reasoning_enabled,
                    reasoning_effort=cfg.reasoning_effort, provider_ignore=cfg.provider_ignore,
                    structured_actions=cfg.structured_actions)
                return IndependentPolicy(own_client, cfg, events)
            engine = IndependentEngine(cfg, task, policy_factory, events, out,
                                       pool_root=None if demo else ROOT / 'artifacts/baseline-sandbox-leases')
        elif cfg.condition == 'swarm':
            from .swarm import SwarmEngine
            engine = SwarmEngine(cfg, task, policy, events, out)
        else:
            engine = Engine(cfg, task, policy, events, out)
        engine.run()
        manifest['status'] = 'completed'
    except BaseException as e:
        failure = e
        manifest['status'] = 'budget_stopped' if isinstance(e, BudgetStop) else 'interrupted'
        manifest['error'] = type(e).__name__ + ': ' + str(e)
        events.emit('stopped', reason=manifest['error'])
    finally:
        if engine:
            try:
                engine.finalize()
            except BaseException as e:
                manifest['finalization_error'] = type(e).__name__ + ': ' + str(e)
                manifest['status'] = 'finalization_failed'
                failure = failure or e
        manifest['finished_at'] = time.time()
        manifest['usage'] = ledger.summary(ident) if ledger else {'reported_usd': 0, 'accounted_usd': 0, 'calls': 0}
        if ledger:
            ledger.close()
        write_json(out / 'manifest.json', manifest)
        if (out / 'state.json').exists():
            replay(out)
        analyze(out)
    result = {'out': str(out), 'status': manifest['status'], 'usage': manifest['usage'],
              'replay': str(out / 'replay.html')}
    print(json.dumps(result, indent=2))
    if failure:
        raise failure
    return result


def build(task):
    if not preflight()['ready']:
        raise ValueError('Start Docker first; build made no changes')
    dockerfile = task.root / 'Dockerfile'
    if dockerfile.exists():
        argv = ['docker', 'build', '--platform', 'linux/amd64', '-t', task.data['image'], str(task.root)]
    else:
        argv = ['docker', 'pull', '--platform', 'linux/amd64', task.data['image']]
    subprocess.run(argv, check=True)
    # Pin the resolved image content for subsequent inference.
    info = json.loads(run(['docker', 'image', 'inspect', task.data['image']]).stdout)[0]
    task.data['image_tag'] = task.data['image']
    task.data['image'] = info['Id']
    write_json(task.root / 'task.json', task.data)
    return {'image': info['Id'], 'tag': task.data['image_tag']}


def evaluate(path, offline_runtime=None):
    root = Path(path).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['status'] != 'completed' or manifest['backend'] != 'openrouter':
        raise ValueError('Only a completed paid inference run can receive a benchmark score')
    destination = root / 'evaluation.json'
    if destination.exists():
        raise FileExistsError('Evaluation already recorded; preserve this artifact')
    task = Task.load(manifest['task_path'])
    if sha(task.root / 'task.json') != manifest['task_manifest_sha256'] or tree_digest(task.root) != manifest['task_input_sha256']:
        raise ValueError('Prepared task inputs changed after inference; restore the recorded inputs before evaluating')
    if offline_runtime:
        if task.data['benchmark'] != 'nl2repo':
            raise ValueError('--offline-runtime is only supported by the NL2Repo final evaluator')
        task.data = {**task.data, 'evaluation_offline_runtime': offline_runtime}
    started = time.time()
    try:
        result = task.final(root / 'submission')
    except Exception as error:
        # A grading failure is evidence, not an absent result file or zero score.
        result = {'status': 'evaluator_exception', 'score': None,
                  'error': type(error).__name__ + ': ' + str(error)}
    result['evaluation_seconds'] = round(time.time() - started, 3)
    result['evaluator_source_hashes'] = {
        str(p.relative_to(ROOT)): sha(p)
        for p in sorted((ROOT / 'codetest').glob('*.py')) if p.is_file()}
    state = json.loads((root / 'state.json').read_text())
    # A research-only overlay score is never promoted to an official result.
    credits = terminal_credits(state, result.get('score'))
    write_json(destination, {**result, 'terminal_credits': credits,
                            'credit_note': ('No rewards in this baseline.' if state['config'].get('condition') in BASELINES
                                            else 'Post-hoc fixed reward pool; no inference or future-episode effect.')})
    manifest['evaluation'] = result
    write_json(root / 'manifest.json', manifest)
    replay(root)
    analyze(root)
    return result


def main():
    parser = argparse.ArgumentParser(description='Code Testing v1 — k=20 repository coalitions')
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('plan'); p.add_argument('--config')
    p = commands.add_parser('prepare'); p.add_argument('benchmark', choices=['programbench', 'commit0', 'nl2repo']); p.add_argument('task_id'); p.add_argument('--out', required=True)
    p = commands.add_parser('preflight'); p.add_argument('--task'); p.add_argument('--paid', action='store_true')
    p = commands.add_parser('build'); p.add_argument('--task', required=True)
    for name in ('run', 'demo'):
        p = commands.add_parser(name); p.add_argument('--config'); p.add_argument('--out', required=True)
        if name == 'run':
            p.add_argument('--task', required=True)
        else:
            p.add_argument('--team-evolution', action='store_true', help='Scripted team growth, departures and regrouping')
    p = commands.add_parser('evaluate'); p.add_argument('directory')
    p.add_argument('--offline-runtime', help='NL2Repo final-only image containing upstream tests and preinstalled build dependencies')
    p = commands.add_parser('replay'); p.add_argument('directory')
    commands.add_parser('budget')
    p = commands.add_parser('analyze'); p.add_argument('directory')
    p = commands.add_parser('budget-retain'); p.add_argument('call_id'); p.add_argument('--reason', required=True)
    p = commands.add_parser('serve'); p.add_argument('directory'); p.add_argument('--port', type=int, default=8795)
    args = parser.parse_args()
    try:
        if args.command == 'plan':
            cfg = config(args.config); result = {'config': cfg.as_dict(), 'allocation': cfg.estimate(), 'campaign_cap_usd': Ledger.LIMIT, 'ledger': str(LEDGER)}
        elif args.command == 'prepare':
            result = prepare(args.benchmark, args.task_id, args.out)
        elif args.command == 'preflight':
            result = preflight(Task.load(args.task) if args.task else None, args.paid)
        elif args.command == 'build':
            result = build(Task.load(args.task))
        elif args.command in {'run', 'demo'}:
            cfg = config(args.config)
            if args.command == 'demo' and not args.config:
                cfg.rounds, cfg.work_calls = 4, 4
                if args.team_evolution:
                    cfg.rounds, cfg.membership_every = 6, 1
                cfg.max_rounds = cfg.rounds
            task = Task.load(args.task) if args.command == 'run' else None
            return execute(cfg, task, args.out, demo=args.command == 'demo',
                           team_evolution=getattr(args, 'team_evolution', False))
        elif args.command == 'evaluate':
            result = evaluate(args.directory, args.offline_runtime)
        elif args.command == 'replay':
            result = {'replay': str(replay(args.directory))}
        elif args.command == 'budget':
            result = Ledger(LEDGER).summary()
        elif args.command == 'budget-retain':
            result = Ledger(LEDGER).retain_failed_reservation(args.call_id, args.reason)
        elif args.command == 'analyze':
            report = analyze(args.directory)
            result = {'analysis': str(Path(args.directory).resolve() / 'analysis.md'), 'status': report['status']}
        else:
            from functools import partial
            from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
            handler = partial(SimpleHTTPRequestHandler, directory=str(Path(args.directory).resolve()))
            print(f'http://127.0.0.1:{args.port}/replay.html', flush=True)
            ThreadingHTTPServer(('127.0.0.1', args.port), handler).serve_forever()
            return
        print(json.dumps(result, indent=2))
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as e:
        parser.exit(2, str(e) + '\n')
