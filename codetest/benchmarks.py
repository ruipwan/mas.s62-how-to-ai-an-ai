"""Benchmark-specific preparation; solver observations contain public inputs only."""
from dataclasses import dataclass
import io
import json
from pathlib import Path
import re
import shlex
import shutil
import sys
import tarfile
import urllib.request
import xml.etree.ElementTree as ET

from .io import write_json, sha
from .sandbox import DockerSandbox, copy_source, run, safe_extract

PINS = {
    'programbench': ('facebookresearch/ProgramBench', '27f02157c785f8da3647aa6dbbe6b9137f99f10e'),
    'nl2repo': ('multimodal-art-projection/NL2RepoBench', '781a1da1ee41fb8edb0bed22f586d69111610edf'),
}


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def junit(text):
    root = ET.fromstring(text)
    cases = list(root.iter('testcase'))
    # Fail closed on collection errors or duplicate identities; no tests != success.
    identities = [(c.get('classname'), c.get('name')) for c in cases]
    if not cases or len(set(identities)) != len(cases):
        return {'status': 'invalid_test_report', 'score': None, 'passed': 0, 'total': len(cases)}
    errors = sum(c.find('error') is not None for c in cases)
    ok = [not any(c.find(k) is not None for k in ('failure', 'error', 'skipped')) for c in cases]
    passed = sum(ok)
    # Per-test identities let the swarm condition turn failing tests into open tasks.
    failing = sorted(f'{c.get("classname") or "?"}::{c.get("name") or "?"}' for c, good in zip(cases, ok) if not good)
    return {'status': 'collection_error' if errors else 'scored', 'score': None if errors else passed / len(cases),
            'passed': passed, 'total': len(cases), 'errors': errors, 'failing': failing}


def overlay_source(reference, source, destination, protected):
    """Audit actual replacements; a protected submitted filename is not a replacement."""
    reference, source, destination = map(Path, (reference, source, destination))
    copy_source(reference, destination)
    applied, ignored = [], []
    for path in sorted(source.rglob('*')):
        rel = path.relative_to(source)
        if path.is_symlink():
            raise ValueError(f'Symlink not supported in evaluation overlay: {rel}')
        if not path.is_file():
            continue
        if rel.name in protected or any(str(rel) == x or str(rel).startswith(x.rstrip('/') + '/') for x in protected):
            ignored.append(str(rel))
            continue
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        applied.append(str(rel))
    retained = sorted(str(p.relative_to(reference)) for p in reference.rglob('*')
                      if p.is_file() and str(p.relative_to(reference)) not in applied)
    return {'applied_source_files': applied, 'ignored_protected_submissions': ignored,
            'retained_reference_files': retained}


@dataclass
class Task:
    root: Path
    data: dict
    sandbox_memory: str = '2g'
    sandbox_cpus: str = '2'

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        data = json.loads(path.read_text())
        if data['benchmark'] not in {'programbench', 'commit0', 'nl2repo', 'fixture'}:
            raise ValueError('Unknown benchmark')
        if data['id'] in {'.', '..'} or not re.fullmatch(r'[A-Za-z0-9_.-]+', data['id']):
            raise ValueError('Task id contains unsafe characters')
        for name in data.get('protected_paths', []):
            if Path(name).is_absolute() or '..' in Path(name).parts or name in {'', '.'}:
                raise ValueError('Unsafe protected path')
        return cls(path.parent, data)

    @property
    def seed(self):
        return self.root / 'seed'

    def public(self):
        return {'id': self.data['id'], 'benchmark': self.data['benchmark'],
                'problem': (self.root / 'SPEC.md').read_text(),
                'workspace': '/workspace', 'specification_file': '/task/SPEC.md',
                'reference': '/oracle/executable' if self.data['benchmark'] == 'programbench' else None,
                'test_visibility': self.data['test_visibility']}

    def sandbox(self, source, *, reference=True):
        box = DockerSandbox(self.data['image'], programbench=reference and self.data['benchmark'] == 'programbench',
                            timeout=self.data.get('command_timeout', 60), memory=self.sandbox_memory,
                            cpus=self.sandbox_cpus)
        try:
            box.load(source)
            box.put(self.root / 'SPEC.md', '/task/SPEC.md', readonly=True)
            if self.data['benchmark'] == 'commit0':
                for name in self.data.get('protected_paths', []):
                    original = self.seed / name
                    if original.exists():
                        box.put(original, '/workspace/' + name, readonly=True)
            return box
        except BaseException:
            box.close()
            raise

    def check(self, source, report_dir):
        """Visible deterministic checks only. Hidden final tests are never used here."""
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        with self.sandbox(source, reference=False) as box:
            build = box.execute(self.data.get('build_command', 'true'), timeout=180)
            write_json(report_dir / 'build.json', build)
            if build['exit_code'] != 0:
                return {'build': False, 'status': 'build_failed', 'score': None,
                        'diagnostic': build['output'][-2500:]}
            cmd = self.data.get('visible_test_command')
            if not cmd:
                return {'build': True, 'status': 'build_only', 'score': None}
            # Keep failure detail in the bounded JUnit artifact. Repeating long
            # tracebacks/captured output on stdout can hit the sandbox's 2 MiB
            # ceiling, destroy the container, and lose an otherwise complete XML.
            # These are display options only; test selection and scoring stay fixed.
            result = box.execute(cmd + ' --tb=no --show-capture=no --junitxml=/tmp/codetest-results.xml', timeout=300)
            write_json(report_dir / 'test-command.json', result)
            # XML routinely exceeds shell-output display limits; copy it as a bounded artifact.
            xml_path = report_dir / 'junit.xml'
            copied = run(['docker', 'cp', box.name + ':/tmp/codetest-results.xml', str(xml_path)], check=False)
            xml_text = xml_path.read_text() if copied.returncode == 0 and xml_path.stat().st_size < 8_000_000 else ''
            try:
                parsed = junit(xml_text)
            except ET.ParseError:
                parsed = {'status': 'invalid_test_report', 'score': None}
            if result['exit_code'] not in (0, 1):
                parsed.update(status='test_harness_error', score=None)
            return {'build': True, **parsed, 'diagnostic': result['output'][-2500:] if result['exit_code'] else ''}

    def final(self, submission_root):
        """Final native evaluator is separate from the inference lifecycle."""
        root = Path(submission_root)
        if self.data['benchmark'] == 'programbench':
            # Use this environment's installed package even when its bin directory
            # was not added to PATH (for example, .venv/bin/python run.py).
            native = [sys.executable, '-c', 'from programbench.cli.main import app; app()']
            result = run([*native, 'eval', str(root), '--workers', '1',
                          '--branch-workers', '1', '--docker-cpus', '2'], timeout=21600, check=False)
            (root / 'native-eval.log').write_bytes(result.stdout + result.stderr)
            if result.returncode:
                return {'status': 'native_evaluator_error', 'score': None}
            # Official ignore lists/branch filtering are applied by info, never raw pass counting.
            info = run([*native, 'info', str(root)], timeout=120, check=False)
            (root / 'official-summary.txt').write_bytes(info.stdout + info.stderr)
            if info.returncode:
                return {'status': 'native_summary_error', 'score': None}
            from .programbench_score import summarize
            return {**summarize(root, self.data['id']), 'official_summary': str(root / 'official-summary.txt'),
                    'native_parameters': {'workers': 1, 'branch_workers': 1, 'docker_cpus': 2}}
        if self.data['benchmark'] == 'commit0':
            return self.check(root / self.data['id'] / 'source', root / 'final-check')
        if self.data['benchmark'] == 'nl2repo':
            return self._nl2repo_final(root)
        return {'status': 'fixture', 'score': None}

    def _nl2repo_final(self, root):
        """Reproduce upstream overlay, using its reference image only after inference ends.

        This is the upstream overlay protocol, including its residual-reference-source
        limitation. Raw command output and the published denominator are preserved.
        """
        from .sandbox import MAX_ARCHIVE
        source = root / self.data['id'] / 'source'
        image = self.data.get('evaluation_offline_runtime') or self.data['evaluation_image']
        # DockerSandbox normally erases /workspace. Extract reference with a stopped container;
        # it is never mounted or shown to the solver.
        ident = run(['docker', 'create', '--platform', 'linux/amd64', '--network', 'none', image]).stdout.decode().strip()
        try:
            image = json.loads(run(['docker', 'inspect', ident]).stdout)[0]['Image']
            blob = run(['docker', 'cp', ident + ':/workspace/.', '-'], timeout=120).stdout
        finally:
            run(['docker', 'rm', '-f', ident], check=False)
        if len(blob) > MAX_ARCHIVE:
            raise ValueError('Reference workspace exceeds v1 artifact limit')
        reference = root / 'private-evaluation-reference'
        reference.mkdir(exist_ok=False)
        safe_extract(blob, reference)
        overlay = root / 'evaluation-overlay'
        protected = set(json.loads((self.root / 'private/test_files.json').read_text()))
        protected.update({'setup.py', 'pyproject.toml', 'setup.cfg', 'pytest.ini', 'tox.ini',
                          'requirements.txt', 'requirements-dev.txt', 'requirements-test.txt',
                          'requirements_test.txt', 'requirements_dev.txt'})
        audit = overlay_source(reference, source, overlay, protected)
        write_json(root / 'overlay-audit.json', audit)
        commands = json.loads((self.root / 'private/test_commands.json').read_text())
        outputs, reports = [], []
        retained = audit['retained_reference_files']
        write_json(root / 'retained-reference-files.json', retained)
        with DockerSandbox(image, user='0') as box:
            box.load(overlay)
            for index, command in enumerate(commands):
                junit_path = f'/tmp/codetest-final-{index}.xml'
                if self.data.get('evaluation_offline_runtime') and command.strip() == 'pip install -e .':
                    command = 'python -m pip install --no-build-isolation --no-deps -e .'
                instrumented = command + ' --junitxml=' + junit_path if 'pytest' in command else command
                outputs.append(box.execute('export PYTHONPATH=/workspace; ' + instrumented, timeout=300))
                if 'pytest' in command:
                    local = root / f'native-tests-{index}.xml'
                    copied = run(['docker', 'cp', box.name + ':' + junit_path, str(local)], check=False)
                    try:
                        if copied.returncode or local.stat().st_size > 8_000_000:
                            raise ValueError('Missing or oversized XML report')
                        reports.append(junit(local.read_text()))
                    except (ValueError, ET.ParseError):
                        reports.append({'status': 'invalid_test_report', 'score': None, 'passed': 0})
        write_json(root / 'native-commands.json', outputs)
        passed = sum(r['passed'] for r in reports)
        total = self.data['expected_tests']
        error = (not reports or any(r['score'] is None for r in reports) or passed > total
                 or any(o['status'] != 'ok' or o['exit_code'] not in ((0, 1) if 'pytest' in o['command'] else (0,)) for o in outputs))
        return {'status': 'evaluator_error' if error else 'scored_overlay',
                'score': None if error else min(1, passed / total), 'passed': passed, 'total': total,
                'protocol': 'upstream_overlay_offline_with_junit', 'reports': reports,
                'evaluation_image_id': image,
                'offline_build_dependency_runtime': bool(self.data.get('evaluation_offline_runtime')),
                'retained_reference_file_count': len(retained),
                'retained_reference_audit': str(root / 'retained-reference-files.json'),
                'limitation': 'Upstream overlay retains reference files absent from submission; audit before scientific claims.'}


def programbench_runtime(destination, reference_image):
    """Relocate the execute-only oracle during image build, before restricted runs."""
    if not re.fullmatch(r'[A-Za-z0-9_./:@-]+', reference_image):
        raise ValueError('Invalid reference image')
    (Path(destination) / 'Dockerfile').write_text(
        f'FROM {reference_image}\n'
        'USER 0\n'
        'RUN mkdir -p /oracle /reference-docs && '
        'mv /workspace/executable /oracle/executable && '
        'cp -a /workspace/. /reference-docs/ && '
        'rm -rf /reference-docs/.git /workspace && '
        'chown -R 0:0 /oracle /reference-docs && '
        'chmod 0111 /oracle/executable && chmod 0555 /oracle && '
        'chmod -R a+rX /reference-docs && mkdir /workspace && chmod 0777 /workspace\n'
        'WORKDIR /workspace\n')


def prepare(benchmark, task_id, destination):
    """Download public task inputs and write inspectable build/evaluation manifests."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    seed = destination / 'seed'
    seed.mkdir()
    data = {'id': task_id, 'benchmark': benchmark, 'command_timeout': 60}
    if benchmark == 'programbench':
        if not re.fullmatch(r'[a-z0-9_.-]+__[a-z0-9_.-]+', task_id):
            raise ValueError('Use an official ProgramBench instance id')
        repo, pin = PINS[benchmark]
        # Verify the task exists at the pinned version, without loading tests into the solver.
        metadata = fetch(f'https://raw.githubusercontent.com/{repo}/{pin}/src/programbench/data/tasks/{task_id}/task.yaml')
        (destination / 'task.yaml').write_bytes(metadata)
        spec = ('Reconstruct the reference program from its normal CLI behavior and bundled documentation. '
                'Read /reference-docs. Query /oracle/executable; do not decompile, trace, copy, wrap, or modify it. '
                'Write an independent implementation in /workspace. An executable compile.sh must create '
                './executable from source. No internet or original source. Do not commit generated executables. '
                'Publish reproducible probes using the probe tool. Hidden tests are final-only.')
        reference_image = 'programbench/' + task_id.replace('__', '_1776_') + ':task_cleanroom_v6'
        data.update(image='code-testing-v1/programbench-' + task_id.replace('__', '-') + ':v1',
                    reference_image=reference_image,
                    build_command='test -x compile.sh && ./compile.sh && test -x executable',
                    test_visibility='agent-generated probes only; official tests final-only', source_revision=pin)
        (seed / '.gitignore').write_text('executable\nbuild/\n__pycache__/\n')
        programbench_runtime(destination, reference_image)
    elif benchmark == 'nl2repo':
        repo, pin = PINS[benchmark]
        if not re.fullmatch(r'[a-zA-Z0-9_.-]+', task_id):
            raise ValueError('Invalid task id')
        base = f'https://raw.githubusercontent.com/{repo}/{pin}/test_files/{task_id}/'
        spec = fetch(base + 'start.md').decode()
        private = destination / 'private'
        private.mkdir()
        for name in ('test_commands.json', 'test_files.json', 'test_case_count.txt'):
            (private / name).write_bytes(fetch(base + name))
        data.update(image='code-testing-v1/nl2repo-' + task_id.lower() + ':v1',
                    evaluation_image='ghcr.io/multimodal-art-projection/nl2repobench/' + task_id + ':1.0',
                    expected_tests=int((private / 'test_case_count.txt').read_text()),
                    test_visibility='document-only; upstream tests/configs final-only', source_revision=pin,
                    build_command='python -m compileall -q .')
        (destination / 'requirements-runtime.txt').write_text('# Add only non-target dependencies needed by this task.\n')
        (destination / 'Dockerfile').write_text(
            'FROM python:3.12-slim\n'
            'RUN apt-get update && apt-get install -y --no-install-recommends git build-essential && rm -rf /var/lib/apt/lists/*\n'
            'COPY requirements-runtime.txt /tmp/requirements-runtime.txt\n'
            'RUN python -m pip install --no-cache-dir pytest setuptools wheel -r /tmp/requirements-runtime.txt\n'
            'RUN python -m venv --system-site-packages /opt/codetest-venv && chown -R 65534:65534 /opt/codetest-venv\n'
            'WORKDIR /workspace\n')
    elif benchmark == 'commit0':
        url = ('https://datasets-server.huggingface.co/rows?dataset=wentingzhao%2Fcommit0_combined'
               '&config=default&split=test&offset=0&length=100')
        dataset = json.loads(fetch(url))
        matches = [r['row'] for r in dataset['rows'] if r['row']['repo'].split('/')[-1] == task_id]
        if len(matches) != 1:
            raise ValueError('Task must match one released Commit0 repository')
        row = matches[0]
        write_json(destination / 'dataset-row.json', row)
        blob = fetch('https://api.github.com/repos/' + row['repo'] + '/tarball/' + row['base_commit'])
        staging = destination / 'download'
        staging.mkdir()
        safe_extract(blob, staging)
        folders = list(staging.iterdir())
        if len(folders) != 1:
            raise ValueError('Unexpected source archive layout')
        copy_source(folders[0], seed)
        shutil.rmtree(staging)
        testdir = row['test']['test_dir'].rstrip('/')
        spec = ('Implement the missing library functions in the supplied commit-zero scaffold. '
                'Read its documentation, signatures and visible tests. Preserve the test suite. '
                'No retrieval of original source or installed target package. '
                'Source directory: ' + row['src_dir'] + '. Specification reference: ' + row['setup']['specification'])
        install_command = row['setup'].get('install') or 'python -m pip install -e .'
        install_command = re.sub(r'(?<!\w)(?:python -m )?pip install\b',
                                 'python -m pip install --no-deps --no-build-isolation', install_command)
        data.update(image='code-testing-v1/commit0-' + task_id + ':latest',
                    build_command=install_command,
                    visible_test_command=row['test']['test_cmd'] + ' ' + shlex.quote(testdir),
                    test_visibility='original tests visible, protected and restored for validation',
                    protected_paths=[testdir] + [n for n in ('pytest.ini', 'tox.ini', 'setup.cfg', 'pyproject.toml', 'conftest.py') if (seed / n).exists()],
                    source_revision=row['base_commit'])
        # Dependencies are prepared from the stripped scaffold, never reference_commit.
        setup = row['setup']
        install = ['python -m pip install pytest setuptools wheel build']
        # Editable installs are offline during inference; provision build backends now.
        import tomllib
        if (seed / 'pyproject.toml').is_file():
            backend = tomllib.loads((seed / 'pyproject.toml').read_text()).get('build-system', {}).get('requires', [])
            if backend:
                install.append('python -m pip install ' + ' '.join(shlex.quote(p) for p in backend))
        install.extend(setup.get('pre_install') or [])
        packages = setup.get('packages') or []
        if isinstance(packages, str):
            packages = [packages]
        install.extend('python -m pip install ' + ('-r ' if '.txt' in p else '') + shlex.quote(p) for p in packages)
        if setup.get('pip_packages'):
            install.append('python -m pip install ' + ' '.join(shlex.quote(p) for p in setup['pip_packages']))
        dockerfile = (f'FROM python:{setup["python"]}-slim\n'
                      'RUN apt-get update && apt-get install -y --no-install-recommends git build-essential && rm -rf /var/lib/apt/lists/*\n'
                      'WORKDIR /opt/scaffold\nCOPY seed/ .\n'
                      'RUN ' + ' && '.join(install) + '\nRUN rm -rf /opt/scaffold\n'
                      'RUN python -m venv --system-site-packages /opt/codetest-venv && chown -R 65534:65534 /opt/codetest-venv\n'
                      'WORKDIR /workspace\n')
        (destination / 'Dockerfile').write_text(dockerfile)
    else:
        raise ValueError('prepare supports programbench, commit0, nl2repo')
    (destination / 'SPEC.md').write_text(spec)
    (seed / 'PROJECT_NOTES.md').write_text('# Public project notes\n\nNo accepted contributions yet.\n')
    data['spec_sha256'] = sha(destination / 'SPEC.md')
    write_json(destination / 'task.json', data)
    return data
