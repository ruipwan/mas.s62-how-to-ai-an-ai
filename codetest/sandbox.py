"""Shell commands run only in disposable, network-isolated Docker containers."""
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
import time
import uuid

MAX_ARCHIVE = 32 * 1024 * 1024


def run(argv, *, timeout=60, check=True, data=None, env=None):
    result = subprocess.run(argv, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=timeout, env=env)
    if check and result.returncode:
        raise RuntimeError(f"{argv[0]} failed ({result.returncode}): {result.stderr.decode(errors='replace')[-1500:]}")
    return result


def safe_extract(blob, destination):
    """No archive member can write outside destination or introduce a symlink."""
    if len(blob) > MAX_ARCHIVE:
        raise ValueError("Artifact exceeds 32 MiB")
    destination = Path(destination)
    with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
        members = archive.getmembers()
        if len(members) > 10000 or sum(m.size for m in members) > MAX_ARCHIVE:
            raise ValueError("Expanded artifact exceeds limits")
        for m in members:
            p = PurePosixPath(m.name)
            if p.is_absolute() or '..' in p.parts or m.issym() or m.islnk() or not (m.isfile() or m.isdir()):
                raise ValueError(f"Unsafe archive member: {m.name}")
            if '.git' in p.parts or '__pycache__' in p.parts:
                continue
            target = destination.joinpath(*p.parts)
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError(f'Archive destination escapes through existing path: {m.name}')
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(m).read())
                target.chmod(0o755 if m.mode & 0o111 else 0o644)


def copy_source(source, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for p in sorted(Path(source).rglob('*')):
        rel = p.relative_to(source)
        if any(part in {'.git', '__pycache__', '.venv', '.pytest_cache'} for part in rel.parts):
            continue
        if p.is_symlink():
            raise ValueError(f"Symlink not supported in v1 source: {rel}")
        if p.is_file():
            target = destination / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)


class DockerSandbox:
    def __init__(self, image, *, programbench=False, timeout=60, platform='linux/amd64', user='65534:65534', memory='2g', cpus='2'):
        self.image, self.timeout = image, timeout
        self.user = user
        self.name = 'codetest-' + uuid.uuid4().hex[:12]
        self.alive = False
        # No socket, host mounts, API keys, host networking, or privileged mode.
        run(['docker', 'run', '-d', '--name', self.name, '--platform', platform,
             '--network', 'none', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
             '--pids-limit', '256', '--memory', memory, '--cpus', cpus,
             '--ulimit', 'fsize=67108864:67108864', '--entrypoint', '/bin/sh',
             image, '-c', 'sleep 21600'])
        self.alive = True
        try:
            if programbench:
                self.root('test -x /oracle/executable && test ! -r /oracle/executable && '
                          'test -d /reference-docs && test ! -d /reference-docs/.git')
            else:
                # Candidate verification cannot call the oracle, including in a
                # prepared ProgramBench runtime whose oracle is baked into its image.
                self.root('if [ -d /oracle ]; then chmod u+w /oracle; rm -rf /oracle; fi')
            self.root('rm -rf /workspace; mkdir -p /workspace /tmp/agent-home; '
                      'chmod 0777 /workspace /tmp/agent-home')
            # Runtime images must contain timeout; fail before paid calls otherwise.
            self.root('command -v timeout >/dev/null')
        except BaseException:
            self.close()
            raise

    def root(self, command):
        return run(['docker', 'exec', '--user', '0', self.name, '/bin/sh', '-ec', command])

    def load(self, source):
        # Public .git history and host repository configuration must never cross this boundary.
        with tempfile.TemporaryDirectory(prefix='codetest-copy-') as temp:
            copy_source(source, temp)
            self.copy_in(temp, '/workspace', contents=True)
        self.root('chmod -R a+rwX /workspace')

    def copy_in(self, source, destination, *, contents=False):
        # Docker Desktop can preserve the macOS uid on copied files. With all
        # capabilities dropped, uid 0 cannot chmod files owned by that host uid.
        # Normalize archive ownership instead of granting extra capabilities.
        size = 0
        def owned(info):
            nonlocal size
            if not (info.isfile() or info.isdir()):
                raise ValueError('Container inputs must be regular files or directories')
            size += info.size
            if size > MAX_ARCHIVE:
                raise ValueError('Container input exceeds 32 MiB')
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            info.mode = 0o755 if info.isdir() or info.mode & 0o111 else 0o644
            return info
        with io.BytesIO() as data:
            with tarfile.open(fileobj=data, mode='w') as archive:
                archive.add(str(source), arcname='.' if contents else PurePosixPath(destination).name,
                            filter=owned)
            parent = destination if contents else str(PurePosixPath(destination).parent)
            run(['docker', 'cp', '-', self.name + ':' + parent], data=data.getvalue())

    def put(self, source, destination, *, readonly=False):
        # Paths here come from the trusted task manifest, never from model JSON.
        import shlex
        quoted = shlex.quote(destination)
        # A repeated fetch can replace a root-owned tree made read-only below.
        # With all capabilities dropped, even uid 0 needs directory write bits
        # to remove its children. Restore owner permissions only; the agent uid
        # never gains write access, and the replacement is locked again below.
        self.root('if [ -e ' + quoted + ' ] && [ ! -L ' + quoted + ' ]; then '
                  'chmod -R u+w ' + quoted + '; fi; rm -rf ' + quoted +
                  '; mkdir -p ' + shlex.quote(str(PurePosixPath(destination).parent)))
        self.copy_in(source, destination)
        if readonly:
            self.root('chmod -R a-w ' + shlex.quote(destination))

    def execute(self, command, *, timeout=None):
        if not self.alive:
            raise RuntimeError('Sandbox was terminated')
        if not isinstance(command, str) or len(command) > 64000 or '\x00' in command:
            raise ValueError('Invalid shell command')
        seconds = min(timeout or self.timeout, 300)
        started = time.monotonic()
        argv = ['docker', 'exec', '--user', self.user, '-w', '/workspace',
                '-e', 'HOME=/tmp/agent-home', '-e', 'PYTHONDONTWRITEBYTECODE=1',
                '-e', 'PIP_NO_INDEX=1', self.name, 'timeout', '-k', '2', str(seconds),
                '/bin/sh', '-c', 'if [ -d /opt/codetest-venv/bin ]; then export PATH=/opt/codetest-venv/bin:$PATH; fi; ' + command]
        # Spool output, not an unbounded in-memory capture. fsize also bounds container files.
        with tempfile.TemporaryFile() as output:
            proc = subprocess.Popen(argv, stdout=output, stderr=subprocess.STDOUT)
            status = 'ok'
            while proc.poll() is None:
                if time.monotonic() - started > seconds + 10 or output.tell() > 2 * 1024 * 1024:
                    status = 'timeout_or_output_limit'
                    self.close()  # Kill the entire container, including background descendants.
                    proc.kill()
                    break
                time.sleep(0.05)
            proc.wait()
            size = output.tell()
            output.seek(0)
            raw = output.read(12000).decode(errors='replace')
            if size > 12000:
                output.seek(max(0, size - 6000))
                raw = raw[:6000] + '\n[... output truncated ...]\n' + output.read(6000).decode(errors='replace')
        return {'command': command, 'output': raw, 'exit_code': proc.returncode,
                'status': status, 'output_truncated': size > 12000,
                'seconds': round(time.monotonic() - started, 3)}

    def snapshot(self, destination):
        # Use tar as an unprivileged user, then validate every path before extracting on host.
        result = run(['docker', 'exec', '--user', '65534:65534', self.name, 'tar',
                      '--exclude=.git', '--exclude=.venv', '--exclude=__pycache__',
                      '--exclude=.pytest_cache', '-C', '/workspace', '-czf', '-', '.'], timeout=60)
        Path(destination).mkdir(parents=True, exist_ok=False)
        safe_extract(result.stdout, destination)

    def close(self):
        if self.alive:
            run(['docker', 'rm', '-f', self.name], check=False)
            self.alive = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class Repository:
    def __init__(self, path, seed):
        self.path = Path(path)
        copy_source(seed, self.path)
        self.git('init', '-q')
        self.git('config', 'user.email', 'local@code-testing.invalid')
        self.git('config', 'user.name', 'Code Testing v1')
        self.git('add', '--all')
        self.git('commit', '-qm', 'Initial sanitized scaffold', '--allow-empty')

    def git(self, *args):
        return self.git_bytes(*args).decode().strip()

    def git_bytes(self, *args):
        # Agent-written .gitattributes must not invoke filters from the host's git config.
        environment = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
        environment.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL='/dev/null',
                           GIT_TERMINAL_PROMPT='0', GIT_CONFIG_COUNT='0')
        return run(['git', '-C', str(self.path), '-c', 'core.hooksPath=/dev/null', *args], env=environment).stdout

    @property
    def version(self):
        return self.git('rev-parse', 'HEAD')

    def replace(self, candidate, expected):
        if expected != self.version:
            raise ValueError('Stale candidate base commit')
        for p in self.path.iterdir():
            if p.name != '.git':
                shutil.rmtree(p) if p.is_dir() else p.unlink()
        copy_source(candidate, self.path)
        # copy2 keeps timestamps, so a same-size rewrite can match git's cached stat data and be
        # skipped. Reset the index to HEAD without stat data so every file is compared by content.
        self.git('read-tree', 'HEAD')
        self.git('add', '--all')
        tracked = self.git('ls-files')
        if any(n.startswith('.git/') for n in tracked.splitlines()):
            raise ValueError('Invalid tracked path')
        patch = self.git('diff', '--cached', '--binary')
        if not patch:
            return None
        self.git('commit', '-qm', 'Accepted coalition contribution')
        return patch

    def package(self, destination):
        with tarfile.open(destination, 'w:gz') as archive:
            names = self.git_bytes('ls-files', '-z').decode().split('\0')
            for name in sorted(n for n in names if n):
                p = self.path / name
                if p.is_symlink() or not p.is_file():
                    raise ValueError('Submission contains a nonregular file')
                archive.add(p, arcname=name, recursive=False)
