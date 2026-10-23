"""Reproducible ProgramBench observations, independent of untrusted agent files."""
import base64
import hashlib
import json
from pathlib import PurePosixPath
import shlex


def validate_probe(raw):
    args, stdin, files, capture = raw.get('args', []), raw.get('stdin', ''), raw.get('files', {}), raw.get('capture', [])
    if (not isinstance(args, list) or len(args) > 64 or any(not isinstance(a, str) or '\0' in a for a in args)
            or not isinstance(stdin, str) or not isinstance(files, dict) or not isinstance(capture, list)
            or any(not isinstance(v, str) for v in files.values()) or len(files) > 12 or len(capture) > 12):
        raise ValueError('Invalid probe')
    for name in [*files, *capture]:
        if not isinstance(name, str) or not name or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:
            raise ValueError('Probe file path must be relative and contained')
    probe = dict(args=args, stdin=stdin, files=files, capture=capture)
    if len(json.dumps(probe).encode()) > 12000:
        raise ValueError('Probe input exceeds 12 KB')
    return probe


def observation(box, probe, executable):
    # The executable and input data are passed as literals, never shell interpolation.
    encoded = base64.b64encode(json.dumps(probe).encode()).decode()
    script = '''import base64,json,pathlib,subprocess,tempfile
p=json.loads(base64.b64decode(ENCODED))
with tempfile.TemporaryDirectory(prefix='codetest-probe-') as d:
 for n,v in p['files'].items():
  f=pathlib.Path(d,n);f.parent.mkdir(parents=True,exist_ok=True);f.write_text(v)
 try:
  r=subprocess.run([EXECUTABLE,*p['args']],input=p['stdin'].encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=d,timeout=10)
  data={'stdout':base64.b64encode(r.stdout[:16384]).decode(),'stderr':base64.b64encode(r.stderr[:16384]).decode(),'exit_code':r.returncode,'files':{}}
  data['truncated']=len(r.stdout)>16384 or len(r.stderr)>16384
  for name in p['capture']:
   f=pathlib.Path(d,name)
   if not f.resolve().is_relative_to(pathlib.Path(d)) or f.is_symlink():raise ValueError('output path escapes probe')
   raw=f.read_bytes() if f.is_file() else None
   if raw is not None and len(raw)>16384: data['truncated']=True
   data['files'][name]=base64.b64encode(raw[:16384]).decode() if raw is not None else None
  print(json.dumps(data))
 except subprocess.TimeoutExpired:print(json.dumps({'error':'timeout'}))
'''.replace('ENCODED', repr(encoded)).replace('EXECUTABLE', repr(executable))
    result = box.execute('python3 -c ' + shlex.quote(script), timeout=20)
    if result['exit_code'] != 0 or result['status'] != 'ok':
        raise ValueError('Probe execution failed')
    value = json.loads(result['output'])
    if value.get('error'):
        raise ValueError('Reference probe timed out after 10 seconds')
    if value.get('truncated'):
        raise ValueError('Probe output exceeded 16 KiB per stream/file; choose a smaller observation')
    return value


def readable_observation(value, limit=1800):
    """Readable previews are separate from exact byte comparisons used by the scorer."""
    def preview(encoded):
        if encoded is None:
            return None
        raw = base64.b64decode(encoded)
        return {'text': raw.decode('utf-8', errors='replace')[:limit], 'bytes': len(raw),
                'preview_truncated': len(raw.decode('utf-8', errors='replace')) > limit}
    return {'stdout': preview(value['stdout']), 'stderr': preview(value['stderr']),
            'exit_code': value['exit_code'], 'captured_files': list(value.get('files', {}))}


def discover(task, raw):
    probe = validate_probe(raw)
    results = []
    for _ in range(2):
        with task.sandbox(task.seed) as box:
            results.append(observation(box, probe, '/oracle/executable'))
    if results[0] != results[1]:
        raise ValueError('Nonreproducible observation; not accepted as evidence')
    digest = hashlib.sha256(json.dumps(probe, sort_keys=True).encode()).hexdigest()[:16]
    return {'id': digest, 'input': probe, 'expected': results[0]}


def score(task, source, bank):
    if not bank:
        return {'status': 'no_public_probes', 'score': None, 'passed': 0, 'total': 0}
    with task.sandbox(source, reference=False) as box:
        build = box.execute(task.data['build_command'], timeout=180)
        if build['exit_code']:
            return {'status': 'build_failed', 'score': None, 'passed': 0, 'total': len(bank)}
        passed = 0
        for row in bank:
            try:
                passed += observation(box, row['input'], '/workspace/executable') == row['expected']
            except (ValueError, json.JSONDecodeError):
                pass
    return {'status': 'public_probe_score', 'score': passed / len(bank), 'passed': passed,
            'total': len(bank), 'note': 'Agent-generated diagnostic; not the hidden benchmark score'}
