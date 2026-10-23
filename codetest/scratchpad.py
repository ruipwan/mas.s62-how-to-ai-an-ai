"""Run-local shared notes and immutable, explicitly retrievable candidate patches.

Peer claims are untrusted evidence. This module never edits the public repository,
selects a contribution, executes code, or reads a final evaluation.
"""
import json
from pathlib import Path
import re
import tempfile

from .io import write_json, sha
from .sandbox import Repository, copy_source


class Scratchpad:
    MAX_ENTRIES = 120

    def __init__(self, root, emit):
        self.root, self.emit = Path(root), emit
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'index.json'
        saved = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.entries = saved.get('entries', {})
        self.artifacts = saved.get('artifacts', {})
        self.version = saved.get('version', 0)
        self.persist()

    def persist(self):
        write_json(self.path, {'version': self.version, 'entries': self.entries,
                              'artifacts': self.artifacts})

    def record(self, event, **values):
        # Append-only history remains available even after an entry is updated.
        with (self.root / 'history.jsonl').open('a') as f:
            f.write(json.dumps({'event': event, **values}, ensure_ascii=False) + '\n')
        self.emit(event, **values)

    def write(self, actor, note, round):
        if not isinstance(note, dict) or set(note) != {'key', 'revision', 'kind', 'status', 'text', 'dependencies'}:
            raise ValueError('Scratchpad note fields must match the schema')
        key = note['key']
        if not isinstance(key, str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}', key):
            raise ValueError('Note key must be 1–64 letters, digits, dots, underscores or hyphens')
        if type(note['revision']) is not int or note['revision'] < 0:
            raise ValueError('revision must be an integer: 0 creates a note')
        if note['kind'] not in ('task', 'finding', 'interface', 'blocker') or note['status'] not in ('open', 'claimed', 'blocked', 'done'):
            raise ValueError('Unknown note kind or status')
        if not isinstance(note['text'], str) or not 1 <= len(note['text']) <= 1600:
            raise ValueError('Note text must contain 1–1600 characters')
        deps = note['dependencies']
        if not isinstance(deps, list) or len(deps) > 12 or any(not isinstance(d, str) or len(d) > 100 for d in deps):
            raise ValueError('At most 12 dependency keys, paths or artifact IDs, each <=100 characters')
        old = self.entries.get(key)
        if old and old['owner'] != actor:
            raise ValueError('Only the note owner can update it; post a separate note to challenge or offer a handoff')
        if note['revision'] != (old['revision'] if old else 0):
            raise ValueError('Revision conflict; read the current note before updating')
        if old is None and len(self.entries) >= self.MAX_ENTRIES:
            raise ValueError('Scratchpad is full; update one of your existing notes')
        self.version += 1
        entry = {**note, 'revision': note['revision'] + 1, 'owner': actor,
                 'updated_round': round, 'version': self.version}
        self.entries[key] = entry
        self.persist()
        self.record('scratchpad_write', agent=actor, round=round, entry=entry)
        return {'status': 'saved', 'key': key, 'revision': entry['revision']}

    def share(self, proposal, base_source, round):
        ident = f'r{round + 1:02}-' + proposal['team'].replace(':', '-')
        if not re.fullmatch(r'[a-zA-Z0-9_.-]+', ident) or ident in self.artifacts:
            raise ValueError('Invalid or duplicate candidate artifact ID')
        destination = self.root / 'artifacts' / ident
        destination.mkdir(parents=True)
        copy_source(proposal['source'], destination / 'source')
        # Generate the patch against the actual base without mutating public Git.
        with tempfile.TemporaryDirectory(prefix='codetest-shared-patch-') as temp:
            repo = Repository(Path(temp) / 'repo', base_source)
            diff = repo.replace(Path(proposal['source']), repo.version)
        patch = destination / 'patch.diff'
        patch.write_text((diff + '\n') if diff else '')
        check = {k: proposal['check'][k] for k in ('build', 'status', 'score', 'passed', 'total', 'errors')
                 if k in proposal['check']}
        metadata = {'id': ident, 'round': round, 'team': proposal['team'],
                    'authors': proposal['members'], 'base_commit': proposal['base_commit'],
                    'changed_paths': proposal['changed_paths'], 'check': check,
                    'patch_sha256': sha(patch),
                    'source_sha256': {str(p.relative_to(destination / 'source')): sha(p)
                                      for p in sorted((destination / 'source').rglob('*')) if p.is_file()}}
        write_json(destination / 'metadata.json', metadata)
        self.artifacts[ident] = {k: v for k, v in metadata.items() if k != 'source_sha256'}
        self.version += 1
        self.persist()
        self.record('scratchpad_artifact', round=round, artifact=self.artifacts[ident])
        return ident

    def fetch(self, ident, actor, current_base, box, round):
        # The model supplies a registry key, never a host or container path.
        if not isinstance(ident, str) or ident not in self.artifacts:
            raise ValueError('Unknown artifact ID; consult /public/scratchpad.json')
        source = self.root / 'artifacts' / ident
        metadata = json.loads((source / 'metadata.json').read_text())
        actual = {str(p.relative_to(source / 'source')): sha(p)
                  for p in (source / 'source').rglob('*') if p.is_file()}
        if actual != metadata['source_sha256'] or sha(source / 'patch.diff') != metadata['patch_sha256']:
            raise ValueError('Shared artifact integrity check failed')
        destination = '/public/candidates/' + ident
        box.put(source, destination, readonly=True)
        result = {'status': 'fetched', 'artifact_id': ident, 'path': destination,
                  'base_commit': metadata['base_commit'], 'same_base': current_base == metadata['base_commit'],
                  'authors': metadata['authors'], 'changed_paths': metadata['changed_paths'],
                  'note': 'Inspect patch.diff and source/. No files have been applied. Reconcile stale bases and test before proposing.'}
        self.record('scratchpad_fetch', agent=actor, round=round, **result)
        return result

    def observation(self):
        ordered = sorted(self.entries.values(), key=lambda e: e['version'])
        recent = [{**a, 'changed_path_count': len(a['changed_paths']),
                   'changed_paths': [p[:120] for p in a['changed_paths'][:16]]}
                  for a in list(self.artifacts.values())[-8:]]
        return {'version': self.version, 'index_file': '/public/scratchpad.json',
                'entry_count': len(ordered), 'artifact_count': len(self.artifacts),
                'entry_index': [{'key': e['key'], 'owner': e['owner'], 'revision': e['revision'],
                                 'status': e['status']} for e in ordered],
                'recent_entries': ordered[-8:], 'recent_artifacts': recent}
