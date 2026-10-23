"""Voluntary, run-local contribution contracts; all prices are virtual wealth.

An agreed contract binds existing reference observations, immutable artifact
bytes and an expiry before delivery. No hidden tests, model judge, extra agent
calls, automatic source merge or newly minted wealth are involved.
"""
import copy
import math
from pathlib import Path, PurePosixPath
import json

from .io import sha, write_json


class ContractMarket:
    MAX_RECORDS = 120
    PRICES = (.025, .05, .1)

    def __init__(self, root, agents, pad, emit):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.path = self.root / 'index.json'
        self.agents = {a.name: a for a in agents}
        self.pad, self.emit = pad, emit
        self.offers, self.contracts = {}, {}
        self.persist()

    @property
    def escrow(self):
        return math.fsum(c['price'] for c in self.contracts.values()
                         if c['status'] in ('funded', 'accepted', 'ready'))

    def persist(self):
        write_json(self.path, {'offers': self.offers, 'contracts': self.contracts,
                              'escrow': self.escrow,
                              'wallets': {n: a.wealth for n, a in self.agents.items()}})

    def record(self, event, round, **values):
        self.persist()
        row = {'event': event, 'round': round, **values,
               'escrow': self.escrow,
               'wallets': {n: a.wealth for n, a in self.agents.items()}}
        with (self.root / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        self.emit(event, **{k: v for k, v in row.items() if k != 'event'})

    def paths(self, values):
        if (not isinstance(values, list) or not 1 <= len(values) <= 8
                or len(set(values)) != len(values)):
            raise ValueError('Declare 1–8 distinct source paths')
        for value in values:
            p = PurePosixPath(value) if isinstance(value, str) else None
            if (p is None or not value or len(value) > 160 or p.is_absolute()
                    or '..' in p.parts or str(p) != value or p.suffix == '.md'
                    or any(n in {'.git', '.venv', 'tests', 'build', 'executable', '__pycache__'} for n in p.parts)):
                raise ValueError('Contract paths must be contained implementation files')
        return list(values)

    def artifact(self, actor, ident, paths):
        if ident not in self.pad.artifacts:
            raise ValueError('Unknown saved artifact')
        root = self.pad.root / 'artifacts' / ident
        meta = json.loads((root / 'metadata.json').read_text())
        if actor not in meta['authors']:
            raise ValueError('Only an artifact author can offer or deliver it')
        actual = {str(p.relative_to(root / 'source')): sha(p)
                  for p in (root / 'source').rglob('*') if p.is_file() and not p.is_symlink()}
        if actual != meta['source_sha256'] or sha(root / 'patch.diff') != meta['patch_sha256']:
            raise RuntimeError('Immutable contract artifact integrity failed')
        if any(p not in meta['changed_paths'] or p not in actual for p in paths):
            raise ValueError('Each contracted path must be a saved changed implementation file')
        return {'id': ident, 'metadata_sha256': sha(root / 'metadata.json'),
                'base_commit': meta['base_commit'], 'paths_sha256': {p: actual[p] for p in paths}}

    def checks(self, ids, bank):
        if not isinstance(ids, list) or not 1 <= len(ids) <= 6 or len(set(ids)) != len(ids):
            raise ValueError('Choose 1–6 distinct previously reproduced probe IDs')
        known = {p['id']: p for p in bank}
        if any(p not in known for p in ids):
            raise ValueError('Acceptance uses only buyer-visible previously reproduced probes')
        rows = copy.deepcopy([known[p] for p in ids])
        if len(json.dumps(rows).encode()) > 96000:
            raise ValueError('Acceptance observations exceed 96 KB')
        return rows

    def funded(self, actor, raw, round, bank, offer=None):
        if len(self.contracts) >= self.MAX_RECORDS:
            raise ValueError('Contract registry full')
        price = offer['price'] if offer else raw['price']
        if type(price) not in (float, int) or price not in self.PRICES:
            raise ValueError('Price must be .025, .05 or .1 virtual wealth')
        provider = offer['provider'] if offer else raw.get('provider')
        if provider is not None and (provider not in self.agents or provider == actor):
            raise ValueError('Provider must be another run-local agent or null')
        expiry = raw['expires_in']
        if type(expiry) is not int or expiry not in (1, 2, 3):
            raise ValueError('Choose an expiry 1–3 rounds ahead')
        paths = self.paths(raw['paths'])
        if offer and any(p not in offer['paths'] for p in paths):
            raise ValueError('Purchase paths must be a subset of the offer')
        checks = self.checks(raw['probe_ids'], bank)
        if self.agents[actor].wealth < price:
            raise ValueError('Insufficient available wealth for escrow')
        artifact = None
        if offer:
            artifact = self.artifact(provider, offer['artifact']['id'], paths)
            if artifact['metadata_sha256'] != offer['artifact']['metadata_sha256']:
                raise RuntimeError('Offer artifact changed')
        ident = f'contract-{len(self.contracts) + 1:03}'
        row = {'id': ident, 'buyer': actor, 'provider': provider, 'price': price,
               'description': self.description(raw['description']), 'paths': paths,
               'checks': checks, 'created_round': round, 'expires_round': round + expiry,
               'status': 'funded', 'artifact': artifact,
               'offer': offer['id'] if offer else None}
        self.agents[actor].wealth -= price
        self.contracts[ident] = row
        self.record('contract_funded', round, contract=copy.deepcopy(row))
        return {'status': 'funded', 'id': ident, 'reserved_wealth': price,
                'note': 'Provider must explicitly accept these frozen conditions'}

    @staticmethod
    def description(text):
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 800:
            raise ValueError('Contract description must contain 1–800 characters')
        return text.strip()

    def perform(self, actor, raw, round, bank, *, source=None, verify=None):
        if actor not in self.agents or not isinstance(raw, dict):
            raise ValueError('Invalid contract actor/action')
        operation = raw.get('operation')
        if operation == 'offer':
            if len(self.offers) >= self.MAX_RECORDS:
                raise ValueError('Offer registry full')
            paths = self.paths(raw['paths'])
            if type(raw['price']) not in (float, int) or raw['price'] not in self.PRICES:
                raise ValueError('Invalid offer price')
            ident = f'offer-{len(self.offers) + 1:03}'
            row = {'id': ident, 'provider': actor, 'price': raw['price'], 'paths': paths,
                   'description': self.description(raw['description']),
                   'artifact': self.artifact(actor, raw['artifact'], paths)}
            self.offers[ident] = row
            self.record('contract_offer', round, offer=row)
            return {'status': 'offered', 'id': ident}
        if operation == 'request':
            return self.funded(actor, raw, round, bank)
        if operation == 'fund':
            offer = self.offers.get(raw['id'])
            if offer is None or offer['provider'] == actor:
                raise ValueError('Choose another agent’s existing offer')
            return self.funded(actor, raw, round, bank, offer)
        row = self.contracts.get(raw.get('id'))
        if row is None:
            raise ValueError('Unknown contract ID')
        if row['status'] not in ('funded', 'accepted', 'ready'):
            raise ValueError('Contract already closed')
        if round >= row['expires_round']:
            raise ValueError('Contract expired')
        if operation == 'accept':
            if row['status'] != 'funded' or actor == row['buyer'] or row['provider'] not in (None, actor):
                raise ValueError('Only the eligible provider can accept')
            row['provider'] = actor
            row['status'] = 'ready' if row['artifact'] else 'accepted'
            self.record('contract_accepted', round, id=row['id'], provider=actor, status=row['status'])
            return {'status': row['status'], 'id': row['id']}
        if operation == 'deliver':
            if row['status'] != 'accepted' or actor != row['provider']:
                raise ValueError('Only the accepted provider can deliver')
            row['artifact'] = self.artifact(actor, raw['artifact'], row['paths'])
            row['status'] = 'ready'
            self.record('contract_delivered', round, id=row['id'], artifact=row['artifact'])
            return {'status': 'ready', 'id': row['id'], 'artifact': row['artifact']['id']}
        if operation == 'cancel':
            if actor != row['provider'] and not (actor == row['buyer'] and row['status'] == 'funded'):
                raise ValueError('Only provider decline or buyer cancellation before acceptance is allowed')
            return self.refund(row, round, 'cancelled')
        if operation != 'settle' or actor != row['buyer'] or row['status'] != 'ready':
            raise ValueError('Only the buyer can settle a ready contract')
        if source is None or verify is None:
            raise RuntimeError('Trusted settlement verifier missing')
        # Recheck immutable provenance before examining the actual integrated source.
        frozen = self.artifact(row['provider'], row['artifact']['id'], row['paths'])
        if frozen != row['artifact']:
            raise RuntimeError('Agreed artifact changed')
        root = Path(source).resolve()
        matches = {}
        for path, expected in frozen['paths_sha256'].items():
            target = root / path
            matches[path] = (target.resolve().is_relative_to(root) and not target.is_symlink()
                             and target.is_file() and sha(target) == expected)
        if not all(matches.values()):
            return self.refund(row, round, 'failed', evidence={'integrated_paths': matches})
        # Verifier exceptions preserve escrow and interrupt work; they are not coding failures.
        try:
            result = verify(root, copy.deepcopy(row['checks']))
        except Exception as error:
            raise RuntimeError('Contract verifier infrastructure failed; escrow retained') from error
        if not isinstance(result, dict) or result.get('status') not in {'public_probe_score', 'build_failed'}:
            raise RuntimeError('Settlement verifier returned no valid observed outcome')
        evidence = {'integrated_paths': matches, 'check_result': result}
        if result.get('score') != 1 or result.get('passed') != len(row['checks']) or result.get('total') != len(row['checks']):
            return self.refund(row, round, 'failed', evidence=evidence)
        self.agents[row['provider']].wealth += row['price']
        row.update(status='paid', closed_round=round, evidence=evidence)
        self.record('contract_paid', round, id=row['id'], buyer=actor,
                    provider=row['provider'], amount=row['price'], artifact=row['artifact'], evidence=evidence)
        return {'status': 'paid', 'id': row['id'], 'transferred_wealth': row['price'], **evidence}

    def refund(self, row, round, status, **values):
        self.agents[row['buyer']].wealth += row['price']
        row.update(status=status, closed_round=round, **values)
        self.record('contract_refunded', round, id=row['id'], buyer=row['buyer'],
                    amount=row['price'], reason=status, **values)
        return {'status': status, 'id': row['id'], 'returned_wealth': row['price'], **values}

    def expire(self, round, *, terminal=False):
        for row in self.contracts.values():
            if row['status'] in ('funded', 'accepted', 'ready') and (terminal or round >= row['expires_round']):
                self.refund(row, round, 'terminal_unused' if terminal else 'expired')

    def observation(self):
        offers = list(self.offers.values())[-16:]
        opened = [c for c in self.contracts.values() if c['status'] in ('funded', 'accepted', 'ready')]
        return {'index_file': '/public/contracts.json', 'escrow_wealth': self.escrow,
                'offers': offers, 'open_contracts': [
                    {**{k: v for k, v in c.items() if k != 'checks'},
                     'probe_ids': [p['id'] for p in c['checks']]} for c in opened[-16:]],
                'contract_count': len(self.contracts),
                'paid_count': sum(c['status'] == 'paid' for c in self.contracts.values())}
