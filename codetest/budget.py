"""One durable campaign ledger, including reservations made before HTTP sends."""
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time
import uuid

import requests
from urllib3.exceptions import ConnectTimeoutError, ReadTimeoutError

from .api_diagnostics import response_diagnostics, validate_api_key


class BudgetStop(RuntimeError):
    pass


class BudgetBusy(RuntimeError):
    """No request sent; wait for another live experiment's request slot."""


def money(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Expected nonnegative finite money")
    return float(value)


class Ledger:
    LIMIT = 100.0
    PARALLEL_REQUESTS = 2

    def __init__(self, path, *, parallel=False, parallel_requests=None):
        self.path = Path(path).resolve()
        self.parallel = parallel
        self.parallel_requests = self.PARALLEL_REQUESTS if parallel_requests is None else parallel_requests
        if type(self.parallel_requests) is not int or not 1 <= self.parallel_requests <= 10:
            raise ValueError('parallel_requests must be between 1 and 10')
        self._leases = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, cap REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS calls (
                  id TEXT PRIMARY KEY, run TEXT NOT NULL, phase TEXT, agent TEXT,
                  model TEXT, reserved REAL NOT NULL, actual REAL, status TEXT NOT NULL,
                  receipt TEXT, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS transport_attempts (
                  call_id TEXT PRIMARY KEY, run TEXT NOT NULL, request_id TEXT NOT NULL,
                  attempt INTEGER NOT NULL, request_hash TEXT NOT NULL,
                  UNIQUE(run, request_id, attempt));
            """)

    @contextlib.contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA synchronous=FULL")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def register(self, run, cap):
        if not 0 < money(cap) <= self.LIMIT:
            raise ValueError("Run cap must be between zero and $100")
        with self.connect() as db:
            db.execute("INSERT INTO runs VALUES (?,?)", (run, cap))
        if self.parallel:
            path = self._lease_path(run)
            path.parent.mkdir(exist_ok=True)
            lease = path.open('a+b')
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                lease.close()
                raise
            self._leases[run] = lease

    def _lease_path(self, run):
        name = hashlib.sha256(run.encode()).hexdigest()
        return self.path.parent / 'request-leases' / (name + '.lock')

    def attach(self, run, cap):
        """Continue an audited episode without resetting its spent or reserved cap."""
        if not self.parallel:
            raise ValueError('Recovery requires a live parallel request lease')
        with self.connect() as db:
            row = db.execute('SELECT cap FROM runs WHERE id=?', (run,)).fetchone()
        if row is None or not math.isclose(row['cap'], money(cap), abs_tol=1e-12):
            raise ValueError('Original episode cap must match exactly')
        if self.summary(run)['pending'] or self.summary()['blocked']:
            raise BudgetStop('Outstanding request requires accounting review before continuation')
        path = self._lease_path(run); path.parent.mkdir(exist_ok=True)
        lease = path.open('a+b')
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            lease.close(); raise
        self._leases[run] = lease

    def _owner_alive(self, run):
        path = self._lease_path(run)
        if not path.exists():
            return False
        with path.open('rb') as lease:
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(lease, fcntl.LOCK_UN)
            return False

    def recover_orphans(self):
        """Keep abandoned reservations in full and block further spending."""
        orphaned = []
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            for row in db.execute("SELECT id,run FROM calls WHERE status='parallel_pending'").fetchall():
                if not self._owner_alive(row['run']):
                    db.execute("UPDATE calls SET status='uncertain' WHERE id=?", (row['id'],))
                    orphaned.append(row['id'])
        return orphaned

    def close(self):
        for lease in self._leases.values():
            lease.close()
        self._leases.clear()
        self.recover_orphans()

    def summary(self, run=None):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM calls" + (" WHERE run=?" if run else ""),
                              (run,) if run else ()).fetchall()
        actual = math.fsum(r["actual"] or 0 for r in rows)
        reserved = math.fsum(r["reserved"] for r in rows if r["actual"] is None)
        return {"reported_usd": actual, "unconfirmed_usd": reserved,
                "accounted_usd": actual + reserved, "calls": len(rows),
                "blocked": any(r["status"] in {"uncertain", "overrun"} for r in rows),
                "pending": sum(r["status"] in {"pending", "parallel_pending"} for r in rows)}

    def reserve(self, run, amount, phase, agent, model, *, transport=None):
        amount = money(amount)
        if (self.path.parent / "STOP").exists():
            raise BudgetStop("Campaign STOP file exists")
        self.recover_orphans()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT * FROM calls").fetchall()
            if any(r["status"] in {"uncertain", "overrun"} for r in rows):
                raise BudgetStop("Unreconciled prior call; reservation retained, no automatic replay")
            if self.parallel:
                if run not in self._leases or self._leases[run].closed:
                    raise BudgetStop('Parallel experiment does not own its live request lease')
                # Legacy workers already in progress recognize pending, uncertain,
                # and overrun. They count every NULL actual as reserved, including
                # parallel_pending. A distinct live status lets them continue
                # unchanged without weakening either shared dollar limit.
                if any(r['status'] == 'pending' for r in rows):
                    raise BudgetBusy('Waiting for the existing serial request to settle')
                if sum(r['status'] == 'parallel_pending' for r in rows) >= self.parallel_requests:
                    raise BudgetBusy('Parallel request slots are occupied')
                status = 'parallel_pending'
            else:
                if any(r['status'] in {'pending', 'parallel_pending'} for r in rows):
                    raise BudgetStop('Unreconciled prior call; reservation retained, no automatic replay')
                status = 'pending'
            total = math.fsum(r["actual"] if r["actual"] is not None else r["reserved"] for r in rows)
            local = math.fsum(r["actual"] if r["actual"] is not None else r["reserved"]
                              for r in rows if r["run"] == run)
            cap = db.execute("SELECT cap FROM runs WHERE id=?", (run,)).fetchone()
            if not cap or total + amount > self.LIMIT or local + amount > cap["cap"]:
                raise BudgetStop("Next call exceeds campaign or episode budget")
            attempt = None
            if transport is not None:
                previous = db.execute(
                    "SELECT a.*,c.status,c.receipt FROM transport_attempts a JOIN calls c ON c.id=a.call_id "
                    "WHERE a.run=? AND a.request_id=? ORDER BY a.attempt DESC",
                    (run, transport['request_id'])).fetchall()
                if previous:
                    last = previous[0]
                    if last['request_hash'] != transport['request_hash']:
                        raise ValueError('Stable transport request ID was reused with different request contents')
                    if (last['status'] != 'reserved_error' or
                            json.loads(last['receipt'] or '{}').get('failure_kind') != 'transport_timeout'):
                        raise BudgetStop('Prior transport attempt is not an acknowledged timeout; no replay')
                    attempt = last['attempt'] + 1
                else:
                    attempt = transport['initial_attempts_used'] + 1
                if attempt > transport['max_retries'] + 1:
                    raise BudgetStop('Transport timeout retry limit reached; reservations retained')
                if attempt > 1:
                    retry_reserved = db.execute(
                        'SELECT COALESCE(SUM(c.reserved),0) FROM calls c JOIN transport_attempts a '
                        'ON c.id=a.call_id WHERE a.run=? AND a.attempt>1', (run,)).fetchone()[0]
                    if retry_reserved + amount > transport['retry_reserve_cap_usd']:
                        raise BudgetStop('Next retry exceeds the episode retry reservation limit')
            ident = uuid.uuid4().hex
            db.execute("INSERT INTO calls VALUES (?,?,?,?,?,?,NULL,?,NULL,?)",
                       (ident, run, phase, agent, model, amount, status, time.time()))
            if transport is not None:
                db.execute('INSERT INTO transport_attempts VALUES (?,?,?,?,?)',
                           (ident, run, transport['request_id'], attempt, transport['request_hash']))
        return ident

    def settle(self, ident, actual, receipt):
        actual = money(actual)
        with self.connect() as db:
            row = db.execute("SELECT * FROM calls WHERE id=?", (ident,)).fetchone()
            if not row or row["status"] not in {"pending", "parallel_pending"}:
                raise ValueError("Call is not pending")
            status = "overrun" if actual > row["reserved"] + 1e-9 else "charged"
            db.execute("UPDATE calls SET actual=?,status=?,receipt=? WHERE id=?",
                       (actual, status, json.dumps(receipt), ident))
        if status == "overrun":
            raise BudgetStop("Provider charge exceeded reservation; campaign blocked")

    def uncertain(self, ident):
        with self.connect() as db:
            db.execute("UPDATE calls SET status='uncertain' WHERE id=? AND status IN ('pending','parallel_pending')", (ident,))

    def transport_attempt(self, ident):
        with self.connect() as db:
            row = db.execute('SELECT * FROM transport_attempts WHERE call_id=?', (ident,)).fetchone()
        return dict(row) if row else None

    def retain_transport_timeout(self, ident, run, error_type):
        """Atomically keep an opted-in timeout's entire allowance without a global lock gap."""
        if error_type not in {'Timeout', 'ReadTimeout', 'ConnectTimeout', 'ReadTimeoutError', 'ConnectTimeoutError'}:
            raise ValueError('Only an identified transport timeout may be retained automatically')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT c.* FROM calls c JOIN transport_attempts a ON a.call_id=c.id '
                             'WHERE c.id=? AND c.run=?', (ident, run)).fetchone()
            if not row or row['status'] not in {'pending', 'parallel_pending'} or row['actual'] is not None:
                raise ValueError('Only a pending opted-in transport attempt may be retained')
            receipt = {'accounting': 'full reservation retained, actual charge unknown',
                       'failure_kind': 'transport_timeout', 'error_type': error_type,
                       'retained_at': time.time()}
            db.execute("UPDATE calls SET status='reserved_error',receipt=? WHERE id=?",
                       (json.dumps(receipt), ident))
        return {'call_id': ident, 'retained_usd': row['reserved'], **receipt}

    def retain_failed_reservation(self, ident, reason):
        """Close an inspected failed request without releasing its dollar allowance.

        This is a manual audit operation, never an automatic HTTP retry. It does
        not claim a zero charge or a provider-confirmed charge. Pending requests
        and confirmed overruns cannot be acknowledged this way.
        """
        if not isinstance(reason, str) or len(reason.strip()) < 20:
            raise ValueError('A concrete audit reason is required')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM calls WHERE id=?', (ident,)).fetchone()
            if not row or row['status'] != 'uncertain' or row['actual'] is not None:
                raise ValueError('Only an uncertain failed call can retain its reservation')
            receipt = {'accounting': 'full reservation retained, actual charge unknown',
                       'reason': reason.strip(), 'reviewed_at': time.time()}
            db.execute("UPDATE calls SET status='reserved_error',receipt=? WHERE id=?",
                       (json.dumps(receipt), ident))
        return {'call_id': ident, 'retained_usd': row['reserved'], **receipt}


URLS = {'openrouter': 'https://openrouter.ai/api/v1/chat/completions',
        'openai': 'https://api.openai.com/v1/chat/completions'}


def transport_timeout_type(error):
    """Recognize timeout types, including Requests' body-read ConnectionError wrapper."""
    if isinstance(error, requests.Timeout):
        return ('ConnectTimeout' if isinstance(error, requests.ConnectTimeout) else
                'ReadTimeout' if isinstance(error, requests.ReadTimeout) else 'Timeout')
    if not isinstance(error, requests.ConnectionError):
        return None
    pending, seen = list(error.args) + [error.__cause__, error.__context__], set()
    while pending:
        nested = pending.pop()
        if not isinstance(nested, BaseException) or id(nested) in seen:
            continue
        seen.add(id(nested))
        if isinstance(nested, (ReadTimeoutError, ConnectTimeoutError)):
            return 'ReadTimeoutError' if isinstance(nested, ReadTimeoutError) else 'ConnectTimeoutError'
        pending.extend(list(nested.args) + [nested.__cause__, nested.__context__])
    return None


class OpenRouter:
    """Chat completions with a ledger reservation before every send; OpenRouter or OpenAI directly.

    OpenRouter reports each call's cost. OpenAI reports token usage only, so the cost is computed
    at the configured list prices, which never exceed the caps used for the reservation.
    """
    def __init__(self, model, ledger, run, caps, event, stop, *, reasoning_enabled=None, reasoning_effort=None, provider_ignore=(), structured_actions=False, shared_scratchpad=False, contribution_contracts=False,
                 provider='openrouter', prices=None, connect_timeout=15, read_timeout=180,
                 max_retries=0, retry_backoff=(5, 15), retry_reserve_cap_usd=1.0, initial_attempts_used=0):
        if type(max_retries) is not int or not 0 <= max_retries <= 2:
            raise ValueError('max_retries must be an integer between 0 and 2')
        if type(initial_attempts_used) is not int or not 0 <= initial_attempts_used <= max_retries:
            raise ValueError('initial_attempts_used must be between zero and max_retries')
        if money(connect_timeout) <= 0 or money(read_timeout) <= 0:
            raise ValueError('Transport timeouts must be positive finite seconds')
        if not isinstance(retry_backoff, (list, tuple)) or len(retry_backoff) != 2:
            raise ValueError('retry_backoff must contain two finite nonnegative waits')
        self.retry_backoff = tuple(money(delay) for delay in retry_backoff)
        self.retry_reserve_cap_usd = money(retry_reserve_cap_usd)
        self.connect_timeout, self.read_timeout = connect_timeout, read_timeout
        self.max_retries, self.initial_attempts_used = max_retries, initial_attempts_used
        if provider not in URLS:
            raise ValueError('Unknown provider')
        self.provider, self.prices = provider, prices
        if provider == 'openai':
            model = model.removeprefix('openai/')
            if not prices or any(money(prices[k]) > money(caps[c]) for k, c in
                                 (('prompt', 'prompt'), ('cached', 'prompt'), ('completion', 'completion'))):
                raise ValueError('OpenAI prices are required and may not exceed the price caps')
        self.model, self.ledger, self.run = model, ledger, run
        self.caps, self.event, self.stop = caps, event, Path(stop)
        self.reasoning_enabled = reasoning_enabled
        self.reasoning_effort, self.provider_ignore = reasoning_effort, provider_ignore
        self.structured_actions = structured_actions
        self.shared_scratchpad = shared_scratchpad
        self.contribution_contracts = contribution_contracts
        self.last_completion = {}
        variable = 'OPENAI_API_KEY' if provider == 'openai' else 'OPENROUTER_API_KEY'
        self.key = validate_api_key(os.environ.get(variable), variable)
        for key in ("prompt", "completion"):
            if money(caps[key]) <= 0:
                raise ValueError("Provider price ceilings must be positive")

    def _check_stop(self):
        if self.stop.exists():
            raise BudgetStop('Run STOP file exists')
        if (self.ledger.path.parent / 'STOP').exists():
            raise BudgetStop('Campaign STOP file exists')

    def _retry_wait(self, seconds):
        deadline = time.monotonic() + seconds
        while True:
            self._check_stop()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(.1, remaining))

    def generate(self, messages, max_tokens, phase, agent, *, request_id=None):
        validate_api_key(self.key, 'OPENAI_API_KEY' if self.provider == 'openai' else 'OPENROUTER_API_KEY')
        self.last_completion = {}
        if self.stop.exists():
            raise BudgetStop("Run STOP file exists")
        # Conservative text-byte estimate plus framing. This is not a universal tokenizer proof.
        input_bound = len(json.dumps(messages, ensure_ascii=False).encode()) + 4096
        reserve = (input_bound * self.caps["prompt"] + max_tokens * self.caps["completion"]) / 1e6
        if self.provider == 'openai':
            payload = {"model": self.model, "messages": messages, "max_completion_tokens": max_tokens,
                       "stream": False, "store": False, "response_format": {"type": "json_object"}}
            if self.reasoning_effort is not None:
                payload['reasoning_effort'] = self.reasoning_effort
        else:
            payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                       "stream": False, "response_format": {"type": "json_object"},
                       "provider": {"require_parameters": True, "allow_fallbacks": False,
                                    "sort": "price", "max_price": {**self.caps, "request": 0}}}
            if self.reasoning_enabled is not None:
                payload['reasoning'] = {'enabled': self.reasoning_enabled}
            if self.reasoning_effort is not None:
                payload.setdefault('reasoning', {})['effort'] = self.reasoning_effort
            if self.provider_ignore:
                payload['provider']['ignore'] = list(self.provider_ignore)
        if self.structured_actions:
            from .actions import action_schema
            payload['response_format'] = {'type': 'json_schema', 'json_schema': {
                'name': phase + '_action', 'strict': True, 'schema': action_schema(phase, self.shared_scratchpad, self.contribution_contracts)}}
        if request_id is not None and (not isinstance(request_id, str) or not 1 <= len(request_id) <= 128
                or any(not c.isascii() or not (c.isalnum() or c in '._:-') for c in request_id)):
            raise ValueError('request_id must be a short ASCII transport identifier')
        logical_id = request_id or uuid.uuid4().hex
        transport = None
        if self.max_retries:
            fingerprint = hashlib.sha256(json.dumps(
                {'provider': self.provider, 'payload': payload, 'phase': phase, 'agent': agent},
                sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            transport = {'request_id': logical_id, 'request_hash': fingerprint,
                         'max_retries': self.max_retries, 'initial_attempts_used': self.initial_attempts_used,
                         'retry_reserve_cap_usd': self.retry_reserve_cap_usd}
            self.initial_attempts_used = 0
        while True:
            waiting_since = time.monotonic()
            announced_wait = False
            while True:
                if self.stop.exists():
                    raise BudgetStop('Run STOP file exists')
                try:
                    if transport is None:
                        ident = self.ledger.reserve(self.run, reserve, phase, agent, self.model)
                    else:
                        ident = self.ledger.reserve(self.run, reserve, phase, agent, self.model, transport=transport)
                    break
                except BudgetBusy as busy:
                    if not announced_wait:
                        self.event('api_queue_wait', phase=phase, agent=agent, reason=str(busy))
                        announced_wait = True
                    if time.monotonic() - waiting_since > 600:
                        raise BudgetStop('Request queue exceeded 600 seconds; no request sent')
                    time.sleep(.1)
            if announced_wait:
                self.event('api_queue_ready', phase=phase, agent=agent,
                           waited_seconds=round(time.monotonic() - waiting_since, 3))
            self.event("api_reserved", call_id=ident, phase=phase, agent=agent, reserved_usd=reserve,
                       max_tokens=max_tokens)
            attempt = self.ledger.transport_attempt(ident)['attempt'] if transport is not None else 1
            try:
                headers = {'Authorization': 'Bearer ' + self.key}
                if self.provider == 'openai':
                    headers['X-Client-Request-Id'] = ident
                try:
                    response = requests.post(URLS[self.provider],
                                             headers=headers,
                                             json=payload, timeout=(self.connect_timeout, self.read_timeout), allow_redirects=False)
                except (requests.Timeout, requests.ConnectionError) as error:
                    error_type = transport_timeout_type(error)
                    if not self.max_retries or error_type is None:
                        raise
                    retained = self.ledger.retain_transport_timeout(ident, self.run, error_type)
                    self.event('api_timeout_retained', phase=phase, agent=agent,
                               request_id=logical_id, attempt=attempt, **retained)
                    if attempt >= self.max_retries + 1:
                        raise BudgetStop('Transport timeout retry limit reached; reservations retained') from None
                    delay = self.retry_backoff[attempt - 1]
                    self.event('api_retry_wait', call_id=ident, phase=phase, agent=agent,
                               request_id=logical_id, next_attempt=attempt + 1, wait_seconds=delay)
                    self._retry_wait(delay)
                    continue
                if response.status_code != 200:
                    self.event('api_http_error', call_id=ident, **response_diagnostics(response, self.key))
                    raise RuntimeError(f"{self.provider} HTTP {response.status_code}; no retry")
                data = response.json()
                usage = data.get("usage") or {}
                cost = self.price(usage) if self.provider == 'openai' else money(usage.get("cost"))
                receipt = {"id": data.get("id"), "model": data.get("model"),
                           "provider": 'openai' if self.provider == 'openai' else data.get("provider"), "usage": usage,
                           "max_tokens": max_tokens,
                           "finish_reason": (data.get("choices") or [{}])[0].get("finish_reason")}
                self.ledger.settle(ident, cost, receipt)
                self.last_completion = {**receipt, 'call_id': ident}
                self.event("api_charged", call_id=ident, phase=phase, agent=agent, cost_usd=cost, **receipt)
                content = data["choices"][0]["message"]["content"]
                # A confirmed, charged length-limited response can exhaust reasoning
                # before emitting text. Consume this slot; Policy may enlarge a later
                # allocated response. Never replay the request or release its charge.
                if receipt['finish_reason'] == 'length' and (content is None or isinstance(content, str) and not content.strip()):
                    return ''
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("Empty model response; charged call, no retry")
                return content
            except BaseException:
                self.ledger.uncertain(ident)
                self.event("api_stopped", call_id=ident, phase=phase, agent=agent)
                raise

    def price(self, usage):
        """OpenAI cost from token usage; reasoning tokens are part of completion_tokens."""
        prompt, completion = usage.get('prompt_tokens'), usage.get('completion_tokens')
        cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens') or 0
        if any(type(n) is not int or n < 0 for n in (prompt, completion, cached)):
            raise RuntimeError('OpenAI response has no token usage; charged call, no retry')
        cached = min(cached, prompt)
        return money(((prompt - cached) * self.prices['prompt'] + cached * self.prices['cached']
                      + completion * self.prices['completion']) / 1e6)
