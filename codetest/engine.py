"""Repository adaptation of the existing voluntary-team prototype.

Uses its unchanged TeamManager. Work precedes the auction; only the public-state
mutation is scarce. Model calls are serial so dollar reservations are auditable.
"""
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import shutil
import tempfile

from hayekmas.adapters.teams.team import TeamManager
from .budget import BudgetStop
from .config import BASELINES
from .io import write_json, sha
from .probes import discover, readable_observation, score as probe_score
from .sandbox import Repository, copy_source, safe_extract
from .scratchpad import Scratchpad
from .contracts import ContractMarket


@dataclass
class Agent:
    name: str
    wealth: float
    team_tag: str | None = None
    memory: str = ''


class Engine:
    def __init__(self, config, task, policy, events, out):
        self.cfg, self.task, self.policy, self.events = config, task, policy, events
        self.out = Path(out)
        self.rng = random.Random(config.seed)
        self.agents = [Agent(f'agent-{i:02}', config.initial_wealth) for i in range(config.k)]
        self.teams = TeamManager(self.agents, config.max_team_size)
        self.repo = Repository(self.out / 'public', task.seed)
        self.bank, self.history, self.snapshots = [], [], []
        self.public_board, self.inboxes = {}, {a.name: [] for a in self.agents}
        self.message_serial, self.delivered = 0, set()
        self.round_limit = config.rounds
        self.checkpoints = []
        self.burned = self.dividends = self.terminal_issued = 0.0
        self.high_water = 0.0
        self.denominator = None
        self.step = -1
        self.feedback = {'status': 'not_checked', 'score': None}
        self.scratchpad = Scratchpad(self.out / 'scratchpad', self.emit) if config.shared_scratchpad else None
        self.scratchpad_results = {}
        if config.contribution_contracts and task.data['benchmark'] not in {'programbench', 'fixture'}:
            raise ValueError('This contract version uses existing ProgramBench reference observations')
        if config.regression_gate and task.data['benchmark'] != 'commit0':
            raise ValueError('This regression gate version uses commit0 visible test scores')
        self.contracts = (ContractMarket(self.out / 'contracts', self.agents, self.scratchpad, self.emit)
                          if config.contribution_contracts else None)
        if config.condition == 'random_teams':
            order = list(self.agents)
            self.rng.shuffle(order)
            for i in range(0, len(order), config.random_team_size):
                group = order[i:i + config.random_team_size]
                if len(group) > 1:
                    self.teams.create_team(group, 0)

    def emit(self, kind, **values):
        self.events.emit(kind, **{'round': self.step, **values})

    def roster(self):
        return [{'id': a.name, 'team': a.team_tag or 'solo:' + a.name, 'wealth': round(a.wealth, 8)}
                for a in self.agents]

    def context(self, agent):
        task = self.task.public()
        task['problem'] = task['problem'][:5500] + '\n[Full specification: /task/SPEC.md]'
        notes = self.repo.path / 'PROJECT_NOTES.md'
        context = {'agent': agent.name, 'round': self.step, 'rounds': self.round_limit,
                'initial_rounds': self.cfg.rounds, 'max_rounds': self.cfg.max_rounds,
                'round_extension': self.cfg.round_extension,
                'population_size': self.cfg.k, 'max_team_size': self.cfg.max_team_size,
                'team_size_rule': 'max(1, floor(k/2)); a ceiling, not a target size',
                'wealth': agent.wealth, 'condition': self.cfg.condition,
                'fixed_bid': self.cfg.fixed_bid, 'task': task,
                'base_commit': self.repo.version, 'public_feedback': visible(self.feedback),
                'public_notes': notes.read_text(errors='replace')[:3000] if notes.exists() else '',
                'repository_files': [name for name in self.repo.git('ls-files').splitlines()
                                     if name.endswith(('.py', '.toml', '.md', '.rst', '.sh'))][:60],
                'memory': agent.memory[-2000:], 'public_probe_count': len(self.bank),
                'public_status_board': list(self.public_board.values()),
                'direct_messages': self.inboxes[agent.name][-8:],
                'communication_rule': 'Public status persists per sender; direct messages reach named agents. Team workspaces remain private.',
                'work_calls_per_agent': self.cfg.work_calls}
        if self.scratchpad:
            context['shared_scratchpad'] = self.scratchpad.observation()
            context['scratchpad_result'] = self.scratchpad_results.get(agent.name)
            context['communication_rule'] = 'Public messages and scratchpad notes persist; completed candidate artifacts can be fetched into private workspaces.'
        if self.contracts:
            context['contribution_market'] = self.contracts.observation()
        return context

    def ask(self, phase, agent, **extra):
        visible = list(self.public_board.values()) + self.inboxes[agent.name][-8:]
        unseen = [m['id'] for m in visible if (agent.name, m['id']) not in self.delivered and m['sender'] != agent.name]
        if unseen:
            self.emit('message_delivered', agent=agent.name, message_ids=unseen, phase=phase)
            self.delivered.update((agent.name, mid) for mid in unseen)
        reply = self.policy.ask(phase, agent, {**self.context(agent), **extra})
        if self.scratchpad and reply.get('scratchpad') is not None:
            try:
                result = self.scratchpad.write(agent.name, reply['scratchpad'], self.step)
            except (ValueError, TypeError) as error:
                result = {'status': 'rejected', 'reason': str(error)}
                self.emit('scratchpad_rejected', agent=agent.name, phase=phase, **result)
            self.scratchpad_results[agent.name] = result
        return reply

    def message(self, agent, recipients, message, channel):
        if not isinstance(message, str):
            return ''
        message = message[:1500]
        if message.strip():
            self.message_serial += 1
            mid = f'message-{self.message_serial}'
            self.emit('message', message_id=mid, sender=agent.name, recipients=recipients, text=message, channel=channel)
            item = {'id': mid, 'sender': agent.name, 'round': self.step, 'text': message[:600]}
            if channel == 'public':
                self.public_board[agent.name] = item
            elif channel == 'direct':
                for name in recipients:
                    self.inboxes[name].append(item)
                    self.inboxes[name] = self.inboxes[name][-8:]
        return message

    def route_message(self, agent, members, reply):
        """Deliver optional cross-team communication without granting extra work calls."""
        channel = reply.get('channel', 'team')
        if channel == 'public':
            recipients = [a.name for a in self.agents if a != agent]
        elif channel == 'direct':
            recipients = reply.get('recipients')
            if (not isinstance(recipients, list) or not recipients or
                    any(not isinstance(n, str) or not self.teams.agent(n) or n == agent.name for n in recipients)):
                self.emit('invalid_action', agent=agent.name, phase='communication', reason='Invalid direct recipients')
                return ''
            recipients = sorted(set(recipients))
        elif channel == 'team':
            recipients = [n for n in members if n != agent.name]
        else:
            self.emit('invalid_action', agent=agent.name, phase='communication', reason='Unknown channel')
            return ''
        return self.message(agent, recipients, reply.get('message'), channel)

    def form(self):
        if not self.cfg.condition.startswith('voluntary') or self.step % self.cfg.membership_every:
            return
        self.teams.invitations.clear()
        order = list(self.agents)
        self.rng.shuffle(order)
        roster, proposals, discussion = self.roster(), [], []
        for a in order:
            d = self.ask('membership', a, roster=roster, discussion=discussion[-4:])
            target = d.get('invite')
            if type(d.get('leave')) is not bool or (target is not None and
                    (not isinstance(target, str) or not self.teams.agent(target) or target == a.name)):
                self.emit('invalid_action', agent=a.name, phase='membership')
                continue
            proposals.append((a, d['leave'], target))
            msg = self.message(a, [b.name for b in self.agents if b != a], d.get('message'), 'public')
            discussion.append({'author': a.name, 'text': msg})
        for a, leave, _ in proposals:
            if leave and a.team_tag:
                old, dissolved = self.teams.leave_team(a)
                self.emit('leave', agent=a.name, old_team=old, dissolved=dissolved)
        for a, _, target in proposals:
            if target:
                try:
                    invite = self.teams.invite(a, self.teams.agent(target), self.step)
                    self.emit('invitation', **invite)
                except ValueError as e:
                    self.emit('invitation_skipped', agent=a.name, target=target, reason=str(e))
        self.rng.shuffle(order)
        for a in order:
            invitations = [v for v in self.teams.invitations.values() if v['to'] == a.name
                           and self.teams.agent(v['from']).team_tag == v['team']]
            if a.team_tag or not invitations:
                continue
            d = self.ask('join', a, invitations=invitations, roster=self.roster())
            self.message(a, [v['from'] for v in invitations], d.get('message'), 'direct')
            selected = d.get('invitation')
            if selected is None:
                continue
            try:
                tag, sender = self.teams.accept_invite(a, selected, self.step)
                self.emit('join', agent=a.name, sender=sender, team=tag)
            except ValueError as e:
                self.emit('invalid_action', agent=a.name, phase='join', reason=str(e))
        self.teams.invitations.clear()

    def working_teams(self):
        """Every agent works each round, solo or in its current team."""
        return list(self.teams.active_teams(self.step))

    def restore_protected(self, candidate):
        for rel in self.task.data.get('protected_paths', []):
            dest, source = candidate / rel, self.task.seed / rel
            if dest.exists():
                shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
            if source.is_dir():
                copy_source(source, dest)
            elif source.is_file():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)

    def candidate(self, team):
        members = [self.teams.agent(n) for n in team.member_ids]
        directory = self.out / 'candidates' / f'round-{self.step:02}' / team.team_id.replace(':', '-')
        directory.mkdir(parents=True)
        transcript = []
        order = list(members)
        self.rng.shuffle(order)
        for a in order:
            d = self.ask('discuss', a, members=team.member_ids, discussion=transcript[-6:])
            msg = self.route_message(a, team.member_ids, d)
            transcript.append({'author': a.name, 'text': msg})
        work = self.work(team, order, directory, transcript)
        proposal = self.assess(team, directory, work)
        check, changed, probes, workers = proposal['check'], proposal['changed'], work['probes'], work['workers']
        code_changed = proposal['code_changed']
        for a in order:
            d = self.ask('ballot', a, members=team.member_ids, candidate_check=visible(check),
                         changed=changed, new_probe_count=len(probes), discussion=transcript[-4:])
            yes, amount = d.get('yes'), d.get('pledge')
            valid = type(yes) is bool
            if self.cfg.condition == 'voluntary_fixed':
                valid = valid and type(d.get('authorize_fixed')) is bool
                yes = yes is True and d.get('authorize_fixed') is True and a.wealth >= self.cfg.fixed_bid
                amount = 0.0
            else:
                valid = valid and type(amount) in (int, float) and amount in (0, .025, .05, .1) and amount <= a.wealth
            proposal['votes'][a.name] = valid and yes is True
            proposal['pledges'][a.name] = float(amount) if valid and yes else 0.0
            self.emit('ballot', agent=a.name, team=team.team_id, yes=proposal['votes'][a.name],
                      pledge=proposal['pledges'][a.name], valid=valid, reason=d.get('message', ''))
        yes = sum(proposal['votes'].values())
        if self.cfg.condition == 'voluntary_fixed' and yes > len(members) / 2:
            proposal['pledges'] = {n: self.cfg.fixed_bid / yes if v else 0 for n, v in proposal['votes'].items()}
        proposal['bid'] = math.fsum(proposal['pledges'].values())
        proposal['participants'] = [n for n in workers if proposal['votes'].get(n)]
        # No-op drafts do not earn path credit. Failed code builds do not enter public state.
        regressed = self.regressed(check, code_changed)
        if self.cfg.regression_gate:
            proposal['regressed'] = regressed
            if regressed:
                self.emit('regression_blocked', team=team.team_id, score=check.get('score'),
                          total=check.get('total'), high_water=self.high_water, denominator=self.denominator)
        proposal['eligible'] = bool(yes > len(members) / 2 and proposal['bid'] > 0 and proposal['participants']
                                    and ((changed and check.get('build')) or (probes and not code_changed))
                                    and not regressed)
        self.emit('candidate', **{k: v for k, v in proposal.items() if k not in ('source', 'check')}, check=visible(check))
        write_json(directory / 'candidate.json', proposal)
        return proposal

    def work(self, team, order, directory, transcript, extra=None):
        """Equal round-robin work slots in one private workspace; writes directory/source."""
        probes, workers, fetched = [], set(), set()
        personal_results = {a.name: [] for a in order}
        base = self.repo.version
        with self.task.sandbox(self.repo.path) as box:
            bank_path = directory / 'public-probes.jsonl'
            bank_path.write_text(''.join(json.dumps(x) + '\n' for x in self.bank))
            box.put(bank_path, '/public/probes.jsonl', readonly=True)
            # Round robin work on the same private workspace; no member gets bonus calls.
            for slot in range(self.cfg.work_calls):
                for a in order:
                    reply = self.ask('work', a, members=team.member_ids, slot=slot,
                                     your_recent_work=personal_results[a.name][-2:],
                                     discussion=transcript[-5:], **(extra or {}))
                    if self.scratchpad:
                        box.put(self.scratchpad.path, '/public/scratchpad.json', readonly=True)
                    if self.contracts:
                        box.put(self.contracts.path, '/public/contracts.json', readonly=True)
                    tool = reply.get('tool')
                    if not isinstance(tool, str):
                        tool = 'invalid'
                    result = {'status': 'invalid_action'}
                    try:
                        if tool in {'inspect', 'edit', 'test'}:
                            result = box.execute(reply.get('command'))
                            if result['status'] == 'ok':
                                workers.add(a.name)
                        elif tool == 'probe' and self.task.data['benchmark'] == 'programbench':
                            entry = discover(self.task, reply)
                            if entry['id'] not in {p['id'] for p in probes + self.bank}:
                                probes.append(entry)
                            private_path = directory / 'private-probes.jsonl'
                            private_path.write_text(''.join(json.dumps({**p, 'readable': readable_observation(p['expected'], 16384)}) + '\n' for p in probes))
                            box.put(private_path, '/public/private-probes.jsonl', readonly=True)
                            result = {'status': 'reproduced', 'id': entry['id'],
                                      'observation': readable_observation(entry['expected']),
                                      'evidence_file': '/public/private-probes.jsonl'}
                            workers.add(a.name)
                        elif tool == 'message' and isinstance(reply.get('message'), str) and reply['message'].strip():
                            result = {'status': 'message'}
                        elif tool == 'fetch' and self.scratchpad:
                            result = self.scratchpad.fetch(reply.get('command'), a.name, base, box, self.step)
                            fetched.update(n for n in result['authors'] if n not in team.member_ids)
                        elif tool == 'contract' and self.contracts:
                            action = reply.get('contract')
                            if isinstance(action, dict) and action.get('operation') == 'settle':
                                with tempfile.TemporaryDirectory(prefix='codetest-contract-source-') as temp:
                                    source = Path(temp) / 'source'
                                    box.snapshot(source)
                                    result = self.contracts.perform(a.name, action, self.step, self.bank + probes,
                                                                   source=source, verify=lambda source, checks: probe_score(self.task, source, checks))
                            else:
                                result = self.contracts.perform(a.name, action, self.step, self.bank + probes)
                            self.assert_accounting()
                        else:
                            raise ValueError('Unknown tool or unavailable reference')
                    except (ValueError, TypeError, json.JSONDecodeError) as e:
                        result = {'status': 'invalid_action', 'error': str(e)}
                    msg = self.route_message(a, team.member_ids, reply)
                    if self.cfg.memory_trace:
                        # Bounded trace of this agent's own actions, not only its latest note.
                        a.memory = (a.memory + f"\n[r{self.step} s{slot} {tool}:{result.get('status')}] {msg}")[-2000:]
                    else:
                        a.memory = msg
                    entry = {'author': a.name, 'tool': tool, 'message': msg,
                             'result': json.dumps(result, ensure_ascii=False)[:4500]}
                    transcript.append(entry)
                    personal_results[a.name].append(entry)
                    self.emit('tool', agent=a.name, team=team.team_id, slot=slot, tool=tool, result=result)
                    if not box.alive:
                        raise RuntimeError('Candidate sandbox stopped; allocated work incomplete')
            source = directory / 'source'
            box.snapshot(source)
        return {'base': base, 'probes': probes, 'workers': workers, 'fetched_from': sorted(fetched),
                'personal_results': personal_results}

    def assess(self, team, directory, work):
        """Package, check and share the workspace snapshot; no votes or eligibility here."""
        source = directory / 'source'
        probes = work['probes']
        self.restore_protected(source)
        # Validate exactly the files that will be submitted, on a fresh checkout.
        if self.task.data['benchmark'] == 'programbench' and (source / 'executable').is_file():
            (source / 'executable').unlink()
        staged = Repository(directory / 'tracked-repository', source)
        staged.package(directory / 'candidate.tar.gz')
        source = directory / 'tracked-source'
        safe_extract((directory / 'candidate.tar.gz').read_bytes(), source)
        check = self.task.check(source, directory / 'check')
        if self.task.data['benchmark'] == 'programbench' and check.get('build'):
            check['probes'] = probe_score(self.task, source, self.bank + probes)
        # Evidence/notes-only proposals can be useful while the implementation is incomplete.
        changed = tree_digest(source) != tree_digest(self.repo.path)
        changed_paths = changed_files(self.repo.path, source)
        code_changed = any(not name.endswith('.md') for name in changed_paths)
        proposal = {'team': team.team_id, 'members': team.member_ids, 'base_commit': work['base'],
                    'source': str(source), 'check': check, 'changed': changed, 'changed_paths': changed_paths,
                    'code_changed': code_changed, 'new_probes': probes, 'workers': sorted(work['workers']),
                    'fetched_from': work['fetched_from'], 'votes': {}, 'pledges': {}}
        if self.scratchpad and changed:
            proposal['shared_artifact'] = self.scratchpad.share(proposal, self.repo.path, self.step)
        return proposal

    def regressed(self, check, code_changed):
        """Opt-in gate: a code change may not score below the public high water mark.

        Ties stay eligible. Before any valid visible score exists nothing can regress.
        A missing score or a changed test denominator after that counts as a regression.
        """
        if not (self.cfg.regression_gate and code_changed and self.task.data['benchmark'] == 'commit0'):
            return False
        if self.denominator is None:
            return False
        score = check.get('score')
        return score is None or check.get('total') != self.denominator or score < self.high_water

    def apply(self, proposals):
        eligible = [c for c in proposals if c['eligible']]
        if not eligible:
            self.emit('no_publication', reason='No buildable, changed candidate with majority consent and positive bid')
            return
        best = max(c['bid'] for c in eligible)
        ties = [c for c in eligible if math.isclose(c['bid'], best, abs_tol=1e-10)]
        winner = self.rng.choice(ties)
        if winner['base_commit'] != self.repo.version:
            self.emit('rejected', team=winner['team'], reason='stale_base')
            return
        patch = self.repo.replace(Path(winner['source']), winner['base_commit']) if winner['changed'] else None
        if not patch and not winner['new_probes']:
            self.emit('rejected', team=winner['team'], reason='no_tracked_changes')
            return
        patch_path = self.out / 'patches' / f'round-{self.step:02}.diff'
        patch_path.parent.mkdir(exist_ok=True)
        patch_path.write_text(patch or '')
        for name, pledge in winner['pledges'].items():
            self.teams.agent(name).wealth -= pledge
        previous = self.history[-1]['participants'] if self.history else []
        if previous:
            for name in previous:
                self.teams.agent(name).wealth += winner['bid'] / len(previous)
            self.emit('transfer', senders=winner['pledges'], recipients=previous,
                      amount=winner['bid'], team=winner['team'])
        else:
            self.burned += winner['bid']
            self.emit('burn', amount=winner['bid'], senders=winner['pledges'])
        before = self.feedback
        self.bank.extend(winner['new_probes'])
        (self.out / 'probes.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in self.bank))
        self.feedback = winner['check']
        if self.task.data['benchmark'] == 'programbench':
            self.feedback = {**self.feedback, **probe_score(self.task, self.repo.path, self.bank)}
        dividend = 0.0
        score = self.feedback.get('score')
        # No reward from agent-made probes or hidden NL2Repo tests during inference.
        if self.task.data['benchmark'] == 'commit0' and score is not None:
            total = self.feedback.get('total')
            if self.denominator is None:
                self.denominator = total
                self.high_water = score
            elif total == self.denominator:
                dividend = self.cfg.progress_alpha * max(0, score - self.high_water)
                self.high_water = max(score, self.high_water)
            else:
                self.emit('feedback_incomparable', reason='visible test denominator changed', total=total,
                          expected=self.denominator)
        for name in winner['participants']:
            self.teams.agent(name).wealth += dividend / len(winner['participants'])
        self.dividends += dividend
        record = {'round': self.step, 'team': winner['team'], 'participants': sorted(winner['participants']),
                  'bid': winner['bid'], 'commit': self.repo.version, 'before': visible(before),
                  'after': visible(self.feedback),
                  'dividend': dividend, 'patch': str(patch_path.relative_to(self.out)),
                  'new_probe_count': len(winner['new_probes']), 'changed_paths': winner.get('changed_paths', [])}
        self.history.append(record)
        self.emit('publication', **{k: v for k, v in record.items() if k != 'round'})
        self.assert_accounting()

    def assert_accounting(self):
        expected = self.cfg.k * self.cfg.initial_wealth - self.burned + self.dividends + self.terminal_issued
        actual = math.fsum(a.wealth for a in self.agents) + (self.contracts.escrow if self.contracts else 0)
        if any(a.wealth < -1e-8 for a in self.agents) or not math.isclose(expected, actual, abs_tol=1e-7):
            raise AssertionError('Wealth conservation failed')

    def snapshot(self):
        row = {'round': self.step, 'agents': self.roster(), 'commit': self.repo.version,
               'feedback': visible(self.feedback), 'accepted': len(self.history), 'probes': len(self.bank)}
        self.snapshots.append(row)
        self.emit('snapshot', **{k: v for k, v in row.items() if k != 'round'})

    def run(self):
        self.emit('started', config=self.cfg.as_dict(), task=self.task.public())
        self.feedback = self.task.check(self.repo.path, self.out / 'initial-check')
        if self.task.data['benchmark'] == 'commit0' and self.feedback.get('score') is not None:
            self.high_water, self.denominator = self.feedback['score'], self.feedback.get('total')
        self.snapshot()
        self.step = -1
        while self.step + 1 < self.round_limit:
            self.step += 1
            self.events.round = self.step
            if (self.out / 'STOP').exists():
                raise BudgetStop('Run STOP file exists')
            if self.contracts:
                self.contracts.expire(self.step)
                self.assert_accounting()
            self.form()
            self.emit('membership', agents=self.roster())
            teams = self.working_teams()
            self.rng.shuffle(teams)
            proposals = [self.candidate(team) for team in teams]
            self.apply(proposals)
            self.snapshot()
            if self.step + 1 == self.round_limit:
                ready, reason = self.answer_ready()
                checkpoint = {'completed_rounds': self.step + 1, 'answer_ready': ready,
                              'criterion': reason, 'publications': len(self.history),
                              'public_feedback': visible(self.feedback), 'next_limit': self.round_limit}
                if not ready and self.round_limit < self.cfg.max_rounds:
                    checkpoint['next_limit'] = min(self.cfg.max_rounds, self.round_limit + self.cfg.round_extension)
                    self.emit('rounds_extended', previous_limit=self.round_limit,
                              new_limit=checkpoint['next_limit'], reason=reason)
                    self.round_limit = checkpoint['next_limit']
                    self.policy.extend_for_no_answer(self.step + 1, reason)
                checkpoint['response_tokens'] = self.policy.token_state()
                self.checkpoints.append(checkpoint)
                self.emit('answer_checkpoint', **checkpoint)
                write_json(self.out / 'checkpoints.json', self.checkpoints)
            write_json(self.out / 'state.json', self.state())
        if self.contracts:
            self.contracts.expire(self.step, terminal=True)
            self.assert_accounting()
            write_json(self.out / 'state.json', self.state())

    def answer_ready(self):
        """Observable delivery gate, never a hidden-test claim or a quality threshold."""
        paths = changed_files(self.task.seed, self.repo.path)
        source_suffixes = {'.py', '.pyi', '.rs', '.go', '.c', '.cc', '.cpp', '.h', '.hpp',
                           '.js', '.ts', '.jsx', '.tsx', '.java', '.rb', '.sh', '.cs', '.swift'}
        source_present = any(Path(p).suffix in source_suffixes and (self.repo.path / p).is_file()
                             and (self.repo.path / p).read_bytes().strip() for p in paths)
        if not self.history or not source_present:
            return False, 'No accepted source change in the current public repository'
        if not self.feedback.get('build'):
            return False, 'The public source has not passed its visible build check'
        if self.task.data['benchmark'] == 'commit0' and self.feedback.get('score') is None:
            return False, 'The public package cannot yet produce a valid visible test report'
        return True, 'Changed public source passes the visible delivery gate; benchmark correctness is still unproven'

    def finalize(self):
        """Always export the current public state, even after an interrupted paid call."""
        root = self.out / 'submission' / self.task.data['id']
        root.mkdir(parents=True, exist_ok=True)
        self.repo.package(root / 'submission.tar.gz')
        # Export precisely the tracked submission, excluding generated/ignored files.
        from .sandbox import safe_extract
        safe_extract((root / 'submission.tar.gz').read_bytes(), root / 'source')
        self.emit('finalized', path=str(root.relative_to(self.out)), commit=self.repo.version,
                  charged=False, credit=False, artifact_sha256=sha(root / 'submission.tar.gz'))
        write_json(self.out / 'state.json', self.state())
        return root.parent

    def state(self):
        return {'config': self.cfg.as_dict(), 'agents': self.roster(), 'history': self.history,
                'round_limit': self.round_limit, 'completed_rounds': len([s for s in self.snapshots if s['round'] >= 0]),
                'checkpoints': self.checkpoints, 'public_status_board': list(self.public_board.values()),
                'response_tokens': self.policy.token_state(),
                'shared_scratchpad': self.scratchpad.observation() if self.scratchpad else None,
                **({'contribution_market': {'offers': self.contracts.offers, 'contracts': self.contracts.contracts,
                                            'escrow': self.contracts.escrow}} if self.contracts else {}),
                'snapshots': self.snapshots, 'burned': self.burned, 'progress_dividends': self.dividends,
                'terminal_issued': self.terminal_issued, 'public_feedback': self.feedback,
                'probe_count': len(self.bank), 'model_calls': self.policy.calls}


def visible(check):
    """Test results as agents and logs see them: counts, without per-test ids.

    junit() keeps the failing test ids for the swarm task board and merge verification.
    Classic conditions never showed them, so they stay internal unless a flag shows them.
    """
    return {k: v for k, v in check.items() if k != 'failing'}


def tree_digest(root):
    """Ignore generated caches and ProgramBench executables when detecting contributions."""
    import hashlib
    digest = hashlib.sha256()
    for p in sorted(Path(root).rglob('*')):
        rel = p.relative_to(root)
        if any(n in {'.git', '.venv', '__pycache__', '.pytest_cache', 'build', 'executable'} for n in rel.parts):
            continue
        if p.is_file():
            digest.update(str(rel).encode()); digest.update(p.read_bytes())
            digest.update(str(p.stat().st_mode & 0o111).encode())
    return digest.hexdigest()


def changed_files(before, after):
    def files(root):
        return {str(p.relative_to(root)): (sha(p), p.stat().st_mode & 0o111) for p in Path(root).rglob('*') if p.is_file()
                and not any(n in {'.git', '.venv', '__pycache__', '.pytest_cache', 'build', 'executable'}
                            for n in p.relative_to(root).parts)}
    a, b = files(Path(before)), files(Path(after))
    return sorted(n for n in a.keys() | b.keys() if a.get(n) != b.get(n))


def terminal_credits(state, score):
    """Post-evaluation accounting only; never feeds a hidden result back to inference."""
    if state['config'].get('condition') in BASELINES:
        return {}
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        return {}
    history = state['history']
    if not history:
        return {}
    pool = state['config']['terminal_reward'] * score
    credits = Counter()
    for item in history:
        for name in item['participants']:
            credits[name] += pool / len(history) / len(item['participants'])
    return dict(credits)
