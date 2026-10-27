"""Swarm condition: a stigmergic task board, claims, a verified merge queue and bounties.

No membership, discussion or ballot phases. Each round the harness turns the public
repository's failing visible tests into open tasks (one per test module). Active
agents each make one claim call: open a group on a task, join a claimant's group, or
rest. Groups work in private workspaces with the usual equal work allowance. All
candidates are checked, ranked by verified gain and merged one 3-way patch at a time
onto the moving public head, re-checking after each merge. A merge may not fail any
test that currently passes. Virtual money is issued only when a visible test flips
to pass in the public repository; authors of fetched artifacts receive a royalty.
"""
import math

from hayekmas.adapters.teams.team import Team
from .engine import Engine
from .io import write_json


class SwarmEngine(Engine):
    def __init__(self, config, task, policy, events, out):
        super().__init__(config, task, policy, events, out)
        if task.data['benchmark'] not in {'commit0', 'fixture'}:
            raise ValueError('Swarm uses per-test visible results: commit0 (or the scripted fixture) only')
        self.first_failed = {}      # test id -> round in which the public repository first failed it
        self.groups = []            # round-local: (Team, task id)
        self.specialty = {a.name: {} for a in self.agents}  # agent -> {task id: tests flipped}
        self.bounties_issued = 0.0

    # ----- task board -----------------------------------------------------------------

    def public_failing(self):
        """Failing visible tests in the public repository, or None before a valid score exists."""
        if self.feedback.get('score') is None or self.feedback.get('failing') is None:
            return None
        failing = set(self.feedback['failing'])
        for test in failing:
            self.first_failed.setdefault(test, max(self.step, 0))
        return failing

    def bounty(self, test):
        age = max(self.step, 0) - self.first_failed.get(test, max(self.step, 0))
        return self.cfg.bounty_base * (1 + self.cfg.bounty_growth * age)

    def task_capacity(self, bootstrap):
        return 2 * self.cfg.crowding_cap if bootstrap else self.cfg.crowding_cap

    def board(self):
        """Open tasks ordered by bounty. Bootstrap: one task until the package builds and scores."""
        failing = self.public_failing()
        if failing is None:
            tasks = {'bootstrap': {'id': 'bootstrap', 'failing': None, 'bounty': None,
                                   'description': 'Make the package build and the visible test suite collect and run; '
                                                  'every test that passes in the first scored merge earns bounty_base.',
                                   'sample_tests': [], 'oldest_round': max(self.step, 0)}}
        else:
            tasks = {}
            for test in sorted(failing):
                module, _, name = test.partition('::')
                row = tasks.setdefault(module, {'id': module, 'failing': 0, 'bounty': 0.0, 'sample_tests': [],
                                                'oldest_round': self.first_failed[test]})
                row['failing'] += 1
                row['bounty'] += self.bounty(test)
                row['oldest_round'] = min(row['oldest_round'], self.first_failed[test])
                if len(row['sample_tests']) < 12:
                    row['sample_tests'].append(name)
            for row in tasks.values():
                row['bounty'] = round(row['bounty'], 6)
        groups = {}
        for team, task in self.groups:
            groups.setdefault(task, []).append({'leader': team.member_ids[0], 'size': len(team.member_ids),
                                                'room': len(team.member_ids) < self.cfg.max_team_size})
        capacity = self.task_capacity(failing is None)
        for tid, row in tasks.items():
            row['groups'] = groups.get(tid, [])
            row['open_slots'] = max(0, capacity - len(row['groups']))
        ordered = sorted(tasks.values(), key=lambda t: (-(t['bounty'] or math.inf), t['id']))
        return {'bootstrap': failing is None, 'open_tasks': len(ordered), 'tasks': ordered,
                'rules': f'At most {capacity} independent groups per task; a group holds up to '
                         f'{self.cfg.max_team_size} agents. Bounties grow by {self.cfg.bounty_growth:g} of '
                         f'bounty_base per round a test stays failing; royalty_rate {self.cfg.royalty_rate:g} '
                         'of a bounty goes to authors of artifacts you fetched.'}

    def context(self, agent):
        context = super().context(agent)
        # The agent-facing view carries counts, not two hundred test ids.
        context['public_feedback'] = {k: v for k, v in self.feedback.items() if k != 'failing'}
        context['task_board'] = self.board()
        context['your_record'] = self.specialty[agent.name]
        context['communication_rule'] = ('Claims, scratchpad notes and completed candidate artifacts persist; '
                                         'messages reach named agents; private workspaces stay private.')
        return context

    # ----- claims (replaces team formation) ------------------------------------------

    def form(self):
        for a in self.agents:
            a.team_tag = None
        self.groups = []
        board = self.board()
        tasks = {t['id']: t for t in board['tasks']}
        capacity = self.task_capacity(board['bootstrap'])
        slots = {tid: capacity for tid in tasks}
        # Scale effort to the open work: wake at most two agents per open group slot.
        awake = min(len(self.agents), 2 * sum(slots.values()))
        order = sorted(self.agents, key=lambda a: (-a.wealth, self.rng.random()))
        leaders = {}  # leader name -> (Team, task id)
        resting = [a.name for a in order[awake:]]
        for a in order[:awake]:
            reply = self.ask('claim', a)
            task, join = reply.get('task'), reply.get('join')
            reason = None
            if task is None:
                reason = 'rest'
            elif task not in tasks:
                reason = 'unknown task'
            elif join is not None:
                target = leaders.get(join)
                if target is None or target[1] != task:
                    reason = 'join must name the leader of a group on that task'
                elif len(target[0].member_ids) >= self.cfg.max_team_size:
                    reason = 'that group is full'
                else:
                    target[0].member_ids.append(a.name)
            elif slots[task] <= 0:
                reason = 'no open slot; join a group with room or claim another task'
            else:
                slots[task] -= 1
                leaders[a.name] = (Team(f'swarm:{a.name}', [a.name], self.step), task)
            if reason == 'rest':
                resting.append(a.name)
            elif reason:
                self.emit('invalid_action', agent=a.name, phase='claim', reason=reason)
                resting.append(a.name)
            self.emit('claim', agent=a.name, task=task if not reason else None, join=join,
                      approach=str(reply.get('approach', ''))[:300], accepted=reason is None)
            # Later claimants see the groups that exist so far.
            self.groups = list(leaders.values())
        for team, task in self.groups:
            if len(team.member_ids) > 1:
                for name in team.member_ids:
                    self.teams.agent(name).team_tag = team.team_id
            else:
                team.team_id = 'solo:' + team.member_ids[0]
        if resting:
            self.emit('rest', agents=sorted(resting), open_tasks=board['open_tasks'])

    def working_teams(self):
        return [team for team, _ in self.groups]

    # ----- candidates (no discussion, no ballot) -------------------------------------

    def candidate(self, team):
        task = next(t for group, t in self.groups if group is team)
        directory = self.out / 'candidates' / f'round-{self.step:02}' / team.team_id.replace(':', '-')
        directory.mkdir(parents=True)
        order = [self.teams.agent(n) for n in team.member_ids]
        self.rng.shuffle(order)
        view = next((t for t in self.board()['tasks'] if t['id'] == task), {'id': task})
        work = self.work(team, order, directory, [], extra={'claimed_task': view})
        proposal = self.assess(team, directory, work)
        check = proposal['check']
        proposal.update(task=task, bid=0.0, votes={}, pledges={},
                        participants=proposal['workers'] or list(team.member_ids))
        # Only scored code changes can enter the merge queue; notes live in the scratchpad.
        proposal['eligible'] = bool(proposal['changed'] and proposal['code_changed'] and check.get('build')
                                    and check.get('score') is not None and check.get('failing') is not None
                                    and proposal.get('shared_artifact'))
        self.emit('candidate', **{k: v for k, v in proposal.items() if k != 'source'})
        write_json(directory / 'candidate.json', proposal)
        return proposal

    # ----- merge queue (replaces the auction) ----------------------------------------

    def apply(self, proposals):
        eligible = [c for c in proposals if c['eligible']]
        if not eligible:
            self.emit('no_publication', reason='No buildable, scored code change this round')
            return
        public = self.public_failing()

        def expected_gain(c):
            if public is None:
                return c['check']['passed']
            failing = set(c['check']['failing'])
            return len(public - failing) - len(failing - public)

        ranked = sorted(eligible, key=lambda c: (-expected_gain(c), len(c['changed_paths']), self.rng.random()))
        merged = sum(self.merge(c) is not None for c in ranked)
        if not merged:
            self.emit('no_publication', reason='Every candidate conflicted or failed verification on the public tree')

    def merge(self, c):
        before = self.feedback
        before_failing = self.public_failing()
        result = self.verified_merge(c)
        if result is None:
            return None
        check, patch, mode, flipped = result
        after_failing = set(check['failing'])
        patch_path = self.out / 'patches' / f'round-{self.step:02}-{c["team"].replace(":", "-")}.diff'
        patch_path.parent.mkdir(exist_ok=True)
        patch_path.write_text(patch or '')
        bootstrap = before_failing is None
        paid, royalty, royalty_to = self.pay(c, flipped, bootstrap, check)
        self.feedback = check
        if self.task.data['benchmark'] == 'commit0':
            if self.denominator is None:
                self.denominator = check.get('total')
            self.high_water = max(self.high_water, check['score'])
        for test in after_failing:
            self.first_failed.setdefault(test, self.step)
        for test in flipped:
            self.first_failed.pop(test, None)
        compact = lambda f: {k: v for k, v in f.items() if k != 'failing'}
        record = {'round': self.step, 'team': c['team'], 'task': c['task'], 'participants': sorted(c['participants']),
                  'bid': 0.0, 'commit': self.repo.version, 'before': compact(before), 'after': compact(check),
                  'dividend': paid, 'royalty': royalty, 'royalty_to': royalty_to, 'mode': mode,
                  'flipped': flipped, 'tests_paid': check['passed'] if bootstrap else len(flipped),
                  'patch': str(patch_path.relative_to(self.out)), 'new_probe_count': 0,
                  'changed_paths': c.get('changed_paths', [])}
        self.history.append(record)
        self.emit('publication', **{k: v for k, v in record.items() if k != 'round'})
        self.assert_accounting()
        return record

    def pay(self, c, flipped, bootstrap, check):
        """Mint virtual money only against verified test flips; split work share, then royalty."""
        total = self.cfg.bounty_base * check['passed'] if bootstrap else math.fsum(self.bounty(t) for t in flipped)
        royalty_to = [n for n in c.get('fetched_from', []) if self.teams.agent(n)]
        royalty = total * self.cfg.royalty_rate if royalty_to else 0.0
        authors = [n for n in c['participants'] if self.teams.agent(n)] or list(c['members'])
        share = (total - royalty) / len(authors)
        for name in authors:
            self.teams.agent(name).wealth += share
            record = self.specialty[name]
            record[c['task']] = record.get(c['task'], 0) + (check['passed'] if bootstrap else len(flipped))
        for name in royalty_to:
            self.teams.agent(name).wealth += royalty / len(royalty_to)
        self.dividends += total
        self.bounties_issued += total
        self.emit('bounty', team=c['team'], task=c['task'], tests=check['passed'] if bootstrap else len(flipped),
                  total=total, share=share, authors=authors, royalty=royalty, royalty_to=royalty_to)
        return total, royalty, royalty_to

    def state(self):
        return {**super().state(), 'bounties_issued': self.bounties_issued, 'specialty': self.specialty,
                'task_board': self.board()}
