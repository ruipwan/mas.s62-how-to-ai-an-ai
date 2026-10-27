"""Fixed active population with local strategy mutation inside a coding task.

This opt-in variant changes strategy allocation between rounds, not model weights.
Mutation selects from eight fixed heuristics with a seeded local RNG; no LLM
reflection or strategy-optimization call is made. It is not the cross-episode
evolution protocol from the original EoM paper.
"""
import hashlib
import random

from .engine import Agent, Engine


STRATEGY_INSTRUCTIONS = (
    'Inspect existing implementation and interfaces before editing.',
    'Make a focused change and verify it with an appropriate visible check.',
    'Look for a small reproducible failure before proposing a repair.',
    'Check boundary cases and preserve behavior that already passes visible checks.',
    'Read available public evidence before duplicating work by another agent.',
    'Keep interfaces stable and record assumptions that another contributor must verify.',
    'When an approach fails repeatedly, inspect the failure before choosing a different approach.',
    'Prefer a minimal testable implementation before adding refinements.',
)
INITIAL_STRATEGY = STRATEGY_INSTRUCTIONS[:2]
MAX_STRATEGY_INSTRUCTIONS = 4
NOTE_LIMIT = 800


def strategy_text(instructions):
    return ' '.join(instructions)


def strategy_hash(instructions):
    return hashlib.sha256(strategy_text(instructions).encode()).hexdigest()


def mutate_strategy(instructions, rng):
    """One bounded, seeded local mutation; no inference or task answers are involved."""
    instructions = list(instructions)
    available = [item for item in STRATEGY_INSTRUCTIONS if item not in instructions]
    operation = rng.choice(['replace', 'add'] if len(instructions) < MAX_STRATEGY_INSTRUCTIONS else ['replace', 'remove'])
    if operation == 'add':
        instructions.append(rng.choice(available))
    elif operation == 'remove':
        del instructions[rng.randrange(len(instructions))]
    else:
        instructions[rng.randrange(len(instructions))] = rng.choice(available)
    return tuple(instructions), operation


class OffspringEngine(Engine):
    def __init__(self, config, task, policy, events, out):
        if not config.offspring:
            raise ValueError('OffspringEngine requires explicit offspring opt-in')
        if config.condition != 'voluntary_pooled' or config.contribution_contracts:
            raise ValueError('Offspring supports voluntary_pooled without contribution contracts')
        if task.data['benchmark'] not in {'commit0', 'fixture'}:
            raise ValueError('Offspring supports Commit0 and the scripted fixture')
        super().__init__(config, task, policy, events, out)
        # Keep population-choice randomness independent of formation and merge ordering.
        self.offspring_rng = random.Random(f'offspring-v1:{config.seed}')
        # Clone and mutation controls must consume the same population-choice stream.
        self.mutation_rng = random.Random(f'offspring-mutation-v1:{config.seed}')
        self.retired_accounts = {}
        self.lineage = {a.name: {'id': a.name, 'parent': None, 'root': a.name, 'generation': 0,
                                'born_round': 0, 'retired_round': None,
                                'strategy_sha256': strategy_hash(INITIAL_STRATEGY)} for a in self.agents}
        self.strategies = {a.name: INITIAL_STRATEGY for a in self.agents}
        self.inherited_notes = {a.name: {} for a in self.agents}
        self.next_agent_number = config.k
        self.births = 0
        self.parent_invested = 0.0
        self.evolved_boundaries = set()

    def financial_account(self, name):
        return self.retired_accounts.get(name) or super().financial_account(name)

    def financial_accounts(self):
        return list(self.agents) + list(self.retired_accounts.values())

    def context(self, agent):
        result = super().context(agent)
        result.update(strategy=strategy_text(self.strategies[agent.name]),
                      lineage=dict(self.lineage[agent.name]),
                      inherited_notes=dict(self.inherited_notes[agent.name]),
                      offspring_rules=(
                          'Strategy is a mutable working heuristic; fixed task rules remain authoritative. '
                          'Inherited notes are unverified parent notes and public repository notes. '
                          'Verify their claims against the current source and visible checks. '
                          'A child has its own identity and no inherited team membership or contribution credit.'))
        return result

    def charge_rent(self):
        self.evolve_before_round()
        super().charge_rent()

    def _extreme(self, agents, *, richest):
        value = (max if richest else min)(a.wealth for a in agents)
        ties = sorted((a for a in agents if a.wealth == value), key=lambda a: a.name)
        return self.offspring_rng.choice(ties)

    def evolve_before_round(self):
        """Replace at most one mature agent at a scheduled round boundary."""
        if (self.step <= 0 or self.step >= self.round_limit or self.step % self.cfg.offspring_every
                or self.step in self.evolved_boundaries):
            return
        self.evolved_boundaries.add(self.step)
        mature = [a for a in self.agents if self.step - self.lineage[a.name]['born_round'] >= 2]
        if not mature or len(self.agents) < 2:
            self.emit('offspring_skipped', reason='no mature replacement candidate')
            return
        victim = self._extreme(mature, richest=False)
        endowment = self.cfg.offspring_endowment
        if endowment + 1e-12 < self.cfg.rent + self.cfg.fixed_bid:
            self.emit('offspring_skipped', reason='endowment cannot fund rent and fixed bid',
                      retired_candidate=victim.name, endowment=endowment)
            return
        # Both selection controls draw from the same fundable population. Sampling an
        # insolvent parent first would give random selection a different birth rate.
        parents = [a for a in self.agents if a is not victim
                   and a.wealth - endowment >= self.cfg.rent - 1e-12]
        if not parents:
            self.emit('offspring_skipped', reason='parent cannot fund child and retain rent',
                      retired_candidate=victim.name, endowment=endowment, eligible_parents=0)
            return
        parent = (self._extreme(parents, richest=True) if self.cfg.offspring_selection == 'wealth'
                  else self.offspring_rng.choice(sorted(parents, key=lambda a: a.name)))
        instructions, operation = self.strategies[parent.name], 'clone'
        if self.cfg.offspring_mutation:
            instructions, operation = mutate_strategy(instructions, self.mutation_rng)
        notes = self.repo.path / 'PROJECT_NOTES.md'
        inherited = {'parent_memory': parent.memory[-NOTE_LIMIT:],
                     'public_notes': notes.read_text(errors='replace')[:NOTE_LIMIT] if notes.exists() else ''}
        name = f'agent-{self.next_agent_number:02}'
        self.next_agent_number += 1
        child = Agent(name, endowment)
        parent_before = parent.wealth
        parent.wealth -= endowment
        self.parent_invested += endowment
        self.births += 1
        self.lineage[victim.name]['retired_round'] = self.step
        self.retired_accounts[victim.name] = victim
        old_team, dissolved = self.teams.leave_team(victim)
        if old_team:
            self.emit('leave', agent=victim.name, old_team=old_team, dissolved=dissolved,
                      reason='offspring retirement')
        self.teams.invitations = {key: value for key, value in self.teams.invitations.items()
                                  if victim.name not in {value['from'], value['to']}}
        # TeamManager keeps this list by reference. Rebinding would leave a stale population.
        self.agents[self.agents.index(victim)] = child
        self.inboxes.pop(victim.name, None)
        self.inboxes[name] = []
        self.resting.discard(victim.name)
        self.scratchpad_results.pop(victim.name, None)
        self.lineage[name] = {'id': name, 'parent': parent.name, 'root': self.lineage[parent.name]['root'],
                              'generation': self.lineage[parent.name]['generation'] + 1,
                              'born_round': self.step, 'retired_round': None,
                              'strategy_sha256': strategy_hash(instructions)}
        self.strategies[name] = instructions
        self.inherited_notes[name] = inherited
        self.teams.active_teams(self.step)
        self.emit('agent_retired', agent=victim.name, wealth=victim.wealth,
                  parent=self.lineage[victim.name]['parent'], root=self.lineage[victim.name]['root'],
                  generation=self.lineage[victim.name]['generation'], reason='offspring replacement')
        self.emit('offspring_investment', parent=parent.name, child=name, amount=endowment,
                  parent_wealth_before=parent_before, parent_wealth_after=parent.wealth)
        self.emit('agent_born', agent=name, birth_id=f'birth-{self.births:04}', parent=parent.name,
                  root=self.lineage[name]['root'], generation=self.lineage[name]['generation'],
                  retired_agent=victim.name, endowment=endowment,
                  parent_strategy_sha256=strategy_hash(self.strategies[parent.name]),
                  strategy_sha256=self.lineage[name]['strategy_sha256'], mutation=self.cfg.offspring_mutation,
                  mutation_operation=operation, population_size=len(self.agents))
        self.assert_accounting()

    def state(self):
        result = super().state()
        result['retired_accounts'] = [
            {'id': name, 'wealth': agent.wealth, 'retired_round': self.lineage[name]['retired_round']}
            for name, agent in sorted(self.retired_accounts.items())]
        result['offspring'] = {
            'version': 1, 'births': self.births, 'retired_count': len(self.retired_accounts),
            'parent_invested': self.parent_invested, 'selection': self.cfg.offspring_selection,
            'mutation': self.cfg.offspring_mutation,
            'lineage': [dict(self.lineage[name]) for name in sorted(self.lineage)],
            'strategies': {name: strategy_text(value) for name, value in sorted(self.strategies.items())},
            'inherited_notes': {name: dict(value) for name, value in sorted(self.inherited_notes.items())},
            'archive': [{**row, 'lineage': dict(self.lineage[row['id']]),
                         'strategy': strategy_text(self.strategies[row['id']])}
                        for row in result['retired_accounts']],
        }
        return result
