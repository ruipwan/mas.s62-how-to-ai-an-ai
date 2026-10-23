"""Small, explicit experiment surface; k counts agents, not concurrent requests."""
from dataclasses import asdict, dataclass
import math

BASELINES = {'single_agent', 'independent_accumulate', 'decentralized_vote', 'centralized', 'hybrid'}


@dataclass
class Config:
    k: int = 20
    rounds: int = 10
    max_rounds: int = 50
    round_extension: int = 10
    work_calls: int = 8
    random_team_size: int = 2
    membership_every: int = 1
    condition: str = 'voluntary_pooled'
    seed: int = 7
    initial_wealth: float = 1.0
    progress_alpha: float = 10.0
    terminal_reward: float = 10.0
    fixed_bid: float = 0.1
    model: str = 'qwen/qwen3-coder-next'
    reasoning_enabled: bool | None = None
    reasoning_effort: str | None = None
    structured_actions: bool = False
    provider_ignore: tuple[str, ...] = ()
    control_tokens: int = 1024
    work_tokens: int = 4096
    adaptive_tokens: bool = False
    max_response_tokens: int = 128000
    parallel_experiments: bool = False
    shared_scratchpad: bool = False
    contribution_contracts: bool = False
    independent_concurrency: int = 10
    sandbox_concurrency: int = 1
    synthesis_calls: int = 16
    orchestrator_in_population: bool = True
    baseline_memory_mb: int = 768
    context_chars: int = 96000
    run_cap_usd: float = 4.0
    input_price_cap: float = 0.30
    output_price_cap: float = 1.20

    def __post_init__(self):
        if type(self.orchestrator_in_population) is not bool:
            raise ValueError('orchestrator_in_population must be boolean')
        if type(self.shared_scratchpad) is not bool:
            raise ValueError('shared_scratchpad must be boolean')
        if type(self.contribution_contracts) is not bool:
            raise ValueError('contribution_contracts must be boolean')
        if self.contribution_contracts and (not self.shared_scratchpad or self.condition != 'voluntary_pooled'):
            raise ValueError('Contribution contracts require voluntary_pooled and shared_scratchpad')
        if type(self.parallel_experiments) is not bool:
            raise ValueError('parallel_experiments must be boolean')
        if type(self.adaptive_tokens) is not bool:
            raise ValueError('adaptive_tokens must be boolean')
        if type(self.structured_actions) is not bool:
            raise ValueError('structured_actions must be boolean')
        if self.reasoning_enabled is not None and type(self.reasoning_enabled) is not bool:
            raise ValueError('reasoning_enabled must be true, false, or null (provider default)')
        if self.reasoning_effort not in {None, 'none', 'low', 'medium', 'high', 'xhigh', 'max'}:
            raise ValueError('Unknown reasoning effort')
        if self.reasoning_enabled is False and self.reasoning_effort not in {None, 'none'}:
            raise ValueError('Disabled reasoning conflicts with a positive reasoning effort')
        if not isinstance(self.provider_ignore, (list, tuple)) or any(not isinstance(x, str) or not x for x in self.provider_ignore):
            raise ValueError('provider_ignore must contain provider identifiers')
        if self.condition not in {'solo', 'random_teams', 'voluntary_fixed', 'voluntary_pooled'} | BASELINES:
            raise ValueError('Unknown condition')
        for name in ('k', 'rounds', 'max_rounds', 'round_extension', 'work_calls', 'random_team_size', 'membership_every',
                     'control_tokens', 'work_tokens', 'max_response_tokens', 'context_chars',
                     'independent_concurrency', 'sandbox_concurrency', 'synthesis_calls', 'baseline_memory_mb'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + ' must be a positive integer')
        if self.independent_concurrency > 10 or self.sandbox_concurrency > 3:
            raise ValueError('Maximum concurrency: 10 independent agents, 3 sandboxes')
        if self.condition in BASELINES and (self.shared_scratchpad or not self.parallel_experiments):
            raise ValueError('Baselines require parallel_experiments and no market scratchpad')
        if self.condition == 'single_agent' and self.k != 1:
            raise ValueError('The single-agent baseline requires k=1')
        if self.condition in {'centralized', 'hybrid'} and self.worker_count < 1:
            raise ValueError('An orchestrator needs at least one worker')
        if not 256 <= self.baseline_memory_mb <= 2048:
            raise ValueError('Baseline sandbox memory must be 256–2048 MiB')
        if self.adaptive_tokens and not max(self.work_tokens, self.control_tokens) <= self.max_response_tokens <= 128000:
            raise ValueError('Adaptive response cap must cover initial limits and cannot exceed 128000')
        if not self.rounds <= self.max_rounds <= 50:
            raise ValueError('Require rounds <= max_rounds <= 50')
        if not 1 <= self.max_team_size <= self.k or self.context_chars < 12000:
            raise ValueError('Invalid team or context limit')
        if self.condition == 'random_teams' and self.random_team_size > self.max_team_size:
            raise ValueError('Random team size exceeds the team cap')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        for name in ('initial_wealth', 'progress_alpha', 'terminal_reward', 'fixed_bid',
                     'run_cap_usd', 'input_price_cap', 'output_price_cap'):
            n = getattr(self, name)
            if type(n) not in (int, float) or not math.isfinite(n) or n < 0:
                raise ValueError(name + ' must be finite and nonnegative')
        if not 0 < self.run_cap_usd <= 100 or min(self.input_price_cap, self.output_price_cap, self.fixed_bid) <= 0:
            raise ValueError('Invalid budget or price ceiling')

    @property
    def worker_count(self):
        return self.k - int(self.condition in {'centralized', 'hybrid'} and self.orchestrator_in_population)

    @property
    def max_team_size(self):
        # A singleton population remains solo; odd populations round down.
        return max(1, self.k // 2)

    def as_dict(self):
        data = {**asdict(self), 'max_team_size': self.max_team_size}
        if not self.contribution_contracts:
            data.pop('contribution_contracts')  # Preserve legacy disabled serialization.
        return data

    def estimate(self):
        # Every agent receives the same work allocation in all four conditions.
        work = (self.worker_count if self.condition in BASELINES else self.k) * self.max_rounds * self.work_calls
        formation = ((self.max_rounds - 1) // self.membership_every + 1) * 2 * self.k
        if self.condition in {'solo', 'random_teams'}:
            formation = 0
        control = 2 * self.k * self.max_rounds + formation
        synthesis = 0
        if self.condition in BASELINES:
            control = 0
            blocks = 1 + math.ceil((self.max_rounds - self.rounds) / self.round_extension)
            if self.condition in {'centralized', 'hybrid'}:
                control += self.max_rounds
            if self.condition in {'decentralized_vote', 'hybrid'}:
                control += self.worker_count * self.max_rounds
            if self.condition == 'decentralized_vote':
                control += blocks * self.worker_count
            if self.condition in {'independent_accumulate', 'centralized', 'hybrid'}:
                synthesis = blocks * self.synthesis_calls
        # Byte bound deliberately assumes four UTF-8 bytes per configured character.
        bound = 4 * (self.context_chars + 6000) + 4096
        max_work = self.max_response_tokens if self.adaptive_tokens else self.work_tokens
        max_control = self.max_response_tokens if self.adaptive_tokens else self.control_tokens
        ceiling = ((work + control + synthesis) * bound * self.input_price_cap
                   + ((work + synthesis) * max_work + control * max_control) * self.output_price_cap) / 1e6
        return {'initial_rounds': self.rounds, 'max_rounds': self.max_rounds,
                'adaptive_tokens': self.adaptive_tokens,
                'initial_response_limits': {'work': self.work_tokens, 'control': self.control_tokens},
                'maximum_response_limits': {'work': max_work, 'control': max_control},
                'initial_work_calls': (self.worker_count if self.condition in BASELINES else self.k) * self.rounds * self.work_calls,
                'work_output_ceiling_per_agent_round': self.work_calls * max_work,
                'work_output_ceiling_per_max_team_round': self.max_team_size * self.work_calls * max_work,
                'work_output_ceiling_per_population_round': self.k * self.work_calls * max_work,
                'work_calls_allocated': work, 'control_calls_max': control,
                'synthesis_calls_max': synthesis,
                'max_calls': work + control + synthesis, 'conservative_reservation_usd': round(ceiling, 4),
                'run_cap_usd': self.run_cap_usd,
                'note': 'Ceiling, not expected cost. A run may stop before its full allocation.'}
