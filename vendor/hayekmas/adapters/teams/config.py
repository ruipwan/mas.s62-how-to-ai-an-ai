from dataclasses import dataclass, fields
import math


@dataclass
class TeamConfig:
    interaction_protocol: str = "legacy"
    round_schedule: str = "compact"
    token_profile: str = "standard"
    discussion_tokens: int = 2048
    formation_context_tokens: int = 6144
    pledge_context_tokens: int = 3072
    coordination_fee_lambda: float = 0.0
    objective_mode: str = "society"
    team_bid_rule: str = "voluntary"
    team_base_bid: float = 0.1
    terminal_policy: str = "funded"
    shared_draft_passes: int = 0
    condition: str = "dynamic"
    seed: int = 7
    num_agents: int = 12
    rounds: int = 12
    max_steps: int = 3
    initial_wealth: float = 20.0
    reward: float = 12.0
    reflection_cost: float = 2.0
    bid_cost_rate: float = 1.0
    negotiation_interval: int = 5
    max_team_size: int = 4
    bidding_turns: int = 2
    formation_turns: int = 3
    discussion_turns: int = 2
    finalization_enabled: bool = True
    bidding_mode: str = "negotiated"
    collaboration_mode: str = "discussion"
    bid_tokens: int = 256
    fixed_team_size: int = 2
    action_tokens: int = 256
    solution_tokens: int = 256
    judge_tokens: int = 1024
    environment_tokens: int = 2048
    inspect_tokens: int = 256
    update_tokens: int = 256
    evidence_tokens: int = 2048
    summary_tokens: int = 192
    context_tokens: int = 8192
    max_calls: int = 10000
    evolution_enabled: bool = False
    population_cap_multiplier: int = 2
    birth_interval: int = 5
    num_births_per_interval: int = 2
    p_a: float = 0.0
    p_b: float = 1.0
    periodical_good_p: float = 0.5
    rent: float = 0.0
    rent_interval: int = 5

    def __post_init__(self):
        if self.terminal_policy not in {"funded", "public_work"}:
            raise ValueError("Unknown terminal_policy")
        if (self.terminal_policy != "funded" or self.shared_draft_passes) and self.interaction_protocol != "rounds":
            raise ValueError("Public-work finalization and shared drafts require rounds")
        if self.objective_mode not in {"society", "wealth"}:
            raise ValueError("Unknown objective_mode")
        if self.team_bid_rule not in {"voluntary", "fixed"}:
            raise ValueError("Unknown team_bid_rule")
        if type(self.team_base_bid) not in (int, float) or not math.isfinite(self.team_base_bid) or self.team_base_bid <= 0:
            raise ValueError("team_base_bid must be finite and positive")
        if self.team_bid_rule == "fixed" and (self.interaction_protocol != "rounds" or self.round_schedule != "compact"):
            raise ValueError("Fixed team bids require compact rounds")
        if type(self.evolution_enabled) is not bool:
            raise ValueError("evolution_enabled must be boolean")
        if self.evolution_enabled and (self.interaction_protocol != "rounds" or self.condition != "dynamic"):
            raise ValueError("Population evolution requires dynamic round teaming")
        for name in ("p_a", "p_b", "periodical_good_p"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite probability")
        if self.p_a + self.p_b > 1:
            raise ValueError("p_a + p_b must not exceed 1")
        if self.interaction_protocol not in {"legacy", "rounds"}:
            raise ValueError("interaction_protocol must be legacy or rounds")
        if self.round_schedule not in {"compact", "full"}:
            raise ValueError("round_schedule must be compact or full")
        if self.token_profile not in {"standard", "solve_first"}:
            raise ValueError("token_profile must be standard or solve_first")
        if self.token_profile == "solve_first" and (self.interaction_protocol != "rounds" or self.round_schedule != "compact"):
            raise ValueError("solve_first token allocation requires compact round teaming")
        if self.interaction_protocol == "rounds" and (
            self.bidding_mode != "negotiated" or self.collaboration_mode != "discussion"
            or not self.finalization_enabled
        ):
            raise ValueError("rounds protocol requires negotiated bids, free discussion and finalization")
        if type(self.finalization_enabled) is not bool:
            raise ValueError("finalization_enabled must be a boolean")
        if self.bidding_mode not in {"negotiated", "sealed"}:
            raise ValueError("bidding_mode must be negotiated or sealed")
        if self.collaboration_mode not in {"discussion", "reviewed"}:
            raise ValueError("collaboration_mode must be discussion or reviewed")
        if self.collaboration_mode == "reviewed" and not self.finalization_enabled:
            raise ValueError("reviewed collaboration requires finalization_enabled")
        if self.condition not in {"individual", "random_fixed", "self_selected_fixed", "dynamic"}:
            raise ValueError("Unknown team condition")
        for name in ("initial_wealth", "reward", "reflection_cost", "bid_cost_rate", "coordination_fee_lambda", "rent"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.coordination_fee_lambda > 0 and (self.interaction_protocol != "rounds" or self.condition != "dynamic"):
            raise ValueError("Positive coordination fees require dynamic round teaming so members can decline and leave")
        if self.bid_cost_rate > 1:
            raise ValueError("bid_cost_rate must be in [0, 1]")
        for f in fields(self):
            if f.type is int:
                value = getattr(self, f.name)
                if type(value) is not int or value < (0 if f.name in {"shared_draft_passes", "seed", "birth_interval", "num_births_per_interval", "rent_interval"} else 1):
                    raise ValueError(f"{f.name} must be a positive integer (seed may be zero)")
        if min(self.formation_context_tokens, self.pledge_context_tokens) < 1024:
            raise ValueError("Control context budgets must be at least 1024 tokens")
        if (
            min(
                self.action_tokens,
                self.bid_tokens,
                self.discussion_tokens,
                self.solution_tokens,
                self.judge_tokens,
                self.inspect_tokens,
                self.update_tokens,
                self.summary_tokens,
            )
            < 64
        ):
            raise ValueError("Generation and summary budgets must be at least 64 tokens")
        if self.context_tokens < 2048 or self.evidence_tokens < 128:
            raise ValueError("context_tokens >= 2048 and evidence_tokens >= 128 are required")
        if self.evidence_tokens + self.update_tokens + 1024 > self.context_tokens:
            raise ValueError("context_tokens must leave room for evidence, strategy and instructions")
        if self.fixed_team_size > self.max_team_size:
            raise ValueError("fixed_team_size exceeds the engineering team-size cap")


class TokenBudget:
    """Explicit reference tokenizer; provider max_tokens limits generation separately.

    This is an o200k_base text-token budget, not a claim about every provider's
    native tokenizer, chat framing, or hidden reasoning tokens.
    """

    def __init__(self):
        import tiktoken

        self.encoding = tiktoken.get_encoding("o200k_base")

    def count(self, text):
        return len(self.encoding.encode(text, disallowed_special=()))

    def clip(self, text, limit, *, tail=False):
        tokens = self.encoding.encode(str(text), disallowed_special=())
        if limit <= 0:
            return ""
        selected = tokens[-limit:] if tail else tokens[:limit]
        # Decode only complete UTF-8 characters at a truncation boundary.
        return self.encoding.decode_bytes(selected).decode("utf-8", errors="ignore")
