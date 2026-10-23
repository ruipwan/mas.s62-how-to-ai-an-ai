"""Metered text-only OpenRouter client implementing EoM's LLMClient interface."""

import json
import math
import os
from pathlib import Path

import requests
from hayekmas.utils.llm import LLMClient


class SpendLimitExceeded(RuntimeError):
    pass


class OpenRouterClient(LLMClient):
    api_name = "openrouter"
    MAX_EMPTY_RETRIES = 1
    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, model, *, budget, api_key=None, api_key_file=None, temperature=0.7):
        super().__init__(model, temperature=temperature)
        expected = {"max_usd", "prompt_usd_per_million", "completion_usd_per_million"}
        if set(budget) != expected or any(
            type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in budget.values()
        ):
            raise ValueError(f"budget requires positive finite values for {sorted(expected)}")
        self.budget = dict(budget)
        self.key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.key and api_key_file:
            self.key = Path(api_key_file).expanduser().read_text().strip()
        if not self.key:
            raise ValueError("Set OPENROUTER_API_KEY or model.extra_kwargs.api_key_file")
        self.records = []
        self.sink = None
        self.blocked = False

    def usage(self):
        from .token_allocation import phase_costs

        return {
            "max_usd": self.budget["max_usd"],
            "reported_cost_usd": math.fsum(r.get("cost_usd", 0) for r in self.records),
            "reserved_unconfirmed_usd": math.fsum(r["reserved_usd"] for r in self.records if "cost_usd" not in r),
            "requests": len(self.records),
            "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in self.records),
            "completion_tokens": sum(r.get("completion_tokens", 0) for r in self.records),
            "blocked": self.blocked,
            "phase_costs": phase_costs(self.records),
        }

    def record(self, event, record):
        if self.sink:
            self.sink({"event": event, **record})

    def _generate_impl(self, prompt, system_prompt=None, max_tokens=None, temperature=None, stop=None,
                       reasoning_effort=None, kind="unknown", **kwargs):
        if self.blocked:
            raise SpendLimitExceeded(
                "An earlier request has uncertain or unexpectedly high cost; inspect api_usage.jsonl"
            )
        if kwargs:
            raise ValueError("Unsupported OpenRouter generation options")
        if reasoning_effort not in {None, "none", "minimal", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("Unsupported reasoning effort")
        if not isinstance(kind, str) or not kind:
            raise ValueError("Request kind must be nonempty text")
        max_tokens = max_tokens or self.default_max_tokens
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [
            {"role": "user", "content": prompt}
        ]
        # Conservative text estimate, with framing margin; NOT a universal tokenizer bound.
        # The provider-side key cap remains the authoritative financial limit.
        estimated_input = len(json.dumps(messages, ensure_ascii=False).encode("utf-8")) + 1024
        reserve = (
            estimated_input * self.budget["prompt_usd_per_million"]
            + max_tokens * self.budget["completion_usd_per_million"]
        ) / 1_000_000
        usage = self.usage()
        if usage["reported_cost_usd"] + usage["reserved_unconfirmed_usd"] + reserve > self.budget["max_usd"]:
            raise SpendLimitExceeded("Next request's cost reservation exceeds this run's USD budget")
        record = {"request": len(self.records) + 1, "reserved_usd": reserve, "status": "pending",
                  "kind": kind, "reasoning_effort": reasoning_effort, "max_tokens": max_tokens}
        self.records.append(record)
        self.record("api_reserved", record)  # Persist BEFORE sending a potentially billable request.
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
            "temperature": self.default_temperature if temperature is None else temperature,
            "provider": {
                "sort": "price",
                "require_parameters": True,
                "allow_fallbacks": False,
                "max_price": {
                    "prompt": self.budget["prompt_usd_per_million"],
                    "completion": self.budget["completion_usd_per_million"],
                    "request": 0,
                },
            },
        }
        if payload["temperature"] is None:
            del payload["temperature"]
        if stop:
            payload["stop"] = stop
        if reasoning_effort is not None:
            payload["reasoning"] = {"effort": reasoning_effort}
        try:
            response = requests.post(
                self.URL,
                headers={"Authorization": f"Bearer {self.key}"},
                json=payload,
                timeout=120,
                allow_redirects=False,
            )
            if response.status_code != 200:
                # Do not log response bodies or headers, which may contain sensitive values.
                raise RuntimeError(f"OpenRouter HTTP {response.status_code}; no automatic retry")
            data = response.json()
            native = data.get("usage") or {}
            cost = native.get("cost")
            if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
                raise RuntimeError("OpenRouter returned no valid usage.cost; stopping with reservation retained")
            record.update(
                cost_usd=cost,
                generation_id=data.get("id"),
                model=data.get("model"),
                provider=data.get("provider"),
                status="charged",
            )
            for key in ("prompt_tokens", "completion_tokens"):
                value = native.get(key)
                if type(value) is int and value >= 0:
                    record[key] = value
            reasoning = (native.get("completion_tokens_details") or {}).get("reasoning_tokens")
            if type(reasoning) is int and reasoning >= 0:
                record["reasoning_tokens"] = reasoning
            self.record("api_charged", record)
            if cost > reserve + 1e-9:
                raise SpendLimitExceeded("Reported cost exceeded the reservation; stopping to review pricing")
            text = data["choices"][0]["message"]["content"]
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError("OpenRouter returned no answer text; no automatic retry")
            return text
        except BaseException:
            self.blocked = True
            record["status"] = "charged_error" if "cost_usd" in record else "uncertain"
            self.record("api_stopped", record)
            raise
