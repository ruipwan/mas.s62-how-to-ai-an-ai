"""Matched-task evaluation: voluntary teams, direct single agent, independent pass@k.

The sampling pool is shared by single-agent and pass@k statistics. Test attempts
never see reference rubrics, other samples, judge feedback, or prior test tasks.
"""

import argparse
import ast
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

from .agent import TeamAction
from .config import TeamConfig, TokenBudget
from .env import ExactTaskEnv, ResearchTaskEnv, demo_tasks, load_tasks
from .mas import BudgetExceeded, TeamMAS, dumps
from .policy import DemoPolicy, INSTRUCTIONS, ModelPolicy
from .replay import write_replay
from .report import write_json
from .runtime import source_hash


BASELINE_SYSTEM = (
    'Solve the entire task accurately. Return one JSON object with an "answer" field. '
    "For a scientific task, include your complete solution, derivations, assumptions, and conclusions "
    "inside that answer string. For an exact-answer task, put only the final value in answer. "
    'You may use an optional "work" field for other working. You have no external tools.'
)


def estimate_pass_at_k(n, correct, k):
    if any(type(v) is not int for v in (n, correct, k)) or not 0 <= correct <= n or not 1 <= k <= n:
        raise ValueError("pass@k requires 0 <= correct <= n and 1 <= k <= n, all integers")
    if n - correct < k:
        return 1.0
    return 1.0 - math.prod(1.0 - k / i for i in range(n - correct + 1, n + 1))


def parse_answer(response, tokens, limit):
    if not isinstance(response, str) or tokens.count(response) > limit:
        return None, "missing or over-budget response"
    try:
        result = json.loads(response, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (ValueError, TypeError):
        return None, "malformed JSON"
    answer = result.get("answer") if isinstance(result, dict) else None
    if type(answer) not in (str, int, float) or not str(answer).strip():
        return None, "missing answer"
    if isinstance(answer, (int, float)) and not math.isfinite(answer):
        return None, "nonfinite answer"
    return answer, None


def append_jsonl(path, value, *, sync=False):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(dumps(value) + "\n")
        stream.flush()
        if sync:
            os.fsync(stream.fileno())


class ExperimentPolicy:
    """One budget/call meter shared by all methods, replicates and graders."""

    def __init__(self, delegate, config, max_calls, out):
        self.delegate, self.config, self.max_calls, self.out = delegate, config, max_calls, Path(out)
        self.label = delegate.label
        self.tokens = TokenBudget()
        self.client = self  # TeamMAS's environment judge uses this metered interface.
        self.context = {}
        self.totals = Counter()
        self.native = getattr(delegate, "client", None)
        if hasattr(self.native, "usage"):
            self.native.sink = lambda event: append_jsonl(
                self.out / "api_usage.jsonl", {**self.context, **event}, sync=True
            )

    def usage(self):
        return self.native.usage() if hasattr(self.native, "usage") else None

    def remaining_calls(self):
        return max(0, self.max_calls - self.totals["calls"])

    def snapshot(self):
        result = {
            k: self.totals[k]
            for k in (
                "calls",
                "judge_calls",
                "reference_input_tokens",
                "reference_output_tokens",
                "requested_output_tokens",
            )
        }
        native = self.usage()
        result.update(
            cost_usd=native["reported_cost_usd"] if native else (0.0 if self.label == "scripted_demo" else None),
            native_prompt_tokens=native["prompt_tokens"] if native else None,
            native_completion_tokens=native["completion_tokens"] if native else None,
        )
        return result

    def invoke(self, kind, prompt, system, cap, fn):
        if self.totals["calls"] >= self.max_calls:
            raise BudgetExceeded("Global comparison call budget exhausted")
        count = self.tokens.count(prompt) + self.tokens.count(system or "")
        if count > self.config.context_tokens:
            raise BudgetExceeded("Comparison request exceeds context_tokens")
        self.totals.update(
            calls=1, judge_calls=int(kind == "judge"), reference_input_tokens=count, requested_output_tokens=cap
        )
        request = int(self.totals["calls"])
        append_jsonl(
            self.out / "requests.jsonl",
            {
                **self.context,
                "event": "request",
                "request": request,
                "kind": kind,
                "system": system,
                "prompt": prompt,
                "max_tokens": cap,
            },
        )
        response = fn()
        self.totals["reference_output_tokens"] += self.tokens.count(response) if isinstance(response, str) else 0
        append_jsonl(
            self.out / "requests.jsonl",
            {**self.context, "event": "response", "request": request, "kind": kind, "response": response},
        )
        return response

    def respond(self, phase, agent, observation, max_tokens):
        prompt = dumps({"instruction": INSTRUCTIONS[phase], "observation": observation})
        return self.invoke(
            phase,
            prompt,
            agent.get_system_prompt(),
            max_tokens,
            lambda: self.delegate.respond(phase, agent, observation, max_tokens),
        )

    def generate(self, prompt, system_prompt=None, max_tokens=None):
        cap = max_tokens or self.config.solution_tokens
        kind = "solve" if system_prompt else "judge"

        def generate():
            if self.native:
                options = {}
                if getattr(self.native, "api_name", None) == "openrouter":
                    options["kind"] = kind
                    if kind == "judge" and self.config.token_profile == "solve_first":
                        options["reasoning_effort"] = "medium"
                return self.native.generate(prompt, system_prompt=system_prompt, max_tokens=cap, **options)
            if kind == "judge":
                raise ValueError("Scripted demo cannot grade scientific rubrics")
            problem = json.loads(prompt)["task"]["problem"]
            try:
                answer = sum(ast.literal_eval(problem.split(":", 1)[1].strip()))
            except (SyntaxError, ValueError, TypeError, IndexError):
                answer = None
            return dumps({"answer": answer})

        return self.invoke(kind, prompt, system_prompt, cap, generate)


def usage_delta(after, before):
    return {k: after[k] - before[k] if after[k] is not None and before[k] is not None else None for k in after}


def summed_usage(rows):
    if not rows:
        return {}
    return {k: sum(row[k] for row in rows) if all(row[k] is not None for row in rows) else None for k in rows[0]}


def make_env(task, environment, config, policy):
    if environment == "researchworld":
        return ResearchTaskEnv(
            task, config.reward, lambda prompt: policy.generate(prompt, max_tokens=config.judge_tokens)
        )
    return ExactTaskEnv(task, config.reward)


def evaluation_copy(source, seed, policy):
    """Copy training state once per test task, discarding all subsequent test feedback."""
    clone = TeamMAS(replace(source.config, rounds=1, seed=seed), policy)
    clone.population, clone.team_manager, clone.public_summaries, clone.inboxes = deepcopy(
        (source.population, source.team_manager, source.public_summaries, source.inboxes)
    )
    clone.training = False
    clone.invitations = clone.team_manager.invitations
    clone.invitations.clear()
    clone.initial_total = math.fsum(agent.wealth for agent in clone.agents)
    clone.events.clear()
    clone.emit(
        "initialized",
        condition=clone.config.condition,
        backend=policy.label,
        config=asdict(clone.config),
        roster=clone.roster(),
        agents=clone.state()["agents"],
    )
    return clone


def team_episode(engine, task, environment, policy, destination, *, formation, reflection):
    destination.mkdir(parents=True, exist_ok=False)
    engine.event_sink = lambda event: append_jsonl(destination / "events.jsonl", event)
    append_jsonl(destination / "events.jsonl", engine.events[0])
    before = policy.snapshot()
    write_replay(engine, destination)
    try:
        metric = engine.run_one_episode(
            make_env(task, environment, engine.config, policy), formation=formation, reflection=reflection
        )
        write_replay(engine, destination, "complete")
        write_json(destination / "population.json", engine.state())
    except BaseException:
        write_replay(engine, destination, "interrupted")
        write_json(destination / "interrupted_state.json", engine.state())
        raise
    final = next((e for e in reversed(engine.events) if e["event"] == "submission" and e.get("final")), None)
    return {
        "score": metric["score"],
        "answer": final["answer"] if final else None,
        "invalid_actions": metric["invalid_actions"],
        "step_count": metric["step_count"],
        "usage": usage_delta(policy.snapshot(), before),
    }


def attempt(task, environment, policy):
    before = policy.snapshot()
    raw = policy.generate(
        dumps({"task": task.public()}), system_prompt=BASELINE_SYSTEM, max_tokens=policy.config.solution_tokens
    )
    answer, error = parse_answer(raw, policy.tokens, policy.config.solution_tokens)
    env = make_env(task, environment, policy.config, policy)
    env.initialize()
    if error is None:
        env.apply(TeamAction(answer, "single-agent", final=True))
    return {
        "answer": answer,
        "score": env.get_terminal_score(),
        "invalid": error,
        "usage": usage_delta(policy.snapshot(), before),
    }


def validate(raw):
    allowed = {
        "domain",
        "teams",
        "backend",
        "model",
        "dataset",
        "split",
        "out",
        "plots",
        "environment",
        "budget",
        "comparison",
    }
    if set(raw) - allowed:
        raise ValueError(f"Unknown config fields: {set(raw) - allowed}")
    settings = {
        "samples": 12,
        "ks": [1, 2, 4, 8, 12],
        "seeds": [7],
        "task_seed": 7,
        "train_rounds": 0,
        "train_dataset": None,
        "max_total_calls": 20000,
        "pass_threshold": None,
        **raw.get("comparison", {}),
    }
    if set(settings) - {
        "samples",
        "ks",
        "seeds",
        "task_seed",
        "train_rounds",
        "train_dataset",
        "max_total_calls",
        "pass_threshold",
    }:
        raise ValueError("Unknown comparison setting")
    for k in ("samples", "max_total_calls", "task_seed", "train_rounds"):
        if type(settings[k]) is not int or settings[k] < (0 if k in {"task_seed", "train_rounds"} else 1):
            raise ValueError(f"Invalid comparison.{k}")
    ks, seeds = settings["ks"], settings["seeds"]
    if not isinstance(ks, list) or not ks or any(type(k) is not int or not 1 <= k <= settings["samples"] for k in ks):
        raise ValueError("ks must be nonempty integers between 1 and samples")
    if len(ks) != len(set(ks)):
        raise ValueError("ks must be distinct")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(s) is not int or s < 0 for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seeds must be distinct nonnegative integers")
    cfg = TeamConfig(**raw.get("teams", {}))
    environment, backend = raw.get("environment", "exact"), raw.get("backend", "demo")
    if environment not in {"exact", "researchworld"} or backend not in {"llm", "demo"}:
        raise ValueError("Unknown environment/backend")
    if environment == "researchworld" and (backend != "llm" or not raw.get("dataset")):
        raise ValueError("Scientific comparison requires an llm backend and explicit dataset")
    threshold = settings["pass_threshold"]
    if threshold is None:
        threshold = 0.5 if environment == "researchworld" else 1.0
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("pass_threshold must be in (0,1]")
    if environment == "exact" and threshold != 1:
        raise ValueError("Exact tasks use pass_threshold=1")
    settings["pass_threshold"] = threshold
    tasks = (
        load_tasks(raw["dataset"], raw.get("split", "test"))
        if raw.get("dataset")
        else demo_tasks(settings["task_seed"], cfg.rounds)
    )
    if len(tasks) < cfg.rounds:
        raise ValueError(f"Requested {cfg.rounds} tasks; dataset contains only {len(tasks)}")
    tasks = tasks[: cfg.rounds]
    training = []
    if settings["train_rounds"]:
        if not settings["train_dataset"]:
            raise ValueError("train_rounds requires train_dataset")
        training = load_tasks(settings["train_dataset"], "train")
        if len(training) < settings["train_rounds"]:
            raise ValueError("Not enough training tasks")
        training = training[: settings["train_rounds"]]
        if {t.id for t in training} & {t.id for t in tasks} or {t.problem.strip() for t in training} & {
            t.problem.strip() for t in tasks
        }:
            raise ValueError("Training and test tasks overlap")
    return cfg, settings, environment, backend, tasks, training


def summarize(pairs, ks, seeds):
    rows = []
    for method in ["team", "single_agent", *[f"pass@{k}" for k in ks]]:
        per_seed = []
        usages = []
        for seed in seeds:
            selected = [p for p in pairs if p["seed"] == seed]
            values = []
            for pair in selected:
                if method == "team":
                    values.append(float(pair["team"]["passed"]))
                    usages.append(pair["team"]["usage"])
                elif method == "single_agent":
                    values.append(float(pair["samples"][0]["passed"]))
                    usages.append(pair["samples"][0]["usage"])
                else:
                    k = int(method.split("@")[1])
                    values.append(pair["pass_at_k"][str(k)])
                    pool = summed_usage([s["usage"] for s in pair["samples"]])
                    usages.append(
                        {
                            key: value * k / len(pair["samples"]) if value is not None else None
                            for key, value in pool.items()
                        }
                    )
            per_seed.append(statistics.mean(values))
        rows.append(
            {
                "method": method,
                "pass_rate": statistics.mean(per_seed),
                "mean_score": statistics.mean(
                    p["team"]["score"] if method == "team" else p["samples"][0]["score"] for p in pairs
                )
                if method in {"team", "single_agent"}
                else None,
                "replicate_rates": per_seed,
                "replicate_sd": statistics.stdev(per_seed) if len(per_seed) > 1 else None,
                "usage": summed_usage(usages),
                "usage_basis": "expected cost of k random samples incl. grading; pool reused"
                if method.startswith("pass@")
                else "measured calls incl. grading; single-agent reuses first sample",
            }
        )
    return rows


def compare(raw, *, out=None):
    cfg, settings, environment, backend, tasks, training = validate(raw)
    destination = Path(out or raw.get("out", "runs/baseline-comparison")).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None
    manifest = {
        "status": "running",
        "source_sha256": source_hash(),
        "backend": backend,
        "environment": environment,
        "dataset": raw.get("dataset"),
        "dataset_split": raw.get("split", "test"),
        "teams": asdict(cfg),
        "comparison": settings,
        "model": {k: raw.get("model", {}).get(k) for k in ("api", "name")},
        "model_options": {"temperature": raw.get("model", {}).get("extra_kwargs", {}).get("temperature", 0.7)},
        "budget": raw.get("budget"),
        "dataset_sha256": digest(raw.get("dataset")),
        "train_dataset_sha256": digest(settings["train_dataset"]),
        "tasks": [{"id": t.id, "subject": t.subject} for t in tasks],
        "train_ids": [t.id for t in training],
        "test_protocol": "fresh copy per task; no test reflection or cross-task feedback; within-task payments retained",
        "seed_scope": "local scheduling and ties only; provider samples are stochastic, not guaranteed seed-reproducible",
    }
    write_json(destination / "manifest.json", manifest)
    pairs, train_runs = [], []
    policy = None
    from .evaluation_report import write_report

    def persist():
        complete = manifest["status"] == "complete"
        result = {
            "status": manifest["status"],
            "backend": backend,
            "environment": environment,
            "dataset_split": manifest["dataset_split"],
            "planned_pairs": len(tasks) * len(settings["seeds"]),
            "completed_pairs": len(pairs),
            "ks": settings["ks"],
            "samples_per_task": settings["samples"],
            "pass_threshold": settings["pass_threshold"],
            "tasks": manifest["tasks"],
            "pairs": pairs,
            "training": train_runs,
            "summary": summarize(pairs, settings["ks"], settings["seeds"]) if complete else [],
            "total_actual_usage": policy.snapshot() if policy else None,
            "provider": policy.usage() if policy else None,
            "note": "Pass@k is oracle success, not an answer-selection algorithm. Costs differ; no equal-compute claim. Scripted results only test plumbing.",
        }
        write_json(destination / "comparison.json", result)
        write_json(destination / "manifest.json", manifest)
        write_report(result, destination)
        return result

    try:
        delegate = DemoPolicy(cfg.seed) if backend == "demo" else ModelPolicy(raw.get("model", {}), raw.get("budget"))
        policy = ExperimentPolicy(delegate, cfg, settings["max_total_calls"], destination)
        persist()
        for seed in settings["seeds"]:
            if backend == "demo":
                policy.delegate = DemoPolicy(seed)
            base = TeamMAS(replace(cfg, seed=seed, rounds=max(1, len(training))), policy)
            train_before = policy.snapshot()
            if training:
                train_out = destination / f"seed-{seed}" / "training"
                train_out.mkdir(parents=True)
                base.event_sink = lambda e: append_jsonl(train_out / "events.jsonl", e)
                append_jsonl(train_out / "events.jsonl", base.events[0])
                try:
                    for task in training:
                        policy.context = {"method": "team", "phase": "train", "seed": seed, "task_id": task.id}
                        base.run_one_episode(make_env(task, environment, cfg, policy))
                        write_replay(base, train_out)
                    write_replay(base, train_out, "complete")
                    write_json(train_out / "population.json", base.state())
                except BaseException:
                    write_replay(base, train_out, "interrupted")
                    write_json(train_out / "interrupted_state.json", base.state())
                    raise
            train_runs.append(
                {"seed": seed, "tasks": len(training), "usage": usage_delta(policy.snapshot(), train_before)}
            )
            for index, task in enumerate(tasks):
                context = {"phase": "test", "seed": seed, "task_id": task.id}
                policy.context = {**context, "method": "team"}
                engine = evaluation_copy(base, seed + index, policy)
                relative = Path(f"seed-{seed}") / f"task-{index + 1:03d}" / "team"
                team = team_episode(
                    engine,
                    task,
                    environment,
                    policy,
                    destination / relative,
                    formation=cfg.interaction_protocol == "rounds" or not bool(training),
                    reflection=False,
                )
                team["passed"] = team["answer"] is not None and team["score"] >= settings["pass_threshold"]
                team["replay"] = str(relative / "replay.html")
                append_jsonl(destination / "outcomes.jsonl", {**context, "method": "team", **team})
                samples = []
                for sample in range(settings["samples"]):
                    policy.context = {**context, "method": "independent", "sample": sample}
                    item = attempt(task, environment, policy)
                    item["sample"] = sample
                    item["passed"] = item["invalid"] is None and item["score"] >= settings["pass_threshold"]
                    samples.append(item)
                    append_jsonl(destination / "outcomes.jsonl", {**policy.context, **item})
                correct = sum(s["passed"] for s in samples)
                pairs.append(
                    {
                        **context,
                        "subject": task.subject,
                        "problem": task.problem,
                        "team": team,
                        "samples": samples,
                        "pass_at_k": {str(k): estimate_pass_at_k(len(samples), correct, k) for k in settings["ks"]},
                    }
                )
                persist()
        manifest["status"] = "complete"
        return {"out": str(destination), **persist()}
    except BaseException as error:
        manifest.update(status="interrupted", error_type=type(error).__name__)
        persist()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="global_configs/compare_science_openrouter.json")
    parser.add_argument("--out")
    parser.add_argument("--tasks", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--ks", help="Comma-separated k values, each <= samples")
    parser.add_argument("--seeds", help="Comma-separated local run seeds")
    parser.add_argument("--train-tasks", type=int)
    parser.add_argument("--validate-only", action="store_true", help="Inspect tasks/config; no model or paid requests")
    args = parser.parse_args()
    raw = json.loads(Path(args.config).read_text())
    settings = raw.setdefault("comparison", {})
    if args.tasks is not None:
        raw.setdefault("teams", {})["rounds"] = args.tasks
    if args.samples is not None:
        settings["samples"] = args.samples
    if args.train_tasks is not None:
        settings["train_rounds"] = args.train_tasks
    for key in ("ks", "seeds"):
        if getattr(args, key) is not None:
            settings[key] = [int(v) for v in getattr(args, key).split(",")]
    if args.validate_only:
        cfg, settings, env, backend, tasks, training = validate(raw)
        print(
            json.dumps(
                {
                    "environment": env,
                    "backend": backend,
                    "evaluation_tasks": len(tasks),
                    "dataset_split": raw.get("split", "test"),
                    "train_tasks": len(training),
                    "subjects": dict(Counter(t.subject or "unspecified" for t in tasks)),
                    "task_ids": [t.id for t in tasks],
                    "samples_per_task": settings["samples"],
                    "ks": settings["ks"],
                    "paid_requests": 0,
                },
                indent=2,
            )
        )
        return
    result = compare(raw, out=args.out)
    print(
        json.dumps(
            {k: result[k] for k in ("out", "status", "completed_pairs", "summary", "total_actual_usage")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
