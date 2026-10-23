"""Config launcher and CLI for the team extension."""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from .config import TeamConfig
from .env import ExactTaskEnv, ResearchTaskEnv, demo_tasks, load_tasks
from .mas import TeamMAS, dumps
from .policy import DemoPolicy, ModelPolicy
from .report import analyze, write_json
from .replay import write_replay


ROOT = Path(__file__).resolve().parents[3]


def source_hash():
    digest = hashlib.sha256()
    paths = sorted((ROOT / "hayekmas").rglob("*.py")) + sorted(
        (ROOT / "hayekmas" / "adapters" / "teams").glob("*.html")
    )
    for path in paths + [ROOT / "main.py"]:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run(raw, *, out=None, plots=True):
    allowed = {"domain", "teams", "backend", "model", "dataset", "split", "out", "plots", "environment", "budget"}
    if set(raw) - allowed:
        raise ValueError(f"Unknown top-level config fields: {set(raw) - allowed}")
    cfg = TeamConfig(**raw.get("teams", {}))
    backend = raw.get("backend", "demo")
    if backend not in {"demo", "llm"}:
        raise ValueError("backend must be demo or llm")
    environment = raw.get("environment", "exact")
    if environment not in {"exact", "researchworld"}:
        raise ValueError("environment must be exact or researchworld")
    if environment == "researchworld" and (backend != "llm" or not raw.get("dataset")):
        raise ValueError("researchworld requires an llm backend and a rubric dataset")
    policy = DemoPolicy(cfg.seed) if backend == "demo" else ModelPolicy(raw.get("model", {}), raw.get("budget"))
    dataset = raw.get("dataset")
    tasks = load_tasks(dataset, raw.get("split", "train")) if dataset else demo_tasks(cfg.seed, cfg.rounds)
    if len(tasks) < cfg.rounds:
        raise ValueError("Dataset must contain at least rounds distinct tasks in the selected split")
    destination = Path(out or raw.get("out", f"runs/teams-{cfg.condition}-{cfg.seed}")).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    try:
        upstream = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        upstream = "unknown"
    manifest = {
        "upstream_commit": upstream,
        "source_sha256": source_hash(),
        "python": platform.python_version(),
        "backend": policy.label,
        "environment": environment,
        "config": asdict(cfg),
        "split": raw.get("split", "train"),
        "model": {key: raw.get("model", {}).get(key) for key in ("api", "name")} if backend == "llm" else None,
        "dataset_sha256": hashlib.sha256(Path(dataset).read_bytes()).hexdigest() if dataset else None,
        "task_ids": [task.id for task in tasks[: cfg.rounds]],
        "budget": raw.get("budget"),
        "status": "running",
        "note": "Optional episode-boundary population evolution; no team treasury or diversity reward. "
        "Reviewed collaboration assigns temporary draft/check/revise duties; discussion mode does not.",
    }
    write_json(destination / "manifest.json", manifest)
    client = getattr(policy, "client", None)
    if hasattr(client, "usage"):

        def api_event(event):
            with (destination / "api_usage.jsonl").open("a", encoding="utf-8") as ledger:
                ledger.write(dumps(event) + "\n")
                ledger.flush()
                os.fsync(ledger.fileno())

        client.sink = api_event
    engine = None
    with (destination / "events.jsonl").open("w", encoding="utf-8") as stream:

        def emit(event):
            stream.write(dumps(event) + "\n")
            stream.flush()

        try:
            engine = TeamMAS(cfg, policy, emit)
            write_replay(engine, destination)
            # Keep a pristine reference; each test gets a disposable copy. The
            # simple runner starts untrained; the dedicated evaluator can supply
            # a trained source. Only audit/usage counters span independent tests.
            source = engine
            testing = raw.get("split", "train") == "test"
            for index, task in enumerate(tasks[: cfg.rounds]):
                if testing:
                    from .evaluation import evaluation_copy
                    previous = engine
                    engine = evaluation_copy(source, cfg.seed + index, policy)
                    engine.config = cfg
                    engine.round = index
                    engine.events, engine.metrics = previous.events, previous.metrics
                    for counter in ("calls", "requested_output_tokens", "reference_input_tokens", "invalid_actions"):
                        setattr(engine, counter, getattr(previous, counter))
                    engine.event_sink = emit
                    engine.emit("evaluation_reset", agents=engine.state()["agents"],
                                reason="fresh source snapshot; previous test feedback discarded")
                env = (
                    ResearchTaskEnv(task, cfg.reward, engine.judge)
                    if environment == "researchworld"
                    else ExactTaskEnv(task, cfg.reward)
                )
                engine.run_one_episode(env, training=not testing, reflection=not testing)
                write_replay(engine, destination)
            summary = analyze(engine, destination, plots=plots and raw.get("plots", True))
            write_json(destination / "population.json", engine.state())
            write_json(destination / "usage.json", engine.state()["usage"])
            # A separate event stream per agent facilitates post-hoc behavior coding.
            agent_dir = destination / "agents"
            agent_dir.mkdir()
            for name in [a.name for a in engine.agents] + [a["name"] for a in engine.retired_agents]:
                with (agent_dir / f"{name}.jsonl").open("w", encoding="utf-8") as target:
                    for event in engine.events:
                        involved = [event.get(k) for k in ("agent", "author", "sender", "recipient", "from", "to")]
                        involved += event.get("members", [])
                        involved += list(event.get("credits", {}))
                        involved += list(event.get("contributions", {}))
                        if name in involved:
                            target.write(dumps(event) + "\n")
            manifest["status"] = "complete"
            write_replay(engine, destination, "complete")
        except BaseException as error:
            manifest["status"] = "interrupted"
            manifest["error_type"] = type(error).__name__
            if engine:
                write_replay(engine, destination, "interrupted")
                write_json(destination / "interrupted_state.json", engine.state())
                write_json(destination / "usage.json", engine.state()["usage"])
            raise
        finally:
            write_json(destination / "manifest.json", manifest)
    return {"out": str(destination), **summary}


def main(raw=None):
    if raw is not None:
        print(json.dumps(run(raw), indent=2))
        return
    parser = argparse.ArgumentParser(description="EoM voluntary team prototype")
    parser.add_argument("--config", default="global_configs/teams_demo.json")
    parser.add_argument("--out")
    parser.add_argument("--condition", choices=["individual", "random_fixed", "self_selected_fixed", "dynamic"])
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    raw = json.loads(Path(args.config).read_text())
    raw.setdefault("teams", {})
    for key in ("condition", "rounds", "seed"):
        if getattr(args, key) is not None:
            raw["teams"][key] = getattr(args, key)
    print(json.dumps(run(raw, out=args.out, plots=not args.no_plots), indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileExistsError) as error:
        print(f"teams: {error}", file=sys.stderr)
        raise SystemExit(2)
