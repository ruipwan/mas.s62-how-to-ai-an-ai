"""Descriptive metrics only: these outputs do not label agents with roles."""

from collections import Counter
import csv
import json
from pathlib import Path
import statistics


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def gini(values):
    values = sorted(values)
    total = sum(values)
    if not values or not total:
        return 0.0
    n = len(values)
    return sum((2 * i - n - 1) * value for i, value in enumerate(values, 1)) / (n * total)


def analyze(engine, out, plots=True):
    out = Path(out)
    metrics, events = engine.metrics, engine.events
    names = sorted({n for row in metrics for key in ("wealth", "paid") for n in row[key]})
    previous = None
    for row in metrics:
        row["wealth_gini"] = gini(row["wealth"].values())
        row["contribution_gini"] = gini(row["contributions"].values())
        current = row["membership"]
        row["membership_turnover"] = (sum(current.get(n, "absent") != previous.get(n, "absent") for n in names) / max(1, len(names))) if previous else 0.0
        previous = current
    write_json(out / "metrics.json", metrics)
    features = []
    edges = Counter((e["sender"], e["recipient"]) for e in events if e["event"] == "message")
    for name in names:
        messages = [e for e in events if e["event"] == "team_message" and e["agent"] == name]
        proposals = [e for e in events if e["event"] == "candidate" and e["author"] == name]
        features.append(
            {
                "agent": name,
                "messages": len(messages),
                "proposals": len(proposals),
                "first_proposals": sum(e["id"] == "candidate-0" for e in proposals),
                "mean_message_tokens": statistics.mean(engine.tokens.count(e["text"]) for e in messages)
                if messages
                else 0,
                "mean_contribution_fraction": statistics.mean(row["contribution_fraction"].get(name, 0) for row in metrics)
                if metrics
                else 0,
                "invitations": sum(e["event"] == "invited" and e["from"] == name for e in events),
                "leaves": sum(e["event"] == "left" and e["agent"] == name and e["team"] is not None for e in events),
                "reflections": sum(e["event"] == "reflection_paid" and e["agent"] == name for e in events),
                "bids_paid": sum(row["paid"].get(name, 0) for row in metrics),
                "bid_income": sum(row["bid_income"].get(name, 0) for row in metrics),
                "reward_income": sum(row["reward_income"].get(name, 0) for row in metrics),
                "final_wealth": engine.lookup(name).wealth if engine.lookup(name) else None,
            }
        )
    with (out / "agent_features.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(features[0]))
        writer.writeheader()
        writer.writerows(features)
    write_json(
        out / "communication_graph.json",
        [{"source": a, "target": b, "messages": count} for (a, b), count in sorted(edges.items())],
    )
    scalar_keys = [
        "round",
        "task_id",
        "score",
        "reward",
        "winner",
        "step_count",
        "bid_paid_total",
        "bid_transfer_total",
        "bid_burn_total",
        "wealth_gini",
        "contribution_gini",
        "membership_turnover",
        "invalid_actions",
    ]
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)
    summary = {
        "backend": engine.policy.label,
        "condition": engine.config.condition,
        "completed_rounds": len(metrics),
        "mean_score": statistics.mean(row["score"] or 0 for row in metrics) if metrics else 0,
        "final_team_sizes": [len(group) for group in engine.groups().values()],
        "final_wealth_gini": gini(a.wealth for a in engine.agents),
        "decision_calls": engine.calls,
        "invalid_actions": engine.invalid_actions,
        "accounting": engine.state()["accounting"],
        "provider_usage": engine.state()["usage"]["provider"],
        "interpretation": "Scripted demo validates plumbing only; no emergent-role claim."
        if engine.policy.label == "scripted_demo"
        else "Descriptive run. Role emergence requires repeated-seed analysis and post-hoc behavioral annotation.",
    }
    write_json(out / "summary.json", summary)
    if plots and metrics:
        render_plots(engine, edges, out)
    return summary


def render_plots(engine, edges, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np

    metrics = engine.metrics
    names = sorted({n for row in metrics for key in ("wealth", "paid") for n in row[key]})
    rounds = [row["round"] for row in metrics]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle(f"EoM teams · {engine.config.condition} · {engine.policy.label}", fontsize=17)
    for i, name in enumerate(names):
        axes[0, 0].plot(
            rounds, [row["wealth"].get(name, float("nan")) for row in metrics], label=name, color=plt.get_cmap("tab20")(i % 20)
        )
    axes[0, 0].set(title="Individual wealth", xlabel="Episode", ylabel="Wealth")
    axes[0, 0].legend(fontsize=7, ncol=3)
    tags = sorted({tag for row in metrics for tag in row["membership"].values() if tag})
    colors = {tag: i + 1 for i, tag in enumerate(tags)}
    grid = np.array([[colors.get(row["membership"][name], 0) if name in row["membership"] else float("nan") for row in metrics] for name in names])
    axes[0, 1].imshow(grid, aspect="auto", interpolation="nearest", cmap="tab20", vmin=0, vmax=max(1, len(tags)))
    axes[0, 1].set(title="Membership (0 = independent)", xlabel="Episode", yticks=range(len(names)), yticklabels=names)
    axes[0, 1].tick_params(axis="y", labelsize=7)
    fractions = np.array([[row["contribution_fraction"].get(name, 0) for row in metrics] for name in names])
    heat = axes[1, 0].imshow(fractions, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    axes[1, 0].set(title="Final pledge / opening wealth", xlabel="Episode", yticks=range(len(names)), yticklabels=names)
    axes[1, 0].tick_params(axis="y", labelsize=7)
    fig.colorbar(heat, ax=axes[1, 0], shrink=0.8)
    graph = nx.DiGraph()
    graph.add_nodes_from(names)
    graph.add_weighted_edges_from((a, b, count) for (a, b), count in edges.items())
    # Circular layout keeps labels and directed edges legible even when strong
    # within-team weights would collapse a force layout into overlapping nodes.
    positions = nx.circular_layout(graph)
    nx.draw_networkx(
        graph,
        positions,
        ax=axes[1, 1],
        node_color="#c8dded",
        node_size=900,
        font_size=7,
        width=[1 + np.log1p(d["weight"]) / 2 for _, _, d in graph.edges(data=True)],
        arrowsize=10,
    )
    axes[1, 1].set_title("Observed message graph (including bid negotiation)")
    axes[1, 1].axis("off")
    fig.savefig(out / "overview.png", dpi=160)
    fig.savefig(out / "overview.svg")
    plt.close(fig)
