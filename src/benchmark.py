from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    if not expected:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for e in expected if e.lower() in answer_lower)
    ratio = hits / len(expected)
    if ratio >= 1.0:
        return 1.0
    if ratio >= 0.5:
        return 0.5
    return 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Length and keyword overlap heuristic."""
    if not answer.strip():
        return 0.0
    score = recall_points(answer, expected)
    # Bonus for reasonable response length (not too short, not too long)
    length = len(answer)
    if 10 <= length <= 500:
        score = min(1.0, score + 0.1)
    return round(score, 3)


def run_agent_benchmark(
    agent_name: str,
    agent,
    conversations: list[dict[str, Any]],
    config,
) -> BenchmarkRow:
    total_agent_tokens = 0
    total_prompt_tokens = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    total_compactions = 0
    memory_growth = 0

    for conv in conversations:
        user_id: str = conv["user_id"]
        conv_id: str = conv["id"]
        turns: list[str] = conv["turns"]
        recall_questions: list[dict] = conv.get("recall_questions", [])

        # Feed all turns using the conversation thread
        for turn in turns:
            agent.reply(user_id, conv_id, turn)

        total_agent_tokens += agent.token_usage(conv_id)
        total_prompt_tokens += agent.prompt_token_usage(conv_id)
        total_compactions += agent.compaction_count(conv_id)

        # Ask recall questions in a FRESH thread (cross-session recall)
        recall_thread = f"{conv_id}-recall"
        for rq in recall_questions:
            question = rq["question"]
            expected = rq.get("expected_contains", [])
            result = agent.reply(user_id, recall_thread, question)
            answer = result["response"]
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

        # Memory growth for advanced agent
        if hasattr(agent, "memory_file_size"):
            memory_growth += agent.memory_file_size(user_id)

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=round(avg_recall, 3),
        response_quality=round(avg_quality, 3),
        memory_growth_bytes=memory_growth,
        compactions=total_compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    try:
        from tabulate import tabulate

        headers = [
            "Agent",
            "Agent tokens only",
            "Prompt tokens processed",
            "Cross-session recall",
            "Response quality",
            "Memory growth (bytes)",
            "Compactions",
        ]
        data = [
            [
                r.agent_name,
                r.agent_tokens_only,
                r.prompt_tokens_processed,
                f"{r.recall_score:.3f}",
                f"{r.response_quality:.3f}",
                r.memory_growth_bytes,
                r.compactions,
            ]
            for r in rows
        ]
        return tabulate(data, headers=headers, tablefmt="github")
    except ImportError:
        lines = ["| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |"]
        lines.append("|---|---|---|---|---|---|---|")
        for r in rows:
            lines.append(
                f"| {r.agent_name} | {r.agent_tokens_only} | {r.prompt_tokens_processed} "
                f"| {r.recall_score:.3f} | {r.response_quality:.3f} | {r.memory_growth_bytes} | {r.compactions} |"
            )
        return "\n".join(lines)


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)

    std_path = config.data_dir / "conversations.json"
    stress_path = config.data_dir / "advanced_long_context.json"

    std_conversations = load_conversations(std_path)
    stress_conversations = load_conversations(stress_path)

    # ── Standard Benchmark ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STANDARD BENCHMARK  (data/conversations.json)")
    print("=" * 60)

    baseline_std = BaselineAgent(config=config, force_offline=True)
    advanced_std = AdvancedAgent(config=config, force_offline=True)

    std_rows = [
        run_agent_benchmark("Baseline", baseline_std, std_conversations, config),
        run_agent_benchmark("Advanced", advanced_std, std_conversations, config),
    ]
    print(format_rows(std_rows))

    # ── Long-Context Stress Benchmark ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("LONG-CONTEXT STRESS BENCHMARK  (data/advanced_long_context.json)")
    print("=" * 60)

    baseline_stress = BaselineAgent(config=config, force_offline=True)
    advanced_stress = AdvancedAgent(config=config, force_offline=True)

    stress_rows = [
        run_agent_benchmark("Baseline", baseline_stress, stress_conversations, config),
        run_agent_benchmark("Advanced", advanced_stress, stress_conversations, config),
    ]
    print(format_rows(stress_rows))

    # ── Analysis ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    _print_analysis(std_rows, stress_rows)


def _print_analysis(std_rows: list[BenchmarkRow], stress_rows: list[BenchmarkRow]) -> None:
    baseline_std = next(r for r in std_rows if r.agent_name == "Baseline")
    advanced_std = next(r for r in std_rows if r.agent_name == "Advanced")
    baseline_stress = next(r for r in stress_rows if r.agent_name == "Baseline")
    advanced_stress = next(r for r in stress_rows if r.agent_name == "Advanced")

    lines = [
        "",
        "1. Cross-session recall:",
        f"   Baseline (standard): {baseline_std.recall_score:.3f}",
        f"   Advanced (standard): {advanced_std.recall_score:.3f}",
        "   -> Advanced recalls better because facts are persisted in User.md across threads.",
        "      Baseline has no persistent memory and forgets everything on a new thread.",
        "",
        "2. Token cost on short conversations:",
        f"   Baseline prompt tokens: {baseline_std.prompt_tokens_processed}",
        f"   Advanced prompt tokens: {advanced_std.prompt_tokens_processed}",
        "   -> Advanced can cost MORE on short threads because it injects User.md + summary",
        "      overhead on every turn, while Baseline only carries session messages.",
        "",
        "3. Compact memory on the stress benchmark:",
        f"   Baseline prompt tokens: {baseline_stress.prompt_tokens_processed}",
        f"   Advanced prompt tokens: {advanced_stress.prompt_tokens_processed}",
        f"   Advanced compactions:    {advanced_stress.compactions}",
        "   -> On very long threads compact memory caps the context window, so Advanced",
        "      processes far fewer prompt tokens than Baseline (which grows unbounded).",
        "      Compact mainly optimises 'prompt tokens processed', not agent output tokens.",
        "",
        "4. Memory file growth:",
        f"   Advanced memory growth (standard): {advanced_std.memory_growth_bytes} bytes",
        f"   Advanced memory growth (stress):   {advanced_stress.memory_growth_bytes} bytes",
        "   -> User.md grows over time and is a real cost. Risks include storing wrong facts,",
        "      unbounded file growth, and needing guardrails such as confidence thresholds",
        "      and conflict-handling for corrections.",
        "",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
