from __future__ import annotations

from pathlib import Path

import pytest

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens


def make_config(tmp_path: Path) -> LabConfig:
    cfg = load_config()
    # Override paths and thresholds for isolated testing
    cfg.state_dir = tmp_path / "state"
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    (cfg.state_dir / "profiles").mkdir(parents=True, exist_ok=True)
    cfg.compact_threshold_tokens = 100   # low threshold so compaction triggers quickly
    cfg.compact_keep_messages = 2
    return cfg


# ── User.md read / write / edit ─────────────────────────────────────────────

def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")

    # Default read returns empty profile header
    text = store.read_text("alice")
    assert "alice" in text.lower() or text.strip() != ""

    # Write and read back
    store.write_text("alice", "# Alice\n- tên: Alice\n- nơi ở: Hà Nội\n")
    assert "Alice" in store.read_text("alice")
    assert "Hà Nội" in store.read_text("alice")

    # Edit one occurrence
    changed = store.edit_text("alice", "Hà Nội", "Đà Nẵng")
    assert changed is True
    assert "Đà Nẵng" in store.read_text("alice")
    assert "Hà Nội" not in store.read_text("alice")

    # Edit non-existent text returns False
    changed = store.edit_text("alice", "Không tồn tại", "X")
    assert changed is False

    # file_size > 0 after write
    assert store.file_size("alice") > 0


# ── upsert_fact ──────────────────────────────────────────────────────────────

def test_upsert_fact_insert_and_update(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    store.upsert_fact("bob", "tên", "Bob")
    assert "Bob" in store.read_text("bob")

    # Update existing fact
    store.upsert_fact("bob", "tên", "Robert")
    text = store.read_text("bob")
    assert "Robert" in text
    assert text.count("- tên:") == 1   # no duplicate line


# ── compact memory trigger ───────────────────────────────────────────────────

def test_compact_trigger(tmp_path: Path) -> None:
    mgr = CompactMemoryManager(threshold_tokens=50, keep_messages=2)
    thread = "t1"

    # Append enough content to exceed 50-token threshold
    long_msg = "a " * 60   # ~30 tokens per message, 2 messages = 60 tokens
    mgr.append(thread, "user", long_msg)
    mgr.append(thread, "assistant", long_msg)
    mgr.append(thread, "user", long_msg)   # this should trigger compaction

    assert mgr.compaction_count(thread) >= 1
    ctx = mgr.context(thread)
    # After compaction, kept messages <= keep_messages
    assert len(ctx["messages"]) <= 2
    # Summary should contain something
    assert ctx["summary"] != ""


def test_compact_does_not_trigger_on_short_thread(tmp_path: Path) -> None:
    mgr = CompactMemoryManager(threshold_tokens=1000, keep_messages=4)
    mgr.append("t2", "user", "hi")
    mgr.append("t2", "assistant", "hello")
    assert mgr.compaction_count("t2") == 0


# ── cross-session recall ─────────────────────────────────────────────────────

def test_cross_session_recall(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)

    adv = AdvancedAgent(config=cfg, force_offline=True)
    base = BaselineAgent(config=cfg, force_offline=True)

    user_id = "carol"
    session_1 = "s1"

    # Teach both agents in session 1
    adv.reply(user_id, session_1, "Chào bạn, mình tên là Carol.")
    adv.reply(user_id, session_1, "Mình đang làm data scientist ở TP.HCM.")
    base.reply(user_id, session_1, "Chào bạn, mình tên là Carol.")
    base.reply(user_id, session_1, "Mình đang làm data scientist ở TP.HCM.")

    # Ask in a FRESH session (cross-session)
    session_2 = "s2-new"
    adv_result = adv.reply(user_id, session_2, "Mình tên gì và làm nghề gì?")
    base_result = base.reply(user_id, session_2, "Mình tên gì và làm nghề gì?")

    # Advanced should recall (User.md is persistent)
    assert "carol" in adv_result["response"].lower() or "Carol" in adv_result["response"]

    # Baseline should NOT recall across sessions
    assert "carol" not in base_result["response"].lower()


# ── prompt load reduction with compact ───────────────────────────────────────

def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)   # threshold=100 tokens

    adv = AdvancedAgent(config=cfg, force_offline=True)
    base = BaselineAgent(config=cfg, force_offline=True)

    user_id = "dave"
    thread = "long-thread"

    # Send many long messages so baseline accumulates prompt tokens unboundedly
    long_msg = "Mình muốn chia sẻ rất nhiều thông tin về công việc hằng ngày của mình. " * 3

    for i in range(8):
        adv.reply(user_id, thread, f"{long_msg} (turn {i})")
        base.reply(user_id, thread, f"{long_msg} (turn {i})")

    adv_prompt = adv.prompt_token_usage(thread)
    base_prompt = base.prompt_token_usage(thread)
    adv_compactions = adv.compaction_count(thread)

    # Compaction must have happened at least once
    assert adv_compactions >= 1, f"Expected compactions, got {adv_compactions}"

    # Advanced should process fewer prompt tokens than baseline on a long thread
    # (because compact reduces the context window each turn)
    assert adv_prompt < base_prompt, (
        f"Expected advanced ({adv_prompt}) < baseline ({base_prompt}) prompt tokens"
    )


# ── Bonus: confidence threshold ───────────────────────────────────────────────

def test_confidence_threshold_blocks_questions(tmp_path: Path) -> None:
    """Facts should NOT be written when message is a question."""
    from memory_store import extract_profile_updates_with_confidence

    question = "Bạn có biết mình tên gì không? Và mình đang làm nghề gì?"
    result = extract_profile_updates_with_confidence(question)
    assert result == {}, f"Questions should not produce facts, got: {result}"


def test_confidence_threshold_accepts_assertions(tmp_path: Path) -> None:
    """Direct assertions should pass the confidence gate."""
    from memory_store import extract_profile_updates_with_confidence, CONFIDENCE_THRESHOLD

    msg = "Mình tên là Alice và đang làm MLOps engineer tại Hà Nội."
    result = extract_profile_updates_with_confidence(msg)
    assert "tên" in result, "Should extract name from direct assertion"
    _, conf = result["tên"]
    assert conf >= CONFIDENCE_THRESHOLD


# ── Bonus: conflict handling ──────────────────────────────────────────────────

def test_conflict_handling_overwrites_old_fact(tmp_path: Path) -> None:
    """A correction message should replace the stale fact, not add a duplicate."""
    store = UserProfileStore(tmp_path / "profiles")

    store.upsert_fact("eve", "nơi ở", "Hà Nội", confidence=0.85)
    assert "Hà Nội" in store.read_text("eve")

    # Correction: move to Đà Nẵng
    store.upsert_fact("eve", "nơi ở", "Đà Nẵng", confidence=0.95)
    text = store.read_text("eve")

    assert "Đà Nẵng" in text
    assert "Hà Nội" not in text, "Old location should be replaced, not kept"
    assert text.count("- nơi ở:") == 1, "No duplicate fact lines"


def test_conflict_correction_via_agent(tmp_path: Path) -> None:
    """Advanced agent should update the fact when user sends an explicit correction."""
    cfg = make_config(tmp_path)
    adv = AdvancedAgent(config=cfg, force_offline=True)
    user_id = "frank"

    adv.reply(user_id, "s1", "Mình đang làm backend engineer.")
    adv.reply(user_id, "s1", "Đính chính: mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.")

    profile = adv.profile_store.read_text(user_id)
    assert "MLOps engineer" in profile
    assert profile.count("- nghề nghiệp:") == 1, "Should not have two profession lines"


# ── Bonus: memory decay ───────────────────────────────────────────────────────

def test_decay_score_recent_fact_is_high(tmp_path: Path) -> None:
    """A fact written today should have a decay score close to 1.0."""
    store = UserProfileStore(tmp_path / "profiles")
    store.upsert_fact("grace", "tên", "Grace", confidence=0.9)
    score = store.decay_score("grace", "tên")
    assert score >= 0.9, f"Fresh fact should score >= 0.9, got {score}"


def test_decay_score_missing_fact_is_zero(tmp_path: Path) -> None:
    """A fact that was never written should score 0.0."""
    store = UserProfileStore(tmp_path / "profiles")
    score = store.decay_score("heidi", "tên")
    assert score == 0.0


def test_prioritized_facts_order(tmp_path: Path) -> None:
    """prioritized_facts() should return most relevant facts first."""
    import datetime
    store = UserProfileStore(tmp_path / "profiles")

    # Write one fact with today's date, one simulating an older date
    store.upsert_fact("ivan", "tên", "Ivan", confidence=0.9)
    # Manually inject an old fact to simulate decay
    old_line = "- nghề nghiệp: old_job [c:0.80 t:2020-01-01 m:1]"
    text = store.read_text("ivan")
    store.write_text("ivan", text + old_line + "\n")

    ranked = store.prioritized_facts("ivan")
    keys = [r[0] for r in ranked]
    # "tên" (today) should rank above "nghề nghiệp" (2020)
    assert keys.index("tên") < keys.index("nghề nghiệp")
