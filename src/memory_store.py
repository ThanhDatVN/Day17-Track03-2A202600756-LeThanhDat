from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    return max(1, len(text) // 4)


# ── Bonus: Confidence threshold ──────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.65


def _score_fact(key: str, message: str, value: str) -> float:
    """Return a 0.0-1.0 confidence score for an extracted fact.

    High score = strong signal (direct assertion or explicit correction).
    Low score = ambiguous (indirect, short, or embedded in noise).
    Facts below CONFIDENCE_THRESHOLD are NOT written to User.md.
    """
    msg_lower = message.lower()

    # Explicit corrections are the most reliable signal
    correction_signals = [
        "dinh chinh", "khong con", "chuyen sang", "thay doi", "cap nhat",
        "da thay", "sai roi", "chinh xac la",
    ]
    # Compare without diacritics by stripping them is complex; use substring search instead
    correction_keywords = [
        "đính chính", "không còn", "chuyển sang", "thay đổi", "cập nhật",
        "đã thay", "sai rồi", "chính xác là", "đã đổi",
    ]
    if any(kw in msg_lower for kw in correction_keywords):
        return 0.92

    # Direct first-person assertions
    direct = {
        "tên": [r"mình tên(?:\s+là)?\s+\S", r"tên mình là", r"tôi tên là"],
        "nghề nghiệp": [r"hiện(?:\s+đang)?\s+là\s+\S", r"chuyển sang\s+\S", r"giờ làm"],
        "nơi ở": [r"đang ở\s+\S", r"hiện ở\s+\S", r"mình ở\s+\S"],
        "đồ uống": [r"đồ uống yêu thích"],
        "phong cách trả lời": [r"muốn bạn trả lời", r"hãy trả lời"],
    }
    for pat in direct.get(key, []):
        if re.search(pat, msg_lower):
            return 0.85

    # Moderate: clear statement, long enough value, not a question
    if len(value) >= 3 and "?" not in message[-30:]:
        return 0.72

    return 0.50  # Below threshold → won't be written


# ── Bonus: Conflict detection ────────────────────────────────────────────────

_CORRECTION_RE = re.compile(
    r"(?:đính\s+chính|không\s+còn|chuyển\s+sang|đã\s+thay|không\s+phải\s+là\s+\S+\s+nữa|"
    r"đã\s+đổi|sai\s+rồi|cập\s+nhật\s+lại|nhớ\s+lấy\s+thông\s+tin\s+mới)",
    re.IGNORECASE,
)


def is_correction_message(message: str) -> bool:
    """True when the message explicitly corrects a previously stated fact."""
    return bool(_CORRECTION_RE.search(message))


# ── User profile store ───────────────────────────────────────────────────────

def _strip_metadata(val: str) -> str:
    """Remove inline metadata tag [c:… t:… m:…] from a fact value."""
    return re.sub(r"\s*\[c:[^\]]*\]", "", val).strip()


@dataclass
class UserProfileStore:
    """Persistent storage for User.md files keyed by user_id."""

    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, user_id: str) -> Path:
        safe = re.sub(r"[^\w\-]", "_", user_id)
        return self.root_dir / f"{safe}.md"

    def read_text(self, user_id: str) -> str:
        p = self.path_for(user_id)
        if p.exists():
            return p.read_text(encoding="utf-8")
        return f"# User Profile: {user_id}\n\n"

    def write_text(self, user_id: str, content: str) -> Path:
        p = self.path_for(user_id)
        p.write_text(content, encoding="utf-8")
        return p

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        current = self.read_text(user_id)
        if search_text not in current:
            return False
        updated = current.replace(search_text, replacement, 1)
        self.write_text(user_id, updated)
        return True

    def file_size(self, user_id: str) -> int:
        p = self.path_for(user_id)
        if p.exists():
            return p.stat().st_size
        return 0

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse key→value, stripping decay metadata."""
        text = self.read_text(user_id)
        result: dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("-") and ":" in line:
                line = line.lstrip("- ")
                key, _, val = line.partition(":")
                result[key.strip().lower()] = _strip_metadata(val)
        return result

    def upsert_fact(
        self,
        user_id: str,
        key: str,
        value: str,
        confidence: float = 1.0,
    ) -> None:
        """Insert or update a fact.

        Bonus — stores inline metadata for decay scoring:
          - c: confidence at write time
          - t: ISO date of last update
          - m: mention count (incremented on each update)

        Bonus — conflict handling: if a correction is detected the old value
        is replaced unconditionally; the mention count still increments so
        the system knows this fact has been revisited.
        """
        today = datetime.date.today().isoformat()
        text = self.read_text(user_id)
        pattern = re.compile(rf"^- {re.escape(key)}:.*$", re.MULTILINE | re.IGNORECASE)
        existing = pattern.search(text)

        mentions = 1
        if existing:
            m_count = re.search(r"\[c:[^]]*m:(\d+)", existing.group(0))
            mentions = int(m_count.group(1)) + 1 if m_count else 2

        new_line = f"- {key}: {value} [c:{confidence:.2f} t:{today} m:{mentions}]"

        if existing:
            text = pattern.sub(new_line, text)
        else:
            if not text.endswith("\n"):
                text += "\n"
            text += new_line + "\n"
        self.write_text(user_id, text)

    # Bonus: memory decay ────────────────────────────────────────────────────

    def decay_score(self, user_id: str, key: str) -> float:
        """Return 0.0–1.0 relevance score based on recency and mention count.

        Facts written today score 1.0; score decays ~0.05 per day, floored at 0.1.
        Frequently mentioned facts decay more slowly.
        """
        text = self.read_text(user_id)
        pattern = re.compile(rf"^- {re.escape(key)}:.*$", re.MULTILINE | re.IGNORECASE)
        m = pattern.search(text)
        if not m:
            return 0.0

        line = m.group(0)

        # Extract date
        t_match = re.search(r"t:(\d{4}-\d{2}-\d{2})", line)
        if not t_match:
            return 0.5
        try:
            written = datetime.date.fromisoformat(t_match.group(1))
        except ValueError:
            return 0.5

        age_days = (datetime.date.today() - written).days

        # Extract mention count (more mentions → slower decay)
        m_match = re.search(r"m:(\d+)", line)
        mentions = int(m_match.group(1)) if m_match else 1
        decay_rate = max(0.01, 0.05 / max(1, mentions ** 0.5))

        score = max(0.1, 1.0 - age_days * decay_rate)
        return round(score, 3)

    def prioritized_facts(self, user_id: str) -> list[tuple[str, str, float]]:
        """Return [(key, value, decay_score)] sorted by relevance descending."""
        result = []
        for key, val in self.facts(user_id).items():
            score = self.decay_score(user_id, key)
            result.append((key, val, score))
        result.sort(key=lambda x: x[2], reverse=True)
        return result


# ── Profile extraction ───────────────────────────────────────────────────────

_QUESTION_RE = re.compile(
    r"(?:bạn\s+(?:có\s+)?(?:biết|nhớ|thể)|nhắc\s+lại|bạn\s+(?:có\s+)?thể\s+cho|mình\s+(?:muốn\s+hỏi|hỏi)|"
    r"(?:có\s+)?thể\s+(?:cho\s+)?(?:mình|tôi)\s+biết|\?)",
    re.IGNORECASE,
)


_QUESTION_VALUE_WORDS = {"gì", "không", "đâu", "nào", "sao", "chưa", "là gì", "của mình là gì"}
_NOISE_LOCATION_WORDS = {
    "đây", "đó", "nơi", "chỗ", "từ", "các", "trước", "giúp", "nhớ",
    "hiện", "tại", "hiện tại", "đang", "vẫn", "này", "mức", "phần",
    "level", "bước", "góc", "nhóm", "team",
}
_NEGATION_RE = re.compile(
    r"(?:không\s+còn|không\s+phải|chứ\s+không|chứ\s+không\s+còn)",
    re.IGNORECASE,
)


def _is_noise_value(val: str) -> bool:
    """True if the extracted value looks like a question fragment or noise."""
    v = val.lower().strip()
    if any(v.endswith(w) for w in _QUESTION_VALUE_WORDS):
        return True
    if any(v.startswith(w) for w in _QUESTION_VALUE_WORDS):
        return True
    # Reject values with more than 5 words (likely captured a sentence)
    if len(v.split()) > 5:
        return True
    return False


def _extract_raw_facts(message: str) -> dict[str, str]:
    """Extract candidate facts without confidence filtering."""
    facts: dict[str, str] = {}

    # Name — stop at common conjunctions / punctuation so we don't capture sentences
    m = re.search(
        r"(?:tên\s+(?:mình|tôi|là)|mình\s+tên(?:\s+là)?|tôi\s+tên(?:\s+là)?)\s+"
        r"([A-Za-zÀ-ỹà-ỹ0-9_]+(?:\s+[A-Za-zÀ-ỹà-ỹ0-9_]+)?)"
        r"(?=\s*(?:và|,|\.|$|\n|\s+(?:đang|là|ở|tại|nhé|nhưng|với)))",
        message, re.IGNORECASE,
    )
    if m:
        name = m.group(1).strip()
        if 2 <= len(name) <= 40:
            facts["tên"] = name

    # Location — require a city/place name (no more than 2 words)
    loc_patterns = [
        r"(?:hiện\s+)?(?:đang\s+)?(?:mình\s+)?(?:ở|tại)\s+"
        r"([A-Za-zÀ-ỹà-ỹ]+(?:\s+[A-Za-zÀ-ỹà-ỹ]+)?)"
        r"(?=\s*(?:và|,|nhé|chứ|để|trong|vài|[.\n]|$))",
        r"chuyển\s+(?:sang|về|đến)\s+([A-Za-zÀ-ỹà-ỹ]+(?:\s+[A-Za-zÀ-ỹà-ỹ]+)?)"
        r"(?=\s*(?:và|,|nhé|chứ|để|[.\n]|$))",
        r"nơi\s+ở\s+(?:là\s+)?([A-Za-zÀ-ỹà-ỹ]+(?:\s+[A-Za-zÀ-ỹà-ỹ]+)?)(?:[.,\n]|$)",
    ]
    for pat in loc_patterns:
        m = re.search(pat, message, re.IGNORECASE)
        if m:
            loc = m.group(1).strip().rstrip(".,")
            loc_words = set(loc.lower().split())
            if (2 <= len(loc) <= 25
                    and not (loc_words & _NOISE_LOCATION_WORDS)
                    and not _is_noise_value(loc)):
                facts["nơi ở"] = loc
                break

    # Profession — priority: explicit new-role signals first, then general pattern
    # Skip matches that are preceded by negation ("không còn là X", "chứ không là X")
    _PROF_KW = r"(?:engineer|developer|designer|manager|analyst|scientist|teacher|doctor|architect|mlops)"
    _PROF_CAP = rf"((?:\w+\s+){{0,3}}{_PROF_KW}\s*(?:engineer)?)"
    _PROF_STOP = r"(?=\s*(?:cho|ở|tại|và|,|nhé|chứ|nữa|$|\.))"

    prof: str | None = None
    # 1. Explicit role-change phrases (highest priority)
    m = re.search(rf"(?:chuyển\s+sang|giờ\s+(?:là|làm))\s+{_PROF_CAP}{_PROF_STOP}", message, re.IGNORECASE)
    if m:
        prof = m.group(1).strip()
    # 2. General "mình/tôi [đang] làm/là X" — require first-person subject right before verb
    #    to avoid catching "nhớ là mình làm X" or "Bạn nhớ là X"
    if not prof:
        _SUBJ_VERB = r"(?:(?:mình|tôi)\s+(?:(?:hiện\s+)?(?:đang\s+)?(?:làm|là)\s+)|hiện\s+(?:đang\s+)?(?:là|làm)\s+)"
        for mm in re.finditer(rf"{_SUBJ_VERB}{_PROF_CAP}{_PROF_STOP}", message, re.IGNORECASE):
            prefix = message[max(0, mm.start() - 50): mm.start()]
            if _NEGATION_RE.search(prefix):
                continue
            prof = mm.group(1).strip()
            break

    if prof:
        prof = re.sub(r"\s+(?:nữa|này|đó|kia)\s*$", "", prof, flags=re.IGNORECASE).strip().rstrip("., ")
        if 3 <= len(prof) <= 50 and not _is_noise_value(prof):
            facts["nghề nghiệp"] = prof

    # Drink — only accept after "là/:" pattern or explicit "thích uống"
    m = re.search(
        r"(?:đồ\s+uống\s+yêu\s+thích\s+(?:là|:)\s*)([^.,\n?]+)",
        message, re.IGNORECASE,
    )
    if not m:
        m = re.search(r"thích\s+uống\s+([^.,\n?]+)", message, re.IGNORECASE)
    if m:
        drink = m.group(1).strip()
        if len(drink) <= 40 and not _is_noise_value(drink):
            facts["đồ uống"] = drink

    # Reply style
    m = re.search(
        r"(?:(?:muốn\s+(?:bạn\s+)?|hãy\s+)trả\s+lời\s+)([^.,\n?]+)",
        message, re.IGNORECASE,
    )
    if m:
        style = m.group(1).strip()
        if len(style) <= 80 and not _is_noise_value(style):
            facts["phong cách trả lời"] = style

    return facts


def extract_profile_updates(message: str) -> dict[str, str]:
    """Return only facts whose confidence >= CONFIDENCE_THRESHOLD (bonus gate)."""
    # Skip mostly-question messages
    question_count = len(re.findall(r"\?", message))
    if question_count >= 2:
        return {}
    if _QUESTION_RE.search(message) and "?" in message and len(message) < 120:
        return {}

    raw = _extract_raw_facts(message)
    return {
        k: v for k, v in raw.items()
        if _score_fact(k, message, v) >= CONFIDENCE_THRESHOLD
    }


def extract_profile_updates_with_confidence(message: str) -> dict[str, tuple[str, float]]:
    """Return {key: (value, confidence)} for all facts above threshold."""
    question_count = len(re.findall(r"\?", message))
    if question_count >= 2:
        return {}
    if _QUESTION_RE.search(message) and "?" in message and len(message) < 120:
        return {}

    raw = _extract_raw_facts(message)
    result: dict[str, tuple[str, float]] = {}
    for k, v in raw.items():
        score = _score_fact(k, message, v)
        if score >= CONFIDENCE_THRESHOLD:
            result[k] = (v, score)
    return result


# ── Compact memory ───────────────────────────────────────────────────────────

_SUMMARY_SNIPPET_CHARS = 60
_SUMMARY_MAX_TOKENS = 200


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Produce a compact summary — intentionally short to save prompt tokens."""
    if not messages:
        return ""
    selected = messages[-max_items:] if len(messages) > max_items else messages
    parts: list[str] = []
    for msg in selected:
        role = msg.get("role", "?")[0].upper()
        content = msg.get("content", "").strip()
        if content:
            snippet = content[:_SUMMARY_SNIPPET_CHARS].replace("\n", " ")
            parts.append(f"[{role}] {snippet}")
    return " | ".join(parts)


@dataclass
class CompactMemoryManager:
    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _init_thread(self, thread_id: str) -> None:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }

    def append(self, thread_id: str, role: str, content: str) -> None:
        self._init_thread(thread_id)
        s = self.state[thread_id]
        msgs: list[dict] = s["messages"]  # type: ignore[assignment]
        msgs.append({"role": role, "content": content})

        total_tokens = sum(estimate_tokens(m["content"]) for m in msgs)
        total_tokens += estimate_tokens(str(s["summary"]))

        if total_tokens > self.threshold_tokens and len(msgs) > self.keep_messages:
            old = msgs[: -self.keep_messages]
            kept = msgs[-self.keep_messages:]
            new_summary = summarize_messages(old)
            prev_summary = str(s["summary"])
            combined = (prev_summary + " || " + new_summary).strip(" |") if prev_summary else new_summary
            max_chars = _SUMMARY_MAX_TOKENS * 4
            if len(combined) > max_chars:
                combined = combined[-max_chars:]
            s["summary"] = combined
            s["messages"] = kept
            s["compactions"] = int(s["compactions"]) + 1  # type: ignore[arg-type]

    def context(self, thread_id: str) -> dict[str, object]:
        self._init_thread(thread_id)
        return self.state[thread_id]

    def compaction_count(self, thread_id: str) -> int:
        self._init_thread(thread_id)
        return int(self.state[thread_id]["compactions"])  # type: ignore[arg-type]
