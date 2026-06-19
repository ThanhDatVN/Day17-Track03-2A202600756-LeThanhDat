from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
    extract_profile_updates_with_confidence,
    is_correction_message,
)


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: short-term + User.md persistent memory + compact memory."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None

        if not force_offline:
            self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent and not self.force_offline:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        # 1. Extract profile facts with confidence scores and persist
        # Bonus: confidence threshold — only writes facts that pass the gate
        # Bonus: conflict handling — corrections get boosted confidence and
        #        overwrite stale values (upsert_fact replaces unconditionally)
        is_correction = is_correction_message(message)
        facts_with_conf = extract_profile_updates_with_confidence(message)
        for key, (value, confidence) in facts_with_conf.items():
            effective_conf = min(1.0, confidence + 0.1) if is_correction else confidence
            self.profile_store.upsert_fact(user_id, key, value, confidence=effective_conf)

        # 2. Append to compact memory
        self.compact_memory.append(thread_id, "user", message)

        # 3. Estimate prompt context
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )

        # 4. Generate response
        response = self._offline_response(user_id, thread_id, message)

        # 5. Append assistant response
        self.compact_memory.append(thread_id, "assistant", response)
        agent_tokens = estimate_tokens(response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + agent_tokens

        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        profile_text = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary = str(ctx.get("summary", ""))
        messages: list[dict] = ctx.get("messages", [])  # type: ignore[assignment]
        recent_text = " ".join(m["content"] for m in messages)
        return (
            estimate_tokens(profile_text)
            + estimate_tokens(summary)
            + estimate_tokens(recent_text)
        )

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        msg_lower = message.lower()
        profile_text = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary = str(ctx.get("summary", ""))
        messages: list[dict] = ctx.get("messages", [])  # type: ignore[assignment]

        # Build a searchable blob: profile + summary + recent messages
        all_context = profile_text + "\n" + summary + "\n" + " ".join(
            m["content"] for m in messages
        )

        from memory_store import _strip_metadata

        def find_in_context(patterns: list[str]) -> str | None:
            for pat in patterns:
                m = re.search(pat, all_context, re.IGNORECASE)
                if m:
                    return _strip_metadata(m.group(1).strip().rstrip(".,"))
            return None

        # Tên
        if any(kw in msg_lower for kw in ["tên", "name", "mình là ai"]):
            name = find_in_context([
                r"- tên:\s*(.+)",
                r"(?:tên\s+(?:mình|tôi|là)|mình\s+tên(?:\s+là)?)\s+([A-Za-zÀ-ỹà-ỹ0-9_\s]+?)(?:[.,\n]|$)",
            ])
            if name:
                parts = [f"Tên bạn là **{name}**."]
                # Also add other facts if comprehensive question
                if any(kw in msg_lower for kw in ["nhắc lại", "tóm tắt", "mô tả"]):
                    prof = find_in_context([r"- nghề nghiệp:\s*(.+)"])
                    loc = find_in_context([r"- nơi ở:\s*(.+)"])
                    drink = find_in_context([r"- đồ uống:\s*(.+)"])
                    style = find_in_context([r"- phong cách trả lời:\s*(.+)"])
                    if prof:
                        parts.append(f"Nghề nghiệp: {prof}.")
                    if loc:
                        parts.append(f"Nơi ở: {loc}.")
                    if drink:
                        parts.append(f"Đồ uống yêu thích: {drink}.")
                    if style:
                        parts.append(f"Style trả lời: {style}.")
                return " ".join(parts)

        # Nghề nghiệp
        if any(kw in msg_lower for kw in ["nghề", "làm gì", "công việc", "nghề nghiệp"]):
            prof = find_in_context([r"- nghề nghiệp:\s*(.+)"])
            if prof:
                loc = find_in_context([r"- nơi ở:\s*(.+)"])
                ans = f"Nghề nghiệp hiện tại của bạn là **{prof}**."
                if loc and any(kw in msg_lower for kw in ["ở", "nơi"]):
                    ans += f" Bạn đang ở {loc}."
                return ans

        # Nơi ở
        if any(kw in msg_lower for kw in ["ở đâu", "nơi ở", "thành phố", "đang ở"]):
            loc = find_in_context([r"- nơi ở:\s*(.+)"])
            if loc:
                return f"Bạn đang ở **{loc}**."

        # Đồ uống
        if any(kw in msg_lower for kw in ["đồ uống", "uống gì", "thích uống", "đồ uống yêu thích"]):
            drink = find_in_context([r"- đồ uống:\s*(.+)"])
            if drink:
                return f"Đồ uống yêu thích của bạn là **{drink}**."

        # Style
        if any(kw in msg_lower for kw in ["style", "phong cách", "trả lời như thế nào", "kiểu trả lời"]):
            style = find_in_context([r"- phong cách trả lời:\s*(.+)"])
            if style:
                return f"Style trả lời bạn thích: **{style}**."

        # Món ăn
        if any(kw in msg_lower for kw in ["món ăn", "ăn gì", "món yêu thích"]):
            food = find_in_context([
                r"(?:món\s+ăn\s+yêu\s+thích\s+(?:là|:)\s*)([^.,\n]+)",
                r"(mì\s+quảng|phở|bánh\s+mì|cơm[^.,\n]*)",
            ])
            if food:
                return f"Món ăn yêu thích của bạn là **{food}**."

        # Nuôi con gì
        if any(kw in msg_lower for kw in ["nuôi", "thú cưng", "con gì"]):
            pet = find_in_context([
                r"(?:nuôi\s+(?:một\s+)?(?:bé\s+|con\s+)?)(corgi|mèo|chó|thỏ|hamster)[^.,\n]*",
                r"(?:con\s+)(corgi|mèo|chó|thỏ)",
            ])
            if pet:
                # Try to find the name
                name_m = re.search(r"(?:corgi|mèo|chó)\s+(?:tên\s+)?([A-Za-zÀ-ỹà-ỹ]+)", all_context, re.IGNORECASE)
                if name_m:
                    return f"Bạn nuôi một bé **{pet}** tên **{name_m.group(1)}**."
                return f"Bạn nuôi một bé **{pet}**."

        # Comprehensive summary / recall
        if any(kw in msg_lower for kw in ["nhắc lại", "tóm tắt", "mô tả", "biết gì", "là ai"]):
            parts: list[str] = []
            name = find_in_context([r"- tên:\s*(.+)"])
            prof = find_in_context([r"- nghề nghiệp:\s*(.+)"])
            loc = find_in_context([r"- nơi ở:\s*(.+)"])
            drink = find_in_context([r"- đồ uống:\s*(.+)"])
            style = find_in_context([r"- phong cách trả lời:\s*(.+)"])
            if name:
                parts.append(f"- Tên: {name}")
            if prof:
                parts.append(f"- Nghề nghiệp: {prof}")
            if loc:
                parts.append(f"- Nơi ở: {loc}")
            if drink:
                parts.append(f"- Đồ uống yêu thích: {drink}")
            if style:
                parts.append(f"- Style trả lời: {style}")
            if parts:
                return "Đây là những gì mình nhớ về bạn:\n" + "\n".join(parts)

        turn = len([m for m in messages if m["role"] == "user"])
        return f"Được rồi, mình đã ghi nhận và lưu thông tin của bạn (lượt {turn + 1})."

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        from langchain_core.messages import HumanMessage

        facts = extract_profile_updates(message)
        for key, value in facts.items():
            self.profile_store.upsert_fact(user_id, key, value)

        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )

        profile_text = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary = str(ctx.get("summary", ""))
        system_content = f"User profile:\n{profile_text}\n\nConversation summary:\n{summary}"

        config = {"configurable": {"thread_id": thread_id}}
        result = self.langchain_agent.invoke(
            {
                "messages": [HumanMessage(content=message)],
                "system": system_content,
            },
            config=config,
        )
        response = result["messages"][-1].content
        self.compact_memory.append(thread_id, "assistant", response)
        agent_tokens = estimate_tokens(response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + agent_tokens

        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _maybe_build_langchain_agent(self) -> None:
        try:
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent
            from model_provider import build_chat_model

            llm = build_chat_model(self.config.model)
            checkpointer = MemorySaver()
            self.langchain_agent = create_react_agent(llm, tools=[], checkpointer=checkpointer)
        except Exception:
            self.langchain_agent = None
