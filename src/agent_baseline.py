from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: within-session memory only, no persistent User.md."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None

        if not force_offline:
            self._maybe_build_langchain_agent()

    def _get_session(self, thread_id: str) -> SessionState:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        return self.sessions[thread_id]

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent and not self.force_offline:
            return self._reply_live(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self._get_session(thread_id).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self._get_session(thread_id).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._get_session(thread_id)
        session.messages.append({"role": "user", "content": message})

        # Estimate prompt context = all messages so far
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in session.messages)
        session.prompt_tokens_processed += prompt_tokens

        response = self._generate_response(session, message)

        session.messages.append({"role": "assistant", "content": response})
        agent_tokens = estimate_tokens(response)
        session.token_usage += agent_tokens

        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _generate_response(self, session: SessionState, message: str) -> str:
        msg_lower = message.lower()

        # Simple recall from within-session messages only
        if any(kw in msg_lower for kw in ["tên", "name"]):
            for m in session.messages:
                if m["role"] == "user":
                    import re
                    match = re.search(
                        r"(?:tên\s+(?:mình|tôi|là)|mình\s+tên(?:\s+là)?)\s+([A-Za-zÀ-ỹà-ỹ0-9_\s]+?)(?:[.,\n]|$)",
                        m["content"], re.IGNORECASE,
                    )
                    if match:
                        name = match.group(1).strip()
                        return f"Tên bạn là {name}."

        if any(kw in msg_lower for kw in ["đồ uống", "uống gì", "thích uống"]):
            for m in session.messages:
                if m["role"] == "user" and ("cà phê" in m["content"].lower() or "trà" in m["content"].lower() or "uống" in m["content"].lower()):
                    import re
                    match = re.search(r"(cà phê[^.,\n]*|trà[^.,\n]*)", m["content"], re.IGNORECASE)
                    if match:
                        return f"Đồ uống yêu thích của bạn là {match.group(1).strip()}."

        if any(kw in msg_lower for kw in ["ở đâu", "nơi ở", "thành phố"]):
            for m in session.messages:
                if m["role"] == "user":
                    import re
                    match = re.search(
                        r"(?:ở|tại|sống tại)\s+([A-Za-zÀ-ỹà-ỹ\s]+?)(?:[.,\n]|$)",
                        m["content"], re.IGNORECASE,
                    )
                    if match:
                        loc = match.group(1).strip()
                        if 2 <= len(loc) <= 30:
                            return f"Bạn đang ở {loc}."

        turn = len([m for m in session.messages if m["role"] == "user"])
        return f"Được rồi, mình đã ghi nhận thông tin này (lượt {turn} trong session)."

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        from langchain_core.messages import HumanMessage
        session = self._get_session(thread_id)
        session.messages.append({"role": "user", "content": message})

        prompt_tokens = sum(estimate_tokens(m["content"]) for m in session.messages)
        session.prompt_tokens_processed += prompt_tokens

        config = {"configurable": {"thread_id": thread_id}}
        result = self.langchain_agent.invoke(
            {"messages": [HumanMessage(content=message)]},
            config=config,
        )
        response = result["messages"][-1].content

        session.messages.append({"role": "assistant", "content": response})
        agent_tokens = estimate_tokens(response)
        session.token_usage += agent_tokens

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
