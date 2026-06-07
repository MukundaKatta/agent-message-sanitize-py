"""agent-message-sanitize-py — sanitize and normalize LLM message lists."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SanitizeResult:
    """Result of sanitizing a message list."""

    messages: list[dict]
    removed: int = 0
    issues: list[str] = field(default_factory=list)


_VALID_ROLES = frozenset({"system", "user", "assistant", "tool", "function"})


def sanitize(
    messages: list[dict[str, Any]],
    *,
    strip_empty: bool = True,
    enforce_alternating: bool = False,
    system_first: bool = True,
    max_messages: int | None = None,
    allowed_roles: set[str] | None = None,
    strip_unknown_keys: bool = False,
) -> SanitizeResult:
    """
    Sanitize a list of LLM messages.

    Args:
        messages: List of message dicts with at least a "role" key.
        strip_empty: Remove messages with empty or whitespace-only "content".
        enforce_alternating: Remove consecutive messages with the same role
            (keeps the last one in each run).
        system_first: Move system messages to the front of the list.
        max_messages: Truncate to the most recent N messages (preserving system).
        allowed_roles: Set of allowed role strings. Messages with other roles are removed.
        strip_unknown_keys: Remove keys other than "role" and "content" from each message.

    Returns:
        SanitizeResult with the sanitized message list and metadata.
    """
    result = list(messages)
    issues: list[str] = []
    removed = 0
    roles = allowed_roles or _VALID_ROLES

    # Remove messages with unknown roles
    before = len(result)
    result = [m for m in result if m.get("role") in roles]
    diff = before - len(result)
    if diff:
        issues.append(f"Removed {diff} message(s) with unknown role.")
        removed += diff

    # Strip empty content
    if strip_empty:
        before = len(result)
        result = [m for m in result if str(m.get("content") or "").strip()]
        diff = before - len(result)
        if diff:
            issues.append(f"Removed {diff} empty message(s).")
            removed += diff

    # Move system messages to the front
    if system_first:
        system_msgs = [m for m in result if m.get("role") == "system"]
        other_msgs = [m for m in result if m.get("role") != "system"]
        if system_msgs and result and result[0].get("role") != "system":
            issues.append(f"Moved {len(system_msgs)} system message(s) to front.")
        result = system_msgs + other_msgs

    # Enforce alternating roles (remove consecutive same-role messages)
    if enforce_alternating:
        deduped: list[dict] = []
        for msg in result:
            if deduped and deduped[-1].get("role") == msg.get("role"):
                deduped[-1] = msg  # keep the last one
                removed += 1
                issues.append(f"Merged consecutive '{msg.get('role')}' messages.")
            else:
                deduped.append(msg)
        result = deduped

    # Truncate to max_messages (preserve system messages at front)
    if max_messages is not None and len(result) > max_messages:
        before = len(result)
        system_msgs = [m for m in result if m.get("role") == "system"]
        others = [m for m in result if m.get("role") != "system"]
        if len(system_msgs) >= max_messages:
            # System messages alone exceed the budget: keep the most recent
            # system messages and drop everything else so the final list never
            # exceeds max_messages.
            system_msgs = system_msgs[-max_messages:]
            others = []
        else:
            keep_others = max_messages - len(system_msgs)
            others = others[-keep_others:] if keep_others else []
        result = system_msgs + others
        trimmed = before - len(result)
        if trimmed > 0:
            issues.append(
                f"Trimmed {trimmed} message(s) to fit max_messages={max_messages}."
            )
            removed += trimmed

    # Strip unknown keys
    if strip_unknown_keys:
        allowed_keys = {"role", "content"}
        result = [{k: v for k, v in m.items() if k in allowed_keys} for m in result]

    return SanitizeResult(messages=result, removed=removed, issues=issues)


def strip_empty(messages: list[dict]) -> list[dict]:
    """Remove messages with empty or whitespace-only content."""
    return sanitize(messages, strip_empty=True).messages


def move_system_first(messages: list[dict]) -> list[dict]:
    """Move all system messages to the beginning."""
    return sanitize(messages, system_first=True).messages


def enforce_roles(messages: list[dict], allowed: set[str]) -> list[dict]:
    """Remove messages with roles not in *allowed*."""
    return sanitize(messages, allowed_roles=allowed).messages


def normalize_content(messages: list[dict]) -> list[dict]:
    """Strip leading/trailing whitespace from message content."""
    result = []
    for m in messages:
        msg = dict(m)
        if isinstance(msg.get("content"), str):
            msg["content"] = msg["content"].strip()
        result.append(msg)
    return result


__all__ = [
    "sanitize",
    "SanitizeResult",
    "strip_empty",
    "move_system_first",
    "enforce_roles",
    "normalize_content",
]
