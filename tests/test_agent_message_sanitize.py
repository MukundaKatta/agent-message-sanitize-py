"""Tests for agent-message-sanitize-py."""

from agent_message_sanitize import (
    sanitize,
    SanitizeResult,
    strip_empty,
    move_system_first,
    enforce_roles,
    normalize_content,
)

MSGS = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
]


def test_passthrough_clean_messages():
    result = sanitize(MSGS)
    assert result.removed == 0
    assert len(result.messages) == 3


def test_strip_empty_content():
    msgs = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "   "},
    ]
    result = sanitize(msgs, strip_empty=True)
    assert len(result.messages) == 1
    assert result.removed == 2


def test_strip_empty_disabled():
    msgs = [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "hello"},
    ]
    result = sanitize(msgs, strip_empty=False)
    assert len(result.messages) == 2


def test_system_first():
    msgs = [
        {"role": "user", "content": "Hello"},
        {"role": "system", "content": "Be helpful."},
    ]
    result = sanitize(msgs, system_first=True)
    assert result.messages[0]["role"] == "system"


def test_system_already_first_no_move():
    result = sanitize(MSGS, system_first=True)
    assert result.messages[0]["role"] == "system"
    assert result.removed == 0


def test_allowed_roles_filters_unknown():
    msgs = [
        {"role": "user", "content": "Hello"},
        {"role": "bot", "content": "Hi"},
        {"role": "assistant", "content": "Hey"},
    ]
    result = sanitize(msgs, allowed_roles={"user", "assistant"})
    assert len(result.messages) == 2
    assert all(m["role"] in {"user", "assistant"} for m in result.messages)


def test_max_messages_truncates():
    msgs = [
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "3"},
        {"role": "assistant", "content": "4"},
    ]
    result = sanitize(msgs, max_messages=2)
    assert len(result.messages) == 2
    assert result.messages[-1]["content"] == "4"  # keeps most recent


def test_max_messages_preserves_system():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "3"},
    ]
    result = sanitize(msgs, max_messages=2)
    # system + 1 most recent other
    assert result.messages[0]["role"] == "system"
    assert len(result.messages) == 2


def test_max_messages_caps_when_system_exceeds_budget():
    msgs = [
        {"role": "system", "content": "s1"},
        {"role": "system", "content": "s2"},
        {"role": "system", "content": "s3"},
        {"role": "user", "content": "u1"},
    ]
    result = sanitize(msgs, max_messages=2)
    # Final list must never exceed max_messages, even if system msgs alone do.
    assert len(result.messages) == 2
    # Most recent system messages are kept; non-system dropped.
    assert [m["content"] for m in result.messages] == ["s2", "s3"]
    assert result.removed == 2


def test_max_messages_system_count_equals_budget():
    msgs = [
        {"role": "system", "content": "a"},
        {"role": "system", "content": "b"},
        {"role": "user", "content": "u"},
    ]
    result = sanitize(msgs, max_messages=2)
    assert len(result.messages) == 2
    assert [m["content"] for m in result.messages] == ["a", "b"]


def test_enforce_alternating_removes_consecutive():
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "reply"},
    ]
    result = sanitize(msgs, enforce_alternating=True)
    assert len(result.messages) == 2
    assert result.messages[0]["content"] == "second"  # keeps last


def test_strip_unknown_keys():
    msgs = [
        {"role": "user", "content": "Hello", "extra_field": "value", "ts": 123},
    ]
    result = sanitize(msgs, strip_unknown_keys=True)
    assert "extra_field" not in result.messages[0]
    assert "ts" not in result.messages[0]
    assert "role" in result.messages[0]
    assert "content" in result.messages[0]


def test_sanitize_result_type():
    result = sanitize(MSGS)
    assert isinstance(result, SanitizeResult)
    assert isinstance(result.messages, list)


def test_strip_empty_helper():
    msgs = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": ""},
    ]
    cleaned = strip_empty(msgs)
    assert len(cleaned) == 1


def test_move_system_first_helper():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "sys"},
    ]
    reordered = move_system_first(msgs)
    assert reordered[0]["role"] == "system"


def test_enforce_roles_helper():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "moderator", "content": "removed"},
    ]
    cleaned = enforce_roles(msgs, {"user", "assistant"})
    assert all(m["role"] in {"user", "assistant"} for m in cleaned)


def test_normalize_content_strips_whitespace():
    msgs = [
        {"role": "user", "content": "  hello  "},
        {"role": "assistant", "content": "\nworld\n"},
    ]
    normalized = normalize_content(msgs)
    assert normalized[0]["content"] == "hello"
    assert normalized[1]["content"] == "world"


def test_empty_input():
    result = sanitize([])
    assert result.messages == []
    assert result.removed == 0


def test_issues_reported():
    msgs = [
        {"role": "user", "content": ""},
    ]
    result = sanitize(msgs, strip_empty=True)
    assert len(result.issues) > 0
