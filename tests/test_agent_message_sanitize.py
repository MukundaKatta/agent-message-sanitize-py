"""Tests for agent-message-sanitize-py.

These tests use only the Python standard library ``unittest`` module so they can
be run without any third-party dependencies::

    python3 -m unittest discover -s tests

The package lives under ``src/`` so, when it is not installed, we add that
directory to ``sys.path`` before importing it.
"""

from __future__ import annotations

import os
import sys
import unittest

# Allow the tests to import the package straight from ``src/`` without an install.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from agent_message_sanitize import (  # noqa: E402
    SanitizeResult,
    enforce_roles,
    move_system_first,
    normalize_content,
    sanitize,
    strip_empty,
)

MSGS = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
]


class SanitizeBasicsTest(unittest.TestCase):
    def test_passthrough_clean_messages(self):
        result = sanitize(MSGS)
        self.assertEqual(result.removed, 0)
        self.assertEqual(len(result.messages), 3)

    def test_empty_input(self):
        result = sanitize([])
        self.assertEqual(result.messages, [])
        self.assertEqual(result.removed, 0)
        self.assertEqual(result.issues, [])

    def test_sanitize_result_type(self):
        result = sanitize(MSGS)
        self.assertIsInstance(result, SanitizeResult)
        self.assertIsInstance(result.messages, list)
        self.assertIsInstance(result.removed, int)
        self.assertIsInstance(result.issues, list)

    def test_does_not_mutate_input_list(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": ""},
        ]
        original = [dict(m) for m in msgs]
        sanitize(msgs, strip_empty=True, strip_unknown_keys=True)
        # The caller's list and dicts must be untouched.
        self.assertEqual(msgs, original)
        self.assertEqual(len(msgs), 2)


class StripEmptyTest(unittest.TestCase):
    def test_strip_empty_content(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "   "},
        ]
        result = sanitize(msgs, strip_empty=True)
        self.assertEqual(len(result.messages), 1)
        self.assertEqual(result.removed, 2)

    def test_strip_empty_disabled(self):
        msgs = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "hello"},
        ]
        result = sanitize(msgs, strip_empty=False)
        self.assertEqual(len(result.messages), 2)

    def test_missing_content_treated_as_empty(self):
        msgs = [
            {"role": "user"},  # no content key
            {"role": "assistant", "content": "hi"},
        ]
        result = sanitize(msgs, strip_empty=True)
        self.assertEqual(len(result.messages), 1)
        self.assertEqual(result.messages[0]["content"], "hi")

    def test_non_string_content_preserved(self):
        # Multimodal / structured content (a list) is non-empty and must survive.
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ]
        result = sanitize(msgs, strip_empty=True)
        self.assertEqual(len(result.messages), 1)


class SystemFirstTest(unittest.TestCase):
    def test_system_first(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "Be helpful."},
        ]
        result = sanitize(msgs, system_first=True)
        self.assertEqual(result.messages[0]["role"], "system")

    def test_system_already_first_no_move(self):
        result = sanitize(MSGS, system_first=True)
        self.assertEqual(result.messages[0]["role"], "system")
        self.assertEqual(result.removed, 0)

    def test_system_first_preserves_relative_order(self):
        msgs = [
            {"role": "user", "content": "u1"},
            {"role": "system", "content": "s1"},
            {"role": "assistant", "content": "a1"},
            {"role": "system", "content": "s2"},
        ]
        result = sanitize(msgs, system_first=True)
        self.assertEqual(
            [m["content"] for m in result.messages],
            ["s1", "s2", "u1", "a1"],
        )

    def test_system_first_disabled(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "Be helpful."},
        ]
        result = sanitize(msgs, system_first=False)
        self.assertEqual(result.messages[0]["role"], "user")


class AllowedRolesTest(unittest.TestCase):
    def test_allowed_roles_filters_unknown(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "bot", "content": "Hi"},
            {"role": "assistant", "content": "Hey"},
        ]
        result = sanitize(msgs, allowed_roles={"user", "assistant"})
        self.assertEqual(len(result.messages), 2)
        self.assertTrue(
            all(m["role"] in {"user", "assistant"} for m in result.messages)
        )

    def test_default_allowlist_includes_tool_and_function(self):
        msgs = [
            {"role": "tool", "content": "result"},
            {"role": "function", "content": "out"},
            {"role": "weird", "content": "drop me"},
        ]
        result = sanitize(msgs)
        self.assertEqual(len(result.messages), 2)
        self.assertEqual(result.removed, 1)


class MaxMessagesTest(unittest.TestCase):
    def test_max_messages_truncates(self):
        msgs = [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": "3"},
            {"role": "assistant", "content": "4"},
        ]
        result = sanitize(msgs, max_messages=2)
        self.assertEqual(len(result.messages), 2)
        self.assertEqual(result.messages[-1]["content"], "4")  # keeps most recent

    def test_max_messages_preserves_system(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": "3"},
        ]
        result = sanitize(msgs, max_messages=2)
        self.assertEqual(result.messages[0]["role"], "system")
        self.assertEqual(len(result.messages), 2)

    def test_max_messages_caps_when_system_exceeds_budget(self):
        msgs = [
            {"role": "system", "content": "s1"},
            {"role": "system", "content": "s2"},
            {"role": "system", "content": "s3"},
            {"role": "user", "content": "u1"},
        ]
        result = sanitize(msgs, max_messages=2)
        # Final list must never exceed max_messages, even if system msgs alone do.
        self.assertEqual(len(result.messages), 2)
        # Most recent system messages are kept; non-system dropped.
        self.assertEqual([m["content"] for m in result.messages], ["s2", "s3"])
        self.assertEqual(result.removed, 2)

    def test_max_messages_system_count_equals_budget(self):
        msgs = [
            {"role": "system", "content": "a"},
            {"role": "system", "content": "b"},
            {"role": "user", "content": "u"},
        ]
        result = sanitize(msgs, max_messages=2)
        self.assertEqual(len(result.messages), 2)
        self.assertEqual([m["content"] for m in result.messages], ["a", "b"])

    def test_max_messages_no_truncation_when_under_budget(self):
        result = sanitize(MSGS, max_messages=10)
        self.assertEqual(len(result.messages), 3)
        self.assertEqual(result.removed, 0)


class EnforceAlternatingTest(unittest.TestCase):
    def test_enforce_alternating_removes_consecutive(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "reply"},
        ]
        result = sanitize(msgs, enforce_alternating=True)
        self.assertEqual(len(result.messages), 2)
        self.assertEqual(result.messages[0]["content"], "second")  # keeps last

    def test_enforce_alternating_disabled_by_default(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]
        result = sanitize(msgs)
        self.assertEqual(len(result.messages), 2)


class StripUnknownKeysTest(unittest.TestCase):
    def test_strip_unknown_keys(self):
        msgs = [
            {"role": "user", "content": "Hello", "extra_field": "value", "ts": 123},
        ]
        result = sanitize(msgs, strip_unknown_keys=True)
        self.assertNotIn("extra_field", result.messages[0])
        self.assertNotIn("ts", result.messages[0])
        self.assertIn("role", result.messages[0])
        self.assertIn("content", result.messages[0])

    def test_unknown_keys_kept_by_default(self):
        msgs = [{"role": "user", "content": "Hello", "name": "alice"}]
        result = sanitize(msgs)
        self.assertEqual(result.messages[0].get("name"), "alice")


class HelpersTest(unittest.TestCase):
    def test_strip_empty_helper(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": ""},
        ]
        cleaned = strip_empty(msgs)
        self.assertEqual(len(cleaned), 1)

    def test_move_system_first_helper(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "sys"},
        ]
        reordered = move_system_first(msgs)
        self.assertEqual(reordered[0]["role"], "system")

    def test_enforce_roles_helper(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "moderator", "content": "removed"},
        ]
        cleaned = enforce_roles(msgs, {"user", "assistant"})
        self.assertTrue(all(m["role"] in {"user", "assistant"} for m in cleaned))

    def test_normalize_content_strips_whitespace(self):
        msgs = [
            {"role": "user", "content": "  hello  "},
            {"role": "assistant", "content": "\nworld\n"},
        ]
        normalized = normalize_content(msgs)
        self.assertEqual(normalized[0]["content"], "hello")
        self.assertEqual(normalized[1]["content"], "world")

    def test_normalize_content_leaves_non_string_alone(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        normalized = normalize_content(msgs)
        self.assertEqual(normalized[0]["content"], [{"type": "text", "text": "hi"}])

    def test_normalize_content_does_not_mutate_input(self):
        msgs = [{"role": "user", "content": "  hi  "}]
        normalize_content(msgs)
        self.assertEqual(msgs[0]["content"], "  hi  ")


class IssuesReportingTest(unittest.TestCase):
    def test_issues_reported(self):
        msgs = [{"role": "user", "content": ""}]
        result = sanitize(msgs, strip_empty=True)
        self.assertTrue(len(result.issues) > 0)

    def test_combined_pipeline_reports_multiple_issues(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": ""},  # empty -> removed
            {"role": "bot", "content": "drop"},  # unknown role -> removed
            {"role": "system", "content": "Be brief."},  # moved to front
        ]
        result = sanitize(
            msgs,
            strip_empty=True,
            system_first=True,
            allowed_roles={"system", "user", "assistant"},
        )
        self.assertEqual(result.messages[0]["role"], "system")
        self.assertEqual(result.removed, 2)
        self.assertTrue(len(result.issues) >= 2)


if __name__ == "__main__":
    unittest.main()
