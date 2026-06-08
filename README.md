# agent-message-sanitize-py

Sanitize and normalize LLM message lists before sending them to a chat
completion API. It removes empty messages, fixes system-message ordering,
enforces a role allowlist, optionally collapses consecutive same-role turns,
strips unknown keys, and truncates a conversation to a message budget — all in
a single pass that reports exactly what it changed.

The library is dependency-free and ships type information (PEP 561).

## Install

```bash
pip install agent-message-sanitize-py
```

## Quick start

```python
from agent_message_sanitize import sanitize

messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": ""},          # empty -> removed
    {"role": "system", "content": "Be helpful."},   # moved to front
    {"role": "bot", "content": "ignore me"},        # unknown role -> removed
]

result = sanitize(
    messages,
    strip_empty=True,
    system_first=True,
    max_messages=10,
    allowed_roles={"system", "user", "assistant"},
    enforce_alternating=False,
    strip_unknown_keys=True,
)

print(result.messages)  # cleaned list, system message first
print(result.removed)   # 2
print(result.issues)    # human-readable descriptions of every change
```

`sanitize` never mutates the list or the dicts you pass in; it returns new
objects, so the original `messages` is safe to keep using.

## API

### `sanitize(messages, *, strip_empty=True, enforce_alternating=False, system_first=True, max_messages=None, allowed_roles=None, strip_unknown_keys=False) -> SanitizeResult`

Run the full sanitisation pipeline. The steps are applied in this order:

1. **Role allowlist** — drop any message whose `role` is not in `allowed_roles`
   (defaults to `{"system", "user", "assistant", "tool", "function"}`).
2. **Strip empty** — when `strip_empty` is true, drop messages whose `content`
   is missing, `None`, or whitespace-only. Non-string content (for example a
   list of multimodal parts) is treated as non-empty and kept.
3. **System first** — when `system_first` is true, move every `system` message
   to the front, preserving their relative order and the order of the rest.
4. **Enforce alternating** — when `enforce_alternating` is true, collapse runs
   of consecutive same-role messages, keeping the last message in each run.
5. **Max messages** — when `max_messages` is set, keep at most that many
   messages. System messages are preferred; if they alone exceed the budget the
   most recent system messages are kept and everything else is dropped, so the
   result never exceeds `max_messages`.
6. **Strip unknown keys** — when `strip_unknown_keys` is true, reduce each
   message to just its `role` and `content` keys.

Returns a [`SanitizeResult`](#sanitizeresult).

### `SanitizeResult`

A dataclass describing the outcome:

| Field      | Type              | Description                                          |
| ---------- | ----------------- | ---------------------------------------------------- |
| `messages` | `list[dict]`      | The sanitized message list.                          |
| `removed`  | `int`             | Total number of messages removed or merged away.     |
| `issues`   | `list[str]`       | Human-readable descriptions of each change applied.  |

### Convenience helpers

Each helper wraps `sanitize` for a single common task and returns the cleaned
list directly:

- `strip_empty(messages) -> list[dict]` — remove empty / whitespace-only messages.
- `move_system_first(messages) -> list[dict]` — move system messages to the front.
- `enforce_roles(messages, allowed) -> list[dict]` — drop messages whose role is not in `allowed`.
- `normalize_content(messages) -> list[dict]` — return copies with leading/trailing whitespace stripped from string content (non-string content is left untouched).

## Development

Run the test suite with the standard library only — no third-party
dependencies are required:

```bash
python3 -m unittest discover -s tests
```

## License

MIT
