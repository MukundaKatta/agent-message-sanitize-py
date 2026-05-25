# agent-message-sanitize-py

Sanitize and normalize LLM message lists. Remove empty messages, fix system message ordering, enforce role allowlists, and truncate to budget.

## Install

```bash
pip install agent-message-sanitize-py
```

## Usage

```python
from agent_message_sanitize import sanitize, strip_empty, move_system_first, normalize_content

messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": ""},   # empty — removed
    {"role": "system", "content": "Be helpful."},   # moved to front
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

print(result.messages)  # cleaned list
print(result.removed)   # count of removed messages
print(result.issues)    # list of human-readable issue descriptions

# Helpers
cleaned = strip_empty(messages)
reordered = move_system_first(messages)
normalized = normalize_content(messages)  # strips whitespace from content
```

## License

MIT
