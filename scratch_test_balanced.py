import json
import re
from typing import Any

def clean_llm_text(raw: str) -> str:
    # 1. XML-style thinking tags
    text = re.sub(r"<(?:think|thought)>.*?</(?:think|thought)>", "", raw, flags=re.DOTALL)
    # 2. Conversational thinking headers before code fence or first {
    text = re.sub(r"^(?:Here(?:'s| is) a thinking process:?|Thinking Process:?|Thought:?).*?(?=(?:```|\{))", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()

def extract_balanced_json_candidates(text: str) -> list[str]:
    candidates = []
    # 1. Markdown code blocks
    for m in re.finditer(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL):
        cand = m.group(1).strip()
        if (cand.startswith("{") and cand.endswith("}")) or (cand.startswith("[") and cand.endswith("]")):
            candidates.append(cand)

    # 2. Balanced braces { ... }
    n = len(text)
    i = 0
    while i < n:
        if text[i] == "{":
            start = i
            depth = 0
            in_str = False
            escape = False
            j = i
            while j < n:
                ch = text[j]
                if in_str:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            candidates.append(text[start:j+1])
                            break
                j += 1
        i += 1

    # 3. Balanced brackets [ ... ]
    i = 0
    while i < n:
        if text[i] == "[":
            start = i
            depth = 0
            in_str = False
            escape = False
            j = i
            while j < n:
                ch = text[j]
                if in_str:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == "[":
                        depth += 1
                    elif ch == "]":
                        depth -= 1
                        if depth == 0:
                            candidates.append(text[start:j+1])
                            break
                j += 1
        i += 1

    return candidates

def parse_relaxed_json(raw: str) -> Any:
    cleaned = clean_llm_text(raw)
    try:
        return json.loads(cleaned)
    except (ValueError, json.JSONDecodeError):
        pass

    candidates = extract_balanced_json_candidates(cleaned)
    parsed_candidates = []
    for cand in candidates:
        try:
            val = json.loads(cand)
            parsed_candidates.append(val)
        except (ValueError, json.JSONDecodeError):
            continue

    if not parsed_candidates:
        return {}

    # Prefer candidates with assignment keys
    assignment_keys = (
        "assignments", "task_assignments", "decomposition", "allocation",
        "tasks", "subtasks", "plan", "dispatch", "agent_assignments", "coalitions"
    )
    for p in parsed_candidates:
        if isinstance(p, dict) and any(k in p for k in assignment_keys):
            return p

    # Next prefer any non-empty dict or list
    for p in parsed_candidates:
        if isinstance(p, (dict, list)) and p:
            return p

    return parsed_candidates[0]

# Test cases
test1 = """Here's a thinking process:
1. Analyze User Input: We have agents {uav_0} and tasks {T_0}.
2. Constraints: distance <= 30m.
Return JSON:
{"assignments": {"T_0": ["uav_0"], "T_1": ["uav_1"]}}"""

test2 = """```json
{"assignments": {"T_0": ["uav_0"]}}
```"""

test3 = """Thinking Process:
Let's consider subtasks.
{"other": 123}
Final answer:
{"assignments": {"T_0": ["uav_1"]}}"""

print("Test 1:", parse_relaxed_json(test1))
print("Test 2:", parse_relaxed_json(test2))
print("Test 3:", parse_relaxed_json(test3))
assert parse_relaxed_json(test1) == {"assignments": {"T_0": ["uav_0"], "T_1": ["uav_1"]}}
assert parse_relaxed_json(test2) == {"assignments": {"T_0": ["uav_0"]}}
assert parse_relaxed_json(test3) == {"assignments": {"T_0": ["uav_1"]}}
print("ALL TESTS PASSED!")
