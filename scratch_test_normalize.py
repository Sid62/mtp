import json
import re
from typing import Any

valid_agent_ids = {"uav_0", "uav_1", "vehicle_2", "robot_3"}
valid_subtask_ids = {"T_0", "T_1", "T_2", "T_3"}

def normalize_task_id(tid: Any, valid_subtask_ids: set[str] | None = None) -> str:
    s = str(tid).strip()
    if not valid_subtask_ids:
        return s
    if s in valid_subtask_ids:
        return s
    if s.isdigit() and f"T_{s}" in valid_subtask_ids:
        return f"T_{s}"
    m = re.search(r"(\d+)", s)
    if m:
        num = m.group(1)
        if f"T_{num}" in valid_subtask_ids:
            return f"T_{num}"
        for v in valid_subtask_ids:
            vm = re.search(r"(\d+)", v)
            if vm and vm.group(1) == num:
                return v
    return s

test_cases = [
    ("T_0", "T_0"),
    ("0", "T_0"),
    ("subtask_id_0", "T_0"),
    ("subtask_1", "T_1"),
    ("task_2", "T_2"),
    ("T_3", "T_3"),
]

for inp, expected in test_cases:
    res = normalize_task_id(inp, valid_subtask_ids)
    print(f"normalize_task_id({inp!r}) -> {res!r} (expected {expected!r})")
    assert res == expected

print("All normalize_task_id tests passed!")
