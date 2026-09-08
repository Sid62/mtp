"""AutoHMA-LLM structural alignment data structures.

Lightweight dataclasses representing AutoHMA-aligned concepts:
- ExecutionFeedback: per-agent execution result
- DeviceFeedback: per-domain aggregated review
- ExecutionDirective: Device LLM dispatch output wrapper

These are purely structural wrappers over data that already exists in memory.
They introduce no new computation, no new API calls, and no behavioral change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AssignmentStatus(str, Enum):
    """Canonical lifecycle status of a task-agent assignment."""
    CANDIDATE = "CANDIDATE"
    VALID = "VALID"
    INVALID = "INVALID"
    UNRESOLVED = "UNRESOLVED"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"


@dataclass
class AssignmentRecord:
    """Canonical representation of an assignment."""
    task_id: str
    agent_id: str
    source: str = "unknown"  # "cloud", "device", "reallocation", "consensus", "continuity", "local_reassign"
    status: AssignmentStatus = AssignmentStatus.CANDIDATE
    reason: str = ""
    step: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "source": self.source,
            "status": self.status.value if isinstance(self.status, AssignmentStatus) else str(self.status),
            "reason": self.reason,
            "step": self.step,
        }


@dataclass
class ExecutionFeedback:
    """Per-agent execution result from Generative Agent → Device LLM."""

    agent_id: str = ""
    subtask_id: str = ""
    distance_to_target: float = 0.0
    completed: bool = False
    step: int = 0

    def summary(self) -> str:
        status = "DONE" if self.completed else f"d={self.distance_to_target:.1f}"
        return f"{self.agent_id}:{self.subtask_id}({status})"


@dataclass
class DeviceFeedback:
    """Aggregated Device LLM review of execution in its managed domain.

    AutoHMA flow: Generative Agent → Device LLM → Cloud LLM
    This represents the Device LLM → Cloud LLM upward feedback.
    """

    domain_id: str = ""
    agent_feedbacks: list[ExecutionFeedback] = field(default_factory=list)
    tasks_completed: list[str] = field(default_factory=list)
    tasks_in_progress: list[str] = field(default_factory=list)
    step: int = 0

    def to_context_string(self) -> str:
        """Format as compact text for injection into Cloud LLM prompt."""
        parts = [f"Domain={self.domain_id}"]
        if self.tasks_completed:
            parts.append(f"done=[{','.join(self.tasks_completed)}]")
        if self.tasks_in_progress:
            progress = []
            for fb in self.agent_feedbacks:
                if not fb.completed:
                    progress.append(fb.summary())
            if progress:
                parts.append(f"progress=[{','.join(progress)}]")
        return " ".join(parts)


RESERVED_SCHEMA_KEYS: frozenset[str] = frozenset({
    "agent_id", "subtask_id", "task_id", "assignments", "dispatched",
    "ack", "waypoints", "status", "action", "coalition_id", "mode",
    "domain", "domain_id", "agents", "subtasks", "type", "members",
    "instruction", "target", "result", "success"
})


def normalize_subtask_id(raw_id: Any, valid_subtask_ids: set[str] | None = None) -> str:
    """Normalize raw subtask ID string or int into canonical format (e.g. 'T_0')."""
    import re
    s = str(raw_id).strip()
    if s.startswith(("uav", "robot", "vehicle")):
        return s
    if valid_subtask_ids:
        if s in valid_subtask_ids:
            return s
        if s.isdigit() and f"T_{s}" in valid_subtask_ids:
            return f"T_{s}"
        m = re.search(r"(\d+)", s)
        if m:
            num = m.group(1)
            candidate = f"T_{num}"
            if candidate in valid_subtask_ids:
                return candidate
            for v in valid_subtask_ids:
                vm = re.search(r"(\d+)", v)
                if vm and vm.group(1) == num:
                    return v
    if s.isdigit():
        return f"T_{s}"
    return s


def normalize_device_dispatch(
    result: Any,
    managed_agent_ids: set[str] | None = None,
    valid_subtask_ids: set[str] | None = None,
) -> dict[str, str]:
    """Normalize diverse Device LLM dispatch responses into canonical agent -> subtask mapping.

    Canonical format: dict[str, str] = {agent_id: subtask_id}
    Ensures that schema keys ('agent_id', 'subtask_id', etc.) are NEVER treated as agent IDs.
    """
    if not isinstance(result, (dict, list)):
        return {}

    normalized: dict[str, str] = {}

    def _is_subtask_candidate(s: str) -> bool:
        if not s or not isinstance(s, str):
            return False
        clean = s.strip()
        if clean.lower() in RESERVED_SCHEMA_KEYS:
            return False
        if clean.startswith(("uav", "robot", "vehicle")):
            return False
        if valid_subtask_ids and clean in valid_subtask_ids:
            return True
        if clean.startswith(("T_", "subtask_")) or clean.isdigit():
            return True
        return False

    def _is_valid_agent(aid: Any) -> bool:
        if not aid or not isinstance(aid, str):
            return False
        aid_clean = aid.strip()
        if aid_clean.lower() in RESERVED_SCHEMA_KEYS:
            return False
        if _is_subtask_candidate(aid_clean):
            return False
        if managed_agent_ids is not None and aid_clean not in managed_agent_ids:
            return False
        return True

    def _process_record(rec: dict) -> None:
        if not isinstance(rec, dict):
            return
        aid = rec.get("agent_id") or rec.get("id")
        sid = rec.get("subtask_id") or rec.get("task_id")
        if aid and sid:
            aid_str = str(aid).strip()
            if _is_valid_agent(aid_str):
                norm_sid = normalize_subtask_id(sid, valid_subtask_ids)
                if not valid_subtask_ids or norm_sid in valid_subtask_ids:
                    normalized[aid_str] = norm_sid

    def _process_mapping(mapping: dict) -> None:
        if not isinstance(mapping, dict):
            return
        # First check if mapping itself is a Schema B record
        if "agent_id" in mapping and ("subtask_id" in mapping or "task_id" in mapping):
            _process_record(mapping)
            return

        for k, v in mapping.items():
            if not isinstance(k, str):
                continue
            k_clean = k.strip()
            if k_clean.lower() in RESERVED_SCHEMA_KEYS:
                continue

            # Case 1: k is a subtask ID (Schema A: {"T_6": ["uav_1"]})
            if _is_subtask_candidate(k_clean):
                norm_sid = normalize_subtask_id(k_clean, valid_subtask_ids)
                if not valid_subtask_ids or norm_sid in valid_subtask_ids:
                    agents = v if isinstance(v, list) else [v]
                    for a in agents:
                        if isinstance(a, str) and _is_valid_agent(a.strip()):
                            normalized[a.strip()] = norm_sid
                continue

            # Case 2: k is an agent ID (Inverted Map: {"uav_1": "T_6"})
            if _is_valid_agent(k_clean):
                raw_sid = v[0] if isinstance(v, list) and v else v
                if isinstance(raw_sid, (str, int)) and _is_subtask_candidate(str(raw_sid)):
                    norm_sid = normalize_subtask_id(raw_sid, valid_subtask_ids)
                    if not valid_subtask_ids or norm_sid in valid_subtask_ids:
                        normalized[k_clean] = norm_sid
                continue

    payload: Any = result
    if isinstance(payload, dict):
        if len(payload) == 1:
            k, v = next(iter(payload.items()))
            if isinstance(v, dict) and (k.lower() in ("uav", "robot", "vehicle", "device") or "assignments" in v or "dispatched" in v):
                payload = v

    if isinstance(payload, dict):
        if "assignments" in payload:
            raw_assigns = payload["assignments"]
            if isinstance(raw_assigns, list):
                for item in raw_assigns:
                    if isinstance(item, dict):
                        _process_record(item)
            elif isinstance(raw_assigns, dict):
                _process_mapping(raw_assigns)
            return normalized

        _process_mapping(payload)
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                _process_record(item)

    return normalized


@dataclass
class ExecutionDirective:
    """Wrapper for Device LLM dispatch output.

    AutoHMA flow: Cloud LLM → Device LLM → Generative Agent
    This represents the Device LLM → Generative Agent directive.
    The actual control remains with the existing PID/NMPC/Q-learning;
    this is the structural link that is consumed by the execution path.
    """

    domain_id: str = ""
    dispatch_result: dict[str, Any] = field(default_factory=dict)
    coalitions: list[dict] = field(default_factory=list)
    agent_assignments: dict[str, str] = field(default_factory=dict)
    step: int = 0

    def get_agent_assignment(self, agent_id: str) -> str | None:
        """Extract assigned subtask id for a specific agent from this domain directive."""
        if agent_id in self.agent_assignments:
            return self.agent_assignments[agent_id]
        return None


def format_feedback_for_cloud(
    device_feedbacks: list[DeviceFeedback],
) -> str | None:
    """Format Device LLM feedbacks as a compact context string for Cloud LLM.

    Takes the latest feedback per domain to avoid redundant duplication across steps.
    Includes explicit self-correction instructions for the Cloud planner.
    Returns None if there is no meaningful feedback to include,
    avoiding prompt bloat on the first planning call.
    """
    if not device_feedbacks:
        return None

    # Take the latest feedback per domain
    latest_by_domain: dict[str, DeviceFeedback] = {}
    for fb in device_feedbacks:
        latest_by_domain[fb.domain_id] = fb

    has_content = any(
        fb.tasks_completed or fb.tasks_in_progress
        for fb in latest_by_domain.values()
    )
    if not has_content:
        return None

    lines = [
        "[Previous Execution Feedback & Self-Correction Context]",
        "Instructions: Review previous execution feedback before generating the next plan.",
        "If previous assignments produced incomplete tasks, insufficient progress, or bottlenecks, refine the allocation accordingly.",
        "Do not blindly repeat assignments that proved ineffective in the feedback.",
    ]
    for domain_id in sorted(latest_by_domain.keys()):
        line = latest_by_domain[domain_id].to_context_string()
        if line:
            lines.append(line)
    return "\n".join(lines)


def normalize_to_assignment_records(
    raw: dict[str, Any],
    source: str = "unknown",
    step: int = 0,
    valid_subtask_ids: set[str] | None = None,
) -> list[AssignmentRecord]:
    """Convert raw assignments (task->agents or agent->task) into canonical AssignmentRecords."""
    records: list[AssignmentRecord] = []
    if not isinstance(raw, dict):
        return records

    for k, v in raw.items():
        k_str = str(k).strip()
        if k_str.lower() in RESERVED_SCHEMA_KEYS:
            continue
        if isinstance(v, (list, set, tuple)):
            for item in v:
                aid = str(item).strip()
                if aid and aid.lower() not in RESERVED_SCHEMA_KEYS:
                    sid = normalize_subtask_id(k_str, valid_subtask_ids)
                    records.append(
                        AssignmentRecord(
                            task_id=sid,
                            agent_id=aid,
                            source=source,
                            status=AssignmentStatus.CANDIDATE,
                            step=step,
                        )
                    )
        elif isinstance(v, str):
            v_str = v.strip()
            if v_str.lower() in RESERVED_SCHEMA_KEYS:
                continue
            # Determine which is task and which is agent
            if k_str.startswith(("uav", "robot", "vehicle")):
                sid = normalize_subtask_id(v_str, valid_subtask_ids)
                records.append(
                    AssignmentRecord(
                        task_id=sid,
                        agent_id=k_str,
                        source=source,
                        status=AssignmentStatus.CANDIDATE,
                        step=step,
                    )
                )
            else:
                sid = normalize_subtask_id(k_str, valid_subtask_ids)
                records.append(
                    AssignmentRecord(
                        task_id=sid,
                        agent_id=v_str,
                        source=source,
                        status=AssignmentStatus.CANDIDATE,
                        step=step,
                    )
                )
    return records


def records_to_task_assignments(
    records: list[AssignmentRecord],
    allowed_statuses: set[AssignmentStatus] | None = None,
) -> dict[str, list[str]]:
    """Convert records to task_id -> list[agent_id] dict."""
    res: dict[str, list[str]] = {}
    for r in records:
        if allowed_statuses is None or r.status in allowed_statuses:
            res.setdefault(r.task_id, []).append(r.agent_id)
    return res


def records_to_agent_assignments(
    records: list[AssignmentRecord],
    allowed_statuses: set[AssignmentStatus] | None = None,
) -> dict[str, str]:
    """Convert records to agent_id -> task_id dict (for execution)."""
    res: dict[str, str] = {}
    for r in records:
        if allowed_statuses is None or r.status in allowed_statuses:
            res[r.agent_id] = r.task_id
    return res

