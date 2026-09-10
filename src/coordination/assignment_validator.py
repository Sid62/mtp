"""Deterministic Assignment Validator for Multi-Agent Task Allocation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence
from src.env.agents import AgentFleet
from src.env.scenarios import Subtask
from src.coordination.autohma_structs import (
    AssignmentRecord,
    AssignmentStatus,
    RESERVED_SCHEMA_KEYS,
    normalize_subtask_id,
)


@dataclass
class ValidationReport:
    is_valid: bool = True
    valid_assignments: dict[str, str] = field(default_factory=dict)  # agent_id -> task_id
    rejected_assignments: dict[str, str] = field(default_factory=dict)  # agent_id -> task_id
    rejection_reasons: dict[str, str] = field(default_factory=dict)  # agent_id -> reason
    unresolved_tasks: list[str] = field(default_factory=list)
    records: list[AssignmentRecord] = field(default_factory=list)


def log_assignment_transition(
    task_id: str,
    agent_id: str,
    source: str = "unknown",
    previous_status: str = "UNKNOWN",
    new_status: str = "UNKNOWN",
    mode: int | str = 0,
    step: int = 0,
    reason: str = "",
) -> None:
    """Structured lifecycle logging for assignment mutations (Phase 12)."""
    reason_str = f" reason={reason}" if reason else ""
    print(
        f"[ASSIGNMENT-STATE] "
        f"task={task_id} "
        f"agent={agent_id} "
        f"source={source} "
        f"previous={previous_status} "
        f"new={new_status} "
        f"mode={mode} "
        f"step={step}"
        f"{reason_str}"
    )


def log_assignment_removal(
    task_id: str,
    agent_id: str,
    mode: int | str = 0,
    step: int = 0,
    reason: str = "",
) -> None:
    """Structured lifecycle logging when an assignment is removed from execution."""
    reason_str = f" reason={reason}" if reason else ""
    print(
        f"[ASSIGNMENT-STATE] "
        f"task={task_id} "
        f"agent={agent_id} "
        f"action=REMOVED_FROM_EXECUTION "
        f"mode={mode} "
        f"step={step}"
        f"{reason_str}"
    )


class AssignmentValidator:
    """Strict, deterministic validation independent of the LLM.

    Validates:
    - Task ID exists and is currently active (not completed)
    - Agent ID exists in the fleet
    - Agent is available (not multiply assigned in the same step)
    - Required skills of the subtask are satisfied by the assigned agent
    """

    @staticmethod
    def validate_single(
        agent_id: str,
        subtask_id: str,
        fleet: AgentFleet,
        subtasks: Sequence[Subtask],
        assigned_agents: set[str] | None = None,
        check_skills: bool = True,
        strict_skills: bool = False,
        coalitions: list[dict[str, Any]] | None = None,
    ) -> tuple[bool, str]:
        if not isinstance(agent_id, str) or not agent_id.strip():
            return False, f"invalid_agent_id_format:{agent_id}"

        if agent_id.lower().strip() in RESERVED_SCHEMA_KEYS:
            return False, f"reserved_schema_keyword_as_agent_id:{agent_id}"

        if not isinstance(subtask_id, str) or not subtask_id.strip():
            return False, f"invalid_subtask_id_format:{subtask_id}"

        if subtask_id.lower().strip() in RESERVED_SCHEMA_KEYS:
            return False, f"reserved_schema_keyword_as_subtask_id:{subtask_id}"

        if not fleet.has_agent(agent_id):
            return False, f"unknown_agent:{agent_id}"

        st = next((s for s in subtasks if s.subtask_id == subtask_id), None)
        if st is None:
            return False, f"unknown_task:{subtask_id}"

        if st.completed:
            return False, f"task_already_completed:{subtask_id}"

        if assigned_agents is not None and agent_id in assigned_agents:
            return False, f"agent_multiply_assigned:{agent_id}"

        if check_skills and st.required_skills:
            agent = fleet.get_agent(agent_id)
            required = set(st.required_skills)
            agent_skills = set(agent.skills)

            # Check coalition coverage if coalitions provided
            coalition_skills = set(agent_skills)
            if coalitions:
                for c in coalitions:
                    members = c.get("members", [])
                    if agent_id in members:
                        for m in members:
                            if fleet.has_agent(m):
                                coalition_skills.update(fleet.get_agent(m).skills)
                        break

            if strict_skills:
                if not required.issubset(coalition_skills):
                    missing = sorted(list(required - coalition_skills))
                    return False, f"missing_required_skill:{missing}"
            else:
                # Issue 1 fix: required skills are a hard constraint —
                # the assigned agent/coalition must cover ALL required skills,
                # not just overlap with at least one.
                if not required.issubset(coalition_skills):
                    missing = sorted(list(required - coalition_skills))
                    return False, f"missing_required_skill:{missing}"

        return True, ""

    @classmethod
    def filter_assignments(
        cls,
        candidate_assignments: dict[str, str],  # agent_id -> subtask_id
        fleet: AgentFleet,
        subtasks: Sequence[Subtask],
        check_skills: bool = True,
        strict_skills: bool = False,
        coalitions: list[dict[str, Any]] | None = None,
        log_diagnostics: bool = True,
        source: str = "execution",
        mode: int | str = 0,
        step: int = 0,
    ) -> ValidationReport:
        """Validate candidate agent->subtask assignments and reject invalid mappings."""
        report = ValidationReport(is_valid=True)
        assigned_agents: set[str] = set()

        for aid, sid in candidate_assignments.items():
            valid, reason = cls.validate_single(
                aid, sid, fleet, subtasks,
                assigned_agents=assigned_agents,
                check_skills=check_skills,
                strict_skills=strict_skills,
                coalitions=coalitions,
            )
            if valid:
                report.valid_assignments[aid] = sid
                assigned_agents.add(aid)
                rec = AssignmentRecord(
                    task_id=sid,
                    agent_id=aid,
                    source=source,
                    status=AssignmentStatus.VALID,
                    step=step,
                )
                report.records.append(rec)
            else:
                report.is_valid = False
                report.rejected_assignments[aid] = sid
                report.rejection_reasons[aid] = reason
                rec = AssignmentRecord(
                    task_id=sid,
                    agent_id=aid,
                    source=source,
                    status=AssignmentStatus.INVALID,
                    reason=reason,
                    step=step,
                )
                report.records.append(rec)
                if log_diagnostics:
                    print(
                        f"\n[INVALID_ASSIGNMENT]\n"
                        f"    agent={aid}\n"
                        f"    task={sid}\n"
                        f"    reason={reason}"
                    )
                    log_assignment_transition(
                        task_id=sid,
                        agent_id=aid,
                        source=source,
                        previous_status="CANDIDATE",
                        new_status="INVALID",
                        mode=mode,
                        step=step,
                        reason=reason,
                    )
                    log_assignment_removal(
                        task_id=sid,
                        agent_id=aid,
                        mode=mode,
                        step=step,
                        reason=reason,
                    )

        return report

    @classmethod
    def validate_and_clean_plan(
        cls,
        assignments: dict[str, list[str]],  # task_id -> list[agent_id]
        fleet: AgentFleet,
        subtasks: Sequence[Subtask],
        check_skills: bool = True,
        strict_skills: bool = False,
        coalitions: list[dict[str, Any]] | None = None,
        log_diagnostics: bool = True,
        source: str = "planning",
        mode: int | str = 0,
        step: int = 0,
    ) -> tuple[dict[str, list[str]], ValidationReport]:
        """Validate and clean task -> agent_list assignments, removing completed/invalid entries."""
        cleaned: dict[str, list[str]] = {}
        report = ValidationReport(is_valid=True)
        assigned_agents: set[str] = set()
        active_sids = {s.subtask_id for s in subtasks if not s.completed}

        for sid, agent_list in assignments.items():
            if sid not in active_sids:
                # Omit completed or unknown tasks
                continue
            valid_agents: list[str] = []
            for aid in agent_list:
                valid, reason = cls.validate_single(
                    aid, sid, fleet, subtasks,
                    assigned_agents=assigned_agents,
                    check_skills=check_skills,
                    strict_skills=strict_skills,
                    coalitions=coalitions,
                )
                if valid:
                    valid_agents.append(aid)
                    assigned_agents.add(aid)
                    report.valid_assignments[aid] = sid
                    report.records.append(
                        AssignmentRecord(
                            task_id=sid,
                            agent_id=aid,
                            source=source,
                            status=AssignmentStatus.VALID,
                            step=step,
                        )
                    )
                else:
                    report.is_valid = False
                    report.rejected_assignments[aid] = sid
                    report.rejection_reasons[aid] = reason
                    report.records.append(
                        AssignmentRecord(
                            task_id=sid,
                            agent_id=aid,
                            source=source,
                            status=AssignmentStatus.INVALID,
                            reason=reason,
                            step=step,
                        )
                    )
                    if log_diagnostics:
                        print(
                            f"\n[INVALID_ASSIGNMENT]\n"
                            f"    agent={aid}\n"
                            f"    task={sid}\n"
                            f"    reason={reason}"
                        )
                        log_assignment_transition(
                            task_id=sid,
                            agent_id=aid,
                            source=source,
                            previous_status="CANDIDATE",
                            new_status="INVALID",
                            mode=mode,
                            step=step,
                            reason=reason,
                        )
                        log_assignment_removal(
                            task_id=sid,
                            agent_id=aid,
                            mode=mode,
                            step=step,
                            reason=reason,
                        )
            if valid_agents:
                cleaned[sid] = valid_agents
            else:
                cleaned[sid] = []
                report.unresolved_tasks.append(sid)

        return cleaned, report


def validate_global_assignment_state(
    assignments: dict[str, list[str]] | dict[str, str],
    fleet: AgentFleet,
    subtasks: Sequence[Subtask],
    source: str = "runtime",
    mode: int | str = 0,
    step: int = 0,
    coalitions: list[dict[str, Any]] | None = None,
    strict_skills: bool = False,
) -> list[str]:
    """Diagnostic invariant validator verifying Invariants 1-12 at runtime boundaries (Phase 11).

    Returns a list of violation messages. Does NOT silently modify state.
    """
    violations: list[str] = []
    fleet_agents = {a.agent_id for a in fleet.agents}
    subtask_map = {s.subtask_id: s for s in subtasks}
    completed_sids = {s.subtask_id for s in subtasks if s.completed}
    assigned_agents: set[str] = set()

    # Convert to normalized list of (task_id, agent_id)
    pairs: list[tuple[str, str]] = []
    if isinstance(assignments, dict):
        for k, v in assignments.items():
            if isinstance(v, list):
                for aid in v:
                    pairs.append((k, aid))
            elif isinstance(v, str):
                if k in fleet_agents:
                    pairs.append((v, k))
                else:
                    pairs.append((k, v))

    for sid, aid in pairs:
        # Invariant 1: No nonexistent fleet agent
        if aid not in fleet_agents:
            msg = (
                f"[STATE-INVARIANT-VIOLATION] invariant=INVARIANT_1_NONEXISTENT_AGENT "
                f"task={sid} agent={aid} source={source} mode={mode} step={step}"
            )
            print(msg)
            violations.append(msg)

        # Invariant 2: No nonexistent task
        if sid not in subtask_map:
            msg = (
                f"[STATE-INVARIANT-VIOLATION] invariant=INVARIANT_2_NONEXISTENT_TASK "
                f"task={sid} agent={aid} source={source} mode={mode} step={step}"
            )
            print(msg)
            violations.append(msg)

        # Invariant 3: Completed task can never appear in active assignments
        if sid in completed_sids:
            msg = (
                f"[STATE-INVARIANT-VIOLATION] invariant=INVARIANT_3_COMPLETED_TASK_ACTIVE "
                f"task={sid} agent={aid} source={source} mode={mode} step={step}"
            )
            print(msg)
            violations.append(msg)

        # Invariant 4 / 6: Required skills satisfied
        if sid in subtask_map and aid in fleet_agents:
            st = subtask_map[sid]
            if st.required_skills:
                agent = fleet.get_agent(aid)
                coalition_skills = set(agent.skills)
                if coalitions:
                    for c in coalitions:
                        members = c.get("members", [])
                        if aid in members:
                            for m in members:
                                if fleet.has_agent(m):
                                    coalition_skills.update(fleet.get_agent(m).skills)
                            break
                req = set(st.required_skills)
                if strict_skills:
                    if not req.issubset(coalition_skills):
                        msg = (
                            f"[STATE-INVARIANT-VIOLATION] invariant=INVARIANT_6_STRICT_SKILL_VIOLATION "
                            f"task={sid} agent={aid} required={sorted(list(req))} available={sorted(list(coalition_skills))} "
                            f"source={source} mode={mode} step={step}"
                        )
                        print(msg)
                        violations.append(msg)
                else:
                    # Issue 1 fix: require full skill coverage even in non-strict mode
                    if not req.issubset(coalition_skills):
                        msg = (
                            f"[STATE-INVARIANT-VIOLATION] invariant=INVARIANT_6_MISSING_REQUIRED_SKILLS "
                            f"task={sid} agent={aid} required={sorted(list(req))} available={sorted(list(coalition_skills))} "
                            f"source={source} mode={mode} step={step}"
                        )
                        print(msg)
                        violations.append(msg)

        # Invariant 12: No multiply assigned agents in executable state
        if aid in assigned_agents:
            msg = (
                f"[STATE-INVARIANT-VIOLATION] invariant=INVARIANT_12_AGENT_MULTIPLY_ASSIGNED "
                f"task={sid} agent={aid} source={source} mode={mode} step={step}"
            )
            print(msg)
            violations.append(msg)
        assigned_agents.add(aid)

    return violations
