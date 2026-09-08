"""Deterministic Assignment Validator for Multi-Agent Task Allocation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from src.env.agents import AgentFleet
from src.env.scenarios import Subtask


@dataclass
class ValidationReport:
    is_valid: bool = True
    valid_assignments: dict[str, str] = field(default_factory=dict)
    rejected_assignments: dict[str, str] = field(default_factory=dict)
    rejection_reasons: dict[str, str] = field(default_factory=dict)


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
        subtasks: list[Subtask],
        assigned_agents: set[str] | None = None,
        check_skills: bool = True,
        strict_skills: bool = False,
        coalitions: list[dict[str, Any]] | None = None,
    ) -> tuple[bool, str]:
        if not isinstance(agent_id, str) or not agent_id.strip():
            return False, f"invalid_agent_id_format:{agent_id}"

        from src.coordination.autohma_structs import RESERVED_SCHEMA_KEYS

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
                # Semantic impossibility: agent/coalition has zero matching skills
                if not (required & coalition_skills):
                    missing = sorted(list(required))
                    return False, f"no_matching_skills:required={missing},available={sorted(list(coalition_skills))}"

        return True, ""

    @classmethod
    def filter_assignments(
        cls,
        candidate_assignments: dict[str, str],  # agent_id -> subtask_id
        fleet: AgentFleet,
        subtasks: list[Subtask],
        check_skills: bool = True,
        strict_skills: bool = False,
        coalitions: list[dict[str, Any]] | None = None,
        log_diagnostics: bool = True,
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
            else:
                report.is_valid = False
                report.rejected_assignments[aid] = sid
                report.rejection_reasons[aid] = reason
                if log_diagnostics:
                    print(
                        f"\n[INVALID_ASSIGNMENT]\n"
                        f"    agent={aid}\n"
                        f"    task={sid}\n"
                        f"    reason={reason}"
                    )

        return report
