"""Dataclasses and parser for the contract_parts contract format."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath


# ---------------------------------------------------------------------------
# Contract text format (what agents produce as their DRAFT in architecture phase)
# ---------------------------------------------------------------------------
#
# GOAL
# <one paragraph>
#
# MODULES
# - relative/path/file.py: <one-line description>
#
# INTERFACES
# - function_name(param: Type) -> ReturnType: <description>
#
# RULES
# - <architectural constraint>
#
# OWNERSHIP
# - agent_id: file.py, other/file.py
#
# All sections are required except INTERFACES (may be empty for simple tasks).
# OWNERSHIP must cover every module in MODULES exactly once.
# ---------------------------------------------------------------------------


class ContractValidationError(ValueError):
    pass


@dataclass
class CouncilContract:
    goal: str
    modules: list[tuple[str, str]]          # (relative_path, description)
    interfaces: list[tuple[str, str]]       # (signature, description)
    rules: list[str]
    ownership: dict[str, list[str]]         # agent_id -> [relative_path, ...]
    raw: str = field(repr=False, default="")

    @property
    def module_paths(self) -> list[str]:
        return [path for path, _ in self.modules]

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "modules": [{"path": path, "description": desc} for path, desc in self.modules],
            "interfaces": [{"signature": sig, "description": desc} for sig, desc in self.interfaces],
            "rules": list(self.rules),
            "ownership": {agent_id: list(files) for agent_id, files in self.ownership.items()},
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=False)


@dataclass
class PartAssignment:
    agent_id: str
    owned_files: list[str]


@dataclass
class PartDraft:
    agent_id: str
    owned_files: list[str]
    implementation: str = ""    # raw text from agent (### FILE: blocks)
    failed: bool = False
    implemented_by: str = ""


@dataclass
class PartReview:
    reviewer_id: str
    target_agent_id: str
    target_files: list[str]
    feedback: str = ""
    failed: bool = False


@dataclass
class MergeResult:
    files: dict[str, str]           # relative_path -> content
    merged_text: str                # complete ### FILE: block output
    unresolved: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _extract_section(text: str, header: str) -> str:
    """Return the body of a named section, stripping the header line.
    Section ends at the next all-caps header or end of string.
    Uses [ \\t]* (not \\s*) after the header so blank lines between sections
    are captured in the body and stripped, not consumed by the header match."""
    pattern = re.compile(
        rf"^{re.escape(header)}[ \t]*\n(.*?)(?=\n[A-Z][A-Z ]+[ \t]*\n|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    m = pattern.search(text)
    return m.group(1).strip() if m else ""


def _parse_bullet_pairs(text: str) -> list[tuple[str, str]]:
    """Parse '- key: value' lines into (key, value) pairs.
    Splits on the first ':' — suitable for MODULES and OWNERSHIP where keys
    are file paths or agent IDs that never contain colons."""
    result = []
    for line in text.splitlines():
        line = line.strip().lstrip("- ").strip()
        if not line:
            continue
        idx = line.find(":")
        if idx == -1:
            result.append((line, ""))
        else:
            result.append((line[:idx].strip(), line[idx + 1:].strip()))
    return result


def _parse_interfaces(text: str) -> list[tuple[str, str]]:
    """Parse interface lines like '- sig(p: Type) -> ReturnType: description'.
    Splits on the first ': ' that appears after '->' to avoid splitting on
    colons inside parameter type annotations."""
    result = []
    for line in text.splitlines():
        line = line.strip().lstrip("- ").strip()
        if not line:
            continue
        arrow = line.find("->")
        if arrow != -1:
            # Split on first ': ' after the '->'
            sep = line.find(": ", arrow)
            if sep != -1:
                result.append((line[:sep].strip(), line[sep + 2:].strip()))
                continue
        # Fallback: split on first ':'
        idx = line.find(":")
        if idx == -1:
            result.append((line, ""))
        else:
            result.append((line[:idx].strip(), line[idx + 1:].strip()))
    return result


def _parse_ownership(text: str) -> dict[str, list[str]]:
    """Parse '- agent_id: file.py, other.py' lines."""
    ownership: dict[str, list[str]] = {}
    for agent_id, files_raw in _parse_bullet_pairs(text):
        files = [f.strip() for f in files_raw.split(",") if f.strip()]
        if agent_id and files:
            ownership[agent_id] = files
    return ownership


def _is_safe_relative_path(path: str) -> bool:
    try:
        pure = PurePosixPath(path)
    except Exception:
        return False
    if not path or pure.is_absolute():
        return False
    # pure.parts is () for "." in Python 3.12+ (vacuously passes all()), so guard explicitly
    if not pure.parts:
        return False
    return all(part not in ("", ".", "..") for part in pure.parts)


def parse_contract(text: str, active_agent_ids: list[str] | None = None) -> CouncilContract:
    """
    Parse a contract from agent output text.
    Raises ContractValidationError if the contract is structurally invalid.
    """
    goal = _extract_section(text, "GOAL")
    modules_raw = _extract_section(text, "MODULES")
    interfaces_raw = _extract_section(text, "INTERFACES")
    rules_raw = _extract_section(text, "RULES")
    ownership_raw = _extract_section(text, "OWNERSHIP")

    modules = _parse_bullet_pairs(modules_raw)
    interfaces = _parse_interfaces(interfaces_raw)
    rules = [
        line.strip().lstrip("- ").strip()
        for line in rules_raw.splitlines()
        if line.strip().lstrip("- ").strip()
    ]
    ownership = _parse_ownership(ownership_raw)

    contract = CouncilContract(
        goal=goal,
        modules=modules,
        interfaces=interfaces,
        rules=rules,
        ownership=ownership,
        raw=text,
    )

    _validate(contract, active_agent_ids or [])
    return contract


def _validate(contract: CouncilContract, active_agent_ids: list[str]) -> None:
    if not contract.goal:
        raise ContractValidationError("Contract GOAL is empty.")

    if not contract.modules:
        raise ContractValidationError("Contract MODULES is empty — at least one module required.")

    if not contract.rules:
        raise ContractValidationError("Contract RULES is empty — at least one architectural constraint required.")

    if not contract.ownership:
        raise ContractValidationError("Contract OWNERSHIP is empty.")

    module_paths = set()
    for path, _ in contract.modules:
        if not _is_safe_relative_path(path):
            raise ContractValidationError(f"MODULES contains invalid relative path: {path!r}")
        if path in module_paths:
            raise ContractValidationError(f"MODULES contains duplicate file path: {path!r}")
        module_paths.add(path)

    # Every owned file must be a declared module
    owned_paths: set[str] = set()
    for agent_id, files in contract.ownership.items():
        for f in files:
            if not _is_safe_relative_path(f):
                raise ContractValidationError(f"OWNERSHIP contains invalid relative path: {f!r}")
            if f in owned_paths:
                raise ContractValidationError(
                    f"File '{f}' appears in OWNERSHIP more than once."
                )
            owned_paths.add(f)

    missing_from_modules = owned_paths - module_paths
    if missing_from_modules:
        raise ContractValidationError(
            f"OWNERSHIP references files not in MODULES: {sorted(missing_from_modules)}"
        )

    unowned = module_paths - owned_paths
    if unowned:
        raise ContractValidationError(
            f"MODULES files have no owner in OWNERSHIP: {sorted(unowned)}"
        )

    # Agent IDs must be from the active list (only checked when list is provided)
    if active_agent_ids:
        active_set = set(active_agent_ids)
        unknown = set(contract.ownership) - active_set
        if unknown:
            raise ContractValidationError(
                f"OWNERSHIP references unknown agents: {sorted(unknown)}. "
                f"Active agents: {sorted(active_set)}"
            )


# ---------------------------------------------------------------------------
# Assignment helper
# ---------------------------------------------------------------------------

def build_assignments(contract: CouncilContract) -> list[PartAssignment]:
    """Convert contract.ownership into ordered PartAssignment list."""
    return [
        PartAssignment(agent_id=aid, owned_files=files)
        for aid, files in contract.ownership.items()
    ]


def cross_review_pairs(assignments: list[PartAssignment]) -> list[tuple[str, str]]:
    """
    Return (reviewer_agent_id, target_agent_id) pairs.
    Agent at index i reviews the parts owned by agent at index (i+1) % n.
    Returns an empty list when there is only one agent (self-review is skipped).
    """
    n = len(assignments)
    if n < 2:
        return []
    return [
        (assignments[i].agent_id, assignments[(i + 1) % n].agent_id)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Contract formatting helpers (for prompts)
# ---------------------------------------------------------------------------

def contract_text(contract: CouncilContract) -> str:
    """Render a CouncilContract back to canonical contract format."""
    lines = ["GOAL", contract.goal, ""]

    lines.append("MODULES")
    for path, desc in contract.modules:
        lines.append(f"- {path}: {desc}")
    lines.append("")

    if contract.interfaces:
        lines.append("INTERFACES")
        for sig, desc in contract.interfaces:
            lines.append(f"- {sig}: {desc}")
        lines.append("")

    lines.append("RULES")
    for rule in contract.rules:
        lines.append(f"- {rule}")
    lines.append("")

    lines.append("OWNERSHIP")
    for agent_id, files in contract.ownership.items():
        lines.append(f"- {agent_id}: {', '.join(files)}")

    return "\n".join(lines)
