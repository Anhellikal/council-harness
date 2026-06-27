"""Tests for council/contracts.py — parser, validator, assignments, review pairs."""
import json
import pytest
from council.contracts import (
    CouncilContract,
    ContractValidationError,
    PartAssignment,
    build_assignments,
    contract_text,
    cross_review_pairs,
    parse_contract,
    _is_safe_relative_path,
)

AGENTS = ["llama", "mistral"]

VALID = """\
GOAL
Implement JWT authentication for a FastAPI service.

MODULES
- auth/token.py: token creation and validation
- auth/routes.py: login and refresh endpoints

INTERFACES
- validate_token(token: str) -> Claims: parse and verify a JWT
- login(username: str, password: str) -> AuthResult: authenticate and return tokens

RULES
- routes must not import from db directly
- token validation lives only in auth/token.py

OWNERSHIP
- llama: auth/token.py
- mistral: auth/routes.py
"""


# ---------------------------------------------------------------------------
# parse_contract — happy path
# ---------------------------------------------------------------------------

class TestParseContractValid:
    def test_goal_parsed(self):
        c = parse_contract(VALID, AGENTS)
        assert c.goal == "Implement JWT authentication for a FastAPI service."

    def test_modules_parsed(self):
        c = parse_contract(VALID, AGENTS)
        assert c.modules == [
            ("auth/token.py", "token creation and validation"),
            ("auth/routes.py", "login and refresh endpoints"),
        ]

    def test_interfaces_parsed(self):
        c = parse_contract(VALID, AGENTS)
        assert len(c.interfaces) == 2
        assert c.interfaces[0][0] == "validate_token(token: str) -> Claims"

    def test_rules_parsed(self):
        c = parse_contract(VALID, AGENTS)
        assert c.rules == [
            "routes must not import from db directly",
            "token validation lives only in auth/token.py",
        ]

    def test_ownership_parsed(self):
        c = parse_contract(VALID, AGENTS)
        assert c.ownership == {
            "llama": ["auth/token.py"],
            "mistral": ["auth/routes.py"],
        }

    def test_raw_preserved(self):
        c = parse_contract(VALID, AGENTS)
        assert c.raw == VALID

    def test_preamble_tolerated(self):
        text = "Here is my architecture proposal:\n\n" + VALID
        c = parse_contract(text, AGENTS)
        assert c.goal == "Implement JWT authentication for a FastAPI service."

    def test_empty_interfaces_ok(self):
        text = VALID.replace(
            "INTERFACES\n"
            "- validate_token(token: str) -> Claims: parse and verify a JWT\n"
            "- login(username: str, password: str) -> AuthResult: authenticate and return tokens\n\n",
            "",
        )
        c = parse_contract(text, AGENTS)
        assert c.interfaces == []

    def test_no_active_list_skips_agent_check(self):
        c = parse_contract(VALID)
        assert set(c.ownership.keys()) == {"llama", "mistral"}

    def test_multi_file_ownership(self):
        text = (
            "GOAL\nA task.\n\n"
            "MODULES\n- a.py: mod a\n- b.py: mod b\n- c.py: mod c\n\n"
            "RULES\n- one rule\n\n"
            "OWNERSHIP\n- llama: a.py, b.py\n- mistral: c.py\n"
        )
        c = parse_contract(text, AGENTS)
        assert c.ownership["llama"] == ["a.py", "b.py"]
        assert c.ownership["mistral"] == ["c.py"]


# ---------------------------------------------------------------------------
# parse_contract — validation errors
# ---------------------------------------------------------------------------

class TestParseContractInvalid:
    def test_empty_goal_raises(self):
        text = VALID.replace(
            "GOAL\nImplement JWT authentication for a FastAPI service.\n",
            "GOAL\n\n",
        )
        with pytest.raises(ContractValidationError, match="GOAL"):
            parse_contract(text, AGENTS)

    def test_empty_modules_raises(self):
        text = VALID.replace(
            "MODULES\n- auth/token.py: token creation and validation\n- auth/routes.py: login and refresh endpoints\n",
            "MODULES\n\n",
        )
        with pytest.raises(ContractValidationError, match="MODULES"):
            parse_contract(text, AGENTS)

    def test_empty_rules_raises(self):
        text = VALID.replace(
            "RULES\n- routes must not import from db directly\n- token validation lives only in auth/token.py\n",
            "RULES\n\n",
        )
        with pytest.raises(ContractValidationError, match="RULES"):
            parse_contract(text, AGENTS)

    def test_empty_ownership_raises(self):
        text = VALID.replace(
            "OWNERSHIP\n- llama: auth/token.py\n- mistral: auth/routes.py\n",
            "OWNERSHIP\n\n",
        )
        with pytest.raises(ContractValidationError, match="OWNERSHIP"):
            parse_contract(text, AGENTS)

    def test_file_owned_twice_raises(self):
        text = VALID.replace(
            "- mistral: auth/routes.py",
            "- mistral: auth/token.py",
        )
        with pytest.raises(ContractValidationError, match="more than once"):
            parse_contract(text, AGENTS)

    def test_ownership_file_not_in_modules_raises(self):
        text = VALID.replace(
            "- mistral: auth/routes.py",
            "- mistral: auth/ghost.py",
        )
        with pytest.raises(ContractValidationError, match="not in MODULES"):
            parse_contract(text, AGENTS)

    def test_module_without_owner_raises(self):
        text = VALID.replace(
            "OWNERSHIP\n- llama: auth/token.py\n- mistral: auth/routes.py",
            "OWNERSHIP\n- llama: auth/token.py",
        )
        with pytest.raises(ContractValidationError, match="no owner"):
            parse_contract(text, AGENTS)

    def test_unknown_agent_raises(self):
        with pytest.raises(ContractValidationError, match="unknown agents"):
            parse_contract(VALID, ["llama", "coder"])  # mistral not in active list

    def test_error_message_names_bad_agent(self):
        with pytest.raises(ContractValidationError, match="mistral"):
            parse_contract(VALID, ["llama", "coder"])


# ---------------------------------------------------------------------------
# _is_safe_relative_path
# ---------------------------------------------------------------------------

class TestIsSafeRelativePath:
    def test_simple_filename(self):
        assert _is_safe_relative_path("file.py") is True

    def test_nested_path(self):
        assert _is_safe_relative_path("auth/token.py") is True

    def test_deeply_nested(self):
        assert _is_safe_relative_path("a/b/c/d.py") is True

    def test_absolute_path_rejected(self):
        assert _is_safe_relative_path("/etc/passwd") is False

    def test_parent_traversal_rejected(self):
        assert _is_safe_relative_path("../secret.py") is False

    def test_nested_traversal_rejected(self):
        assert _is_safe_relative_path("a/../../escape.py") is False

    def test_dot_prefix_normalized_and_accepted(self):
        # PurePosixPath normalizes "./" away — "file.py" parts remain, path is safe
        # Consistent with _safe_path in cli.py accepting "./a.py"
        assert _is_safe_relative_path("./file.py") is True

    def test_empty_string_rejected(self):
        assert _is_safe_relative_path("") is False

    def test_double_dot_alone_rejected(self):
        assert _is_safe_relative_path("..") is False

    def test_dot_alone_rejected(self):
        # PurePosixPath(".").parts is () in Python 3.12+ — guard catches it explicitly
        assert _is_safe_relative_path(".") is False


# ---------------------------------------------------------------------------
# Path validation in _validate (via parse_contract)
# ---------------------------------------------------------------------------

class TestPathValidation:
    def _contract_with_module_path(self, path: str) -> str:
        return (
            "GOAL\nTask.\n\n"
            f"MODULES\n- {path}: description\n\n"
            "RULES\n- rule\n\n"
            f"OWNERSHIP\n- llama: {path}\n"
        )

    def test_absolute_path_in_modules_raises(self):
        text = self._contract_with_module_path("/etc/passwd")
        with pytest.raises(ContractValidationError, match="invalid relative path"):
            parse_contract(text, ["llama"])

    def test_parent_traversal_in_modules_raises(self):
        text = self._contract_with_module_path("../escape.py")
        with pytest.raises(ContractValidationError, match="invalid relative path"):
            parse_contract(text, ["llama"])

    def test_absolute_path_in_ownership_raises(self):
        text = (
            "GOAL\nTask.\n\n"
            "MODULES\n- a.py: mod\n\n"
            "RULES\n- rule\n\n"
            "OWNERSHIP\n- llama: /etc/a.py\n"
        )
        with pytest.raises(ContractValidationError):
            parse_contract(text, ["llama"])

    def test_duplicate_module_path_raises(self):
        text = (
            "GOAL\nTask.\n\n"
            "MODULES\n- a.py: first\n- a.py: second\n\n"
            "RULES\n- rule\n\n"
            "OWNERSHIP\n- llama: a.py\n"
        )
        with pytest.raises(ContractValidationError, match="duplicate"):
            parse_contract(text, ["llama"])

    def test_valid_nested_path_accepted(self):
        text = self._contract_with_module_path("auth/routes/login.py")
        c = parse_contract(text, ["llama"])
        assert c.modules[0][0] == "auth/routes/login.py"


# ---------------------------------------------------------------------------
# CouncilContract.module_paths property
# ---------------------------------------------------------------------------

class TestModulePaths:
    def test_returns_paths_only(self):
        c = parse_contract(VALID, AGENTS)
        assert c.module_paths == ["auth/token.py", "auth/routes.py"]

    def test_order_preserved(self):
        text = (
            "GOAL\nTask.\n\n"
            "MODULES\n- z.py: last\n- a.py: first\n- m.py: middle\n\n"
            "RULES\n- rule\n\n"
            "OWNERSHIP\n- llama: z.py, a.py, m.py\n"
        )
        c = parse_contract(text, ["llama"])
        assert c.module_paths == ["z.py", "a.py", "m.py"]


# ---------------------------------------------------------------------------
# CouncilContract.to_dict / to_json
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_dict_keys(self):
        c = parse_contract(VALID, AGENTS)
        d = c.to_dict()
        assert set(d.keys()) == {"goal", "modules", "interfaces", "rules", "ownership"}

    def test_to_dict_goal(self):
        c = parse_contract(VALID, AGENTS)
        assert c.to_dict()["goal"] == "Implement JWT authentication for a FastAPI service."

    def test_to_dict_modules_shape(self):
        c = parse_contract(VALID, AGENTS)
        modules = c.to_dict()["modules"]
        assert modules[0] == {"path": "auth/token.py", "description": "token creation and validation"}

    def test_to_dict_interfaces_shape(self):
        c = parse_contract(VALID, AGENTS)
        ifaces = c.to_dict()["interfaces"]
        assert ifaces[0]["signature"] == "validate_token(token: str) -> Claims"
        assert ifaces[0]["description"] == "parse and verify a JWT"

    def test_to_dict_rules_is_list(self):
        c = parse_contract(VALID, AGENTS)
        rules = c.to_dict()["rules"]
        assert isinstance(rules, list)
        assert "routes must not import from db directly" in rules

    def test_to_dict_ownership_shape(self):
        c = parse_contract(VALID, AGENTS)
        ownership = c.to_dict()["ownership"]
        assert ownership == {"llama": ["auth/token.py"], "mistral": ["auth/routes.py"]}

    def test_to_dict_is_independent_copy(self):
        c = parse_contract(VALID, AGENTS)
        d = c.to_dict()
        d["goal"] = "mutated"
        d["rules"].append("extra")
        # original contract unchanged
        assert c.goal == "Implement JWT authentication for a FastAPI service."
        assert len(c.rules) == 2

    def test_to_json_is_valid_json(self):
        c = parse_contract(VALID, AGENTS)
        parsed = json.loads(c.to_json())
        assert parsed["goal"] == c.goal

    def test_to_json_roundtrips_ownership(self):
        c = parse_contract(VALID, AGENTS)
        parsed = json.loads(c.to_json())
        assert parsed["ownership"] == {"llama": ["auth/token.py"], "mistral": ["auth/routes.py"]}

    def test_to_json_pretty_printed(self):
        c = parse_contract(VALID, AGENTS)
        raw = c.to_json()
        assert "\n" in raw  # indent=2 means multiline

    def test_to_dict_empty_interfaces(self):
        text = (
            "GOAL\nTask.\n\n"
            "MODULES\n- a.py: mod\n\n"
            "RULES\n- rule\n\n"
            "OWNERSHIP\n- llama: a.py\n"
        )
        c = parse_contract(text, ["llama"])
        assert c.to_dict()["interfaces"] == []


# ---------------------------------------------------------------------------
# build_assignments
# ---------------------------------------------------------------------------

class TestBuildAssignments:
    def test_basic_two_agents(self):
        c = parse_contract(VALID, AGENTS)
        assignments = build_assignments(c)
        assert len(assignments) == 2
        ids = [a.agent_id for a in assignments]
        assert "llama" in ids
        assert "mistral" in ids

    def test_owned_files_match(self):
        c = parse_contract(VALID, AGENTS)
        by_id = {a.agent_id: a for a in build_assignments(c)}
        assert by_id["llama"].owned_files == ["auth/token.py"]
        assert by_id["mistral"].owned_files == ["auth/routes.py"]

    def test_multi_file_assignment(self):
        text = (
            "GOAL\nTask.\n\n"
            "MODULES\n- a.py: a\n- b.py: b\n- c.py: c\n\n"
            "RULES\n- rule\n\n"
            "OWNERSHIP\n- llama: a.py, b.py\n- mistral: c.py\n"
        )
        c = parse_contract(text, AGENTS)
        by_id = {a.agent_id: a for a in build_assignments(c)}
        assert by_id["llama"].owned_files == ["a.py", "b.py"]
        assert by_id["mistral"].owned_files == ["c.py"]


# ---------------------------------------------------------------------------
# cross_review_pairs
# ---------------------------------------------------------------------------

class TestCrossReviewPairs:
    def test_single_agent_returns_empty(self):
        assert cross_review_pairs([PartAssignment("solo", ["a.py"])]) == []

    def test_two_agents_mutual(self):
        pairs = cross_review_pairs([
            PartAssignment("llama", ["a.py"]),
            PartAssignment("mistral", ["b.py"]),
        ])
        assert set(pairs) == {("llama", "mistral"), ("mistral", "llama")}

    def test_three_agents_shift_by_one(self):
        pairs = cross_review_pairs([
            PartAssignment("a", ["x.py"]),
            PartAssignment("b", ["y.py"]),
            PartAssignment("c", ["z.py"]),
        ])
        assert pairs == [("a", "b"), ("b", "c"), ("c", "a")]

    def test_four_agents_wraps(self):
        pairs = cross_review_pairs([PartAssignment(x, [f"{x}.py"]) for x in "abcd"])
        assert pairs == [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")]

    def test_no_self_review(self):
        assignments = [PartAssignment(x, [f"{x}.py"]) for x in "abc"]
        for reviewer, target in cross_review_pairs(assignments):
            assert reviewer != target


# ---------------------------------------------------------------------------
# contract_text round-trip
# ---------------------------------------------------------------------------

class TestContractText:
    def test_roundtrip_preserves_structure(self):
        c = parse_contract(VALID, AGENTS)
        rendered = contract_text(c)
        c2 = parse_contract(rendered, AGENTS)
        assert c2.goal == c.goal
        assert c2.modules == c.modules
        assert c2.interfaces == c.interfaces
        assert c2.rules == c.rules
        assert c2.ownership == c.ownership

    def test_always_emits_rules_section(self):
        c = parse_contract(VALID, AGENTS)
        rendered = contract_text(c)
        assert "RULES\n" in rendered

    def test_always_emits_ownership_section(self):
        c = parse_contract(VALID, AGENTS)
        rendered = contract_text(c)
        assert "OWNERSHIP\n" in rendered

    def test_omits_interfaces_when_empty(self):
        text = (
            "GOAL\nTask.\n\n"
            "MODULES\n- a.py: mod\n\n"
            "RULES\n- rule\n\n"
            "OWNERSHIP\n- llama: a.py\n"
        )
        c = parse_contract(text, ["llama"])
        rendered = contract_text(c)
        assert "INTERFACES" not in rendered

    def test_agent_files_joined_with_comma(self):
        text = (
            "GOAL\nTask.\n\n"
            "MODULES\n- a.py: a\n- b.py: b\n- c.py: c\n\n"
            "RULES\n- rule\n\n"
            "OWNERSHIP\n- llama: a.py, b.py\n- mistral: c.py\n"
        )
        c = parse_contract(text, AGENTS)
        rendered = contract_text(c)
        assert "llama: a.py, b.py" in rendered
