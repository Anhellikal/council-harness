"""Tests for deterministic helpers in council/loop.py.

No agent calls — only pure functions that operate on local state.
"""
import pytest
from council.loop import (
    _adoption_draft,
    _convergence_winner,
    _distinct_count,
    _effective_threshold,
    _merge_multifile,
    _parse_response,
)

ACTIVE = {"llama", "mistral", "coder"}


# ---------------------------------------------------------------------------
# _effective_threshold
# ---------------------------------------------------------------------------

class TestEffectiveThreshold:
    def test_no_scaling_needed(self):
        assert _effective_threshold(2, 3, 3) == 2

    def test_scales_down_proportionally(self):
        # 2-of-3 → 2-of-2 (ceil(2 * 2/3) = ceil(1.33) = 2)
        assert _effective_threshold(2, 3, 2) == 2

    def test_scales_down_to_one(self):
        # 2-of-3 → 1-of-1
        assert _effective_threshold(2, 3, 1) == 1

    def test_threshold_clamped_to_active_count(self):
        # configured threshold larger than active → clamp to active
        assert _effective_threshold(5, 5, 2) == 2

    def test_minimum_is_one(self):
        assert _effective_threshold(1, 10, 1) == 1

    def test_zero_config_total_returns_active_count(self):
        assert _effective_threshold(2, 0, 3) == 3

    def test_larger_council(self):
        # 3-of-5 configured, 4 active → ceil(4 * 3/5) = ceil(2.4) = 3
        assert _effective_threshold(3, 5, 4) == 3


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------

class TestParseResponse:
    def test_revise_simple(self):
        text = "ACTION: revise\nDRAFT: some implementation"
        r = _parse_response(text, ACTIVE, "llama")
        assert r == {"action": "revise", "draft": "some implementation"}

    def test_revise_multiline_draft(self):
        text = "ACTION: revise\nDRAFT: def foo():\n    return 42"
        r = _parse_response(text, ACTIVE, "llama")
        assert r is not None
        assert r["draft"] == "def foo():\n    return 42"

    def test_adopt_valid(self):
        text = "ACTION: adopt\nTARGET: mistral"
        r = _parse_response(text, ACTIVE, "llama")
        assert r == {"action": "adopt", "target": "mistral"}

    def test_preamble_before_action_tolerated(self):
        text = "After careful review...\nACTION: adopt\nTARGET: mistral"
        r = _parse_response(text, ACTIVE, "llama")
        assert r is not None
        assert r["action"] == "adopt"

    def test_case_insensitive_action(self):
        text = "ACTION: REVISE\nDRAFT: code"
        r = _parse_response(text, ACTIVE, "llama")
        assert r is not None
        assert r["action"] == "revise"

    def test_target_strips_trailing_punctuation(self):
        text = "ACTION: adopt\nTARGET: mistral."
        r = _parse_response(text, ACTIVE, "llama")
        assert r == {"action": "adopt", "target": "mistral"}

    def test_target_strips_brackets(self):
        text = "ACTION: adopt\nTARGET: [mistral]"
        r = _parse_response(text, ACTIVE, "llama")
        assert r == {"action": "adopt", "target": "mistral"}

    def test_self_adopt_returns_none(self):
        text = "ACTION: adopt\nTARGET: llama"
        assert _parse_response(text, ACTIVE, "llama") is None

    def test_unknown_target_returns_none(self):
        text = "ACTION: adopt\nTARGET: gpt4"
        assert _parse_response(text, ACTIVE, "llama") is None

    def test_missing_draft_returns_none(self):
        assert _parse_response("ACTION: revise\n", ACTIVE, "llama") is None

    def test_whitespace_only_draft_returns_none(self):
        assert _parse_response("ACTION: revise\nDRAFT:   ", ACTIVE, "llama") is None

    def test_missing_target_returns_none(self):
        assert _parse_response("ACTION: adopt\n", ACTIVE, "llama") is None

    def test_unknown_action_returns_none(self):
        assert _parse_response("ACTION: skip\nDRAFT: x", ACTIVE, "llama") is None

    def test_no_action_line_returns_none(self):
        assert _parse_response("just some text", ACTIVE, "llama") is None


# ---------------------------------------------------------------------------
# _convergence_winner / _distinct_count
# ---------------------------------------------------------------------------

class TestConvergence:
    def test_winner_at_threshold(self):
        drafts = {"a": "x", "b": "x"}
        assert _convergence_winner(drafts, 2) == "x"

    def test_winner_above_threshold(self):
        drafts = {"a": "x", "b": "x", "c": "x"}
        assert _convergence_winner(drafts, 2) == "x"

    def test_no_winner_below_threshold(self):
        drafts = {"a": "x", "b": "y", "c": "z"}
        assert _convergence_winner(drafts, 2) is None

    def test_split_vote_no_winner(self):
        drafts = {"a": "x", "b": "x", "c": "y", "d": "y"}
        assert _convergence_winner(drafts, 3) is None

    def test_unanimous_wins(self):
        drafts = {"a": "x", "b": "x", "c": "x"}
        assert _convergence_winner(drafts, 3) == "x"

    def test_distinct_count_all_same(self):
        assert _distinct_count({"a": "x", "b": "x", "c": "x"}) == 1

    def test_distinct_count_all_different(self):
        assert _distinct_count({"a": "x", "b": "y", "c": "z"}) == 3

    def test_distinct_count_mixed(self):
        assert _distinct_count({"a": "x", "b": "y", "c": "x"}) == 2

    def test_single_agent_wins_at_threshold_one(self):
        assert _convergence_winner({"solo": "draft"}, 1) == "draft"


# ---------------------------------------------------------------------------
# _adoption_draft — cycle detection
# ---------------------------------------------------------------------------

class TestAdoptionDraft:
    def test_simple_adopt(self):
        snapshot = {"a": "draft_a", "b": "draft_b"}
        assert _adoption_draft("a", "b", {}, snapshot) == "draft_b"

    def test_adoption_is_not_transitive(self):
        # All round updates apply from the same snapshot.
        # a adopts b, b adopts c in the same round → a gets b's SNAPSHOT draft,
        # not b's post-adoption draft. Transitive chains are intentionally not followed.
        snapshot = {"a": "da", "b": "db", "c": "dc"}
        round_adopts = {"b": "c"}
        assert _adoption_draft("a", "b", round_adopts, snapshot) == "db"

    def test_two_agent_cycle_collapses(self):
        # a→b, b→a: cycle; both should land on min-id (a)'s draft
        snapshot = {"a": "draft_a", "b": "draft_b"}
        round_adopts = {"a": "b", "b": "a"}
        result_a = _adoption_draft("a", "b", round_adopts, snapshot)
        result_b = _adoption_draft("b", "a", round_adopts, snapshot)
        assert result_a == result_b
        assert result_a == "draft_a"  # min id is "a"

    def test_no_cycle_no_chain(self):
        snapshot = {"a": "da", "b": "db"}
        # b hasn't adopted anyone this round
        assert _adoption_draft("a", "b", {}, snapshot) == "db"


# ---------------------------------------------------------------------------
# _merge_multifile
# ---------------------------------------------------------------------------

class TestMergeMultifile:
    def test_new_file_replaces_old(self):
        old = "### FILE: a.py\nold content"
        new = "### FILE: a.py\nnew content"
        result = _merge_multifile(old, new)
        assert "new content" in result
        assert "old content" not in result

    def test_absent_file_kept_from_old(self):
        old = "### FILE: a.py\ncontent_a\n### FILE: b.py\ncontent_b"
        new = "### FILE: a.py\nupdated_a"
        result = _merge_multifile(old, new)
        assert "content_b" in result
        assert "updated_a" in result

    def test_new_file_added(self):
        old = "### FILE: a.py\ncontent_a"
        new = "### FILE: b.py\ncontent_b"
        result = _merge_multifile(old, new)
        assert "content_a" in result
        assert "content_b" in result

    def test_non_multifile_new_returned_as_is(self):
        old = "### FILE: a.py\nold"
        new = "plain text without file markers"
        assert _merge_multifile(old, new) == new

    def test_multiple_files_merged(self):
        old = "### FILE: a.py\na\n### FILE: b.py\nb\n### FILE: c.py\nc"
        new = "### FILE: b.py\nB_updated"
        result = _merge_multifile(old, new)
        assert "a" in result
        assert "B_updated" in result
        assert "c" in result
        assert "\nb\n" not in result
