"""Tests for deterministic helpers in council/cli.py.

Covers path safety, fence stripping, file block parsing, and mode selection logic.
No Click runner, no agent calls, no config files.
"""
import json
import pytest
from pathlib import Path
from council.cli import _parse_files, _safe_path, _strip_fences, _write_pipeline_artifacts
from council.contracts import CouncilContract, MergeResult, PartAssignment, PartDraft, PartReview
from council.loop import CouncilResult, RoundSummary
from council.pipeline import PipelineResult


# ---------------------------------------------------------------------------
# _safe_path
# ---------------------------------------------------------------------------

class TestSafePath:
    def test_valid_relative_path(self, tmp_path):
        result = _safe_path("src/main.py", tmp_path)
        assert result == tmp_path / "src" / "main.py"

    def test_simple_filename(self, tmp_path):
        result = _safe_path("output.py", tmp_path)
        assert result == tmp_path / "output.py"

    def test_nested_valid_path(self, tmp_path):
        result = _safe_path("a/b/c.py", tmp_path)
        assert result == tmp_path / "a" / "b" / "c.py"

    def test_absolute_path_rejected(self, tmp_path):
        assert _safe_path("/etc/passwd", tmp_path) is None

    def test_parent_traversal_rejected(self, tmp_path):
        assert _safe_path("../secret.py", tmp_path) is None

    def test_nested_traversal_rejected(self, tmp_path):
        assert _safe_path("a/../../outside.py", tmp_path) is None

    def test_dot_prefix_accepted(self, tmp_path):
        # ./a.py resolves safely inside output_dir — _safe_path accepts it
        result = _safe_path("./a.py", tmp_path)
        assert result is not None
        assert result == (tmp_path / "a.py").resolve()

    def test_result_is_inside_output_dir(self, tmp_path):
        result = _safe_path("deep/nested/file.py", tmp_path)
        assert result is not None
        assert str(result).startswith(str(tmp_path))


# ---------------------------------------------------------------------------
# _strip_fences
# ---------------------------------------------------------------------------

class TestStripFences:
    def test_strips_plain_fence(self):
        assert _strip_fences("```\ncode here\n```") == "code here"

    def test_strips_language_fence(self):
        assert _strip_fences("```python\ncode here\n```") == "code here"

    def test_no_fence_unchanged(self):
        assert _strip_fences("plain code") == "plain code"

    def test_strips_surrounding_whitespace(self):
        assert _strip_fences("  ```\ncode\n```  ") == "code"

    def test_only_opening_fence_unchanged(self):
        result = _strip_fences("```python\ncode")
        assert "code" in result

    def test_only_closing_fence_unchanged(self):
        result = _strip_fences("code\n```")
        # closing fence stripped but no opening, so content includes first line
        assert "code" in result

    def test_inner_fence_not_stripped(self):
        text = "```python\nif x:\n    ```nested```\n```"
        result = _strip_fences(text)
        assert "```nested```" in result

    def test_multiline_content_preserved(self):
        text = "```\nline1\nline2\nline3\n```"
        assert _strip_fences(text) == "line1\nline2\nline3"

    def test_empty_fence_block(self):
        result = _strip_fences("```\n```")
        assert result == ""


# ---------------------------------------------------------------------------
# _parse_files
# ---------------------------------------------------------------------------

class TestParseFiles:
    def test_single_file_block(self):
        draft = "### FILE: main.py\ndef main(): pass"
        files = _parse_files(draft)
        assert files == [("main.py", "def main(): pass")]

    def test_two_file_blocks(self):
        draft = "### FILE: a.py\ncontent_a\n### FILE: b.py\ncontent_b"
        files = _parse_files(draft)
        assert len(files) == 2
        assert files[0] == ("a.py", "content_a")
        assert files[1] == ("b.py", "content_b")

    def test_strips_code_fences_inside_block(self):
        draft = "### FILE: a.py\n```python\ncode\n```"
        files = _parse_files(draft)
        assert files[0][1] == "code"

    def test_nested_path(self):
        draft = "### FILE: auth/token.py\nimport jwt"
        files = _parse_files(draft)
        assert files[0][0] == "auth/token.py"

    def test_no_markers_returns_empty(self):
        assert _parse_files("just some prose") == []

    def test_fallback_to_heading_style(self):
        draft = "## a.py\ncontent_a\n## b.py\ncontent_b"
        files = _parse_files(draft)
        assert len(files) == 2
        assert files[0][0] == "a.py"
        assert files[1][0] == "b.py"

    def test_file_block_takes_priority_over_heading(self):
        draft = "### FILE: a.py\ncontent\n## b.py\nother"
        files = _parse_files(draft)
        assert len(files) == 1
        assert files[0][0] == "a.py"

    def test_trailing_whitespace_stripped_from_content(self):
        draft = "### FILE: a.py\ncontent   \n### FILE: b.py\nmore"
        files = _parse_files(draft)
        assert files[0][1] == "content"

    def test_three_files_all_extracted(self):
        draft = (
            "### FILE: a.py\naa\n"
            "### FILE: b.py\nbb\n"
            "### FILE: c.py\ncc"
        )
        files = _parse_files(draft)
        assert len(files) == 3
        assert [f[0] for f in files] == ["a.py", "b.py", "c.py"]
        assert [f[1] for f in files] == ["aa", "bb", "cc"]

    def test_path_with_spaces_trimmed(self):
        draft = "### FILE:   spaced.py   \ncontent"
        files = _parse_files(draft)
        assert files[0][0] == "spaced.py"


# ---------------------------------------------------------------------------
# _write_pipeline_artifacts
# ---------------------------------------------------------------------------

class TestWritePipelineArtifacts:
    def test_writes_contract_parts_artifacts(self, tmp_path):
        result = PipelineResult(
            final_draft="### FILE: app/a.py\nprint('a')\n\n### FILE: app/b.py\nprint('b')",
            contract=CouncilContract(
                goal="Build two files.",
                modules=[("app/a.py", "file a"), ("app/b.py", "file b")],
                interfaces=[("ping() -> str", "returns a marker")],
                rules=["keep it tiny"],
                ownership={"llama": ["app/a.py"], "mistral": ["app/b.py"]},
            ),
            assignments=[
                PartAssignment("llama", ["app/a.py"]),
                PartAssignment("mistral", ["app/b.py"]),
            ],
            drafts=[
                PartDraft(
                    agent_id="llama",
                    owned_files=["app/a.py"],
                    implementation="### FILE: app/a.py\nprint('a')",
                    implemented_by="coder-2",
                ),
                PartDraft(
                    agent_id="mistral",
                    owned_files=["app/b.py"],
                    implementation="### FILE: app/b.py\nprint('b')",
                    implemented_by="mistral",
                ),
            ],
            reviews=[
                PartReview(
                    reviewer_id="mistral",
                    target_agent_id="llama",
                    target_files=["app/a.py"],
                    feedback="BUG: missing import",
                ),
            ],
            merge_result=MergeResult(
                files={"app/a.py": "print('a')", "app/b.py": "print('b')"},
                merged_text="### FILE: app/a.py\nprint('a')\n\n### FILE: app/b.py\nprint('b')",
                unresolved=[],
            ),
            integration_feedback="NO ISSUES FOUND",
            syntax_errors=[],
            interface_warnings=[],
            rounds_run=2,
            architecture_result=CouncilResult(
                final_draft="GOAL\nBuild two files.",
                winning_agent_id="llama",
                rounds_run=2,
                consensus_reached=True,
                transcript=[RoundSummary(round_num=1), RoundSummary(round_num=2)],
            ),
        )

        artifact_dir = _write_pipeline_artifacts(tmp_path, result)

        assert artifact_dir == tmp_path / ".council-harness"
        assert (artifact_dir / "contract.json").exists()
        assert (artifact_dir / "contract.txt").exists()
        assert (artifact_dir / "integration_review.txt").read_text() == "NO ISSUES FOUND"
        assert (artifact_dir / "merged_output.txt").read_text() == result.merge_result.merged_text

        summary = json.loads((artifact_dir / "summary.json").read_text())
        assert summary["mode"] == "contract_parts"
        assert summary["architecture_winner_id"] == "llama"
        assert summary["architecture_rounds"] == 2
        assert summary["merge"]["files"] == ["app/a.py", "app/b.py"]

        llama_draft = json.loads((artifact_dir / "drafts" / "llama.json").read_text())
        assert llama_draft["implemented_by"] == "coder-2"
        assert (artifact_dir / "drafts" / "llama.txt").read_text() == "### FILE: app/a.py\nprint('a')"

        llama_review = json.loads((artifact_dir / "reviews" / "llama.json").read_text())
        assert llama_review["reviewer_id"] == "mistral"
        assert (artifact_dir / "reviews" / "llama.txt").read_text() == "BUG: missing import"
