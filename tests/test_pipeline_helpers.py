"""Tests for retry behavior in council/pipeline.py."""
from council.loop import CouncilResult, RoundSummary
from council.pipeline import (
    _next_retry_agent,
    _filter_owned_files,
    _match_owned,
    _extract_file_blocks,
    _strip_block_fences,
    run_pipeline,
)
from council.prompts import contract_retry_prompt


class TestMatchOwned:
    def test_exact_match(self):
        assert _match_owned(["main.py", "server.py"], ["main.py", "server.py"]) == {
            "main.py": "main.py",
            "server.py": "server.py",
        }

    def test_emitted_has_prefix_drift(self):
        # agent emitted dashboard/main.py; contract owns bare main.py
        assert _match_owned(["dashboard/main.py"], ["main.py"]) == {"dashboard/main.py": "main.py"}

    def test_owned_has_prefix_drift(self):
        # agent emitted bare main.py; contract owns src/main.py
        assert _match_owned(["main.py"], ["src/main.py"]) == {"main.py": "src/main.py"}

    def test_ambiguous_basename_not_matched(self):
        # two owned files share a basename -> don't guess
        result = _match_owned(["a/util.py"], ["x/util.py", "y/util.py"])
        assert result == {}

    def test_unowned_file_excluded(self):
        assert _match_owned(["main.py", "evil.py"], ["main.py"]) == {"main.py": "main.py"}


class TestFilterOwnedFiles:
    def test_canonicalizes_drifted_path_to_owned(self):
        raw = "### FILE: dashboard/main.py\nprint('hi')"
        merged, skipped = _filter_owned_files(raw, ["main.py"])
        assert "### FILE: main.py" in merged  # key canonicalized to owned path
        assert "dashboard/main.py" not in merged
        assert skipped == []

    def test_skips_truly_unowned(self):
        raw = "### FILE: main.py\na\n\n### FILE: other.py\nb"
        merged, skipped = _filter_owned_files(raw, ["main.py"])
        assert "### FILE: main.py" in merged
        assert skipped == ["other.py"]


class TestStripBlockFences:
    def test_strips_language_fence(self):
        assert _strip_block_fences("```python\nx = 1\n```") == "x = 1"

    def test_strips_plain_fence(self):
        assert _strip_block_fences("```\nx = 1\n```") == "x = 1"

    def test_no_fence_unchanged(self):
        assert _strip_block_fences("x = 1\ny = 2") == "x = 1\ny = 2"

    def test_leaves_inner_fence(self):
        body = "```python\ntext = '''\n```\n'''\n```"
        # outer pair stripped, inner ``` preserved as content
        assert _strip_block_fences(body) == "text = '''\n```\n'''"


class TestExtractFileBlocksStripsFences:
    def test_fenced_block_yields_parseable_code(self):
        import ast
        raw = "### FILE: main.py\n```python\ndef f():\n    return 1\n```"
        blocks = _extract_file_blocks(raw)
        assert blocks["main.py"] == "def f():\n    return 1"
        ast.parse(blocks["main.py"])  # no SyntaxError

    def test_two_fenced_blocks(self):
        raw = "### FILE: a.py\n```python\na = 1\n```\n\n### FILE: b.py\n```\nb = 2\n```"
        blocks = _extract_file_blocks(raw)
        assert blocks == {"a.py": "a = 1", "b.py": "b = 2"}


class TestValidateInterfaces:
    def _merge(self, files):
        from council.contracts import MergeResult
        return MergeResult(files=files, merged_text="", unresolved=[])

    def _contract(self, interfaces):
        from council.contracts import CouncilContract
        return CouncilContract(goal="g", modules=[], interfaces=interfaces, rules=[], ownership={})

    def test_backtick_contaminated_name_still_matches(self):
        from council.pipeline import _validate_interfaces
        contract = self._contract([("`load_run(x) -> Y`", "loads a run")])
        merge = self._merge({"loader.py": "def load_run(x):\n    return 1"})
        assert _validate_interfaces(contract, merge) == []

    def test_dataclass_interface_not_flagged(self):
        from council.pipeline import _validate_interfaces
        contract = self._contract([("RunData", "the run dataclass")])
        merge = self._merge({"models.py": "class RunData:\n    pass"})
        assert _validate_interfaces(contract, merge) == []

    def test_genuinely_missing_is_flagged(self):
        from council.pipeline import _validate_interfaces
        contract = self._contract([("ghost(a) -> b", "nope")])
        merge = self._merge({"x.py": "def other():\n    pass"})
        warnings = _validate_interfaces(contract, merge)
        assert len(warnings) == 1 and warnings[0].startswith("ghost:")


class TestNextRetryAgent:
    def test_single_agent_returns_none(self):
        assert _next_retry_agent("llama", ["llama"]) is None

    def test_returns_next_agent(self):
        assert _next_retry_agent("llama", ["llama", "mistral", "coder"]) == "mistral"

    def test_wraps_around(self):
        assert _next_retry_agent("coder", ["llama", "mistral", "coder"]) == "llama"

    def test_skips_attempted_ids(self):
        assert _next_retry_agent(
            "llama",
            ["llama", "mistral", "coder"],
            {"llama", "mistral"},
        ) == "coder"

    def test_unknown_current_agent_starts_from_front(self):
        assert _next_retry_agent("ghost", ["llama", "mistral"]) == "llama"


def _architecture_result(contract_text: str, winner_id: str = "llama") -> CouncilResult:
    return CouncilResult(
        final_draft=contract_text,
        winning_agent_id=winner_id,
        rounds_run=1,
        consensus_reached=True,
        transcript=[RoundSummary(round_num=1)],
    )


class TestPipelineRetries:
    def test_contract_retry_prompt_lists_exact_valid_agent_ids(self):
        prompt = contract_retry_prompt(
            "OWNERSHIP references unknown agents: ['dev01', 'dev02']",
            ["coder", "gemma", "mac_gemma"],
        )

        assert "Your contract failed validation" in prompt
        assert "Valid agent IDs for OWNERSHIP" in prompt
        assert "use exactly these, no others" in prompt
        assert "coder, gemma, mac_gemma" in prompt

    def test_pipeline_contract_retry_recovers_with_active_agent_ids(self, monkeypatch):
        bad_contract = "Not a contract at all."
        repaired_contract = (
            "GOAL\nBuild one file.\n\n"
            "MODULES\n- app/main.py: entrypoint\n\n"
            "RULES\n- keep it tiny\n\n"
            "OWNERSHIP\n- coder: app/main.py\n"
        )
        agents = [{"id": "coder"}, {"id": "gemma"}, {"id": "mac_gemma"}]

        monkeypatch.setattr(
            "council.pipeline.run_council",
            lambda **kwargs: _architecture_result(bad_contract, winner_id="coder"),
        )

        seen_retry_prompts: list[str] = []

        def fake_call_agent(agent, prompt):
            if "Your contract failed validation:" in prompt:
                seen_retry_prompts.append(prompt)
                return repaired_contract, {}
            if "You are implementing your assigned files" in prompt:
                return ("### FILE: app/main.py\nprint('ok')", {})
            if "The council has merged all implementation parts" in prompt:
                return ("NO ISSUES FOUND", {})
            raise AssertionError(f"unexpected prompt for {agent['id']}")

        monkeypatch.setattr("council.pipeline.call_agent", fake_call_agent)

        result = run_pipeline(
            task="Build the entrypoint.",
            active_agents=agents,
            config={"rounds": {"max": 3}},
            emit=lambda *_args, **_kwargs: None,
        )

        assert result.contract.goal == "Build one file."
        assert result.contract.ownership == {"coder": ["app/main.py"]}
        assert result.final_draft == "### FILE: app/main.py\nprint('ok')"
        assert len(seen_retry_prompts) == 1
        retry_prompt = seen_retry_prompts[0]
        assert "Contract GOAL is empty." in retry_prompt
        assert "coder, gemma, mac_gemma" in retry_prompt
        assert "use exactly these, no others" in retry_prompt

    def test_retries_failed_implementation_with_next_agent(self, monkeypatch):
        contract = (
            "GOAL\nImplement one file.\n\n"
            "MODULES\n- app/main.py: entrypoint\n\n"
            "RULES\n- keep it tiny\n\n"
            "OWNERSHIP\n- llama: app/main.py\n"
        )
        agents = [{"id": "llama"}, {"id": "mistral"}]

        monkeypatch.setattr(
            "council.pipeline.run_council",
            lambda **kwargs: _architecture_result(contract),
        )

        calls: list[tuple[str, str]] = []

        def fake_call_agent(agent, prompt):
            calls.append((agent["id"], prompt))
            if "You are implementing your assigned files" in prompt:
                if agent["id"] == "llama":
                    raise RuntimeError("timeout")
                return ("### FILE: app/main.py\nprint('ok')", {})
            if "The council has merged all implementation parts" in prompt:
                return ("NO ISSUES FOUND", {})
            raise AssertionError(f"unexpected prompt for {agent['id']}")

        monkeypatch.setattr("council.pipeline.call_agent", fake_call_agent)

        result = run_pipeline(
            task="Build the entrypoint.",
            active_agents=agents,
            config={"rounds": {"max": 3}},
            emit=lambda *_args, **_kwargs: None,
        )

        assert result.final_draft == "### FILE: app/main.py\nprint('ok')"
        assert result.drafts[0].agent_id == "llama"
        assert result.drafts[0].implemented_by == "mistral"
        assert result.drafts[0].failed is False
        assert [agent_id for agent_id, _ in calls] == ["llama", "mistral", "llama"]

    def test_retries_failed_fix_with_next_agent(self, monkeypatch):
        contract = (
            "GOAL\nImplement two files.\n\n"
            "MODULES\n- app/a.py: file a\n- app/b.py: file b\n\n"
            "INTERFACES\n- ping() -> str: returns a marker\n\n"
            "RULES\n- keep output deterministic\n\n"
            "OWNERSHIP\n- llama: app/a.py\n- mistral: app/b.py\n"
        )
        agents = [{"id": "llama"}, {"id": "mistral"}]

        monkeypatch.setattr(
            "council.pipeline.run_council",
            lambda **kwargs: _architecture_result(contract),
        )

        fix_attempts: list[str] = []

        def fake_call_agent(agent, prompt):
            agent_id = agent["id"]
            if "You are implementing your assigned files" in prompt:
                if "You own these files: app/a.py" in prompt:
                    return ("### FILE: app/a.py\nBROKEN = True", {})
                return ("### FILE: app/b.py\nREADY = True", {})
            if "You are reviewing an implementation written by another agent." in prompt:
                if "Files: app/a.py" in prompt:
                    return ("BUG: app/a.py is broken", {})
                return ("NO ISSUES FOUND", {})
            if "Apply the following reviewer feedback to your implementation." in prompt:
                fix_attempts.append(agent_id)
                if agent_id == "llama":
                    raise RuntimeError("fix failed")
                return ("### FILE: app/a.py\nBROKEN = False", {})
            if "The council has merged all implementation parts" in prompt:
                return ("NO ISSUES FOUND", {})
            raise AssertionError(f"unexpected prompt for {agent_id}")

        monkeypatch.setattr("council.pipeline.call_agent", fake_call_agent)

        result = run_pipeline(
            task="Build two tiny modules.",
            active_agents=agents,
            config={"rounds": {"max": 3}},
            emit=lambda *_args, **_kwargs: None,
        )

        draft_by_owner = {draft.agent_id: draft for draft in result.drafts}
        assert draft_by_owner["llama"].implementation == "### FILE: app/a.py\nBROKEN = False"
        assert draft_by_owner["llama"].implemented_by == "mistral"
        assert draft_by_owner["mistral"].implemented_by == "mistral"
        assert fix_attempts == ["llama", "mistral"]

    def test_cross_review_uses_fallback_implementer_as_author(self, monkeypatch):
        contract = (
            "GOAL\nImplement two files.\n\n"
            "MODULES\n- app/a.py: file a\n- app/b.py: file b\n\n"
            "RULES\n- keep it tiny\n\n"
            "OWNERSHIP\n- llama: app/a.py\n- mistral: app/b.py\n"
        )
        agents = [{"id": "llama"}, {"id": "mistral"}]

        monkeypatch.setattr(
            "council.pipeline.run_council",
            lambda **kwargs: _architecture_result(contract),
        )

        review_prompts: list[str] = []

        def fake_call_agent(agent, prompt):
            if "You are implementing your assigned files" in prompt:
                if "You own these files: app/a.py" in prompt and agent["id"] == "llama":
                    raise RuntimeError("timeout")
                if "You own these files: app/a.py" in prompt:
                    return ("### FILE: app/a.py\nprint('a')", {})
                return ("### FILE: app/b.py\nprint('b')", {})
            if "You are reviewing an implementation written by another agent." in prompt:
                review_prompts.append(prompt)
                return ("NO ISSUES FOUND", {})
            if "The council has merged all implementation parts" in prompt:
                return ("NO ISSUES FOUND", {})
            raise AssertionError(f"unexpected prompt for {agent['id']}")

        monkeypatch.setattr("council.pipeline.call_agent", fake_call_agent)

        result = run_pipeline(
            task="Build the entrypoint.",
            active_agents=agents,
            config={"rounds": {"max": 3}},
            emit=lambda *_args, **_kwargs: None,
        )

        draft_by_owner = {draft.agent_id: draft for draft in result.drafts}
        assert draft_by_owner["llama"].implemented_by == "mistral"
        assert len(review_prompts) == 2
        prompt_for_a = next(prompt for prompt in review_prompts if "Files: app/a.py" in prompt)
        assert "Author: mistral" in prompt_for_a
        assert "Author: llama" not in prompt_for_a

    def test_merge_aborts_when_retry_also_fails(self, monkeypatch):
        contract = (
            "GOAL\nImplement one file.\n\n"
            "MODULES\n- app/main.py: entrypoint\n\n"
            "RULES\n- keep it tiny\n\n"
            "OWNERSHIP\n- llama: app/main.py\n"
        )
        agents = [{"id": "llama"}, {"id": "mistral"}]

        monkeypatch.setattr(
            "council.pipeline.run_council",
            lambda **kwargs: _architecture_result(contract),
        )

        attempts: list[str] = []

        def fake_call_agent(agent, prompt):
            if "You are implementing your assigned files" in prompt:
                attempts.append(agent["id"])
                raise RuntimeError("still failing")
            raise AssertionError(f"unexpected prompt for {agent['id']}")

        monkeypatch.setattr("council.pipeline.call_agent", fake_call_agent)

        try:
            run_pipeline(
                task="Build the entrypoint.",
                active_agents=agents,
                config={"rounds": {"max": 3}},
                emit=lambda *_args, **_kwargs: None,
            )
            raise AssertionError("expected run_pipeline to abort")
        except RuntimeError as exc:
            message = str(exc)

        assert attempts == ["llama", "mistral"]
        assert "contract_parts merge aborted due to incomplete implementation" in message
        assert "failed drafts from llama" in message
        assert "llama missing owned file app/main.py" in message

    def test_single_agent_pipeline_does_not_retry(self, monkeypatch):
        contract = (
            "GOAL\nImplement one file.\n\n"
            "MODULES\n- app/main.py: entrypoint\n\n"
            "RULES\n- keep it tiny\n\n"
            "OWNERSHIP\n- llama: app/main.py\n"
        )
        agents = [{"id": "llama"}]

        monkeypatch.setattr(
            "council.pipeline.run_council",
            lambda **kwargs: _architecture_result(contract),
        )

        attempts: list[str] = []

        def fake_call_agent(agent, prompt):
            if "You are implementing your assigned files" in prompt:
                attempts.append(agent["id"])
                raise RuntimeError("timeout")
            raise AssertionError(f"unexpected prompt for {agent['id']}")

        monkeypatch.setattr("council.pipeline.call_agent", fake_call_agent)

        try:
            run_pipeline(
                task="Build the entrypoint.",
                active_agents=agents,
                config={"rounds": {"max": 3}},
                emit=lambda *_args, **_kwargs: None,
            )
            raise AssertionError("expected run_pipeline to abort")
        except RuntimeError as exc:
            message = str(exc)

        assert attempts == ["llama"]
        assert "failed drafts from llama" in message
