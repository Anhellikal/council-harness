# Architecture

Everything a contributor needs to understand this codebase without reading every file. For user-facing docs see [README.md](README.md).

---

## What this project is

A CLI tool (`council run "<task>"`) that coordinates N LLM agents — local models and subscription-based CLI agents — to collaboratively produce a coding implementation. Agents propose, discuss, and converge over multiple rounds. The final agreed draft optionally passes through a writer agent for clean output (or is written straight to disk by a write-enabled CLI writer).

---

## File map

```
council-harness/
├── config.example.yaml   Commented config template. Users copy to config.yaml (gitignored).
├── pyproject.toml        Package metadata + entry point: council.cli:main
└── council/
    ├── agent.py          Call layer. call_agent(), rollcall(), call_all_parallel().
    │                     HTTP for ollama/openai; subprocess for claude_cli/codex_cli.
    ├── contracts.py      contract_parts schema. CouncilContract, PartAssignment,
    │                     PartDraft, PartReview, MergeResult. parse_contract(),
    │                     build_assignments(), cross_review_pairs().
    ├── dashboard.py      Optional Rich live dashboard for run status and convergence.
    ├── prompts.py        All prompt strings (holistic + contract_parts).
    ├── loop.py           holistic logic. run_council() + run_writer() + review/fix phase.
    ├── pipeline.py       contract_parts orchestration. run_pipeline(), merge, validation.
    └── cli.py            Click CLI entry. run command + agent config management +
                          file writing / artifact persistence / tiebreak handling.
```

---

## Data flow

### holistic mode (default)

```
cli.run()
  │
  ├─ load config.yaml
  ├─ rollcall(agents)          → (active: list[dict], missing: list[dict], statuses: dict)
  │
  └─ run_council(task, active, config, emit)        [loop.py]
       │
       ├─ Round 1: call_all_parallel(active, round1_prompt)
       │           → current_drafts: {agent_id: str}
       │
       ├─ Rounds 2–N (parallel per round):
       │   for each agent → iteration_prompt → call_agent (with retries)
       │                  → parse ACTION/TARGET/DRAFT
       │                  → apply to current_drafts atomically from a round-start snapshot
       │   check convergence after each round
       │
       ├─ Review/fix pass (optional) → _run_review_phase(..., winner_id=winning_agent_id)
       │   all agents review in parallel; the winning agent applies combined fixes
       │
       └─ CouncilResult(final_draft, winning_agent_id, rounds_run, consensus_reached,
                        tiebreak_options, reviewed_by, transcript)

cli.run() then:
  ├─ true-tie handling: print proposals, prompt to pick one OR 'r' to restart from round 1
  ├─ run_writer(...) if writer: configured   (passes output_dir for write-enabled CLI writers)
  └─ write files / council.log to --output-dir
```

### contract_parts mode

```
cli.run()
  │
  ├─ load config (mode: contract_parts, or --mode contract_parts)
  ├─ rollcall(agents)
  │
  └─ run_pipeline(task, active, config, emit)        [pipeline.py]
       │
       ├─ [Architecture] run_council(..., prompt_fn=architecture_prompt)
       │   Agents propose GOAL/MODULES/INTERFACES/RULES/OWNERSHIP via REVISE/ADOPT.
       │   → CouncilResult.final_draft = raw contract text
       │
       ├─ _parse_contract_with_retry(...) → CouncilContract
       │   parse_contract validates: all modules owned, no file owned twice, agent IDs known.
       │   On validation failure: retry with contract_retry_prompt (lists valid agent IDs), then abort.
       │
       ├─ build_assignments(contract) → list[PartAssignment]    (deterministic, no agent calls)
       │
       ├─ [Implementation] parallel per agent:
       │   implementation_prompt(task, contract_text, owned_files) → call_agent
       │   _filter_owned_files() keeps only owned files (tolerant path match) and strips fences
       │   → PartDraft(agent_id, owned_files, implementation, failed, implemented_by)
       │   Failed/missing drafts are reassigned to _next_retry_agent() and re-attempted.
       │
       ├─ cross_review_pairs(assignments) → [(reviewer_id, target_id), ...]   (agent i reviews (i+1)%n)
       │
       ├─ [Cross-Review] parallel per reviewer:
       │   cross_review_prompt(...) → PartReview(reviewer_id, target_agent_id, feedback, failed)
       │   A failed/timed-out review is recorded (failed=True) and the fix for that part is skipped.
       │
       ├─ [Fix] parallel per implementer (skipped when review failed or "NO ISSUES FOUND"):
       │   part_fix_prompt(...) → updated PartDraft; failed fixes reassigned to next agent
       │
       ├─ [Merge] deterministic:
       │   _merge_drafts() extracts ### FILE: blocks per draft, combines by ownership.
       │   Aborts (RuntimeError) with per-draft diagnostics if any owned file is missing.
       │   → MergeResult(files, merged_text, unresolved)
       │
       ├─ [Validation] advisory (never aborts):
       │   _validate_syntax()    — ast.parse() each merged file → syntax_errors
       │   _validate_interfaces()— contract INTERFACES present as top-level func/class → interface_warnings
       │
       ├─ [Integration] integration_review_prompt(...) → advisory integration_feedback
       │
       └─ PipelineResult(final_draft, contract, assignments, drafts, reviews, merge_result,
                         integration_feedback, syntax_errors, interface_warnings,
                         rounds_run, architecture_result)

cli.run() then writes files + council.log + .council-harness/ artifacts to --output-dir.
```

---

## Key types

**`CouncilContract`** (contracts.py) — the agreed architecture in contract_parts mode:
```python
goal: str                               # one paragraph system description
modules: list[tuple[str, str]]          # (relative_path, description)
interfaces: list[tuple[str, str]]       # (signature, description) — cross-module boundaries
rules: list[str]                        # architectural constraints reviewers check
ownership: dict[str, list[str]]         # agent_id → [relative_path, ...]
```

**`PartDraft`** (contracts.py) — one agent's implementation output:
```python
agent_id: str            # the owner the work is for
owned_files: list[str]
implementation: str = "" # filtered ### FILE: block text (fences stripped)
failed: bool = False     # True if the agent call errored or owned files were missing
implemented_by: str = "" # who actually produced it (differs from agent_id after a retry)
```

**`PartReview`** (contracts.py):
```python
reviewer_id: str
target_agent_id: str
target_files: list[str]
feedback: str = ""       # VIOLATION:/MISMATCH:/BUG:/COMPAT: bullets
failed: bool = False     # True if the review call errored or timed out
```

**`MergeResult`** (contracts.py):
```python
files: dict[str, str]    # relative_path → content
merged_text: str         # complete ### FILE: block output
unresolved: list[str]    # owned paths that could not be parsed from any PartDraft
```

**`PipelineResult`** (pipeline.py) — returned by `run_pipeline()`:
```python
final_draft: str                  # = merge_result.merged_text
contract: CouncilContract
assignments: list[PartAssignment]
drafts: list[PartDraft]           # after the fix phase
reviews: list[PartReview]
merge_result: MergeResult
integration_feedback: str         # advisory
syntax_errors: list[str]          # from the Validation phase
interface_warnings: list[str]     # from the Validation phase
rounds_run: int                   # architecture convergence rounds
architecture_result: CouncilResult
```

**`CouncilResult`** (loop.py) — returned by `run_council()`:
```python
final_draft: str           # winning implementation text
winning_agent_id: str
rounds_run: int
consensus_reached: bool    # False = majority fallback was used
tiebreak_options: dict     # {agent_id: draft} — set only on a true tie, else None
reviewed_by: str | None    # fixer agent id if the review/fix phase changed the draft
transcript: list[RoundSummary]
```

**Agent config dict** — shape from YAML:
```python
# common
{"id": str, "type": "ollama"|"openai"|"claude_cli"|"codex_cli", "timeout": int, ...}
# ollama/openai also: "model", "url", "max_tokens", "num_ctx", "no_think",
#                     "api_key" | "api_key_env"
# claude_cli/codex_cli: "model" (optional), "can_write" (writer only)
```

---

## Agent call layer (`agent.py`)

`call_agent(agent, prompt, workdir=None)` dispatches on `agent["type"]`:

| type         | backend                          | notes |
|--------------|----------------------------------|-------|
| `ollama`     | `POST {url}/api/generate`        | reports real tok/s + ttft |
| `openai`     | `POST {url}/v1/chat/completions` | Bearer token from `api_key_env`/`api_key` |
| `claude_cli` | `claude -p` subprocess (stdin)   | subscription login; `--allowedTools ""` (gen) or `--permission-mode acceptEdits` (write) |
| `codex_cli`  | `codex exec` subprocess (stdin)  | subscription login; `-s read-only` (gen) or `-s workspace-write` (write); final message read from `-o` file |

- **`workdir`** is honored only by CLI types. `None` → generation: runs sandboxed in a throwaway temp dir, read-only/no tools. Set → writer: runs in that dir with writes enabled. `cli_can_write(agent)` gates this (`can_write: true` + CLI type).
- `ping_agent()` does a real generation for HTTP agents; for CLI agents it just checks the binary is on `PATH`.
- Prompts are sent via **stdin** for CLI agents (avoids the variadic `--allowedTools` swallowing a positional prompt).

---

## Core state in `run_council()`

```python
current_drafts: dict[str, str]   # {agent_id → current draft text}
```

The single source of truth. Convergence logic operates on it:
- **Revise**: `current_drafts[agent_id] = new_draft`
- **Adopt**: `current_drafts[agent_id] = snapshot[target_id]` (snapshot = state at round start)
- **Distinct proposals**: `len(set(current_drafts.values()))`
- **Convergence**: most-common draft count `>= effective_threshold`

Iteration rounds read a **snapshot** before dispatching, so all agents see the same starting state regardless of call order; updates apply atomically after all responses arrive.

---

## File-block parsing & matching (contract_parts)

Agents emit files as `### FILE: <path>` blocks. Two robustness layers in `pipeline.py`:

- **`_strip_block_fences()`** — strips a wrapping ` ```lang … ``` ` fence from each block's body (otherwise every merged file starts with ` ```python ` and fails `ast.parse`). Inner/nested fences are preserved.
- **`_match_owned()` / `_filter_owned_files()`** — match an emitted path to a contract-owned path by exact match first, then by *unambiguous* basename (handles `dashboard/main.py` vs bare `main.py` drift), canonicalizing the key to the owned path. Ambiguous basenames are left unmatched rather than guessed.

On a missing owned file the implementation phase writes the agent's raw prompt+output to `${TMPDIR}/council-pipeline-debug/` via `_dump_debug()` for diagnosis.

---

## Prompt structure

### holistic mode

| Phase | Function | Sent to |
|---|---|---|
| Round 1 | `round1_prompt(task)` | All agents, identical, parallel |
| Round 2–N | `iteration_prompt(...)` | Each agent individually |
| Parse failure | `retry_prompt(bad_response)` | Same agent, up to 3 attempts |
| Review | `review_prompt(draft)` | All active agents, parallel |
| Fix | `fix_prompt(draft, reviews)` | Winning agent |
| Final output | `writer_prompt(task, draft, write_mode=...)` | Writer agent only |

`writer_prompt(..., write_mode=True)` swaps in `_WRITER_WRITE`, instructing a write-enabled CLI writer to create the files with its tools instead of returning text.

### contract_parts mode

| Phase | Function | Sent to |
|---|---|---|
| Architecture round 1 | `architecture_prompt(task, agents)` | All agents, parallel |
| Architecture round 2–N | `architecture_iteration_prompt(...)` | Each agent individually |
| Contract reformat | `contract_retry_prompt(error, agent_ids)` | Winner (on validation failure; lists valid IDs) |
| Implementation | `implementation_prompt(task, contract_text, owned_files)` | Each agent, parallel |
| Cross-review | `cross_review_prompt(...)` | Reviewer agent (shifted by 1) |
| Part fix | `part_fix_prompt(...)` | Original implementer |
| Integration review | `integration_review_prompt(...)` | All agents, parallel |

Architecture rounds reuse the holistic REVISE/ADOPT parse logic — only the prompt text differs; the contract format is the DRAFT content. Cross-review uses `VIOLATION:/MISMATCH:/BUG:/COMPAT:` bullets; integration uses `IMPORT:/GLUE:/DRIFT:/TESTGAP:`.

---

## Consensus threshold

Configured as `consensus_threshold` relative to total configured agents; scaled at runtime to whoever responded:

```python
effective = math.ceil(active_count * config_threshold / config_total)   # clamped to [1, active_count]
```

Special cases: **1 agent** skips all rounds (round 1 is final); **all agents fail** → `active = []`, CLI exits before `run_council`.

---

## Where to make common changes

| Change | File | What to touch |
|---|---|---|
| Tune holistic prompts | `prompts.py` | `_ROUND1`, `_ITERATION`, `_WRITER`, `_WRITER_WRITE`, `_RETRY` |
| Tune contract_parts prompts | `prompts.py` | `_ARCHITECTURE_*`, `_IMPLEMENTATION`, `_CROSS_REVIEW`, `_PART_FIX`, `_INTEGRATION_REVIEW`, `_CONTRACT_RETRY` |
| Change retry count | `loop.py` | `MAX_RETRIES` |
| Add a new agent backend | `agent.py` | `_call_<type>()` + branches in `call_agent()` and `ping_agent()` |
| Adjust CLI agent flags/sandbox | `agent.py` | `_call_claude_cli()` / `_call_codex_cli()` |
| Change convergence logic | `loop.py` | convergence/distinct-count helpers |
| Change contract validation | `contracts.py` | `parse_contract()` validation block |
| Change file matching / fence stripping | `pipeline.py` | `_match_owned()`, `_filter_owned_files()`, `_strip_block_fences()` |
| Change merge or post-merge validation | `pipeline.py` | `_merge_drafts()`, `_validate_syntax()`, `_validate_interfaces()` |
| Add CLI flags | `cli.py` | `@click.option` decorators on `run()` |
| Change output / artifact writing | `cli.py` | `_write_files()`, `_write_pipeline_artifacts()` |
| Change dashboard UI | `dashboard.py` | `CouncilDashboard` render + panel helpers |
| Onboard/update agents | `cli.py` | `agent list` / `add` / `update` subcommands |
```
