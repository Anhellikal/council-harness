"""Prompt templates for each phase of the council loop."""

_MULTIFILE_INSTRUCTION = """
OUTPUT FORMAT — MULTIPLE FILES:
Your response must consist entirely of file blocks. Use this exact format for every file:

### FILE: relative/path/to/file.py
<file content>

### FILE: another/file.py
<file content>

Rules:
• You MUST use exactly "### FILE: path" — do NOT use "## filename" or any other heading format.
• Every line of code must be inside a file block — no text outside them.
• Use relative paths (e.g. models/user.py, services/auth.py).
• In round 1, include every file needed for a complete, runnable implementation.
• In revision rounds, only include ### FILE: blocks for files you are actually changing — omit unchanged files entirely.
• Do not add introductions, summaries, or any prose between or after file blocks."""

_ROUND1 = """\
You are a member of a council of AI models collaborating on a coding task.
In this first round you work independently — other agents will not see your proposal yet.

TASK:
{task}
{context_block}{multifile_instruction}
Provide a complete, working implementation. Think carefully before writing."""


_ITERATION = """\
You are agent [{agent_id}], round {round_num} of {max_rounds}.

TASK:
{task}

CURRENT PROPOSALS FROM THE COUNCIL:
{proposals_block}

Review every proposal above. Choose one of two actions and respond using EXACTLY the format shown — \
no text before ACTION:, nothing after your DRAFT content.

━━━ Option A — revise your own draft ━━━
ACTION: revise
DRAFT: <your updated implementation>

━━━ Option B — adopt another agent's proposal as your own ━━━
ACTION: adopt
TARGET: <agent_id>

Rules:
• You cannot adopt your own draft — use ACTION: revise if you want to keep or tweak it.
• TARGET must be one of the other agent IDs listed above.
• If revising, be focused — only include what changed. Do not rewrite sections that are already correct.
• If the proposals are largely the same, adopt the best one rather than rewriting.
• Start your response with ACTION: — do not write anything before it."""


_RETRY = """\
Your previous response was not in the required format.

Respond using EXACTLY one of these two templates — nothing else:

To revise:
ACTION: revise
DRAFT: <your complete implementation>

To adopt another agent:
ACTION: adopt
TARGET: <agent_id>

Start with ACTION: — no preamble."""


def _build_context_block(context_files: list[tuple[str, str]]) -> str:
    if not context_files:
        return ""
    lines = ["\nCONTEXT FILES:\n"]
    for name, content in context_files:
        lines.append(f"--- {name} ---\n{content}\n")
    lines.append("─" * 60 + "\n")
    return "\n".join(lines)


def round1_prompt(task: str, multifile: bool = False, context_files: list[tuple[str, str]] | None = None) -> str:
    return _ROUND1.format(
        task=task,
        context_block=_build_context_block(context_files or []),
        multifile_instruction=_MULTIFILE_INSTRUCTION if multifile else "",
    )


def iteration_prompt(
    task: str,
    agent_id: str,
    round_num: int,
    max_rounds: int,
    current_drafts: dict[str, str],
    multifile: bool = False,
    context_files: list[tuple[str, str]] | None = None,
) -> str:
    blocks = []
    for aid, draft in current_drafts.items():
        label = f"[{aid}]" + (" ← your current draft" if aid == agent_id else "")
        blocks.append(f"{label}\n{draft}")
    proposals_block = "\n\n" + ("\n" + "─" * 60 + "\n").join(blocks) + "\n"

    base = _ITERATION.format(
        agent_id=agent_id,
        round_num=round_num,
        max_rounds=max_rounds,
        task=task,
        proposals_block=proposals_block,
    )
    return base + _build_context_block(context_files or []) + (_MULTIFILE_INSTRUCTION if multifile else "")


_WRITER = """\
A council of AI agents has reached consensus on the following implementation.

TASK:
{task}

AGREED IMPLEMENTATION:
{draft}

Reproduce the AGREED IMPLEMENTATION above EXACTLY — every character, every line, unchanged.
Do NOT add explanations, comments, preamble, or postamble.
Do NOT fix, improve, reformat, or alter anything.
Do NOT wrap the output in markdown code fences (no ``` blocks).
Output ONLY the raw implementation and nothing else."""


_WRITER_WRITE = """\
A council of AI agents has reached consensus on the following implementation.

TASK:
{task}

AGREED IMPLEMENTATION:
{draft}

Write this implementation to disk in the current working directory using your
file-writing tools. For each `### FILE: <path>` block, create the file at that
exact relative path with the exact contents shown (strip the ### FILE: marker
and any surrounding ``` fences). If there are no ### FILE: markers, write the
whole implementation to a single appropriately named file.
Do NOT modify, fix, or reformat the code. Create only the files described above."""


def writer_prompt(task: str, draft: str, multifile: bool = False, write_mode: bool = False) -> str:
    if write_mode:
        return _WRITER_WRITE.format(task=task, draft=draft)
    base = _WRITER.format(task=task, draft=draft)
    if multifile:
        base += "\nPreserve all ### FILE: markers and paths exactly as they appear."
    return base


_REVIEW = """\
You are a code reviewer. Below is an implementation that a council of AI agents agreed on.

Your job: find every bug. Focus on:
- Missing imports
- Incorrect async/await usage
- Logic errors and off-by-one errors
- Uninitialised variables or missing setup
- Anything that would crash or produce wrong output at runtime

List each issue as a short bullet point. Be specific — name the file and function.
Do NOT rewrite the code. Only list the problems.

IMPLEMENTATION:
{draft}"""


_FIX = """\
Below is an implementation and a list of bugs found by code reviewers.
Fix every identified issue and output the complete corrected implementation.
Do not change anything that is not broken.
{multifile_instruction}
IMPLEMENTATION:
{draft}

BUGS IDENTIFIED:
{reviews}"""


def review_prompt(draft: str) -> str:
    return _REVIEW.format(draft=draft)


def fix_prompt(draft: str, reviews: str, multifile: bool = False) -> str:
    return _FIX.format(
        draft=draft,
        reviews=reviews,
        multifile_instruction=_MULTIFILE_INSTRUCTION + "\n" if multifile else "",
    )


def retry_prompt(bad_response: str) -> str:
    excerpt = bad_response[:400].replace("\n", " ").strip()
    return (
        f"Your previous response could not be parsed:\n  {excerpt!r}\n\n"
        + _RETRY
    )


# ===========================================================================
# contract_parts mode prompts
# ===========================================================================

_ARCHITECTURE_ROUND1 = """\
You are a member of a council designing the architecture for a coding task.
Do NOT write any code. Produce a structured architecture proposal only.

TASK:
{task}
{context_block}
Use EXACTLY this format — every section header must appear on its own line:

GOAL
<one paragraph describing what the system does and its key constraints>

MODULES
- relative/path/file.py: <one-line description of this file's responsibility>

INTERFACES
- function_name(param: Type) -> ReturnType: <one-line description>

RULES
- <one architectural constraint per line — things reviewers will check>

OWNERSHIP
- {agent_id_placeholders}

Guidelines:
- Every module in MODULES must appear in exactly one OWNERSHIP entry.
- INTERFACES are the shared boundaries: functions that cross module boundaries or are \
called by files outside their owning module.
- RULES encode constraints that any agent can verify without running the code \
(e.g. "routes must not import from db directly").
- Distribute files roughly evenly. Each agent should own at least one file.
- Use relative paths (e.g. auth/token.py, not /home/user/auth/token.py)."""


_ARCHITECTURE_ITERATION = """\
You are agent [{agent_id}], architecture round {round_num} of {max_rounds}.

TASK:
{task}

CURRENT ARCHITECTURE PROPOSALS:
{proposals_block}

Review every proposal. Choose one of two actions — respond in EXACTLY the format shown:

━━━ Option A — revise your proposal ━━━
ACTION: revise
DRAFT:
GOAL
<paragraph>

MODULES
- path/file.py: description

INTERFACES
- signature: description

RULES
- constraint

OWNERSHIP
- agent_id: file.py, other.py

━━━ Option B — adopt another agent's proposal ━━━
ACTION: adopt
TARGET: <agent_id>

Rules:
- You cannot adopt your own proposal — use ACTION: revise to keep or refine it.
- TARGET must be one of the other agent IDs listed above.
- Prefer adoption when another proposal is clearly better; revise when you see \
meaningful improvements to make.
- The goal is convergence on the best architecture, not winning.
- Start your response with ACTION: — nothing before it."""


_ARCHITECTURE_RETRY = """\
Your previous architecture proposal was not in the required format.

Respond with EXACTLY one of:

To revise:
ACTION: revise
DRAFT:
GOAL
<paragraph>

MODULES
- path/file.py: description

INTERFACES
- signature: description

RULES
- constraint

OWNERSHIP
- agent_id: file.py, other.py

To adopt:
ACTION: adopt
TARGET: <agent_id>

Start with ACTION: — no preamble."""


_IMPLEMENTATION = """\
You are implementing your assigned files as part of a coordinated council effort.

TASK:
{task}

AGREED CONTRACT:
{contract}

YOUR ASSIGNMENT
You own these files: {owned_files}

Implement ONLY your owned files. The other files will be written by other agents.
Assume every INTERFACE declared in the contract exists and works correctly.
Match every INTERFACE signature exactly — other agents will call your exported functions.
Follow every RULE in the contract without exception.

OUTPUT FORMAT — MULTIPLE FILES:
Use this exact format for every file you write:

### FILE: relative/path/file.py
<file content>

Rules:
- Only include ### FILE: blocks for your owned files — nothing else.
- Every line of code must be inside a file block.
- Do not add prose, summaries, or explanations outside file blocks.
- Include all necessary imports within each file."""


_CROSS_REVIEW = """\
You are reviewing an implementation written by another agent.

TASK:
{task}

AGREED CONTRACT:
{contract}

IMPLEMENTATION TO REVIEW
Author: {author_id}
Files: {file_list}

{implementations}

Check only for the following categories and label each finding accordingly:

VIOLATION: <a RULE from the contract that this code breaks>
MISMATCH:  <an INTERFACE signature that does not match the contract exactly>
BUG:       <logic error, missing import, uninitialised variable, or runtime failure>
COMPAT:    <something that will break integration with another agent's files>

List each issue as one short bullet starting with its label.
Do NOT rewrite or suggest rewrites. Do NOT comment on style.
If you find no issues, write: NO ISSUES FOUND"""


_PART_FIX = """\
Apply the following reviewer feedback to your implementation.

TASK:
{task}

CONTRACT INTERFACES (must match exactly):
{interfaces}

YOUR FILES: {owned_files}

YOUR IMPLEMENTATION:
{implementation}

REVIEWER FEEDBACK:
{review}

Output the complete corrected implementation for your files only.
Fix every flagged issue. Do not change anything that was not flagged.

OUTPUT FORMAT — MULTIPLE FILES:
### FILE: relative/path/file.py
<file content>

Include only your owned files. All lines of code must be inside ### FILE: blocks."""


_INTEGRATION_REVIEW = """\
The council has merged all implementation parts. Review the assembled whole.

TASK:
{task}

AGREED CONTRACT:
{contract}

MERGED IMPLEMENTATION:
{merged}

Check for the following and label each finding:

IMPORT:    <missing import or incorrect import path between modules>
GLUE:      <missing wiring, registration, or initialization between modules>
DRIFT:     <any file that diverged from the agreed INTERFACES or RULES>
TESTGAP:   <a scenario mentioned in RULES that has no test coverage>

List each issue as one short bullet starting with its label.
Name the specific file (and function/line if possible) for each issue.
Do NOT rewrite code.
If you find no issues, write: NO ISSUES FOUND"""


_CONTRACT_RETRY = """\
Your previous response could not be parsed as a contract.

Rewrite your response using EXACTLY this format — every section header on its own line, \
every entry as a bullet starting with "-":

GOAL
<one paragraph>

MODULES
- <src/module.py>: <one-line description>

INTERFACES
- <function_name(params) -> ReturnType>: <description>

RULES
- <constraint>

OWNERSHIP
- <agent_id>: <file1.py>, <file2.py>

Requirements:
- Every module in MODULES must appear in OWNERSHIP exactly once.
- INTERFACES may be empty if there are no cross-module boundaries.
- RULES may not be empty.
- Start with GOAL — nothing before it."""


# ---------------------------------------------------------------------------
# contract_parts public prompt functions
# ---------------------------------------------------------------------------

def architecture_prompt(
    task: str,
    agents: list[dict],
    context_files: list[tuple[str, str]] | None = None,
) -> str:
    agent_ids = [a["id"] for a in agents]
    placeholder = "\n- ".join(f"{aid}: <owned files>" for aid in agent_ids)
    return _ARCHITECTURE_ROUND1.format(
        task=task,
        context_block=_build_context_block(context_files or []),
        agent_id_placeholders=placeholder,
    )


def architecture_iteration_prompt(
    task: str,
    agent_id: str,
    round_num: int,
    max_rounds: int,
    current_proposals: dict[str, str],
) -> str:
    blocks = []
    for aid, proposal in current_proposals.items():
        label = f"[{aid}]" + (" ← your current proposal" if aid == agent_id else "")
        blocks.append(f"{label}\n{proposal}")
    proposals_block = "\n\n" + ("\n" + "─" * 60 + "\n").join(blocks) + "\n"
    return _ARCHITECTURE_ITERATION.format(
        agent_id=agent_id,
        round_num=round_num,
        max_rounds=max_rounds,
        task=task,
        proposals_block=proposals_block,
    )


def architecture_retry_prompt(bad_response: str) -> str:
    excerpt = bad_response[:400].replace("\n", " ").strip()
    return f"Your previous response could not be parsed:\n  {excerpt!r}\n\n" + _ARCHITECTURE_RETRY


def contract_retry_prompt(error: str = "", agent_ids: list[str] | None = None) -> str:
    prefix = f"Your contract failed validation: {error}\n\n" if error else ""
    body = _CONTRACT_RETRY
    if agent_ids:
        valid = ", ".join(agent_ids)
        body += f"\n\nValid agent IDs for OWNERSHIP (use exactly these, no others): {valid}"
    return prefix + body


def implementation_prompt(
    task: str,
    contract: str,
    owned_files: list[str],
) -> str:
    return _IMPLEMENTATION.format(
        task=task,
        contract=contract,
        owned_files=", ".join(owned_files),
    )


def cross_review_prompt(
    task: str,
    contract: str,
    author_id: str,
    owned_files: list[str],
    implementations: str,
) -> str:
    return _CROSS_REVIEW.format(
        task=task,
        contract=contract,
        author_id=author_id,
        file_list=", ".join(owned_files),
        implementations=implementations,
    )


def part_fix_prompt(
    task: str,
    interfaces_text: str,
    owned_files: list[str],
    implementation: str,
    review: str,
) -> str:
    return _PART_FIX.format(
        task=task,
        interfaces=interfaces_text,
        owned_files=", ".join(owned_files),
        implementation=implementation,
        review=review,
    )


def integration_review_prompt(
    task: str,
    contract: str,
    merged: str,
) -> str:
    return _INTEGRATION_REVIEW.format(
        task=task,
        contract=contract,
        merged=merged,
    )
