# Findings

Honest, observational notes from building and running this harness. **This is not a benchmark** it's a single hobby project, a handful of tasks, small models on consumer/homelab hardware, and an N of "a few dozen runs." The point of writing it down is that the harness's value turned out to be *learning these limits*, not the harness in itself.

## TL;DR
A council of **local** LLM agents reliably **converges** on a design and produces **runnable-looking** multi-file code. It does **not** reliably produce *correct, integrated* code without a human review pass. The structure (propose → discuss → converge → cross-review → merge) helps most with *agreement* and least with *correctness*. The correctness safety net — cross-review and validation — is the fragile part, and it's the first thing to break. But the fragility traced back to the *agents*, not the structure: weak agents agree on wrong things and drift from their prompts, and that gap closed once stronger agents were swapped in.


## Setup these notes are based on

- **Local models** (Ollama): `qwen2.5-coder` 7B and 14B variants, Q4_K_M, gemma4, on multiple consumer GPUs and Silicon Macs.
- **Subscription CLI agents**: Claude (`claude -p`) and Codex (`codex exec`), via subprocess — no API, just the logged-in CLI.
- **Tasks**: small-to-medium Python (rate limiter, palindrome, Connect Four, a stdlib web dashboard), single-file and multi-file.

## Local models can't compete with higher models, even when breaking down smaller tasks


1. **Holding agreed upon contracts is hard for local agents.** The 7B/14B Qwen-Coder models write reasonable single functions and small files. Where they fall down is *holding to a shared contract* — emitting the agreed file paths, matching an interface another agent defined, working on the integration/file they were asked to. Several merge failures were path drift (`dashboard/main.py` vs `main.py`) or markdown-fence contamination, not "the model can't code."

2. **Bigger wasn't better for review.** In multiple scenarios the 14B model did **not** clearly out-review the 7B on this workload. Review quality was dominated by *prompting and decoding settings*, not parameter count.

3. **Local models are weak, fast reviewers; strong models are good, slow reviewers.** Local models would rubber-stamp ("NO ISSUES FOUND") in a few seconds. The strong CLI agents found real issues — but the cross-review prompt carries the full contract *plus* a peer's whole implementation.


## What the council structure does and doesn't buy you

**Helps:**
- **Convergence.** REVISE/ADOPT rounds genuinely collapse 3 divergent designs into 1 agreed one. This part works well.
- **Diversity in round 1.** Different agents surface genuinely different approaches before converging — occasionally one catches a constraint the others missed.
- **Partitioning saves tokens.** `contract_parts` (each agent implements only its files) costs far less than holistic (every agent re-emits the whole solution every round). The bigger and more separable the task, the larger the win.

**Doesn't buy you:**
- **Correctness.** Agreement ≠ correct. Three agents can confidently converge on the same wrong interface.
- **Reliable self-QA.** Cross-review is exactly as good as the reviewing model *and* its time budget, and it fails open (a skipped review just… ships).
- **Speed.** `contract_parts` uses fewer tokens but more wall-clock — the design phase is sequential rounds before any code is written.


## Honest conclusion

This project intended to solve one question: can a council of agents deliver reliable code and perform as well as an enterprise model?
After a dozen-plus runs the question changed to "Is my ceiling the agents and not the council-harness itself?" — which turned out to be true after incorporating paid agents (Sonnet, Codex (GPT-5 class)).
The bottleneck was never really the generated code — at least for what we could observe. It was that the agents would *agree on things that were wrong*, or simply not follow their prompts. Swapping in stronger agents largely closed that gap: the contract failures, mismatched functions, and uninstantiated references mostly disappeared.
That's also where the project stopped. Continuing meant fine-tuning the agents, not improving the council-harness.