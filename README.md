<h1 align="center">prompt-architect</h1>

<h4 align="center">Expert prompt-engineering assistant. Prompts as structured artifacts, grounded by a book corpus.</h4>

<p align="center">
  <a href="https://github.com/nuclide-research/prompt-architect/releases"><img src="https://img.shields.io/github/v/release/nuclide-research/prompt-architect?style=flat-square" alt="release"></a>
  <a href="https://github.com/nuclide-research/prompt-architect/blob/main/LICENSE"><img src="https://img.shields.io/github/license/nuclide-research/prompt-architect?style=flat-square" alt="license"></a>
  <a href="https://www.python.org"><img src="https://img.shields.io/badge/python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="python"></a>
  <a href="https://nuclide-research.com"><img src="https://img.shields.io/badge/by-NuClide-blue?style=flat-square" alt="NuClide"></a>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#api">API</a> •
  <a href="#runtime-rag">RAG</a> •
  <a href="#data-models">Models</a> •
  <a href="#scope">Scope</a>
</p>

---

prompt-architect is a Python library that treats a prompt as a structured artifact: role, goal, input and output contracts, constraints, and self-checks. Given a raw prompt and a small context object, it discovers the prompt's implicit structure, names the structural and behavioral problems, refines it into a canonical production-ready form, and generates optimization-focused variants. Every refinement is grounded against an embedded corpus of seven O'Reilly titles using a pure-Python BM25 retriever, so each fix points back to the passage that justifies it.

The heuristics are pure functions with no I/O. The book corpus is optional and lazy: it builds only on the first grounding call, caches to disk, and never touches the network. Every return value is JSON-serializable.

# Features

- Three operations: `analyze` a prompt, `refine` it into a canonical spec, `generate_variants`
- Prompts modeled as structured specs: role, goal, inputs contract, ordered instructions, output contract, constraints, quality checks
- Issue and risk detection: missing role, absent output format, vague instructions, no self-check, hallucination and security risk by context
- Three variants per prompt: `cot` (chain-of-thought), `concise_strict` (format-strict, machine-friendly), `fewshot_skeleton` (few-shot demonstration)
- Runtime RAG: pure-Python BM25 over the bundled books grounds every refinement and answers direct passage queries
- Pluggable retriever: swap BM25 for an embedding backend via the `Retriever` protocol
- Offline, dependency-free, deterministic. No model download, no API calls, no third-party packages
- Every return value is JSON-serializable

# Installation

```bash
git clone https://github.com/nuclide-research/prompt-architect
cd prompt-architect
python -m prompt_architect.architect      # runs the worked example
```

Pure standard library. Requires Python 3.8 or later. No `pip install` step.

```bash
python -m unittest discover -s prompt_architect/tests -t .   # 23 regression tests
```

# API

### `analyze(prompt, context)`

Inspect a raw prompt against best practices. Returns a `PromptAnalysis`: discovered components, structural and behavioral issues, a risk assessment, and high-level suggestions. Never mutates the prompt.

```python
from prompt_architect import PromptArchitect

a = PromptArchitect()
context = {
    "goal": "Summarize AI/LLM infrastructure exposures for security engineers",
    "audience": "security engineers",
    "constraints": {"format": "markdown", "length": "medium"},
    "risk_points": ["hallucination", "security"],
    "domain": "ai_llm_infra_security",
    "inputs_type": "data_summary",
}
a.analyze("Summarize the exposures and tell engineers what to fix.", context)
```

### `refine(prompt, context, ground=True, k=2)`

Refine a raw prompt into a structured `PromptSpec` and rendered string. Returns `prompt_spec`, `rendered_prompt`, `analysis_before`, `analysis_after`, `change_log`, and `grounding`. The goal and each detected issue become retrieval queries against the corpus, k passages each. `ground=False` skips retrieval.

```python
r = a.refine("Summarize the exposures and tell engineers what to fix.", context)
r["rendered_prompt"]   # canonical [ROLE]/[GOAL]/[INPUTS]/[INSTRUCTIONS]/[OUTPUT]/...
r["change_log"]        # ["Added explicit role definition.", "Expanded instructions 1->3", ...]
r["grounding"]         # [{query, passages:[{book, chunk_id, score, text}]}]
```

### `generate_variants(prompt, context, n=3)`

Generate up to N optimization-focused variants from the refined base. Returns `base_spec` and `variants`, each a `PromptVariant` with its own spec and rendered string. Grounding is skipped here to stay lean.

```python
v = a.generate_variants("Summarize the exposures and tell engineers what to fix.", context)
[x["id"] for x in v["variants"]]   # ['cot', 'concise_strict', 'fewshot_skeleton']
```

### `ground(query, k=5, book=None)`

Direct RAG entry point. Returns the top-k book passages for a query, optionally scoped to one book slug.

```python
a.ground("how to specify a json output schema", k=3)
a.ground("context window token limits", k=2, book="farris-how-llms-work")
```

# Runtime RAG

The retriever is pure-Python BM25 over the per-book `_combined.md` extractions under `books/`. It chunks each book into overlapping word windows, builds a term-frequency and inverse-document-frequency index once, and scores every chunk per query. No model download, no network, no third-party packages.

```
books/*/_combined.md  ->  chunk (220-word windows, 40 overlap)  ->  BM25 index (tf, idf, avgdl)
                                                                          |
   query  ->  tokenize  ->  score every chunk  ->  top-k passages  <------+

   score = idf(t) * tf*(k1+1) / (tf + k1*(1 - b + b*dl/avgdl))     k1=1.5  b=0.75
```

The index builds on first use (about 0.6s, 3,225 chunks across 7 books) and caches to `books/.rag_cache.pkl` keyed on the source files' size and mtime (about 0.1s warm). Swap in an embedding backend by passing any object that implements `search(query, k, book)`:

```python
PromptArchitect(retriever=my_embedding_corpus)
```

# Data models

All models are `TypedDict`s in `models.py`, kept JSON-serializable.

```
PromptContext      goal, audience, model_name, model_context_tokens,
                   constraints{style,tone,format,length}, risk_points[],
                   domain, inputs_type

PromptComponents   role, goal, inputs_description, instructions[],
                   output_format, constraints{}, examples_present,
                   quality_check_present

PromptAnalysis     components, issues[], risk_assessment[],
                   suggested_improvements[]

PromptSpec         role, goal, inputs_contract, instructions[],
                   output_contract, constraints{}, quality_checks[], notes[]

PromptVariant      id, description, prompt_spec, rendered_prompt

Retrieved          book, chunk_id, score, text
GroundingResult    query, passages[Retrieved]
```

# Example

```
$ python -c "from prompt_architect import PromptArchitect; \
  a=PromptArchitect(); \
  print('\n'.join(f'[{r[\"score\"]:.1f}] {r[\"book\"]}' \
  for r in a.ground('json output schema guardrails', k=3)))"

[17.4] pai-designing-llm-applications
[15.1] esposito-programming-llms-azure-openai
[14.8] esposito-programming-llms-azure-openai
```

```
$ python -m prompt_architect.architect

=== REFINE ===
change_log:
  Added explicit role definition.
  Specified markdown output with explicit structure.
  Expanded instructions from 1 to 3 ordered steps.
  Introduced a quality check to reduce format drift and hallucination risk.

=== GROUNDING (RAG over books) ===
Q: No explicit ROLE; model may infer an inconsistent persona.
  [16.83] bhatti-docs-for-developers   ...a persona named "Charles" that represents them...
```

# Book corpus

Seven O'Reilly titles ground the design. Full Markdown is bundled under `books/` (per-chapter plus `_combined.md`), extracted with [colophon](https://github.com/nuclide-research/colophon). See [`books/MANIFEST.md`](books/MANIFEST.md) for the URN-to-concern map. The heuristics align with these texts; the RAG layer retrieves from them at runtime.

# Scope

prompt-architect is a local library. It analyzes and rewrites prompt text and retrieves from a local corpus. It makes no network calls, sends no telemetry, and downloads no models. The bundled corpus is the only data it reads. It does not call an LLM; it produces prompts for one.

# Our other projects

- [tome](https://github.com/nuclide-research/tome) — canonical AI/ML platform corpus and passive fingerprinter
- [aimap](https://github.com/nuclide-research/aimap) — AI/ML infrastructure fingerprint scanner
- [colophon](https://github.com/nuclide-research/colophon) — extract an O'Reilly Learning book to local Markdown
- [BARE](https://github.com/nuclide-research/BARE) — semantic exploit-module ranking over scanner findings
- [herald](https://github.com/nuclide-research/herald) — HTTP auth-posture probe

# License

MIT. Part of the NuClide toolchain. Contact: [nuclide-research.com](https://nuclide-research.com)
