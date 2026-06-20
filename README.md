# PromptArchitect

Expert prompt-engineering assistant. Treats prompts as structured artifacts
(role, goal, input/output contracts, constraints, self-checks) and **analyzes**,
**refines**, and **diversifies** them into production-ready forms.

Pure string-processing and heuristics — no external services, no API calls.

## Install / use

```python
from prompt_architect import PromptArchitect

architect = PromptArchitect()

prompt = "Summarize the AI/LLM exposures we found and tell engineers what to fix."
context = {
    "goal": "Summarize AI/LLM infrastructure exposures for security engineers",
    "audience": "security engineers",
    "model_context_tokens": 8000,
    "constraints": {"format": "markdown", "length": "medium", "tone": "neutral"},
    "risk_points": ["hallucination", "security"],
    "domain": "ai_llm_infra_security",
    "inputs_type": "data_summary",
}

analysis = architect.analyze(prompt, context)            # PromptAnalysis
refined  = architect.refine(prompt, context)             # spec + rendered + change_log
variants = architect.generate_variants(prompt, context)  # cot / concise_strict / fewshot_skeleton
```

Run the worked example:

```
python -m prompt_architect.architect
```

## API

| Method | Returns |
|--------|---------|
| `analyze(prompt, context)` | `PromptAnalysis` — components, issues, risk_assessment, suggested_improvements |
| `refine(prompt, context)` | dict — `prompt_spec`, `rendered_prompt`, `analysis_before`, `analysis_after`, `change_log` |
| `generate_variants(prompt, context, n=3)` | dict — `base_spec`, `variants` (each a `PromptVariant`) |
| `ground(query, k=5, book=None)` | list of `Retrieved` — top passages from the book corpus (RAG) |

All returns are JSON-serializable.

## Runtime RAG over the books

`refine()` grounds every refinement against the bundled corpus: the goal and each
detected issue are used as retrieval queries, and the supporting book passages
come back under `refined["grounding"]` (a list of `GroundingResult`). `ground()`
is the direct retrieval entry point.

```python
a = PromptArchitect()
a.ground("how to specify a json output schema", k=3)          # all books
a.ground("context window token limits", k=2,
         book="farris-how-llms-work")                          # scoped to one book

refined = a.refine(prompt, context)        # grounding on by default
refined["grounding"]                       # [{query, passages:[{book, chunk_id, score, text}]}]
a.refine(prompt, context, ground=False)    # skip retrieval
```

The retriever is **pure-Python BM25** over the per-book `_combined.md` files — no
model download, no network, no third-party packages. It builds on first use
(~0.6s, 3,225 chunks) and caches to `books/.rag_cache.pkl` (~0.1s warm). Swap in
an embedding backend by passing any object implementing the `Retriever` protocol
(`search(query, k, book)`):

```python
PromptArchitect(retriever=my_chroma_backed_corpus)
```

## Variants

- **`cot`** — chain-of-thought; forces a `Reasoning` then `Final Answer` split.
- **`concise_strict`** — format-strict, no visible reasoning; deterministic, machine-friendly output.
- **`fewshot_skeleton`** — few-shot demonstration skeleton for consistent structure.

## Structure

```
prompt_architect/
  __init__.py     # package exports
  models.py       # PromptContext, PromptComponents, PromptAnalysis, PromptSpec, PromptVariant
  architect.py    # PromptArchitect + heuristics, rendering, variant builders
  corpus.py       # BookCorpus — pure-Python BM25 RAG retriever over books/
  books/          # reference corpus — see books/MANIFEST.md
```

## Design grounding

Behavior follows best practices from seven O'Reilly titles (Esposito, Pai,
Farris, Cronin, Raschka, Chinchilla, Bhatti et al.). Full text is bundled under
`books/` for reference and for a future retrieval/eval layer — see
[`books/MANIFEST.md`](books/MANIFEST.md). The heuristics are self-contained and
do not load that text at runtime.
