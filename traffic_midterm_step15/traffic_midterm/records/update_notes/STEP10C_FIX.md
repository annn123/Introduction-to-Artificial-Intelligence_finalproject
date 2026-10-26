# Step 10C

Extraction is now a hybrid method, not unrestricted LLM claim generation.
The local LLM extracts case fields. Conservative source rules create recognised
behaviour and responsibility candidates with original quotes, known actors,
polarity and explicitly linked premises. Only unknown subject roles request a
second constrained LLM call. Every result requires human review.

No model-generated text, quote, predicate or premise index is accepted in this
candidate path. Complex clauses, multiple behaviours, conditional language and
unsupported third-party reports are flagged for manual handling. Unrecognised
claims can be omitted: this method does not support arbitrary Chinese syntax,
coreference, nested quotations or general semantic entailment. No recognised
candidate means an explicit manual-input error. Known case-field extraction
still depends on the real local model and can fail validation.

The current example should have A-unlicensed alleged by B, A-liability alleged
by B, with a separate system causal check question. Dashcam mention is not
verified evidence. Rules are development decisions, not model accuracy gains.
Use held-out independent examples and labels to evaluate generalisation.

134 automated tests passed, including reversed roles, other violations,
negation, wrong model keys, unsupported syntax and pipeline integration.
These use transport stubs; no real Ollama inference was run here.
