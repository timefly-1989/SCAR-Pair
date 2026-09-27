# SCAR-Pair major-revision source data

This archive adds post-review controls to the fixed primary reproduction package at Git commit `71896b9c9b701425526325e068880850ed5a7bd7` (`https://github.com/timefly-1989/SCAR-Pair/tree/71896b9c9b701425526325e068880850ed5a7bd7`). The complete archive is published under the versioned public tag `major-revision-2026-09-28-v4` (`https://github.com/timefly-1989/SCAR-Pair/tree/major-revision-2026-09-28-v4/major_revision_2026_v4`). It is not a second independent benchmark. The primary public package supplies the original source-corpus reconstruction, feature cache builder, fold definitions, and frozen reader outputs. Model weights and third-party corpora are not redistributed here.

## New experiments and files

- `reproduction/revision_controls.py`: full baseline refit, query-CE replay, and no-insertion refit on frozen stage-two features.
- `reproduction/revision_mismatch.py`: five independently seeded, prespecified different-tool donor permutations; the current request and candidate pools stay fixed, and 48,743 CE pairs are rescored per permutation (243,715 in total).
- `reproduction/revision_sensitivity.py`: one-factor C and unavailable-CE-floor offset checks.
- `reproduction/revision_reader_audit.py`: per-task, precedence-defined error categories and a post-generation selected-tool-name lock on frozen reader outputs. This is not constrained decoding.
- `reproduction/revision_constrained_reader.py`: a selected-tool-constrained Qwen2.5-1.5B control in which the interface fixes the tool identity and the reader generates only the argument object before deterministic wrapping.
- `reproduction/revision_multireader.py`: matched four-context replications with pinned 4-bit Qwen2.5-3B-Instruct and Phi-3.5-mini-instruct MLX checkpoints.
- `reproduction/revision_statistics.py`: 10,000 shared-source-component bootstrap resamples, 50,000 joint component sign flips, and Holm adjustment within each three-metric family.
- `reproduction/revision_reader_extensions_statistics.py`: component-aware paired inference for the constrained and multi-reader experiments.
- `reproduction/revision_schema_components.py`: matched leave-one-schema-component-out refits and a request-only scoring refit on the common broad candidate pool.
- `reproduction/revision_grammar_reader.py`: token-level argument JSON Schema constraints with LM Format Enforcer 0.11.3, unchanged exact-match scoring, and a deterministic schema-gated local dispatcher that returns auditable receipts without calling an external API.
- `scripts/build_revision_tables.py`, `scripts/build_reader_extensions_table.py`, `scripts/plot_revision_controls.py`: table and vector-figure generators from the JSON reports.
- `reproduction/output/revision_*`: per-task predictions, all 243,715 placebo CE logits with permutation seeds, candidate IDs, and donor mappings, reader records, aggregates, and paired statistics. `publication_reader_predictions.jsonl` and `retrieval_predictions.jsonl` are copied from the pinned primary reproduction package for audit convenience.
- `tables/generated/revision_*.tex`, `figures/revision_controls.pdf/.svg/.png`: derived submission artifacts.
- `supplementary_methods.tex`: exact lightweight-feature coordinates, cross-encoder missing-value equations, and the tables moved from the main manuscript during presentation streamlining.

## Interpretation limits

The Query-CE replay reuses cached request CE scores, but a redundant second pass would have the same 18-dimensional input and equal CE-call count as the schema-view model. The five mismatched-schema runs change only the schema CE view; selected-schema lightweight features and candidate access remain. Both no-insertion refits skip zero-hit **training** tasks in their own ordinary pools (broad SCAR-Pair; strict request-only Query-Pair) but evaluate all test tasks. The name lock changes an output name only after generation, leaving arguments fixed; it is a diagnostic counterfactual. The selected-tool-constrained control changes the generation task: the reader emits an argument object and cannot select a different tool, but it does not impose a grammar over the argument keys or values. The grammar control does impose the translated JSON Schema token by token and fixes the tool identity, but its local dispatcher only demonstrates structural acceptance and a deterministic returned receipt. It does not invoke the unavailable benchmark APIs, reproduce their side effects, or turn a semantically wrong schema-valid call into an exact result. The component ablations operate on pooled information groups and cannot identify a particular parameter field as causally important. The larger-reader runs use 4-bit quantized checkpoints and are reader-family sensitivity analyses. Nonsignificant contrasts do not establish equivalence.

The original primary run used Python 3.12.14, NumPy 2.0.2, scikit-learn 1.6.1, and PyTorch 2.8.0. Post-review refits used Python 3.13.9, NumPy 2.5.3, scikit-learn 1.7.2, and PyTorch 2.13.0 on Apple Silicon MPS; the baseline refit reproduced the primary SCAR-Pair aggregate to three decimals. The grammar run additionally used `lm-format-enforcer==0.11.3`, the pinned primary Qwen checkpoint, greedy decoding, and the original 512-token output limit. The additional quantized readers used `mlx-lm==0.26.4` and `mlx==0.32.2` with batch size one; their reports pin the complete model revisions. Exact binary output may differ across package versions or devices.

## Rebuild

Place these files alongside the pinned primary reproduction package and run from the manuscript root:

```bash
python reproduction/revision_controls.py
python reproduction/revision_mismatch.py
python reproduction/revision_sensitivity.py
python reproduction/revision_reader_audit.py
python reproduction/revision_statistics.py
PYTHONPATH=reproduction python reproduction/revision_constrained_reader.py run
PYTHONPATH=reproduction python reproduction/revision_multireader.py run --model qwen25_3b
PYTHONPATH=reproduction python reproduction/revision_multireader.py run --model phi35_mini
PYTHONPATH=reproduction python reproduction/revision_reader_extensions_statistics.py
PYTHONPATH=reproduction python reproduction/revision_schema_components.py
PYTHONPATH=reproduction python reproduction/revision_grammar_reader.py run
python scripts/build_revision_tables.py
python scripts/build_reader_extensions_table.py
PYTHONPATH=/path/to/nature-figure/scripts python scripts/plot_revision_controls.py
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build supplementary.tex
```

`revision_mismatch.py` requires the pinned MiniLM cross-encoder weights and MPS device as written; edit and revalidate the device setting for other hardware. The archived JSONL files are the actual per-task outputs used for the revised tables and figure, so the paper can be audited without rerunning neural inference.
