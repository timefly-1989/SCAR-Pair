# SCAR-Pair reproduction

This directory rebuilds the experiments from the public Mem2ActBench source
corpora. It intentionally separates values recomputed from public records from
values copied into the manuscript tables.

The public source commit used by the current run is recorded in every output
manifest. The three source JSONL files contain 19,679 rows. Eighteen rows have
no textual turn content; excluding them gives the manuscript's 19,661-document
memory bank exactly. Retrieval evaluation excludes the nine tasks whose
`source_conversation_ids` list is empty, leaving 391 tasks.

## Environment

```bash
uv venv reproduction/.venv --python 3.12
uv pip install --python reproduction/.venv/bin/python -r reproduction/requirements.txt
```

The reported run uses Python 3.12.14 with the pinned package versions in
`requirements.txt`. The Python version and installed package manifest are
recorded in the reproduction outputs.
Inference scripts target Apple Silicon MPS, as used for this run. Porting them
to CUDA or CPU requires a device change and fresh validation; identical outputs
across devices have not been established.

## Run

From the repository root:

```bash
git clone https://github.com/Cantaloupe-M/Mem2ActBench.git reproduction/vendor/Mem2ActBench
git -C reproduction/vendor/Mem2ActBench checkout b00726940b5abbe9bd324bdd7a2cb272f5c62a29
reproduction/.venv/bin/python reproduction/scar_pair_reproduce.py audit
reproduction/.venv/bin/python reproduction/scar_pair_reproduce.py stage1
PYTORCH_ENABLE_MPS_FALLBACK=1 reproduction/.venv/bin/python \
  reproduction/stage2_evaluate.py score --include-broad
reproduction/.venv/bin/python reproduction/stage2_evaluate.py evaluate
reproduction/.venv/bin/python reproduction/statistics_reproduce.py
reproduction/.venv/bin/python reproduction/slice_reproduce.py
reproduction/.venv/bin/python reproduction/depth_grid_reproduce.py all
reproduction/.venv/bin/python reproduction/capacity_learning_reproduce.py capacity
reproduction/.venv/bin/python reproduction/capacity_learning_reproduce.py learning
reproduction/.venv/bin/python reproduction/coverage_reproduce.py
reproduction/.venv/bin/python reproduction/graph_probe_reproduce.py
reproduction/.venv/bin/python reproduction/reader_reproduce.py audit
reproduction/.venv/bin/python reproduction/reader_reproduce.py run
reproduction/.venv/bin/python reproduction/reader_reproduce.py summarize
reproduction/.venv/bin/python reproduction/reader_parser_audit.py
reproduction/.venv/bin/python reproduction/pairwise_controls.py
reproduction/.venv/bin/python reproduction/publication_analysis.py
reproduction/.venv/bin/python reproduction/publication_extra_summaries.py
reproduction/.venv/bin/python reproduction/publication_integrity.py
reproduction/.venv/bin/python scripts/build_publication_tables.py
reproduction/.venv/bin/python scripts/rebuild_manuscript_figures.py
reproduction/.venv/bin/python scripts/build_metrics_by_k.py
```

`stage1` reconstructs the bank, embeds it with the pinned BGE-small checkpoint,
builds BM25/dense/RRF rankings, computes the broad and narrow candidate unions,
and saves lightweight features. `score` runs both MiniLM cross-encoder views on
the narrow pool and, with `--include-broad`, on the matched broad control pool.
`evaluate` fits every pointwise model and SCAR-Pair inside each fold for all four
validation protocols. It writes fold assignments, per-task rankings and metrics,
fit diagnostics, and aggregate results under `reproduction/output/`.

The later commands regenerate every manuscript diagnostic that can be defined
from the released corpus. `depth_grid_reproduce.py` first extends the stored
lightweight features to depth 400 and then evaluates all 16 depth cells while
reusing the already computed depth-200 CE scores. `reader_reproduce.py` writes
one append-only trace after every deterministic Qwen generation, so an
interrupted multi-hour run can be resumed without losing completed outputs.
`publication_analysis.py` is the sole manuscript-authoritative entry point for
paired intervals and randomization tests. It uses shared-source-component
bootstrap resampling and joint component sign flips. The older
`reader_statistics.py` is retained only as a clearly labelled task-level
sensitivity analysis and is neither run by this workflow nor packaged as a
submission result.

The external LoCoMo and long-encoder controls use the separately pinned public
LoCoMo checkout and locally cached Hugging Face revisions:

```bash
git clone https://github.com/snap-research/locomo.git reproduction/vendor/locomo
git -C reproduction/vendor/locomo checkout 3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376
reproduction/.venv/bin/python reproduction/locomo_reproduce.py bge-small --style timestamped --tokenizer unicode
reproduction/.venv/bin/python reproduction/locomo_reproduce.py mgte --style timestamped --tokenizer unicode
reproduction/.venv/bin/python reproduction/locomo_reproduce.py bge-m3 --style timestamped --tokenizer unicode
reproduction/.venv/bin/python reproduction/long_encoder_baselines.py mgte --schema-style full
reproduction/.venv/bin/python reproduction/long_encoder_baselines.py bge-m3 --schema-style full
reproduction/.venv/bin/python reproduction/remaining_diagnostics.py
reproduction/.venv/bin/python reproduction/validate_outputs.py
```

Run the final diagnostics and validation after the LoCoMo controls: they require
all three timestamped/Unicode prediction files. The parser audit records the
effect of rejecting duplicate JSON keys separately from the frozen primary
Reader metrics.

After validation, run `reproduction/.venv/bin/python reproduction/build_bundle.py`
to package the experiment scripts, per-task predictions, validated reports,
figure source data, figure/table generators, manuscript sources, and generated
tables. The ZIP includes a SHA-256 manifest with installed package versions and
source commits; the builder verifies every archived file. Model weights,
third-party source corpora, caches, and logs are excluded. Download the pinned
sources and rerun the commands above to regenerate intermediate caches needed by
prompt reconstruction and model evaluation.

## Explicit replication conventions

This suite is an independent public-data reconstruction rather than historical
code from an earlier unreleased run. The following conventions define the
current manuscript results and must be preserved for exact regeneration:

- conversation text is the newline-joined nonempty `content` fields, without
  tool-call metadata;
- BM25 tokenization uses `lower().split()`, matching the public benchmark's own
  BM25 implementation, with rank-bm25 defaults (`k1=1.5`, `b=0.75`);
- score ties are broken by memory-bank position;
- BM25 min-max scaling and reciprocal ranks use the full memory bank;
- schema views use the literal labels implemented in this script.
- unavailable cross-encoder scores use the within-view minimum minus `1e-6`;
- pointwise models use an unweighted training-candidate `StandardScaler`, then a
  class-balanced logistic regression with `C=1` and an intercept;
- pairwise scaling uses the unweighted population standard deviation of all
  mirrored fold-training differences; the classifier is a no-intercept,
  query-weighted logistic regression with `C=1`.

The validation reports cross-check the reconstructed data, candidates,
predictions, statistics, tables, figures, and manuscript claims before the
bundle is built.
