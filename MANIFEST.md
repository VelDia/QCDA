# QCDA: every file and why it is here

How to use it is in `README.md`.

## Inputs and provenance
| file | role |
|---|---|
| `openalex_papers.csv` | The quantum computing harvest: 544,156 rows, 3 Sep 2026 snapshot. Read by `keyword_arms.py`. |

## The pipeline: `data_availability_pipeline/v2/`
| path | why it is necessary |
|---|---|
| `dap/collect.py` | **New.** Stage 0: harvests any subject from OpenAlex (`--subject`, `--search-in`, `--year-min/max`, `--filter`); resumable; records the query. |
| `dap/schema.py` | The only vocabulary (scheme 2.0.1). Gained `Arm.UNSPECIFIED`, the default for new corpora. |
| `dap/sources.py`, `verify.py`, `store.py`, `batch.py` | Retrieval, link verification, SQLite state, Message Batches. Release change: the contact address is optional and sent only to Crossref and Unpaywall, never in a User-Agent. |
| `dap/redact.py` | **New.** Masks addresses and keys in error text before it is stored (`Store.fail`) or raised (`sources._get`). |
| `dap/classify.py` | Constrained classification. Release changes: prompt `v2.3.0` (subject-neutral, no arm in the message; no coding rule changed); `tool_choice: auto` and more output room on models that reject forced tool use (Opus 5.5, Sonnet 5.5, Fable 5.1/5.2); a price table for current models so `--budget` can stop them. |
| `dap/pipeline.py` | The CLI. Release changes: `collect` subcommand; `--arm-map`/`--arm` removed; `enrol --year-max` no longer defaults to 2025; `--model` pins the model in `--mode api`; `DAP_TEXT_CACHE` saves the exact retrieved text (benchmark only). |
| `tests/` | 297 offline tests: the research suite (guards, integration, pipeline) plus `test_release.py` for the changes above. |
| `requirements.txt` | Dependencies, minimum versions (matplotlib dropped with `plots.py`). |
| `requirements.lock` | **New.** The exact versions both test suites passed with on a fresh Python 3.12 install, 10 Nov 2026. |
| `dap_frame.sqlite` | The case-study state: 41,395 papers. Copied with SQLite's backup API from the 28 Oct version. On 10 Nov 2026 the contact addresses in 357 `failures.message` rows were replaced with `<redacted>` by `dap/redact.py`, and the file was vacuumed; nothing else changed. The original in `scraper/` still holds them. |
| `keyword_qc_papers.csv`, `keyword_arms.py` | The case study's scope: the 38,539-row enrol input and the title-keyword rule that selected it. |
| `scratch/h.py`, `scratch/chk.py` | Manual mode: decided-rule record builders, and the gate to run before any `--import`. |
| `review_issues.md` | The decision log (S1–S220). |
| `../CLAUDE.md` | The rules the case study was run under. |

## Post-analysis: `post-analysis/`
| file | role |
|---|---|
| `review_app.py` | The blind human-review form (never shows the classifier's codes). |
| `review_metrics.py`, `merged.csv` | Classifier against the 70-paper human review. |

## Benchmark: `benchmark/` (new)
| file | role |
|---|---|
| `build_gold.py` | Human-labelled validation set from the human review; frame strata; the as-built predictions. |
| `gold/`, `predictions/` | Its output: 69 gold papers, strata population, as-built codes. |
| `run_pipeline.sh` | The pipeline run unattended on the gold papers: `prepare` once, `classify <model>` per model. |
| `baseline.py` | LLM baselines (Anthropic SDK, or any OpenAI-compatible endpoint): full text plus class definitions, or the pipeline prompt as an ablation. |
| `evaluate.py` | Accuracy, kappa, PABAK, F1, reweighted accuracy, McNemar + bootstrap + Holm, manual-effort and cost tables. |
| `common.py` | Shared vocabulary, scoring rules and the baseline prompt. |
| `tests/` | 11 offline tests, including an end-to-end run against a fake model. |
| `texts/`, `runs/`, `results/` | Created by the runs. `texts/` is publisher full text: is kept local. |

## Left out
-`out*/` exports, per-batch `scratch/` files, `cross_validation/` (the internal qDATA set for cross validation).
