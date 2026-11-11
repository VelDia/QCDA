# QCDA: where do research papers keep their data?

QCDA is a pipeline that takes a field of research as input, collects the papers 
published on the topic via OpenAlex, and records for each paper whether the data 
it uses or produces can actually be reached, and how. 
We have tested it on **quantum computing** (41,395 papers enrolled), 
and with the benchmark that compares the pipeline to plain LLM classification.

---

## What it records

Every paper is coded on two independent axes:

- **Axis A, external data.** Data that existed before the paper and that the
  paper analyses: a dataset, a benchmark, real-world records, prior artifacts.
- **Axis B, self-generated data.** Data the paper produced itself: measurements,
  simulation output, generated instances, training runs, its own collected data.

Each axis gets exactly one of seven cases:

| # | token | meaning |
|---|---|---|
| 1 | `not_applicable` | the axis does not apply: no data analysed (A) or generated (B) |
| 2 | `openly_accessible` | a concrete identifier is given (DOI, URL, accession, repository) |
| 3 | `access_not_provided` | the material is identifiable, but no route to it is given |
| 4 | `insufficiently_specified` | the material cannot even be identified |
| 5 | `access_unresolvable` | a route is given, and it cannot be reached when checked |
| 6 | `on_request` | offered upon reasonable request through contacting the authors |
| 7 | `explicitly_restricted` | stated that it cannot be shared |

Case 5 is never asserted from text. Only the verification step writes it, after
the link has actually been checked.

Alongside the cases, every row records the identifiers found and the quotes as evidence 
that justify each decision. Each quote is checked against the paper's
text before the row is stored. Every row also records a confidence level and a
`needs_human_review` flag.

## How a paper moves through it

```
collect → enrol → metadata → retrieve → classify → verify → export
```

| stage | what it does |
|---|---|
| `collect` | searches OpenAlex for your query and writes the corpus CSV |
| `enrol` | loads the CSV into the database, filters by year, removes duplicate DOIs |
| `metadata` | fetches OpenAlex, Crossref and DataCite records; checks every cited DOI against DataCite |
| `retrieve` | finds a legal full-text copy (publisher TDM, open-access locations, arXiv, Unpaywall), checks it is the right paper and complete, and cuts it to the passages about data |
| `classify` | the model fills a schema generated from the coding scheme; every quote must appear in the passages. A paper with no text and no metadata evidence is never sent to a model: it is stored as flagged (see step 7) |
| `verify` | resolves every identifier the classifier recorded; a dead, paywalled or login-gated route becomes `access_unresolvable` |
| `export` | writes `analysis.csv`, `evidence.csv`, `review_queue.csv` from the database |

Everything lives in one SQLite file (`das.sqlite` below). Every stage is resumable: 
stop it, run the same command again, and it continues where it stopped. 
Running a stage twice never redoes finished papers, unless recoding is intended.

---

## Run it on your own topic

The steps below take a topic and store the classification results and dataset links in
`das.sqlite`. They were tested as written on 10 Nov 2026, from a fresh install
with no keys set, on a 25-paper pilot (the step 3 command with
`--max-papers 25`); the timings and counts quoted come from that run. With no
`ANTHROPIC_API_KEY`, that run stopped at classification: a live `--mode api`
run is the one step not exercised there (see step 7a).

### 0. What you need

| | needed for | without it |
|---|---|---|
| Python 3.10+ | everything | the macOS system Python (3.9) fails on import |
| `ANTHROPIC_API_KEY` | classification in `api` or `batch` mode | use `manual` mode (step 7c), or stop after retrieval |
| `DAP_CONTACT_EMAIL` | Crossref's faster pool; the Unpaywall step | Crossref still works; Unpaywall is skipped, so fewer papers get full text |
| `OPENALEX_API_KEY` (free) | large harvests | fine for a pilot; OpenAlex budgets keyless list queries per day |
| TDM keys (`ELSEVIER_TDM_KEY`, `WILEY_TDM_TOKEN`, `SPRINGER_TDM_KEY`) | full text of subscription journals | only open copies are read; the rest are flagged for a human |

Nothing else is required. The pipeline never bypasses a paywall, login or
CAPTCHA; a paper it cannot read legally is flagged, not guessed.

### 1. Install

```bash
cd data_availability_pipeline/v2
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock          # the exact versions the tests passed with
python -m pytest tests/ -q                # expect: 297 passed
```

`requirements.txt` has the same packages with minimum versions only. Do not run
`pip install dap`: that is an unrelated PyPI package. `dap/` runs in place as
`python -m dap.pipeline`, so **run every command below from
`data_availability_pipeline/v2/` with the venv active.**

### 2. Set the environment

```bash
export DAP_CONTACT_EMAIL="project@institution.edu"   # optional, see below
export OPENALEX_API_KEY="..."                        # optional; free at openalex.org/settings/api
export ANTHROPIC_API_KEY="..."                       # for --mode api / batch
# optional
export ELSEVIER_TDM_KEY=... WILEY_TDM_TOKEN=... SPRINGER_TDM_KEY=...
export DAP_GROBID_URL="http://localhost:8070"        # GROBID, for section-level locators
export DAP_PDF_DIR="/path/to/pdfs"                   # PDFs you obtained yourself: <DOI with / as _>.pdf or <W-id>.pdf
```

**Where `DAP_CONTACT_EMAIL` goes.** To two services only: Crossref, as
`mailto=` (its faster "polite pool"), and Unpaywall, as `email=` (its terms
require one). It is not in the User-Agent, is not sent to OpenAlex (which
identifies callers by `OPENALEX_API_KEY`) or to any publisher, repository or
dataset host, and is never written to the database or the logs: error messages
are passed through `dap/redact.py` before they are stored. A shared project
address works as well as a personal one. When it is unset, every stage prints
one line saying so; that line is not an error.

Every command below names the database with `--db das.sqlite`. `--db` goes
before the command (`python -m dap.pipeline --db das.sqlite stage ...`).
Leave it out and the database is `dap.sqlite` in the current directory.

### 3. Collect the corpus

```bash
python -m dap.pipeline collect --subject "protein folding" \
    --year-min 2021 --year-max 2026 --filter type:article --out corpus.csv
```

- `--subject` is a topic of interest or the research field. 
It takes OpenAlex search syntax: `"exact phrase"`, `AND`, `OR`, `NOT`. 
A subject searched in titles or abstracts may not contain a comma; use `OR`.
- `--search-in` is `title` (narrowest), `title_and_abstract` (default) or
  `fulltext` (broadest, noisy).
- `--filter` adds any OpenAlex filter. `type:article` is worth keeping: without
  it, book chapters and encyclopaedia entries come in, and they have no data
  statement to find. Without it, an earlier 20-paper pilot got full text for 5
papers; with it, the 25-paper pilot got full text for 19.
- `--max-papers 25` makes a pilot. **Run steps 3–9 on a pilot first.**

You will see `N / total works` lines, then `N works in corpus.csv`. The exact
query is saved to `corpus.csv.query.json`. An interrupted harvest resumes when
you rerun the same command; rerunning a *different* query into the same file is
refused, so give each query its own `--out`.

### 4. Enrol

```bash
python -m dap.pipeline --db das.sqlite enrol --input corpus.csv
```

Prints `enrolled N new` and `"enrolled/pending": N`. Enrolling the same CSV
again adds nothing; a DOI already in the database is not enrolled twice.

**The year window.** Every `stage` command works only on papers from 2021
onward unless told otherwise. If your corpus includes earlier years, add
`--year-min 0` (no window) or `--year-min <year>` to **every** `stage` command
in steps 5–8, or those papers are silently left at their current stage.
`--oldest-first` changes the order papers are processed in, nothing else.

### 5. Metadata

```bash
python -m dap.pipeline --db das.sqlite stage metadata --workers 16
```

No key needed. Pilot: 25 papers in 9 s. Ends with
`"metadata/pending": N`.

### 6. Retrieve full text

```bash
python -m dap.pipeline --db das.sqlite stage retrieve --workers 8
```

Pilot: 25 papers in 27 s. Lines starting `MuPDF error` come
from malformed PDFs and are harmless. Ends with `"retrieved/pending": N`.

Check how many papers got text before paying for classification:

```bash
sqlite3 das.sqlite "SELECT json_extract(meta_json,'$.retrieval') AS got,
                           json_extract(meta_json,'$.text_source') AS source, count(*)
                    FROM papers GROUP BY 1, 2"
```

`full_text` means a complete text was read; `metadata_only` means none was
found. In the pilot, with no email and no TDM keys, 19 of 25 papers had full
text. Setting `DAP_CONTACT_EMAIL`, adding TDM keys, or putting PDFs you are
entitled to in `DAP_PDF_DIR` and rerunning this step raises that.

### 7. Classify

**Papers without text are not classified by a model.** When a paper has no
text and no data statement or dataset link in its metadata, the classify step
stores it at once with both axes `not_applicable`, `needs_human_review = 1`,
and `review_reason` "no retrievable text, no data-availability statement and
no dataset link". This costs nothing and happens in every mode. Those
`not_applicable` codes mean "not read", not "no data": leave flagged rows out of
any rate, or settle them by hand (step 10).

#### 7a. Shakedown (always first)

```bash
python -m dap.pipeline --db das.sqlite stage classify --mode api \
    --model claude-opus-5-5 --no-escalate --limit 25 --budget 2.00
python -m dap.pipeline --db das.sqlite export --out shakedown/
```

Read `shakedown/review_queue.csv` and a sample of `shakedown/analysis.csv`
before going on.

- `--model` pins one model, so every row of the corpus comes from one
  protocol; `model_id` records it. `claude-haiku-5-5` is far cheaper,
  `claude-opus-5-5` the most capable.
- `--no-escalate` keeps it to that model. Without `--model` and
  `--no-escalate`, a low-confidence answer is re-asked on a stronger model
  (`DAP_MODEL_FAST`, `DAP_MODEL_STRONG`).
- `--budget` stops the run at a dollar amount, costed from recorded tokens.
  `status` shows the running total.

#### 7b. The whole corpus, with an API key

```bash
# synchronous: results as they come
python -m dap.pipeline --db das.sqlite stage classify --mode api \
    --model claude-opus-5-5 --no-escalate --budget 50

# or the Message Batches API, half price: submit, then collect once it has finished
python -m dap.pipeline --db das.sqlite stage classify --mode batch \
    --model claude-opus-5-5 --budget 50 --flush
python -m dap.pipeline --db das.sqlite stage classify --mode collect    # rerun until nothing is open
```

**Batch mode needs `--flush` for any corpus under 5,000 papers.** Requests are
spooled into files of 5,000 (`DAP_BATCH_SIZE`); without `--flush` a partial
file is not sent and the command prints `nothing ready to submit`. Batch mode
prints its estimated cost first and refuses to submit above `--budget`.
A batch finishes within 24 hours, usually much sooner; `status` counts the open ones.

#### 7c. The whole corpus, without an API key (manual mode)

An agent or a person classifies exported prompts; the import puts every answer
through the same schema check and quote check as the API path.

```bash
export DAP_MANUAL_CODER="claude-opus-5-5"     # who classifies: stored as model_id "manual:<coder>"
python -m dap.pipeline --db das.sqlite stage classify --mode manual --export prompts.jsonl --limit 50
#   ... classify prompts.jsonl into results.jsonl (format below) ...
python scratch/chk.py prompts.jsonl results.jsonl das.sqlite    # offline check: must end with FAILURES: 0
python -m dap.pipeline --db das.sqlite stage classify --mode manual --import results.jsonl
```

- `prompts.jsonl`: the first line holds the system prompt, the output schema
  and instructions; each further line is one paper (`custom_id`, `prompt`).
- `results.jsonl`: one line per paper,
  `{"custom_id": "<as in prompts.jsonl>", "classification": {...}}`, where
  `classification` matches `output_schema` exactly. Tokens must come from the
  schema's enums and every quote must be copied verbatim from the prompt's
  passages, not from its title line.
- **A rejected line costs the paper an attempt, and three rejections retire
  it.** That is why `scratch/chk.py` runs the real gate offline first. A retired
  paper shows as `failed` in `status`; `retry W123 --reason "..."` reinstates it.
- `--limit` counts the papers taken from the queue, including those stored at
  once for lack of text, so an export can hold fewer prompts than `--limit`.
- Keep one coder per corpus. If the coding model changes, give it a new
  `DAP_MANUAL_CODER`, so the rows stay distinguishable.

Repeat export → classify → check → import until `status` shows no
`retrieved/pending`.

### 8. Verify

```bash
python -m dap.pipeline --db das.sqlite stage verify --workers 16
```

Checks every identifier the classifier recorded. No key needed. A paper ends
at `"verified/done"`.

### 9. Results

```bash
python -m dap.pipeline --db das.sqlite status
python -m dap.pipeline --db das.sqlite export --out out/
```

`status` prints the count at each stage and the money spent. A finished corpus
has every paper at `verified/done`, apart from any `failed` (out of attempts)
or `dead` (OpenAlex has no record).

| file | contents |
|---|---|
| `out/analysis.csv` | one row per paper, 58 columns: both cases with confidence and rationale, declared and verified access, identifiers, data type, `needs_human_review`, `review_reason`, `scheme_version`, `prompt_version`, `model_id`, retrieval source |
| `out/evidence.csv` | every quote behind a decision, with its section |
| `out/review_queue.csv` | the papers a human must look at, and why |

The same results, read straight from `das.sqlite`:

| table | holds |
|---|---|
| `papers` | one row per paper: `stage`, `status`, and once classified, the full record as JSON in `record_json` |
| `evidence` | the quotes |
| `verifications` | each identifier checked, with its HTTP outcome |
| `usage` | tokens and cost per model call |
| `failures` | every error and rejection, with its message |

```bash
# the case distribution on each axis, unflagged papers only
sqlite3 das.sqlite "SELECT json_extract(record_json,'$.a_case') AS a,
                           json_extract(record_json,'$.b_case') AS b, count(*)
                    FROM papers WHERE status='done'
                      AND json_extract(record_json,'$.needs_human_review') = 0
                    GROUP BY 1, 2 ORDER BY 3 DESC"
```

### 10. Human review

A flagged paper is settled with `adjudicate`. It is validated against the same
scheme, and a changed identifier is re-verified:

```bash
python -m dap.pipeline --db das.sqlite adjudicate W123 --set b_case=on_request \
    --coder human:yourname --reason "statement offers data on request" --yes
```

The other repair commands are `requeue` (wrong copy retrieved), `retry` (out of
attempts), `recode`, `exclude` (duplicates, notices), `outdated` and
`preprint-das`. `python -m dap.pipeline <command> -h` explains each.

### When something goes wrong

| you see | it means |
|---|---|
| `ANTHROPIC_API_KEY is not set. This stage needs it` | set the key, or use `--mode manual` |
| `DAP_CONTACT_EMAIL is not set: ...` | information only; see step 2 |
| `nothing ready to submit` | batch mode without `--flush` on a small corpus |
| `... belongs to a different query; delete it or use another --out` | `collect` was rerun with a changed query into the same file |
| `... exists with no cursor file` | that CSV is a finished harvest; use another `--out` |
| papers stay at a stage after a run | they are outside the year window (step 4), or `failed`; `status` shows which |
| `OpenAlex asked for a ...s wait: the daily quota is spent` | rerun later or set `OPENALEX_API_KEY`; the harvest resumes |
| `REJECTED — ...` on import | the line failed the schema or quote check; fix it, run `scratch/chk.py`, import again |

### Tests

```bash
python -m pytest tests/ -q                     # from data_availability_pipeline/v2: 297 tests
cd ../../benchmark && python -m pytest tests/ -q   # 11 tests
```

Both suites run offline, against fakes of every external service. They do not
exercise a live `--mode api` run; the 25-paper shakedown in step 7a is that
check.

---

## The quantum computing case study

| | |
|---|---|
| Harvest | OpenAlex `search="quantum computing"` with `filter=concepts.id:C121332964` (the OpenAlex concept "Quantum computing"), snapshot of 3 Sep 2026: 544,156 works (`openalex_papers.csv`). The same request through this pipeline is `collect --subject "quantum computing" --search-in fulltext --filter concepts.id:C121332964`. OpenAlex's index has moved since, so it does not redraw the same 544,156 rows (on 10 Nov 2026 it counted 649,850 up compared to 3 Sep 2026): the CSV is the record of the corpus |
| Scope | the title-keyword rule `kw-1` (`data_availability_pipeline/v2/keyword_arms.py`) keeps 38,539 quantum computing papers (`keyword_qc_papers.csv`); the 5,647-paper venue frame of 15 quantum journals was enrolled first |
| State | `dap_final.sqlite`: 41,395 papers enrolled, 19,415 fully processed (`status` lists the rest) |
| Coding window | 2021–2026; 2026 is a partial year by construction |
| Decisions | every ruling on how a case applies is logged in `data_availability_pipeline/v2/review_issues.md` and stamped on the rows it affects (`scheme_version`, `prompt_version`) |

This corpus was classified in **manual mode**, with an agent as the classifier
behind the same validation gate. Each row records its coder in `model_id`.
Human adjudications are recorded as `human:<name>`. Report rates **by year
band**: 853 papers coded before the window was narrowed to 2021-2026 are still in
the database.

To get one result per paper and draw a stratified sample for human review:

```bash
cd post-analysis
python frame_result.py ../dap_final.sqlite -o frame_samples \
    --random_source random_category --n 10                  # up to 10 papers per result, one CSV each
python -c "import glob, pandas as pd; pd.concat(pd.read_csv(f) for f in sorted(glob.glob('frame_samples/result_*.csv'))).to_csv('review.csv', index=False)"
python review_app.py review.csv \
    --db ../dap_final.sqlite     # blind review form at http://127.0.0.1:8765
python review_metrics.py review.csv --out review_results    # classifier vs. human
```

---

## Benchmark: the pipeline against plain LLM classification

The question is whether the pipeline is more accurate than an LLM that is only
given the class definitions, and whether it leaves a human less to do.

### Systems compared

| system | what it is |
|---|---|
| **pipeline** | the full pipeline in `--mode api` with one pinned model, run unattended |
| **baseline, definitions** | an LLM given the paper's full text and only the two axis definitions and seven case definitions (the ones human reviewers saw), asked for JSON |
| **baseline, pipeline prompt** | ablation: the pipeline's full instructions on the full text, but none of its machinery (no metadata, no passage selection, no validator, no quote check, no link verification). It separates "a better prompt" from "a pipeline" |
| **as-built** | the codes the case-study corpus actually carries (manual mode) |

Run the baselines with the same model as the pipeline, which isolates the
pipeline's contribution, and with models from other vendors.

### Fairness rules

- **Same document.** Retrieval runs once (`run_pipeline.sh prepare`), and the
  exact text each paper's passages were cut from is saved to
  `benchmark/texts/`. Every baseline reads those files, so no difference comes
  from a better or worse PDF. This favours the baselines: in practice someone
  would have to find and download each paper for them.
- **No dropped papers.** A missing, unparseable or out-of-vocabulary answer
  counts as wrong. It is never mapped to the nearest token.
- **Blind gold.** `review_app.py` never shows the classifier's codes to the
  reviewer.
- **One rule set.** The NDA ruling (NDA-restricted data scores
  `on_request` ≡ `explicitly_restricted`) applies to every system.

### Steps

```bash
cd benchmark
python build_gold.py                                    # gold/frame_review.csv (69 papers), strata, as-built predictions
./run_pipeline.sh prepare                               # once: enrol, metadata, retrieve; fills texts/
./run_pipeline.sh classify claude-opus-5-5              # pipeline, pinned model, no escalation
python baseline.py --provider anthropic --model claude-opus-5-5                            # same model, definitions only
python baseline.py --provider anthropic --model claude-opus-5-5 --variant pipeline_prompt  # ablation
python baseline.py --provider openai --model <model> --api-key-env OPENAI_API_KEY          # another vendor
python baseline.py --provider openai --model <model> \
    --base-url https://generativelanguage.googleapis.com/v1beta/openai --api-key-env GEMINI_API_KEY
python evaluate.py --reference pipeline \
    --system pipeline=runs/pipeline_claude-opus-5-5/analysis.csv \
    --system asbuilt=predictions/pipeline_asbuilt.csv \
    --system opus_defs=runs/baseline_claude-opus-5-5-definitions.jsonl \
    --system opus_prompt=runs/baseline_claude-opus-5-5-pipeline_prompt.jsonl \
    --system other_defs=runs/baseline_<model>-definitions.jsonl
```

`results/report.md` holds the tables. `results/*.csv` holds the same figures
per system, axis and class, with the confusion matrices.

### What is measured

**Accuracy and agreement.** Per axis and per paper (both axes right):
- accuracy with a Wilson 95% interval;
- Cohen's kappa, PABAK, and macro and weighted F1, always reported together,
  because the skewed case distribution makes kappa alone misleading;
- per-class precision, recall and F1.

The gold sample is stratified on the as-built result, so a population-weighted
accuracy with a stratified-bootstrap interval is reported beside the sample
figure.

**Significance.** Each system is compared with the reference on the same papers:
- an exact McNemar test;
- a paired-bootstrap interval for the accuracy difference;
- Holm's correction across all comparisons on an axis.

**Manual effort.** A flagged paper goes to a human. Flags mean
`needs_human_review` for the pipeline, and an invalid answer or `low`
confidence for a baseline. The report gives:
- papers flagged, and review hours per 1,000 papers (`--minutes-per-check`;
  replace the default 2 with your measured time);
- the accuracy of what is accepted unreviewed, and the **silent errors** (wrong
  papers nobody is shown);
- accuracy after review;
- two equal-effort views from each system's own confidence ordering: accuracy
  after as many reviews as the reference needs, and the number of reviews
  needed to match the reference's post-review accuracy;
- the area under the risk–coverage curve.

**Credibility and cost.**
- The share of quoted evidence that is not actually in the paper. The
  pipeline's quotes are re-checked independently against `texts/`.
- The share of papers whose text was found with no human step.
- API cost from recorded tokens, and time.

### A gold set for validation

The included gold set is quantum computing. A claim about another field needs
human-verified labels from that field:

1. Run the pipeline on the `subject` (topic of interest).
2. Draw a stratified sample with `frame_result.py`.
3. Have a person code it blind in `review_app.py`.
4. Point `build_gold.py --review <csv> --db <db>` at the result.

Any CSV with `openalex_id`, `gold_a` and `gold_b` (case tokens) works as
`--gold`. `stratum` and `nda` are optional.

### Limits to state in the paper

- The gold labels come from one reviewer. Inter-rater agreement needs a second
  blind coder on an overlapping sample.
- The as-built corpus codes come from several agent coders and some human
  adjudication, so they are not one fixed protocol. That is why the benchmark
  re-runs the pipeline unattended with a pinned model.
- Baselines were handed the text. The effort of obtaining it is not in their
  numbers, and is reported separately as the share of text the pipeline
  retrieved with no human step.

---

## Data you can and cannot redistribute

- `openalex_papers.csv` is OpenAlex metadata (CC0).
- The case-study database's `failures` log is stripped from the contact addresses of
  the people who ran it, inside the URLs of failed Unpaywall lookups. They are
  replaced with `<redacted>`; new failures are redacted as they are written.
- The SQLite databases and `benchmark/texts/` contain **excerpts and full
  texts of publisher articles**. Keep them local. Publish the exported codes
  and short evidence quotes, not the database.

## Layout

```
README.md                      this file
MANIFEST.md                    every file and why it is here
openalex_papers.csv            the quantum computing harvest
dap_final.sqlite               the case-study state
data_availability_pipeline/
  CLAUDE.md                    the working rules the case study was run under
  v2/dap/                      the pipeline (collect.py, schema.py, sources.py, classify.py,
                               verify.py, store.py, batch.py, pipeline.py)
  v2/tests/                    offline test suite
  v2/scratch/h.py, chk.py      manual-mode record builders and the pre-import gate
  v2/keyword_arms.py           the case study's title-keyword scope rule
  v2/review_issues.md          the decision log
post-analysis/                 one result per paper, sampling, blind review app, review metrics
benchmark/                     gold set, baselines, pipeline runner, evaluation
```
