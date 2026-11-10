"""
dap.pipeline — the staged, resumable, parallel runner.

    python -m dap.pipeline collect  --subject "quantum computing" --out papers.csv
    python -m dap.pipeline enrol    --input papers.csv
    python -m dap.pipeline stage metadata  --workers 48
    python -m dap.pipeline stage retrieve  --workers 16
    python -m dap.pipeline stage classify  --mode batch     # spool + submit
    python -m dap.pipeline stage classify  --mode collect   # pull results
    python -m dap.pipeline stage classify  --mode api  --limit 25   # sync
    python -m dap.pipeline stage classify  --mode manual --export p.jsonl
    python -m dap.pipeline stage classify  --mode manual --import r.jsonl
    python -m dap.pipeline stage verify    --workers 32
    python -m dap.pipeline status
    python -m dap.pipeline export --out out/

THE SHAPE. Each paper moves through fixed stages, and each stage is run by its
own command with its own concurrency. That is the whole point: the stages have
nothing in common operationally. Metadata lookups are cheap and can run 50-wide;
PDF parsing is CPU-bound and wants a process pool; classification is either a
per-request API call or a 5,000-paper batch that takes an hour; verification is
polite per-host HTTP. The old `run.py` did all of it in one loop, so a slow PDF
held up an LLM call and a rate-limited host held up everything.

RESUMABILITY. A stage runner claims papers sitting at its input stage, does the
work, and advances them. State lives in SQLite. Kill it anywhere and rerun the
same command: papers at the finished stage are skipped, papers mid-flight are
still at the prior stage and get picked up again.

WHY THREADS, NOT ASYNCIO. The network stages are I/O-bound and the GIL releases
during I/O, so a thread pool at 16-64 workers saturates the per-host rate limits
comfortably. 150,000 requests at 48-wide and ~400 ms each is under half an hour.
asyncio would add a rewrite of `sources.py` for no throughput gain at this scale.
PDF extraction is the exception — it is CPU-bound — and gets a process pool.

CLASSIFY MODES.
  api     — synchronous, one call per paper, with the schema-rejection retry
            loop. For shakedowns, escalation, and the batch's schema failures.
  batch   — spool requests to JSONL, submit to the Message Batches API at half
            price. For the census. `collect` pulls results when they land.
  manual  — the agent (Claude Code / Cowork) is the classifier. Prompts are
            exported as JSONL; the agent writes classifications back as JSONL;
            import validates every one through schema.py and refuses anything
            invalid. This is what v1 did WITHOUT the validation, which is why
            v1's output had to be discarded. Fine for a pilot; not for a census.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .batch import Spool, SpoolEntry, collect as batch_collect, poll as batch_poll, \
    safe_custom_id, submit as batch_submit, estimate_cost
from .classify import (TIER_FAST, TIER_STRONG, ClassificationError, Usage,
                       _assemble, build_request, classify, select_passages,
                       structural_passages, passage_text, _STRONG_AVAILABILITY,
                       has_classifiable_evidence, unclassifiable_record,
                       grounding_text,
                       should_escalate, PROMPT_VERSION)
from .schema import Arm, Retrieval, SchemaError, TextSource, SCHEME_VERSION
from .sources import (Config, PermanentError, TransientError, registry_facts,
                      datacite_datasets, openalex_family_names, retrieve_full_text,
                      _get, _normalise_doi)
from .store import Store, _scheme_family
from .verify import aggregate, classify_pinning, derive_case, verify_identifier

_lock = threading.Lock()


def log(msg: str) -> None:
    with _lock:
        print(msg, flush=True)


def build_config(*, need_llm: bool) -> Config:
    email = os.environ.get("DAP_CONTACT_EMAIL", "").strip()
    if not email:
        log("DAP_CONTACT_EMAIL is not set: Crossref is queried in its public pool "
            "and Unpaywall is skipped. When set, it is sent to those two APIs only.")
    if need_llm and not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        sys.exit("ANTHROPIC_API_KEY is not set. This stage needs it; "
                 "`--mode manual` does not.")
    tdm = {}
    if os.environ.get("ELSEVIER_TDM_KEY"):
        tdm["api.elsevier.com"] = {"X-ELS-APIKey": os.environ["ELSEVIER_TDM_KEY"]}
    if os.environ.get("WILEY_TDM_TOKEN"):
        tdm["api.wiley.com"] = {"Wiley-TDM-Client-Token": os.environ["WILEY_TDM_TOKEN"]}
    if os.environ.get("SPRINGER_TDM_KEY"):
        tdm["articles.springer.com"] = {"X-API-Key": os.environ["SPRINGER_TDM_KEY"]}
    return Config(grobid_url=os.environ.get("DAP_GROBID_URL", "").strip(),
                  contact_email=email,
                  openalex_key=os.environ.get("OPENALEX_API_KEY", ""),
                  pdf_dir=os.environ.get("DAP_PDF_DIR", "").strip(),
                  deny_hosts=frozenset(
                      h.strip().lower().removeprefix("www.")
                      for h in os.environ.get("DAP_DENY_HOSTS", "").split(",")
                      if h.strip()),
                  tdm_headers=tdm)


def _client():
    import anthropic
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


# ---------------------------------------------------------------------------
# Stage 1: metadata  (enrolled -> metadata)
# ---------------------------------------------------------------------------
def _openalex(cfg: Config, oid: str) -> dict:
    wid = oid.rsplit("/", 1)[-1]
    params = {}                         # OpenAlex identifies callers by API key
    if cfg.openalex_key:
        params["api_key"] = cfg.openalex_key
    return _get(cfg, f"https://api.openalex.org/works/{wid}", params=params).json()


def stage_metadata_one(cfg: Config, store: Store, row: dict) -> None:
    oid, doi = row["openalex_id"], row.get("doi") or ""
    try:
        work = _openalex(cfg, oid)
    except PermanentError as e:
        store.fail(oid, "metadata", "openalex_404", e, permanent=True); return
    except TransientError as e:
        store.fail(oid, "metadata", "openalex_transient", e); return

    facts = registry_facts(cfg, doi)
    # Every cited DOI, not a prefix of them. The former [:25] cap hid 68.6% of
    # references, and on the first 200-paper run all three confirmed datasets
    # sat beyond reference 25 — data and software are cited late, after the
    # literature. Uncapped is affordable because `datacite_datasets` is bulk.
    refs = facts.dataset_refs
    hits = datacite_datasets(cfg, refs)
    confirmed = [d for d in refs if _normalise_doi(d).lower() in hits]

    meta = {
        "openalex_id": oid, "doi": doi, "arm": row.get("arm") or Arm.UNSPECIFIED.value,
        "title": work.get("title") or work.get("display_name") or "",
        "year": work.get("publication_year"),
        "venue": ((work.get("primary_location") or {}).get("source") or {}).get("display_name", "") or "",
        "is_oa": bool((work.get("open_access") or {}).get("is_oa")),
        "oa_status": (work.get("open_access") or {}).get("oa_status", ""),
        "das_texts": facts.das_texts,
        "dataset_ids": facts.dataset_ids,
        "confirmed_datasets": confirmed,
        "tdm_links": facts.tdm_links,
        "crossref_found": facts.found,
        # A DOI no registry knows still has authors in OpenAlex, and without
        # them the arXiv title search refuses every candidate.
        "authors": facts.authors or openalex_family_names(work),
        "published_year": facts.published_year,
        "reference_count": facts.reference_count,
        # keep only what retrieval needs from the OpenAlex work, not the whole blob
        "oa_locations": {
            "best": work.get("best_oa_location") or {},
            "open_access": work.get("open_access") or {},
            "locations": [{"landing_page_url": l.get("landing_page_url"),
                           "pdf_url": l.get("pdf_url"),
                           "source": {"display_name": (l.get("source") or {}).get("display_name")}}
                          for l in (work.get("locations") or [])[:8]],
            "ids": work.get("ids") or {},
        },
        # metadata alone settles a paper when it has a DAS and a structural link
        "decisive": bool(facts.has_das and (facts.has_structured_dataset_link or confirmed)),
    }
    store.advance(oid, "metadata", meta)


# ---------------------------------------------------------------------------
# Stage 2: retrieve  (metadata -> retrieved)
# ---------------------------------------------------------------------------
def _extract_passages_cpu(text: str, sections: list | None = None) -> list[dict]:
    """Runs in the process pool: the keyword scan over a long document is the
    only CPU-bound piece besides PDF parsing, which happens inside retrieval."""
    return select_passages(passage_text(text, sections), sections=sections)


def _cache_text(oid: str, ft) -> None:
    """Keep the exact full text the passages were cut from, when asked to.

    The database stores only the passages. A comparison against a classifier
    that reads the whole paper must give that classifier the SAME copy, or a
    difference in accuracy could be a difference in retrieval. Set
    `DAP_TEXT_CACHE=<dir>` and every retrieved text is written to
    `<dir>/<Wid>.txt`, with its source and URL in `<dir>/<Wid>.json`. Off by
    default: the texts are publisher full text and must not be redistributed.
    """
    d = os.environ.get("DAP_TEXT_CACHE", "").strip()
    if not d:
        return
    p = Path(d); p.mkdir(parents=True, exist_ok=True)
    wid = oid.rsplit("/", 1)[-1]
    (p / f"{wid}.txt").write_text(ft.text, encoding="utf-8")
    (p / f"{wid}.json").write_text(json.dumps({
        "openalex_id": oid, "text_source": ft.source.value, "text_url": ft.url,
        "complete": ft.complete, "chars": len(ft.text)}))


def stage_retrieve_one(cfg: Config, store: Store, row: dict, *,
                       metadata_only: bool, pool: ProcessPoolExecutor | None) -> None:
    oid, meta = row["openalex_id"], row["meta"]
    from .sources import CrossrefFacts
    if not meta.get("authors"):
        # Metadata fetched before `authors` was stored carries none (5,630 of the
        # venue frame), and an empty list switches the arXiv title search off.
        # Look them up rather than retrieve with that fallback silently disabled;
        # a failed lookup only leaves the paper where it would have been anyway.
        try:
            meta["authors"] = openalex_family_names(_openalex(cfg, oid))
        except (PermanentError, TransientError):
            pass
    # reference_count must ride along: without it the completeness check compares
    # every text against 0 references and always passes. It did, for all 4,654
    # fetched papers in the venue frame.
    # `authors` and `published_year` ride along for the same reason
    # `reference_count` does: the arXiv title search refuses to return a match
    # it cannot check against them, so a facts object rebuilt without them
    # disables that fallback entirely rather than loosening it.
    facts = CrossrefFacts(found=meta.get("crossref_found", False),
                          das_texts=meta.get("das_texts", []),
                          dataset_ids=meta.get("dataset_ids", []),
                          tdm_links=meta.get("tdm_links", []),
                          authors=meta.get("authors") or [],
                          published_year=meta.get("published_year") or meta.get("year"),
                          reference_count=meta.get("reference_count") or 0)
    work = {"best_oa_location": meta["oa_locations"]["best"],
            "open_access": meta["oa_locations"]["open_access"],
            "locations": meta["oa_locations"]["locations"],
            "ids": meta["oa_locations"]["ids"]}

    passages: list[dict] = []
    attempts: list[dict] = []
    retrieval, source, url = "metadata_only", "none", ""
    if not (metadata_only or meta.get("decisive")):
        try:
            ft = retrieve_full_text(cfg, work=work, doi=meta["doi"],
                                    title=meta["title"], facts=facts)
        except TransientError as e:
            # A preferred copy was only temporarily unavailable. Leave the paper
            # at `metadata` so the next run retries it, rather than storing
            # whatever a worse source gave.
            store.fail(oid, "retrieve", "transient", e); return
        attempts = ft.attempts
        if ft.ok:
            if pool is not None:
                passages = pool.submit(_extract_passages_cpu, ft.text,
                                       ft.sections).result()
            else:
                passages = _extract_passages_cpu(ft.text, ft.sections)
            # A body whose bibliography is missing is partial by definition, however
            # much prose it carries: the end of the document is where data
            # availability statements live.
            retrieval = ("full_text" if (passages and ft.complete)
                         else "partial_text")
            # Matching no keyword is itself a finding — a theory paper with no
            # data says none of these words. Give the classifier the opening and
            # the back matter so it codes that from the paper rather than being
            # gated as though nothing had been retrieved. Computed after
            # `retrieval` so a structural-only paper is never called full_text.
            # The reference list is cut from the text before either pass
            # (`passage_text`), so neither can land in it: a keyword inside a
            # cited title is evidence about what the paper cites.
            if not passages:
                passages = structural_passages(passage_text(ft.text, ft.sections))
                meta["passages_are_structural"] = bool(passages)
            elif not any(set(p["keywords"]) & _STRONG_AVAILABILITY for p in passages):
                # No availability heading anywhere: to code an absence, the
                # classifier must see where a statement would be. Cutting the
                # reference list first means a stray keyword can no longer route
                # a paper to the opening and closing, and in two-column layouts
                # reference fragments share lines with prose and survive the cut
                # — four recoded papers reached the classifier with nothing but
                # such a fragment (26 Sep 2026).
                taken = [(p["offset"], p["offset"] + len(p["text"])) for p in passages]
                extra = [s for s in structural_passages(passage_text(ft.text, ft.sections))
                         if not any(a < s["offset"] + len(s["text"]) and s["offset"] < b
                                    for a, b in taken)]
                if extra:
                    passages = sorted(passages + extra, key=lambda p: p["offset"])
                    meta["structural_context"] = True
            source, url = ft.source.value, ft.url
            meta["text_complete"] = ft.complete
            _cache_text(oid, ft)
        elif not facts.found:
            store.fail(oid, "retrieve", "no_text_no_metadata",
                       "no full text and no Crossref or DataCite record", permanent=True)
            return

    meta["pdf_parser"] = "grobid" if cfg.grobid_url else "pymupdf"
    meta["retrieval_attempts"] = attempts
    meta.update({"passages": passages, "retrieval": retrieval,
                 "text_source": source, "text_url": url})
    store.advance(oid, "retrieved", meta)


# ---------------------------------------------------------------------------
# Stage 3: classify  (retrieved -> classified)   four modes
# ---------------------------------------------------------------------------
def _summary(meta: dict) -> dict:
    return {"das_texts": meta.get("das_texts", []),
            "dataset_ids": meta.get("dataset_ids", []),
            "confirmed_datasets": meta.get("confirmed_datasets", [])}


def _gate_unclassifiable(store: Store, oid: str, meta: dict) -> bool:
    """Refuse to put an evidence-free paper in front of a model.

    Measured on the first live run: 78 of 200 papers (39%) reached `retrieved`
    with no passages, no DAS and no dataset link. The prompt asks the model to
    set needs_human_review in that case, but nothing enforced it — a model can
    quote the title back and land a confident code. Flag here instead, before a
    request is ever made. Returns True if the paper was gated.
    """
    if has_classifiable_evidence(crossref_summary=_summary(meta),
                                 passages=meta.get("passages", [])):
        return False
    store.save_classified(unclassifiable_record(
        openalex_id=oid, doi=meta.get("doi", ""), title=meta.get("title", ""),
        venue=meta.get("venue", ""), year=meta.get("year"),
        arm=Arm.coerce(meta["arm"]),
        retrieval=Retrieval.coerce(meta["retrieval"]),
        text_source=TextSource.coerce(meta["text_source"]),
        text_url=meta.get("text_url", "")))
    return True


def _record_meta(meta: dict) -> dict:
    """What `collect` needs to rebuild a record — everything but the passages."""
    return {k: meta.get(k) for k in ("openalex_id", "doi", "title", "venue", "year",
                                     "arm", "retrieval", "text_source", "text_url")}


def classify_api_one(client, store: Store, row: dict, *, escalate: bool,
                     model: str | None = None) -> None:
    """One synchronous classification. `model` pins a single model for the
    first call (default: the fast tier); escalation, if on, still goes to the
    strong tier. For a benchmark, pin the model and pass --no-escalate, so every
    row comes from one model."""
    oid, meta = row["openalex_id"], row["meta"]
    if _gate_unclassifiable(store, oid, meta):
        log(f"  {oid}  GATED — no evidence, flagged for human review"); return
    common = dict(title=meta["title"], venue=meta["venue"], year=meta["year"],
                  arm=Arm.coerce(meta["arm"]), openalex_id=oid, doi=meta["doi"],
                  crossref_summary=_summary(meta), passages=meta.get("passages", []),
                  retrieval=Retrieval.coerce(meta["retrieval"]),
                  text_source=TextSource.coerce(meta["text_source"]),
                  text_url=meta.get("text_url", ""))
    try:
        rec, usage = classify(client, model=model or TIER_FAST, **common)
        if escalate and should_escalate(rec):
            store.record_usage(oid, usage)
            rec, usage = classify(client, model=TIER_STRONG, **common)
    except (ClassificationError, SchemaError) as e:
        store.fail(oid, "classify", type(e).__name__, e); return
    except Exception as e:
        store.fail(oid, "classify", type(e).__name__, e); return
    store.save_classified(rec)
    store.record_usage(oid, usage)
    log(f"  {oid}  A={rec.axis_a.case.value.value:<26} B={rec.axis_b.case.value.value:<26}"
        f"{'  REVIEW' if rec.needs_human_review else ''}")


def classify_spool(store: Store, spool: Spool, rows: list[dict], *, model: str) -> int:
    n = 0
    for row in rows:
        oid, meta = row["openalex_id"], row["meta"]
        if _gate_unclassifiable(store, oid, meta):
            continue
        params = build_request(model=model, title=meta["title"], venue=meta["venue"],
                               year=meta["year"], arm=Arm.coerce(meta["arm"]),
                               openalex_id=oid, crossref_summary=_summary(meta),
                               passages=meta.get("passages", []))
        spool.add(SpoolEntry(custom_id=safe_custom_id(oid), params=params,
                             meta=_record_meta(meta)))
        n += 1
    return n


def classify_manual_export(store: Store, rows: list[dict], out: Path) -> int:
    """Write prompts for the agent. One JSON object per line."""
    from .classify import SYSTEM_PROMPT
    from .schema import CLASSIFICATION_TOOL_SCHEMA
    with out.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"_header": True, "system_prompt": SYSTEM_PROMPT,
                            "output_schema": CLASSIFICATION_TOOL_SCHEMA,
                            "prompt_version": PROMPT_VERSION,
                            "scheme_version": SCHEME_VERSION,
                            "instructions": (
                                "For each following line, read `prompt` and produce a JSON "
                                "object matching `output_schema` EXACTLY. Write results as "
                                "JSONL: {\"custom_id\": ..., \"classification\": {...}}. "
                                "Every value must be a token from the schema's enums; the "
                                "import will reject anything else and tell you why.")}) + "\n")
        written = 0
        for row in rows:
            oid, meta = row["openalex_id"], row["meta"]
            if _gate_unclassifiable(store, oid, meta):
                continue
            written += 1
            params = build_request(model="manual", title=meta["title"], venue=meta["venue"],
                                   year=meta["year"], arm=Arm.coerce(meta["arm"]),
                                   openalex_id=oid, crossref_summary=_summary(meta),
                                   passages=meta.get("passages", []))
            f.write(json.dumps({"custom_id": safe_custom_id(oid),
                                "openalex_id": oid,
                                "prompt": params["messages"][0]["content"],
                                "meta": _record_meta(meta)}) + "\n")
    return written


def classify_manual_import(store: Store, path: Path) -> dict[str, int]:
    """Validate the agent's classifications through the same gate as an API
    result. Nothing reaches the store without passing schema.py."""
    tally = {"accepted": 0, "rejected": 0, "unknown": 0}
    by_cid = {}
    for r in store.claim("retrieved", limit=100_000):
        by_cid[safe_custom_id(r["openalex_id"])] = r["meta"]

    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        obj = json.loads(line)
        cid = obj.get("custom_id")
        meta = by_cid.get(cid)
        if meta is None:
            tally["unknown"] += 1
            log(f"  {cid}: not at the retrieved stage — skipped")
            continue
        try:
            rec = _assemble(
                obj["classification"],
                openalex_id=meta["openalex_id"], doi=meta.get("doi", ""),
                title=meta.get("title", ""), venue=meta.get("venue", ""),
                year=meta.get("year"), arm=Arm.coerce(meta.get("arm", "unspecified")),
                model="manual:" + os.environ.get("DAP_MANUAL_CODER", "agent"),
                retrieval=Retrieval.coerce(meta.get("retrieval", "metadata_only")),
                text_source=TextSource.coerce(meta.get("text_source", "none")),
                text_url=meta.get("text_url", ""),
                usage=Usage(model="manual"),
                # Same grounding rule as the API path: a quote must come from
                # the passages or DAS the coder was actually shown. Manual mode
                # is the likeliest place for a paraphrase to slip in.
                grounding=grounding_text(crossref_summary=_summary(meta),
                                         passages=meta.get("passages", [])),
                passages=meta.get("passages", []),
                crossref_das=bool(meta.get("das_texts")))
        except (SchemaError, KeyError, TypeError, ValueError) as e:
            tally["rejected"] += 1
            store.fail(meta["openalex_id"], "classify_manual", "SchemaError", str(e))
            log(f"  {cid}: REJECTED — {str(e)[:160]}")
            continue
        store.save_classified(rec)
        tally["accepted"] += 1
    return tally


# ---------------------------------------------------------------------------
# Stage 4: verify  (classified -> verified)
# ---------------------------------------------------------------------------
def stage_verify_one(cfg: Config, store: Store, row: dict) -> None:
    from .schema import (Actual, AxisRecord, Case, Confidence, DataType, Declared,
                         Evidence, Granularity, Judgement, PaperRecord, PaperType,
                         Pinned, Restriction, RepoType)
    oid = row["openalex_id"]
    with store._conn() as c:
        r = c.execute("SELECT record_json FROM papers WHERE openalex_id=?", (oid,)).fetchone()
    flat = json.loads(r["record_json"])

    verifications = []
    for pfx in ("a", "b"):
        idents = [i for i in (flat.get(f"{pfx}_identifiers") or "").split(";") if i]
        if not idents:
            continue
        results = [verify_identifier(cfg, i) for i in idents[:5]]
        verifications += [(f"axis_{pfx}", vr) for vr in results]
        actual = aggregate(results)
        flat[f"{pfx}_actual"] = actual.value
        flat[f"{pfx}_pinned"] = classify_pinning(idents).value
        new_case, note = derive_case(Case.coerce(flat[f"{pfx}_case"]), actual)
        if new_case.value != flat[f"{pfx}_case"]:
            flat[f"{pfx}_case"] = new_case.value
            flat[f"{pfx}_case_rationale"] = (flat.get(f"{pfx}_case_rationale", "") + " | " + note)[:600]
            from .schema import CASE_NUMBER, OPENNESS_RANK
            flat[f"{pfx}_case_num"] = CASE_NUMBER[new_case]
            flat[f"{pfx}_openness_rank"] = OPENNESS_RANK[new_case]

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with store._conn() as c:
        c.execute("BEGIN")
        c.execute("UPDATE papers SET stage='verified', status='done', record_json=?, "
                  "updated_at=? WHERE openalex_id=?", (json.dumps(flat), now, oid))
        for axis, vr in verifications:
            c.execute("INSERT OR REPLACE INTO verifications VALUES (?,?,?,?,?,?,?,?)",
                      (oid, axis, vr.identifier, vr.status.value, vr.http_status,
                       vr.final_url, vr.note, now))
        c.execute("COMMIT")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_pool(rows: list[dict], fn, *, workers: int, budget: float | None,
             store: Store, label: str) -> None:
    stop = threading.Event()

    def guarded(row):
        if stop.is_set():
            return
        if budget and store.total_cost() >= budget:
            stop.set(); return
        try:
            fn(row)
        except Exception as e:                       # never let one paper kill a stage
            store.fail(row["openalex_id"], label, type(e).__name__, str(e)[:500])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(guarded, r) for r in rows]
        for i, f in enumerate(as_completed(futs), 1):
            f.result()
            if i % 50 == 0:
                log(f"[{label}] {i}/{len(rows)}  ${store.total_cost():.2f}")
    if stop.is_set():
        log(f"[{label}] stopped at budget ${budget:.2f}")


def cmd_enrol(a):
    store = Store(a.db)
    df = pd.read_csv(a.input)
    if "openalex_id" not in df.columns:
        sys.exit("input needs an openalex_id column")
    # Arms (study sub-populations) are not part of the general tool. The column
    # stays so databases from the quantum case study still open; a new corpus
    # is `unspecified` throughout. An `arm` column in the input is still honoured.
    if getattr(a, "arm_map", None):
        df = df.merge(pd.read_csv(a.arm_map), on="openalex_id", how="left")
    if "arm" not in df.columns:
        df["arm"] = Arm.UNSPECIFIED.value
    df["arm"] = df["arm"].fillna(Arm.UNSPECIFIED.value)
    # Build one mask against the *current* index. Filtering in two steps left the
    # second mask carrying the pre-filter index, which pandas only rescued by
    # reindexing (with a UserWarning) and newer versions refuse outright.
    yr = pd.to_numeric(df.get("publication_year"), errors="coerce")
    df["year"] = yr
    keep = pd.Series(True, index=df.index)
    if a.year_min: keep &= (yr >= a.year_min)
    if a.year_max: keep &= (yr <= a.year_max)
    df = df[keep]
    # Deduplicate before truncating: the other order makes `--limit N` enrol
    # fewer than N whenever the slice contains a repeated openalex_id.
    #
    # Dedup on the DOI too, and not only the OpenAlex id. OpenAlex mints more
    # than one work id for the same DOI often enough to matter -- 399 DOIs in
    # dap_frame are held by two rows -- and a row enrolled twice is classified
    # twice, because `is_classified` keys on the id. A row with no DOI is kept
    # whatever happens: an empty string is not an identity, so deduplicating on
    # it would collapse every DOI-less paper in the harvest into one row.
    df = df.drop_duplicates(subset=["openalex_id"])
    # A handful of harvested DOIs carry the PDF link's suffix -- EDP Sciences
    # and IOP serve `.../ae333e/pdf`, and OpenAlex stored that as the DOI. The
    # suffixed form is a different string from the real one, so the uniqueness
    # index cannot see that they are the same paper: 15 of the 17 such rows in
    # dap_frame duplicated a row that already held the clean DOI.
    doi = (df["doi"].fillna("").str.strip().str.lower()
                    .str.replace(r"/(pdf|full|abstract|epdf)$", "", regex=True))
    df["doi"] = doi
    df = df[(doi == "") | ~doi.duplicated()]
    if a.limit:    df = df.head(a.limit)
    rows = df[["openalex_id", "doi", "arm"]].fillna("").to_dict("records")
    for r, y in zip(rows, df["year"]):
        r["year"] = int(y) if pd.notna(y) else None
    n = store.enrol(rows)
    log(f"enrolled {n} new ({len(df)} after filters)")
    log(json.dumps(store.stage_counts(), indent=2))


#: The batch window. `enrol` decides what is in the corpus at all; this decides
#: what the NEXT batch works on, and it is deliberately separate: narrowing it
#: never touches a paper already coded, it only changes what gets picked up
#: next. Nothing stored is rewritten or removed by it.
BATCH_YEAR_MIN = 2021


def _window(a) -> dict:
    """Year filter, ordering and DOI scope for a stage's claim, from the CLI."""
    ymin = a.year_min if a.year_min is not None else BATCH_YEAR_MIN
    return {"year_min": None if ymin <= 0 else ymin,
            "newest_first": not a.oldest_first,
            "publisher_doi_only": bool(getattr(a, "publisher_doi_only", False)),
            "skip_coded_dois": bool(getattr(a, "skip_coded_dois", False))}


def cmd_stage(a):
    store = Store(a.db)
    if a.stage == "metadata":
        cfg = build_config(need_llm=False)
        rows = store.claim("enrolled", a.limit or 10**9, getattr(a, "arm", None), **_window(a))
        log(f"metadata: {len(rows)} papers, {a.workers} workers")
        run_pool(rows, lambda r: stage_metadata_one(cfg, store, r),
                 workers=a.workers, budget=None, store=store, label="metadata")

    elif a.stage == "retrieve":
        cfg = build_config(need_llm=False)
        rows = store.claim("metadata", a.limit or 10**9, getattr(a, "arm", None), **_window(a))
        log(f"retrieve: {len(rows)} papers, {a.workers} workers"
            + (" (metadata-only)" if a.metadata_only else ""))
        with ProcessPoolExecutor(max_workers=max(2, os.cpu_count() or 2)) as ppool:
            run_pool(rows, lambda r: stage_retrieve_one(cfg, store, r,
                                                        metadata_only=a.metadata_only,
                                                        pool=ppool),
                     workers=a.workers, budget=None, store=store, label="retrieve")

    elif a.stage == "classify":
        if a.mode == "api":
            build_config(need_llm=True); client = _client()
            rows = store.claim("retrieved", a.limit or 10**9, getattr(a, "arm", None), **_window(a))
            log(f"classify/api: {len(rows)} papers, {a.workers} workers")
            run_pool(rows, lambda r: classify_api_one(client, store, r,
                                                      escalate=not a.no_escalate,
                                                      model=a.model),
                     workers=a.workers, budget=a.budget, store=store, label="classify")
        elif a.mode == "batch":
            build_config(need_llm=True); client = _client()
            spool = Spool(a.spool)
            rows = store.claim("retrieved", a.limit or 10**9, getattr(a, "arm", None), **_window(a))
            n = classify_spool(store, spool, rows, model=a.model or TIER_FAST)
            sync, batched = estimate_cost(n, model=a.model or TIER_FAST)
            log(f"spooled {n} requests  (est. ${batched:.2f} batched vs ${sync:.2f} sync)")
            if a.budget and batched > a.budget:
                sys.exit(f"estimated ${batched:.2f} exceeds --budget {a.budget:.2f}; not submitting")
            batch_submit(client, store, spool, flush=a.flush, dry_run=a.dry_run, log=log)
        elif a.mode == "collect":
            build_config(need_llm=True); client = _client()
            log("open batches:"); batch_poll(client, store, log=log)
            t = batch_collect(client, store, log=log)
            log(json.dumps(t))
        elif a.mode == "manual":
            build_config(need_llm=False)
            if a.export_path:
                rows = store.claim("retrieved", a.limit or 50, getattr(a, "arm", None), **_window(a))
                n = classify_manual_export(store, rows, Path(a.export_path))
                log(f"exported {n} prompts to {a.export_path}")
                log("Classify them, write results as JSONL, then re-run with --import")
            elif a.import_path:
                t = classify_manual_import(store, Path(a.import_path))
                log(json.dumps(t))
            else:
                sys.exit("--mode manual needs --export PATH or --import PATH")

    elif a.stage == "verify":
        cfg = build_config(need_llm=False)
        rows = store.claim("classified", a.limit or 10**9, getattr(a, "arm", None), **_window(a))
        log(f"verify: {len(rows)} papers, {a.workers} workers")
        run_pool(rows, lambda r: stage_verify_one(cfg, store, r),
                 workers=a.workers, budget=None, store=store, label="verify")

    log(json.dumps(store.stage_counts(), indent=2))


def cmd_status(a):
    store = Store(a.db)
    log(json.dumps(store.stage_counts(), indent=2))
    log(f"spent: ${store.total_cost():.2f}")
    ob = store.open_batches()
    if ob:
        log(f"open batches: {len(ob)}")


def cmd_export(a):
    store = Store(a.db)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log(json.dumps(store.export_csv(out / "analysis.csv", out / "evidence.csv",
                                    out / "review_queue.csv")))


def cmd_requeue(a):
    """Send retrieved, unclassified papers back through retrieval — for a paper
    whose stored text is known to be the wrong copy. The store refuses anything
    already classified; every requeue is logged to `failures`."""
    store = Store(a.db)
    for oid in a.ids:
        # The DB stores full OpenAlex URLs; accept the bare W-number too.
        full = oid if oid.startswith("http") else f"https://openalex.org/{oid}"
        why = store.requeue_for_retrieval(full, reason=a.reason)
        log(f"  {full}: " + (f"REFUSED — {why}" if why else "requeued for retrieval"))
    log(json.dumps(store.stage_counts(), indent=2))


def cmd_retry(a):
    """Clear the attempt counter on papers that ran out of tries at their stage.

    A schema rejection in `--mode manual` spends an attempt exactly as a timeout
    does, so a malformed results file can retire papers that have nothing wrong
    with them. The store refuses anything that is not actually failed, or that
    already carries a classification."""
    store = Store(a.db)
    ids = list(a.ids)
    if a.ids_from:
        ids += [ln.strip() for ln in Path(a.ids_from).read_text().splitlines() if ln.strip()]
    for oid in ids:
        full = oid if oid.startswith("http") else f"https://openalex.org/{oid}"
        why = store.clear_attempts(full, reason=a.reason)
        log(f"  {full}: " + (f"REFUSED — {why}" if why else "attempts cleared"))
    log(json.dumps(store.stage_counts(), indent=2))


def cmd_recode(a):
    """Discard a finished paper's text and its codes together, and start it
    over. For a paper classified from the wrong document. Destructive by
    design: the discarded record is written to `failures` first, and `--ids-from`
    reads one id per line so a measured list can be piped in rather than typed."""
    store = Store(a.db)
    ids = list(a.ids)
    if a.ids_from:
        with open(a.ids_from) as f:
            ids += [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if not ids:
        log("no ids given"); return
    if not a.yes:
        log(f"would recode {len(ids)} papers — discarding their codes and text. "
            f"Re-run with --yes to do it.")
        return
    ok = 0
    for oid in ids:
        full = oid if oid.startswith("http") else f"https://openalex.org/{oid}"
        why = store.requeue_for_recode(full, reason=a.reason)
        if why:
            log(f"  {full}: REFUSED — {why}")
        else:
            ok += 1
    log(f"recoded {ok} of {len(ids)}; they are back at stage=metadata")
    log(json.dumps(store.stage_counts(), indent=2))


def cmd_exclude(a):
    """Take papers out of the corpus — a duplicate, a correction notice, or
    (`outdated`) a paper outside the study window. The row and its codes are
    kept; it just stops being counted. Reports only, unless --yes."""
    store = Store(a.db)
    ids = list(a.ids)
    if a.ids_from:
        with open(a.ids_from) as f:
            ids += [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    ids = [oid if oid.startswith("http") else f"https://openalex.org/{oid}" for oid in ids]
    if not a.yes:
        log(f"would mark {len(ids)} papers {a.as_status}: {' '.join(ids)}. "
            f"Re-run with --yes to do it.")
        return
    for full in ids:
        why = store.exclude(full, reason=a.reason, as_status=a.as_status)
        log(f"  {full}: " + (f"REFUSED — {why}" if why else a.as_status))
    log(json.dumps(store.counts(), indent=2))


#: What counts as an availability statement when a preprint is read in full:
#: the headings, plus the stock sentences that stand in for one.
PREPRINT_DAS_PHRASES = ("data availability", "code availability", "availability of data",
                        "data that support the findings", "available from the corresponding author",
                        "upon reasonable request", "on reasonable request")


def preprint_das(cfg: Config, store: Store, oid: str) -> dict:
    """Read a paper's arXiv copy in full and record whether it carries an
    availability statement (review_issues S7).

    Quantum Science and Technology's arXiv copies carry one 22% of the time,
    its published text 48%; a paper coded from its version of record keeps
    this as the measured half of that comparison. The whole text is scanned,
    not passages, so absence means absent from the preprint.
    """
    meta = store.paper_meta(oid)
    urls = [l.get("pdf_url") for l in (meta.get("oa_locations") or {}).get("locations", [])
            if "arxiv.org" in (l.get("pdf_url") or "")]
    urls += [a["url"] for a in meta.get("retrieval_attempts") or []
             if a.get("source") == "arxiv" and a.get("url")]
    if not urls:
        return {"checked": False, "reason": "no arXiv copy known"}
    url = urls[0]
    from .sources import _extract_pdf
    text, _ = _extract_pdf(_get(cfg, url).content)
    low = text.lower()
    found = [p for p in PREPRINT_DAS_PHRASES if p in low]
    version = re.search(r"arxiv:\d{4}\.\d{4,5}(v\d+)", low)
    result = {"checked": True, "source": "arxiv", "url": url,
              "version": version.group(1) if version else "",
              "words": len(text.split()), "das_found": bool(found), "phrases": found,
              "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    store.record_preprint_das(oid, result)
    return result


def cmd_preprint_das(a):
    cfg = build_config(need_llm=False)
    store = Store(a.db)
    for oid in a.ids:
        full = oid if oid.startswith("http") else f"https://openalex.org/{oid}"
        log(f"  {full}: {json.dumps(preprint_das(cfg, store, full))}")


def record_from_flat(flat: dict, evidence: list) -> "PaperRecord":
    """Rebuild a PaperRecord from its stored flat row, so it can go back
    through `validate()`.

    `flat()` keeps each value but only the case rationale and the paper-type
    and case confidences. The rest is filled with a marked placeholder and
    medium confidence: enough for the cross-field rules to run, and never
    written back, since `flat()` stores values only.
    """
    from .schema import (Actual, AxisRecord, Case, Confidence, DataType, Declared,
                         Evidence, Granularity, Judgement, PaperRecord, PaperType,
                         Pinned, Restriction, RepoType)
    stored = "(carried from the stored record)"
    split = lambda v: [x for x in (v or "").split(";") if x]
    ev = [Evidence(id=int(e["ev_id"]), quote=e["quote"], section=e["section"] or "",
                   locator=e["locator"] or "", source=e["source"] or "") for e in evidence]
    ev_ids = [e.id for e in ev]

    def axis(p):
        conf = Confidence.coerce(flat[f"{p}_case_conf"])
        return AxisRecord(
            case=Judgement(flat[f"{p}_case"], conf, flat[f"{p}_case_rationale"],
                           ev_ids if conf is Confidence.HIGH else []),
            data_type=Judgement(flat[f"{p}_data_type"], Confidence.MEDIUM, stored),
            declared=Judgement(flat[f"{p}_declared"], Confidence.MEDIUM, stored),
            actual=Judgement(flat[f"{p}_actual"], Confidence.MEDIUM, stored),
            granularity=Judgement(flat[f"{p}_granularity"], Confidence.MEDIUM, stored),
            pinned=flat[f"{p}_pinned"], restriction=flat[f"{p}_restriction"],
            repo_types=split(flat[f"{p}_repo_types"]),
            identifiers=split(flat[f"{p}_identifiers"]), names=split(flat[f"{p}_names"]),
            generation_code=flat.get(f"{p}_gen_code"),
            generation_params=flat.get(f"{p}_gen_params"),
            seed_specified=flat.get(f"{p}_seed"),
            environment_specified=flat.get(f"{p}_environment"))

    pconf = Confidence.coerce(flat["paper_type_conf"])
    return PaperRecord(
        openalex_id=flat["openalex_id"], doi=flat["doi"], title=flat["title"],
        year=flat["year"], venue=flat["venue"], arm=flat["arm"],
        paper_type=Judgement(flat["paper_type"], pconf, stored,
                             ev_ids if pconf is Confidence.HIGH else []),
        axis_a=axis("a"), axis_b=axis("b"), evidence=ev,
        das_present=bool(flat["das_present"]), das_source=flat["das_source"],
        needs_human_review=bool(flat["needs_human_review"]),
        review_reason=flat.get("review_reason") or "",
        scheme_version=flat["scheme_version"], prompt_version=flat["prompt_version"],
        model_id=flat["model_id"], retrieval=flat["retrieval"],
        text_source=flat["text_source"], text_url=flat["text_url"],
        processed_at=flat["processed_at"],
        input_tokens=flat.get("input_tokens") or 0, output_tokens=flat.get("output_tokens") or 0)


#: Flat fields a human adjudication may not set: machine-derived, or written
#: only by verify.py (CLAUDE.md rule 6), or provenance.
_NOT_ADJUDICABLE = {"a_actual", "b_actual", "a_case_num", "b_case_num",
                    "a_openness_rank", "b_openness_rank", "purely_theoretical",
                    "a_pinned", "b_pinned", "scheme_version", "prompt_version",
                    "model_id", "processed_at", "openalex_id", "input_tokens", "output_tokens"}


def cmd_adjudicate(a):
    """A human coder's settlement of one paper, field by field — e.g. two
    copies merged, or an identifier the extraction broke. Goes through the
    schema gate; the old record is logged; a changed identifier is re-verified.
    Reports the change only, unless --yes."""
    store = Store(a.db)
    oid = a.id if a.id.startswith("http") else f"https://openalex.org/{a.id}"
    sets = {}
    for kv in a.set:
        k, _, v = kv.partition("=")
        try:
            sets[k] = json.loads(v)
        except json.JSONDecodeError:
            sets[k] = v
    with store._conn() as c:
        r = c.execute("SELECT record_json FROM papers WHERE openalex_id=?", (oid,)).fetchone()
        evidence = [dict(e) for e in c.execute(
            "SELECT * FROM evidence WHERE openalex_id=? ORDER BY ev_id", (oid,))]
    if r is None or not r["record_json"]:
        sys.exit(f"{oid}: not enrolled or not classified")
    flat = json.loads(r["record_json"])
    bad = sorted(set(sets) & _NOT_ADJUDICABLE) + sorted(k for k in sets if k not in flat)
    if bad:
        sys.exit(f"cannot adjudicate {bad}: machine-set, verify-only, or not a field")
    new = dict(flat, **sets)
    reverify = False
    for p in ("a", "b"):
        if new[f"{p}_identifiers"] != flat[f"{p}_identifiers"]:
            new[f"{p}_actual"] = "not_attempted"     # re-derived by verify.py, not asserted
            reverify = True
    if getattr(a, "reverify", False):
        # A settled case can change what verify derives even when the
        # identifier is untouched: promoting a row to openly_accessible over
        # an identifier already known to 404 must let derive_case do the
        # promotion, not assert the result here.
        for p in ("a", "b"):
            if new[f"{p}_identifiers"]:
                new[f"{p}_actual"] = "not_attempted"
        reverify = True
    new["model_id"] = a.coder
    new["processed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # An adjudication is a coding made today, under today's rulings: it carries
    # the current scheme version, so a recode under a patch-level ruling (2.0.1,
    # S219) is traceable to it. A major or minor difference is refused, since
    # that is a vocabulary change the old record was never coded against.
    old_family = _scheme_family(flat["scheme_version"])
    if old_family != _scheme_family(SCHEME_VERSION):
        sys.exit(f"cannot adjudicate a row coded under scheme {flat['scheme_version']} "
                 f"in a {SCHEME_VERSION} checkout: migrate it first")
    new["scheme_version"] = SCHEME_VERSION
    rec = record_from_flat(new, evidence).validate()
    for k in sorted(set(sets) | {"model_id", "scheme_version"}
                    | ({"a_actual", "b_actual"} if reverify else set())):
        if flat.get(k) != new.get(k):
            log(f"  {k}: {flat.get(k)!r}\n    -> {new.get(k)!r}")
    if not a.yes:
        log(f"{oid}: valid; would adjudicate{' and re-verify' if reverify else ''}. "
            f"Re-run with --yes to do it.")
        return
    why = store.adjudicate(rec, reason=a.reason, reverify=reverify)
    log(f"{oid}: " + (f"REFUSED — {why}" if why else
                      "adjudicated" + (" — now at classified, run `stage verify`" if reverify else "")))


def cmd_adopt(a):
    """Copy papers with their codes and history from another database — a
    separate run being merged in. Append only: a paper already here is refused
    and listed, so an overlap is shown to a human rather than resolved."""
    store = Store(a.db)
    ids = list(a.ids)
    if a.ids_from:
        with open(a.ids_from) as f:
            ids += [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    ok, refused = 0, collections.Counter()
    for oid in ids:
        full = oid if oid.startswith("http") else f"https://openalex.org/{oid}"
        why = store.adopt(a.src, full, reason=a.reason)
        if why:
            refused[why] += 1
            log(f"  {full}: REFUSED — {why}")
        else:
            ok += 1
    log(f"adopted {ok} of {len(ids)} from {a.src}" +
        (f"; refused: {dict(refused)}" if refused else ""))
    log(json.dumps(store.stage_counts(), indent=2))


def main():
    p = argparse.ArgumentParser(prog="dap.pipeline")
    p.add_argument("--db", default="dap.sqlite")
    sub = p.add_subparsers(dest="cmd", required=True)

    from .collect import add_parser as add_collect
    add_collect(sub)

    e = sub.add_parser("enrol"); e.set_defaults(fn=cmd_enrol)
    e.add_argument("--input", required=True,
                   help="CSV with openalex_id (and doi, publication_year), e.g. from `collect`")
    e.add_argument("--year-min", type=int); e.add_argument("--year-max", type=int)
    e.add_argument("--limit", type=int)

    s = sub.add_parser("stage"); s.set_defaults(fn=cmd_stage)
    s.add_argument("stage", choices=["metadata", "retrieve", "classify", "verify"])
    s.add_argument("--workers", type=int, default=32)
    s.add_argument("--limit", type=int)
    s.add_argument("--budget", type=float)
    s.add_argument("--metadata-only", action="store_true")
    s.add_argument("--mode", choices=["api", "batch", "collect", "manual"], default="api")
    s.add_argument("--model", help="pin the classifier model (api and batch modes)")
    s.add_argument("--no-escalate", action="store_true")
    s.add_argument("--spool", default="spool")
    s.add_argument("--flush", action="store_true", help="submit a partial spool file")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--export", dest="export_path"); s.add_argument("--import", dest="import_path")
    s.add_argument("--year-min", type=int, default=None,
                   help=f"only work on papers published in this year or later "
                        f"(default {BATCH_YEAR_MIN}; 0 disables). Papers whose year is "
                        f"not yet known are kept, since the year arrives with metadata.")
    s.add_argument("--oldest-first", action="store_true",
                   help="claim oldest first; the default is newest first")
    s.add_argument("--publisher-doi-only", action="store_true",
                   help="claim only versions of record: skip rows whose DOI was "
                        "minted by a preprint registrar, and rows whose venue is "
                        "a repository (arXiv, Zenodo, Research Square, SSRN, "
                        "Preprints.org, Figshare, ...). A row with neither a DOI "
                        "nor a venue is kept")
    s.add_argument("--skip-coded-dois", action="store_true",
                   help="skip rows whose DOI is already carried by a paper at "
                        "verified/done; OpenAlex mints two work ids for one DOI "
                        "often enough to matter, and is_classified keys on the id")

    rq = sub.add_parser("requeue", help="send retrieved, unclassified papers back through retrieval")
    rq.set_defaults(fn=cmd_requeue)
    rq.add_argument("ids", nargs="+", help="OpenAlex ids: W123 or https://openalex.org/W123")
    rq.add_argument("--reason", required=True, help="why; logged to the failures table")

    rt = sub.add_parser("retry",
                        help="clear the attempt counter on papers stuck at failed")
    rt.set_defaults(fn=cmd_retry)
    rt.add_argument("ids", nargs="*", help="OpenAlex ids: W123 or the full URL")
    rt.add_argument("--ids-from", default="", help="file with one id per line")
    rt.add_argument("--reason", required=True, help="why; logged to the failures table")

    rc = sub.add_parser("recode",
                        help="discard a classified paper's text AND codes, then redo both")
    rc.set_defaults(fn=cmd_recode)
    rc.add_argument("ids", nargs="*", help="OpenAlex ids: W123 or the full URL")
    rc.add_argument("--ids-from", default="", help="file with one id per line")
    rc.add_argument("--reason", required=True, help="why; logged to the failures table")
    rc.add_argument("--yes", action="store_true",
                    help="actually do it; without this the command only reports")

    ex = sub.add_parser("exclude",
                        help="take papers out of the corpus; row and codes are kept")
    ex.set_defaults(fn=cmd_exclude, as_status="excluded")
    ou = sub.add_parser("outdated",
                        help="mark papers outside the study window; row and codes are kept")
    ou.set_defaults(fn=cmd_exclude, as_status="outdated")
    for q in (ex, ou):
        q.add_argument("ids", nargs="*", help="OpenAlex ids: W123 or the full URL")
        q.add_argument("--ids-from", default="", help="file with one id per line")
        q.add_argument("--reason", required=True, help="why; logged to the failures table")
        q.add_argument("--yes", action="store_true",
                       help="actually do it; without this the command only reports")

    pd_ = sub.add_parser("preprint-das",
                         help="read a paper's arXiv copy in full; record whether it has a statement")
    pd_.set_defaults(fn=cmd_preprint_das)
    pd_.add_argument("ids", nargs="+", help="OpenAlex ids: W123 or the full URL")

    aj = sub.add_parser("adjudicate",
                        help="a human coder's field-by-field settlement of one finished paper")
    aj.set_defaults(fn=cmd_adjudicate)
    aj.add_argument("id", help="OpenAlex id: W123 or the full URL")
    aj.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                    help="a flat record field; VALUE is parsed as JSON when it can be")
    aj.add_argument("--coder", required=True, help="the human coder, e.g. human:diana")
    aj.add_argument("--reason", required=True, help="why; logged with the replaced record")
    aj.add_argument("--reverify", action="store_true",
                    help="send the paper back to the verify stage even though no "
                         "identifier changed. Needed when a settled case would "
                         "change what verify.py derives: `Actual` and any "
                         "promotion to access_unresolvable are written only "
                         "there, never asserted by hand (hard rule 6)")
    aj.add_argument("--yes", action="store_true",
                    help="actually do it; without this the command only reports")

    ad = sub.add_parser("adopt",
                        help="copy papers, codes and history from another database; append only")
    ad.set_defaults(fn=cmd_adopt)
    ad.add_argument("ids", nargs="*", help="OpenAlex ids: W123 or the full URL")
    ad.add_argument("--from", dest="src", required=True, help="the database to copy from")
    ad.add_argument("--ids-from", default="", help="file with one id per line")
    ad.add_argument("--reason", required=True, help="why; logged to the failures table")

    st = sub.add_parser("status"); st.set_defaults(fn=cmd_status)
    x = sub.add_parser("export"); x.set_defaults(fn=cmd_export); x.add_argument("--out", default="out")

    a = p.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
