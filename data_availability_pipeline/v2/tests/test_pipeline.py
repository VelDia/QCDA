"""End-to-end stage machine, offline. This is the test run.py never had.

Every network boundary is faked; the stages, the store transitions, the spool,
the batch bookkeeping, the manual export/import gate and the verify derivation
are real. If this passes, the ASSEMBLY works — not merely the pieces.

    python3 tests/test_pipeline.py
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import pytest
except ImportError:
    from test_guards import pytest

import dap.pipeline as P
import dap.sources as sources
import dap.verify as verify
from dap.batch import Spool, safe_custom_id
from dap.batch import collect as batch_collect
from dap.batch import submit as batch_submit
from dap.sources import Config
from dap.store import Store
from fakes import (
    CROSSREF_BARE,
    CROSSREF_WITH_DAS,
    DATACITE_DATASET,
    DATACITE_DATASET_ATTRS,
    FakeBlock,
    FakeMessage,
    FakeResponse,
    datacite_bulk,
    fake_get_factory,
    good_payload,
)

CFG = Config(contact_email="researcher@institution.edu")
DB = "/tmp/_dap_pipeline.sqlite"

OPENALEX_WORK = {
    "title": "Transmon readout fidelity improvements",
    "publication_year": 2024,
    "primary_location": {"source": {"display_name": "PRX Quantum"}},
    "open_access": {"is_oa": True, "oa_status": "gold", "oa_url": "https://oa.example/paper.pdf"},
    "best_oa_location": {"pdf_url": "https://oa.example/paper.pdf"},
    "locations": [],
    "ids": {"openalex": "https://openalex.org/W1"},
}

ARTICLE = (
    "Transmon readout fidelity improvements were measured on our device. "
    * 60
    + "Data availability: The datasets generated during the current study are "
    "available in the Zenodo repository, "
    "https://doi.org/10.5281/zenodo.10499999 under CC-BY. "
    + "We simulate 10,000 random circuit instances with seed 42. "
    * 20
)


def fresh_store():
    # The DataCite resolver cache is module-level and shared across threads, so
    # it must not leak verdicts between tests.
    with sources._dc_lock:
        sources._dc_cache.clear()
    for suf in ("", "-wal", "-shm"):
        Path(DB + suf).unlink(missing_ok=True)
    s = Store(DB)
    s.enrol(
        [
            {
                "openalex_id": "https://openalex.org/W1",
                "doi": "10.1038/x",
                "arm": "quantum_computing",
            },
            {
                "openalex_id": "https://openalex.org/W2",
                "doi": "10.1145/1",
                "arm": "machine_learning",
            },
        ]
    )
    return s


def with_network(fn):
    routes = {
        "api.openalex.org": FakeResponse(OPENALEX_WORK),
        "api.crossref.org/works/10.1038": FakeResponse(CROSSREF_WITH_DAS),
        "api.crossref.org/works/10.1145": FakeResponse(CROSSREF_BARE),
        "api.datacite.org/dois/10.5061": FakeResponse(DATACITE_DATASET),
        "api.datacite.org": sources.PermanentError("HTTP 404"),
        "oa.example": FakeResponse(
            headers={"Content-Type": "text/html"},
            content=f"<html><body><article>{ARTICLE}</article></body></html>".encode(),
        ),
        "api.springer.com": sources.PermanentError("HTTP 403"),
        "api.unpaywall.org": sources.PermanentError("HTTP 404"),
    }
    orig_get, orig_pget = sources._get, P._get
    sources._get = P._get = fake_get_factory(routes)
    try:
        return fn()
    finally:
        sources._get, P._get = orig_get, orig_pget


# ---------------------------------------------------------------------------
# Regression: the reference list was truncated to the first 25 DOIs.
#
# Measured on the first live 200-paper run: the cap hid 68.6% of references
# (7,861 of 11,465), and ALL THREE confirmed datasets in the whole sample sat
# beyond reference 25. Data and software get cited after the literature, so a
# prefix of the reference list is the part least likely to contain them.
# ---------------------------------------------------------------------------
def test_every_reference_is_checked_not_just_the_first_25():
    s = fresh_store()
    # 40 references; the only dataset is at position 30, past the old cap.
    refs = [{"DOI": f"10.1103/physrevlett.{i}"} for i in range(40)]
    refs[30] = {"DOI": "10.5281/zenodo.999"}
    crossref = {
        "message": {
            "type": "journal-article",
            "container-title": ["PRX Quantum"],
            "published": {"date-parts": [[2024]]},
            "reference": refs,
        }
    }
    queries = []

    def routes(cfg, url, **kw):
        if "api.openalex.org" in url:
            return FakeResponse(OPENALEX_WORK)
        if "api.crossref.org" in url:
            return FakeResponse(crossref)
        if "api.datacite.org" in url:
            q = kw.get("params", {}).get("query", "")
            queries.append(q)
            return FakeResponse(datacite_bulk(("10.5281/zenodo.999", DATACITE_DATASET_ATTRS)))
        raise AssertionError(f"unrouted: {url}")

    orig_get, orig_pget = sources._get, P._get
    sources._get = P._get = routes
    try:
        for row in s.claim("enrolled", 10):
            P.stage_metadata_one(CFG, s, row)
    finally:
        sources._get, P._get = orig_get, orig_pget

    asked = " ".join(queries)
    assert "10.5281/zenodo.999" in asked, "the late dataset reference must be queried"
    assert "10.1103/physrevlett.39" in asked, "reference 40 must not be truncated away"
    w1 = next(r for r in s.claim("metadata", 10) if r["openalex_id"].endswith("W1"))
    assert w1["meta"]["confirmed_datasets"] == ["10.5281/zenodo.999"], (
        "a dataset cited past reference 25 is still Axis A evidence"
    )


def test_metadata_stage_advances_and_carries_facts():
    s = fresh_store()

    def go():
        for row in s.claim("enrolled", 10):
            P.stage_metadata_one(CFG, s, row)

    with_network(go)
    counts = s.stage_counts()
    assert counts.get("metadata/pending") == 2
    w1 = next(r for r in s.claim("metadata", 10) if r["openalex_id"].endswith("W1"))
    assert w1["meta"]["das_texts"], "DAS text from Crossref should ride along"
    assert w1["meta"]["decisive"] is True, "DAS + structural link = decisive"
    w2 = next(r for r in s.claim("metadata", 10) if r["openalex_id"].endswith("W2"))
    assert w2["meta"]["decisive"] is False


def test_retrieve_skips_fulltext_when_metadata_is_decisive():
    s = fresh_store()

    def go():
        for row in s.claim("enrolled", 10):
            P.stage_metadata_one(CFG, s, row)
        for row in s.claim("metadata", 10):
            P.stage_retrieve_one(CFG, s, row, metadata_only=False, pool=None)

    with_network(go)
    rows = {r["openalex_id"][-2:]: r["meta"] for r in s.claim("retrieved", 10)}
    assert rows["W1"]["retrieval"] == "metadata_only", "decisive paper never fetched a PDF"
    assert rows["W1"]["passages"] == []
    assert rows["W2"]["retrieval"] == "full_text", "undecided paper did"
    assert any("zenodo" in p["text"].lower() for p in rows["W2"]["passages"])


def test_manual_export_import_round_trip_is_gated():
    s = fresh_store()

    def go():
        for row in s.claim("enrolled", 10):
            P.stage_metadata_one(CFG, s, row)
        for row in s.claim("metadata", 10):
            P.stage_retrieve_one(CFG, s, row, metadata_only=False, pool=None)

    with_network(go)

    out = Path("/tmp/_dap_prompts.jsonl")
    n = P.classify_manual_export(s, s.claim("retrieved", 10), out)
    assert n == 2
    lines = [json.loads(l) for l in out.open()]
    assert lines[0].get("_header") and "output_schema" in lines[0]
    assert (
        "DATA AVAILABILITY STATEMENT" in lines[1]["prompt"]
        or "DATA AVAILABILITY STATEMENT" in lines[2]["prompt"]
    )

    # the agent answers: one valid, one out-of-vocabulary
    good = good_payload()
    bad = good_payload()
    bad["axis_b"]["case"]["value"] = "fully_open"
    res = Path("/tmp/_dap_results.jsonl")
    with res.open("w") as f:
        f.write(json.dumps({"custom_id": lines[1]["custom_id"], "classification": good}) + "\n")
        f.write(json.dumps({"custom_id": lines[2]["custom_id"], "classification": bad}) + "\n")
    t = P.classify_manual_import(s, res)
    assert t == {"accepted": 1, "rejected": 1, "unknown": 0}
    counts = s.stage_counts()
    assert counts.get("classified/pending") == 1
    assert counts.get("retrieved/failed") == 1, "rejected paper stays put, marked failed"


def test_batch_spool_submit_collect():
    s = fresh_store()

    def go():
        for row in s.claim("enrolled", 10):
            P.stage_metadata_one(CFG, s, row)
        for row in s.claim("metadata", 10):
            P.stage_retrieve_one(CFG, s, row, metadata_only=False, pool=None)

    with_network(go)

    import shutil

    shutil.rmtree("/tmp/_dap_spool", ignore_errors=True)
    spool = Spool("/tmp/_dap_spool")
    n = P.classify_spool(s, spool, s.claim("retrieved", 10), model="fake-model")
    assert n == 2
    assert spool.ready_files() == [], "below BATCH_SIZE nothing is 'ready'"
    assert len(spool.ready_files(include_partial=True)) == 1

    class FakeBatch:
        def __init__(self):
            self.id = "msgbatch_test1"
            self.processing_status = "ended"

    class FakeBatches:
        def __init__(self, results):
            self._r = results

        def create(self, requests):
            assert len(requests) == 2 and all("custom_id" in r and "params" in r for r in requests)
            return FakeBatch()

        def retrieve(self, bid):
            return FakeBatch()

        def results(self, bid):
            return iter(self._r)

    class R:
        def __init__(self, cid, ok=True):
            self.custom_id = cid
            from types import SimpleNamespace as NS

            if ok:
                self.result = NS(type="succeeded", message=FakeMessage([FakeBlock(good_payload())]))
            else:
                self.result = NS(type="errored", message=None, error={"type": "x"})

    cids = [safe_custom_id("https://openalex.org/W1"), safe_custom_id("https://openalex.org/W2")]
    client = type("C", (), {})()
    client.messages = type("M", (), {})()
    client.messages.batches = FakeBatches([R(cids[0]), R(cids[1], ok=False)])

    assert batch_submit(client, s, spool, flush=True, log=lambda *_: None) == 1
    assert s.open_batches() == ["msgbatch_test1"]
    assert spool.ready_files(include_partial=True) == [], "spool deleted after batch recorded"
    assert s.stage_counts().get("classify_queued/pending") == 2

    t = batch_collect(client, s, log=lambda *_: None)
    assert t["succeeded"] == 1 and t["errored"] == 1
    assert s.open_batches() == []
    c = s.stage_counts()
    assert c.get("classified/pending") == 1
    assert c.get("classify_queued/dead") == 1, "errored batch result is parked, not retried forever"


def test_verify_stage_derives_unresolvable_from_dead_link():
    s = fresh_store()

    def go():
        for row in s.claim("enrolled", 10):
            P.stage_metadata_one(CFG, s, row)
        for row in s.claim("metadata", 10):
            P.stage_retrieve_one(CFG, s, row, metadata_only=False, pool=None)

    with_network(go)
    out = Path("/tmp/_dap_prompts.jsonl")
    P.classify_manual_export(s, s.claim("retrieved", 10), out)
    lines = [json.loads(l) for l in out.open()]
    res = Path("/tmp/_dap_results.jsonl")
    with res.open("w") as f:
        f.write(
            json.dumps({"custom_id": lines[1]["custom_id"], "classification": good_payload()})
            + "\n"
        )
    P.classify_manual_import(s, res)

    class _Req:
        import requests as _real

        RequestException = _real.RequestException

        def head(self, *a, **k):
            return FakeResponse(status=404)

        def get(self, *a, **k):
            return FakeResponse(status=404)

    orig = verify.requests
    verify.requests = _Req()
    try:
        for row in s.claim("classified", 10):
            P.stage_verify_one(CFG, s, row)
    finally:
        verify.requests = orig

    with s._conn() as c:
        flat = json.loads(
            c.execute("SELECT record_json FROM papers WHERE stage='verified'").fetchone()[
                "record_json"
            ]
        )
    assert flat["b_actual"] == "not_found"
    assert flat["b_case"] == "access_unresolvable", (
        "openly_accessible + dead link => derived, not asserted"
    )
    assert flat["b_openness_rank"] == 1


def test_export_after_full_run():
    test_verify_stage_derives_unresolvable_from_dead_link()
    s = Store(DB)
    out = s.export_csv("/tmp/_dap_an.csv", "/tmp/_dap_ev.csv", "/tmp/_dap_rq.csv")
    assert out["analysis"] == 1 and out["evidence"] == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ---------------------------------------------------------------------------
# enrol's row filter. Both bugs below are silent: they produce a corpus that is
# quietly not what was asked for, which no downstream stage can detect.
# ---------------------------------------------------------------------------
def _enrol(tmpcsv, **kw):

    a = argparse.Namespace(
        db=DB, input=str(tmpcsv), arm_map=None, year_min=None, year_max=None, limit=None
    )
    for k, v in kw.items():
        setattr(a, k, v)
    for suf in ("", "-wal", "-shm"):
        Path(DB + suf).unlink(missing_ok=True)
    P.cmd_enrol(a)
    s = Store(DB)
    return [r["openalex_id"] for r in s.claim("enrolled", 10**6)]


def _write_csv(tmp_path, rows):
    import csv

    f = tmp_path / "in.csv"
    with open(f, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["openalex_id", "doi", "publication_year"])
        w.writerows(rows)
    return f


def test_both_year_bounds_filter_on_an_aligned_mask(tmp_path, recwarn):
    """Applying year_min then year_max left the second mask carrying the
    pre-filter index. pandas reindexed it with a UserWarning; newer pandas
    raises. Real years arrive as floats ("2018.0"), as they do in the corpus.
    """
    csv_ = _write_csv(
        tmp_path,
        [
            (f"W{i}", f"10.1/{i}", y)
            for i, y in enumerate(["2014.0", "2017.0", "2020.0", "2026.0", "2027.0", ""])
        ],
    )
    got = _enrol(csv_, year_min=2017, year_max=2026)
    assert got == ["W1", "W2", "W3"], "only 2017-2026 rows, in order"
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)], (
        "a misaligned boolean mask must not be silently reindexed"
    )


def test_limit_yields_that_many_distinct_papers(tmp_path):
    """`--limit N` truncated before deduplicating, so a slice containing a
    repeated openalex_id enrolled fewer than N papers.
    """
    csv_ = _write_csv(
        tmp_path,
        [
            ("W1", "10.1/a", "2020.0"),
            ("W1", "10.1/a", "2020.0"),
            ("W2", "10.1/b", "2020.0"),
            ("W3", "10.1/c", "2020.0"),
        ],
    )
    got = _enrol(csv_, limit=3)
    assert got == ["W1", "W2", "W3"], f"asked for 3 distinct papers, got {got}"


def test_enrol_stores_the_harvest_year_so_the_window_applies_at_once(tmp_path):
    """Enrolling 2017-2026 while scheduling only 2021+ needs the year before
    the metadata stage runs; without it every enrolled paper counts as
    'year unknown' and is fetched regardless of age.
    """
    csv_ = _write_csv(
        tmp_path, [("W1", "10.1/a", "2019.0"), ("W2", "10.1/b", "2023.0"), ("W3", "10.1/c", "")]
    )
    assert _enrol(csv_) == ["W1", "W2", "W3"]
    s = Store(DB)
    assert [r["openalex_id"] for r in s.claim("enrolled", 10, year_min=2021)] == ["W2", "W3"], (
        "2019 is outside the window; an unknown year is still kept"
    )


# ---------------------------------------------------------------------------
# Regression: an evidence-free paper must never reach a model.
#
# Measured on the first live run: 78 of 200 papers (39%) reached `retrieved`
# with no passages, no DAS and no dataset link. The prompt asked the model to
# set needs_human_review in that case; nothing enforced it, and a model handed
# only a title can quote the title back and return a confident code that never
# escalates and never flags. Gate before the request, in every mode.
# ---------------------------------------------------------------------------
def _bare_retrieved():
    """One paper at `retrieved` with nothing to read but its title."""
    s = fresh_store()

    def go():
        for row in s.claim("enrolled", 10):
            P.stage_metadata_one(CFG, s, row)
        for row in s.claim("metadata", 10):
            P.stage_retrieve_one(CFG, s, row, metadata_only=True, pool=None)

    with_network(go)
    rows = [r for r in s.claim("retrieved", 10) if r["openalex_id"].endswith("W2")]
    assert rows and not rows[0]["meta"].get("passages")
    assert not rows[0]["meta"].get("das_texts")
    return s, rows


def test_evidence_free_paper_is_never_sent_to_the_api():
    s, rows = _bare_retrieved()

    class Exploding:
        class messages:
            @staticmethod
            def create(**kw):
                raise AssertionError("a paper with no evidence reached the model")

    P.classify_api_one(Exploding(), s, rows[0], escalate=True)
    with s._conn() as c:
        rec = json.loads(
            c.execute(
                "SELECT record_json FROM papers WHERE openalex_id=?", (rows[0]["openalex_id"],)
            ).fetchone()["record_json"]
        )
    assert rec["needs_human_review"] is True
    assert rec["review_reason"], "a flagged paper must say why"
    assert rec["a_case_conf"] == "low" and rec["b_case_conf"] == "low", (
        "nothing here may read as a coded result"
    )
    assert rec["model_id"] == "", "no model was consulted, so none may be recorded"


def test_evidence_free_paper_is_not_spooled_for_batch():
    s, rows = _bare_retrieved()
    import shutil

    shutil.rmtree("/tmp/_dap_spool_gate", ignore_errors=True)
    n = P.classify_spool(s, Spool("/tmp/_dap_spool_gate"), rows, model="fake-model")
    assert n == 0, "batch mode must apply the same gate as the API mode"


def test_evidence_free_paper_is_not_exported_for_manual_coding():
    s, rows = _bare_retrieved()
    out = Path("/tmp/_dap_gate_manual.jsonl")
    out.unlink(missing_ok=True)
    n = P.classify_manual_export(s, rows, out)
    assert n == 0, "manual mode must apply the same gate too"
    body = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert all(o.get("_header") for o in body), "no paper prompts may be written"


def test_a_paper_with_only_a_das_is_still_classifiable():
    """The gate asks whether there is anything to read, not whether full text
    was retrieved. A publisher-deposited DAS is enough on its own.
    """
    s = fresh_store()

    def go():
        for row in s.claim("enrolled", 10):
            P.stage_metadata_one(CFG, s, row)
        for row in s.claim("metadata", 10):
            P.stage_retrieve_one(CFG, s, row, metadata_only=True, pool=None)

    with_network(go)
    w1 = next(r for r in s.claim("retrieved", 10) if r["openalex_id"].endswith("W1"))
    assert not w1["meta"].get("passages") and w1["meta"]["das_texts"]
    assert P.has_classifiable_evidence(
        crossref_summary=P._summary(w1["meta"]), passages=w1["meta"].get("passages", [])
    ), "a DAS with no full text must still be classified, not gated"


def test_a_cited_qiskit_doi_is_not_evidence_about_the_citing_paper():
    """The evidence gate's dataset-id clause is built from the reference list, so
    a cited deposit counted the same as a deposited one. Measured on dap_frame:
    of the 20 classified rows that reached a coder with no passages and no DAS,
    11 passed on one of the two Qiskit Zenodo records alone -- a title, a venue
    and a year, which is exactly what the gate exists to refuse.
    """
    cs = {"das_texts": [], "dataset_ids": [], "confirmed_datasets": ["10.5281/zenodo.2562111"]}
    assert not P.has_classifiable_evidence(crossref_summary=cs, passages=[])


def test_a_real_cited_deposit_still_counts_as_evidence():
    """The clause carries real signal and must keep it: a paper with no
    retrievable text but a confirmed Dryad or project deposit is codable, and
    two rows in the corpus reached `openly_accessible` that way.
    """
    cs = {"das_texts": [], "dataset_ids": [], "confirmed_datasets": ["10.5061/dryad.k6t1rj8"]}
    assert P.has_classifiable_evidence(crossref_summary=cs, passages=[])
    mixed = {
        "das_texts": [],
        "dataset_ids": [],
        "confirmed_datasets": ["10.5281/zenodo.2562111", "10.5061/dryad.k6t1rj8"],
    }
    assert P.has_classifiable_evidence(crossref_summary=mixed, passages=[]), (
        "one ubiquitous record must not mask a real one beside it"
    )


def test_publisher_doi_only_covers_the_preprint_servers_this_corpus_contains():
    """`--publisher-doi-only` is the flag that holds the brief's version-of-record
    scope, and it filters on the DOI prefix. Batch 108 exported an engrXiv
    preprint (`W4413166857`, `10.31224/5086`) because engrXiv was not listed and
    its venue string was empty, so neither the prefix nor the venue test fired.
    Measured on dap_frame: 273 queued rows carry one of the seven prefixes added
    in S100 and the venue test would have caught only 83.
    """
    from dap.store import PREPRINT_DOI_PREFIXES

    for prefix in (
        "10.48550/",
        "10.5281/",
        "10.21203/",
        "10.2139/",
        "10.20944/",
        "10.6084/",
        "10.60893/",
        "10.36227/",
        "10.26434/",
        "10.31219/",
        "10.22541/",
        "10.1101/",
        "10.31224/",
        "10.31234/",
    ):
        assert prefix in PREPRINT_DOI_PREFIXES, f"{prefix} is a preprint registrar"


def test_publisher_doi_only_does_not_reject_journal_prefixes():
    """The boundary. These are the publisher prefixes carrying most of the coded
    corpus; a prefix added to the preprint list by mistake would silently remove
    a whole publisher from the claim.
    """
    from dap.store import PREPRINT_DOI_PREFIXES

    for prefix in (
        "10.1038/",
        "10.1103/",
        "10.1007/",
        "10.1088/",
        "10.1016/",
        "10.22331/",
        "10.1021/",
        "10.1109/",
        "10.3390/",
        "10.1017/",
    ):
        assert prefix not in PREPRINT_DOI_PREFIXES, f"{prefix} is a journal publisher"


def test_the_gate_covers_every_framework_release_not_just_qiskit():
    """The set was two Qiskit records until batch 102 hit a third. Resolving all
    120 distinct `confirmed_datasets` DOIs in the corpus against DataCite found
    19 tool releases, and two rows -- `W4213428754` and `W7135088349` -- had
    already passed the gate on a Cirq and a Qiskit Nature release alone. Each of
    these is a release of a tool the field cites, so none of them is evidence
    about the citing paper.
    """
    for doi in (
        "10.5281/zenodo.2573505",  # Qiskit metapackage 0.44.0
        "10.5281/zenodo.4062499",  # Cirq
        "10.5281/zenodo.7828768",  # Qiskit Nature 0.6.0
        "10.5281/zenodo.4023103",  # qsim
        "10.17637/rh.7000520",  # Quantikz, a drawing package
        "10.5281/zenodo.820864",
    ):  # SageMath
        cs = {"das_texts": [], "dataset_ids": [], "confirmed_datasets": [doi]}
        assert not P.has_classifiable_evidence(crossref_summary=cs, passages=[]), (
            f"{doi} is a cited tool release, not evidence about the citing paper"
        )


def test_a_deposit_released_for_a_paper_is_not_a_framework():
    """The boundary the list must not cross. Frequency cannot draw it: Quantikz
    is excluded at 4 citing papers while these are kept at 1 and 2, because what
    matters is whether the record was released for a paper or for the field.
    """
    for doi in (
        "10.5281/zenodo.1034811",  # pysme, "for upcoming paper"
        "10.15128/r27d278t029",  # one paper's own deposit
        "10.5281/zenodo.7545881",
    ):  # GregDMeyer/IQPwn: v1.0
        cs = {"das_texts": [], "dataset_ids": [], "confirmed_datasets": [doi]}
        assert P.has_classifiable_evidence(crossref_summary=cs, passages=[]), (
            f"{doi} is a paper's own artefact and must stay codable"
        )


def test_manual_import_applies_the_same_grounding_rule_as_the_api():
    """--mode manual routes around classify(), so the grounding check has to be
    applied at import too. It is the likeliest place for a paraphrase to slip in.
    """
    s = fresh_store()

    def go():
        for row in s.claim("enrolled", 10):
            P.stage_metadata_one(CFG, s, row)
        for row in s.claim("metadata", 10):
            P.stage_retrieve_one(CFG, s, row, metadata_only=False, pool=None)

    with_network(go)
    rows = [r for r in s.claim("retrieved", 10) if r["openalex_id"].endswith("W2")]
    cid = safe_custom_id(rows[0]["openalex_id"])

    payload = json.loads(json.dumps(good_payload()))
    payload["evidence"] = [
        {"id": 1, "quote": "the authors state data are available on request"},
        {"id": 2, "quote": "the authors state data are available on request"},
    ]
    path = Path("/tmp/_dap_manual_grounding.jsonl")
    path.write_text(json.dumps({"custom_id": cid, "classification": payload}) + "\n")

    tally = P.classify_manual_import(s, path)
    assert tally["accepted"] == 0 and tally["rejected"] == 1, (
        "an ungrounded quote must be rejected in manual mode too"
    )


# ---------------------------------------------------------------------------
# Regression: batch collect never checked grounding. The spool stores a slim copy
# of each paper's metadata with no passages, so collect called _assemble without
# them — on the path CLAUDE.md designates for the census. A fabricated quote was
# accepted there while the API and manual paths rejected it.
# ---------------------------------------------------------------------------
def _run_batch(payload_for_w1):
    s = fresh_store()

    def go():
        for row in s.claim("enrolled", 10):
            P.stage_metadata_one(CFG, s, row)
        for row in s.claim("metadata", 10):
            P.stage_retrieve_one(CFG, s, row, metadata_only=False, pool=None)

    with_network(go)
    import shutil

    shutil.rmtree("/tmp/_dap_spool_g", ignore_errors=True)
    spool = Spool("/tmp/_dap_spool_g")
    P.classify_spool(s, spool, s.claim("retrieved", 10), model="fake-model")
    from types import SimpleNamespace as NS

    class B:
        id = "msgbatch_g"
        processing_status = "ended"

    cids = [safe_custom_id("https://openalex.org/W1"), safe_custom_id("https://openalex.org/W2")]
    results = [
        NS(
            custom_id=cids[0],
            result=NS(type="succeeded", message=FakeMessage([FakeBlock(payload_for_w1)])),
        ),
        NS(custom_id=cids[1], result=NS(type="errored", message=None)),
    ]

    class Batches:
        def create(self, requests):
            return B()

        def retrieve(self, bid):
            return B()

        def results(self, bid):
            return iter(results)

    client = NS(messages=NS(batches=Batches()))
    batch_submit(client, s, spool, flush=True, log=lambda *_: None)
    return s, batch_collect(client, s, log=lambda *_: None)


def test_batch_collect_rejects_a_fabricated_quote():
    bad = json.loads(json.dumps(good_payload()))
    bad["evidence"] = [
        {"id": 1, "quote": "the authors state data are available on request"},
        {"id": 2, "quote": "the authors state data are available on request"},
    ]
    s, t = _run_batch(bad)
    assert t["invalid"] == 1, "an ungrounded quote must not be accepted on the batch path"
    assert s.stage_counts().get("classified/pending") is None, "nothing may be saved"


def test_batch_collect_still_accepts_a_grounded_record_and_labels_its_das():
    ok = json.loads(json.dumps(good_payload()))
    ok["das_present"] = True
    s, t = _run_batch(ok)
    assert t["succeeded"] == 1 and t["invalid"] == 0
    with s._conn() as c:
        rec = json.loads(
            c.execute(
                "SELECT record_json FROM papers WHERE openalex_id=?", ("https://openalex.org/W1",)
            ).fetchone()["record_json"]
        )
    assert rec["das_source"] == "crossref_assertion", (
        "W1's statement is a Crossref deposit, read back from the stored metadata"
    )


def test_manual_import_labels_a_crossref_das_as_crossref_assertion():
    """The manual-import call site. W1's statement is a Crossref deposit, so a
    dropped crossref_das argument would silently relabel it full_text.
    """
    s = fresh_store()

    def go():
        for row in s.claim("enrolled", 10):
            P.stage_metadata_one(CFG, s, row)
        for row in s.claim("metadata", 10):
            P.stage_retrieve_one(CFG, s, row, metadata_only=False, pool=None)

    with_network(go)
    payload = json.loads(json.dumps(good_payload()))
    payload["das_present"] = True
    path = Path("/tmp/_dap_manual_das.jsonl")
    path.write_text(
        json.dumps(
            {"custom_id": safe_custom_id("https://openalex.org/W1"), "classification": payload}
        )
        + "\n"
    )
    tally = P.classify_manual_import(s, path)
    assert tally["accepted"] == 1
    with s._conn() as c:
        rec = json.loads(
            c.execute(
                "SELECT record_json FROM papers WHERE openalex_id=?", ("https://openalex.org/W1",)
            ).fetchone()["record_json"]
        )
    assert rec["das_source"] == "crossref_assertion"


# ---------------------------------------------------------------------------
# Regression: two silent retrieval failures, both found re-running a live slice.
#
# 1. A throttled location read like a missing one, so the cascade settled for a
#    worse copy (a preprint, or no text) and logged nothing. Found investigating
#    why re-running the same 200-paper slice moved 5 of 189 papers to a
#    different source; the cause of that drift was never recorded.
# 2. stage_retrieve_one rebuilt CrossrefFacts without reference_count, so the
#    completeness check compared every text against 0 references and always
#    passed. In the venue frame all 4,654 fetched papers were marked complete,
#    4,525 of them with ten or more deposited references.
# ---------------------------------------------------------------------------
ARTICLE_PAGE = FakeResponse(
    headers={"Content-Type": "text/html"},
    content=f"<html><body><article>{ARTICLE}</article></body></html>".encode(),
)


def _metadata_done():
    s = fresh_store()

    def go():
        for row in s.claim("enrolled", 10):
            P.stage_metadata_one(CFG, s, row)

    with_network(go)
    return s


def _retrieve_with(s, rows, routes):
    orig = sources._get
    sources._get = fake_get_factory(routes)
    try:
        for row in rows:
            P.stage_retrieve_one(CFG, s, row, metadata_only=False, pool=None)
    finally:
        sources._get = orig


def _row(s, stage, suffix):
    return next(r for r in s.claim(stage, 10) if r["openalex_id"].endswith(suffix))


def test_a_throttled_copy_leaves_the_paper_at_metadata_for_a_retry():
    s = _metadata_done()
    _retrieve_with(
        s,
        s.claim("metadata", 10),
        {
            "oa.example": sources.TransientError("exhausted 3 attempts: HTTP 429"),
            "api.unpaywall.org": sources.PermanentError("HTTP 404"),
        },
    )
    counts = s.stage_counts()
    assert counts.get("metadata/failed") == 1, (
        "W2 must wait for a retry, not be stored as metadata-only"
    )
    assert counts.get("retrieved/pending") == 1, "W1 is decisive and never fetched"
    with s._conn() as c:
        f = c.execute(
            "SELECT error_type, message FROM failures WHERE openalex_id=?",
            ("https://openalex.org/W2",),
        ).fetchone()
    assert f["error_type"] == "transient"
    assert "oa.example" in f["message"] and "429" in f["message"], (
        "the failure must say which location was throttled"
    )
    assert _row(s, "metadata", "W2")["attempts"] == 1, "and it is claimable again"


def test_retrieve_records_every_location_it_tried():
    s = _metadata_done()
    _retrieve_with(s, s.claim("metadata", 10), {"oa.example": ARTICLE_PAGE})
    assert _row(s, "retrieved", "W2")["meta"]["retrieval_attempts"] == [
        {"url": "https://oa.example/paper.pdf", "source": "openalex_oa", "outcome": "ok"}
    ]


def test_a_reference_list_never_becomes_the_papers_evidence():
    """End to end from metadata. The body matches no keyword; the only keyword
    is 'simulation' inside cited titles, in an author-year list sparse enough
    (under the 120 bar) that the per-passage filter kept it. The list is cut
    before selection, so the paper is coded from its own opening and closing.
    """
    s = _metadata_done()
    body = "<h1>Transmon readout fidelity improvements</h1>" + "".join(
        f"<p>The operator norm is bounded below by the spectral gap in case {i}.</p>"
        for i in range(120)
    )
    refs = "".join(
        f"<p>Edward Aftandilian and Raluca Sauciuc. 20{10 + i % 9}. Building a simulation "
        f"of program analysis tools. In Working Conference on Source Code, {i}–{i + 9}.</p>"
        for i in range(40)
    )
    page = FakeResponse(
        headers={"Content-Type": "text/html"},
        content=f"<html><body><article>{body}<h2>References</h2>{refs}"
        f"</article></body></html>".encode(),
    )
    _retrieve_with(s, [_row(s, "metadata", "W2")], {"oa.example": page})
    meta = _row(s, "retrieved", "W2")["meta"]
    assert meta["passages_are_structural"] is True
    assert [p["section"] for p in meta["passages"]] == ["opening", "closing"]
    assert not any("Aftandilian" in p["text"] for p in meta["passages"]), (
        "the closing passage must be the paper's end, not its reference list"
    )


def _preprint_run(s, oid, text):
    orig_get, orig_pdf = P._get, sources._extract_pdf
    P._get = lambda cfg, url, **kw: FakeResponse(content=b"%PDF-1.4")
    sources._extract_pdf = lambda content, max_pages=60, cfg=None: (text, [])
    try:
        return P.preprint_das(CFG, s, oid)
    finally:
        P._get, sources._extract_pdf = orig_get, orig_pdf


def test_preprint_das_reads_the_arxiv_copy_and_survives_a_recode():
    """review_issues S7: the arXiv copy of a QST paper has no statement; the
    version of record, hand-fetched, is what gets coded. The preprint's
    absence is kept as a fact beside the codes, through the recode.
    """
    s, oid = _done_paper()
    with s._conn() as c:
        meta = json.loads(
            c.execute("SELECT meta_json FROM papers WHERE openalex_id=?", (oid,)).fetchone()[0]
        )
        meta["oa_locations"]["locations"].append({"pdf_url": "https://arxiv.org/pdf/2304.05683"})
        c.execute("UPDATE papers SET meta_json=? WHERE openalex_id=?", (json.dumps(meta), oid))
    got = _preprint_run(s, oid, "arXiv:2304.05683v1 [quant-ph] A GHZ state is generated. " * 300)
    assert got["das_found"] is False and got["version"] == "v1" and got["words"] > 1000
    assert s.requeue_for_recode(oid, reason="version of record hand-fetched") == ""
    kept = s.paper_meta(oid)["preprint_das"]
    assert kept["das_found"] is False and kept["url"] == "https://arxiv.org/pdf/2304.05683"


def test_preprint_das_finds_a_statement_anywhere_in_the_text():
    s, oid = _done_paper()
    with s._conn() as c:
        meta = json.loads(
            c.execute("SELECT meta_json FROM papers WHERE openalex_id=?", (oid,)).fetchone()[0]
        )
        meta["retrieval_attempts"] = [
            {"url": "https://arxiv.org/pdf/1", "source": "arxiv", "outcome": "ok"}
        ]
        c.execute("UPDATE papers SET meta_json=? WHERE openalex_id=?", (json.dumps(meta), oid))
    got = _preprint_run(
        s,
        oid,
        "Body. " * 2000 + "Data availability: the data that support the findings are on request.",
    )
    assert got["das_found"] is True and "data availability" in got["phrases"]


def test_a_paper_with_no_availability_heading_still_shows_its_back_matter():
    """End to end from metadata. A single stray keyword (here in a sentence of
    the body; in two-column PDFs, a reference fragment that shares a line with
    prose) must not be the classifier's only evidence: with no availability
    heading anywhere, the opening and closing come along.
    """
    s = _metadata_done()
    body = "<h1>Transmon readout fidelity improvements</h1>" + "".join(
        f"<p>The operator norm is bounded below by the spectral gap in case {i}.</p>"
        for i in range(120)
    )
    page = FakeResponse(
        headers={"Content-Type": "text/html"},
        content=(
            f"<html><body><article>{body}<p>A simulation confirms the bound.</p>{body}"
            f"<p>We thank the referees for their careful reading.</p></article></body></html>"
        ).encode(),
    )
    _retrieve_with(s, [_row(s, "metadata", "W2")], {"oa.example": page})
    meta = _row(s, "retrieved", "W2")["meta"]
    sections = [p["section"] for p in meta["passages"]]
    assert "opening" in sections and "closing" in sections
    assert any("simulation" in p["keywords"] for p in meta["passages"])
    assert "careful reading" in meta["passages"][-1]["text"]
    assert meta.get("structural_context") is True and not meta.get("passages_are_structural")


def test_a_paper_with_an_availability_heading_gets_no_extra_context():
    s = _metadata_done()
    _retrieve_with(s, [_row(s, "metadata", "W2")], {"oa.example": ARTICLE_PAGE})
    meta = _row(s, "retrieved", "W2")["meta"]
    assert not meta.get("structural_context")
    assert all(p["section"] not in ("opening", "closing") for p in meta["passages"])


def test_a_foreign_language_record_page_is_not_stored_as_the_paper():
    """End to end from metadata (review_issues S8): a Spanish DSpace record
    carries none of the marker phrases, and was coded as though it were the
    article. It must be refused like any other record page.
    """
    s = _metadata_done()
    title = "Transmon readout fidelity improvements"
    abstract = "We report measured readout of a superconducting qubit and its error budget. " * 40
    page = FakeResponse(
        headers={"Content-Type": "text/html"},
        content=(
            f"<html><body><main><h1>{title}</h1><p>Citación: Pérez, J. (2024). {title}. "
            f"PRX Quantum 5, 020318.</p><p>Resumen {abstract}</p><p>Exportar a: Pérez J. "
            f"{title}. PRX Quantum. 2024;5:020318.</p></main></body></html>"
        ).encode(),
    )
    _retrieve_with(
        s,
        [_row(s, "metadata", "W2")],
        {
            "oa.example": page,
            "api.unpaywall.org": sources.PermanentError("HTTP 404"),
            "export.arxiv.org": FakeResponse(text="<feed></feed>"),
        },
    )
    meta = _row(s, "retrieved", "W2")["meta"]
    assert meta["retrieval"] == "metadata_only" and meta["passages"] == []
    assert meta["retrieval_attempts"][0]["outcome"] == "not_article"


def test_a_ligature_in_a_repository_url_reaches_the_classifier_folded():
    """End to end from metadata (review_issues S10): the classifier must be
    shown the address the paper prints, or the identifier it copies fails
    verification and the paper is coded access_unresolvable.
    """
    s = _metadata_done()
    page = FakeResponse(
        headers={"Content-Type": "text/html"},
        content=(
            f"<html><body><article>{ARTICLE} Data availability: the code is also "
            f"available at the following repository: github.com/leporini/classiﬁcation."
            f"</article></body></html>"
        ).encode(),
    )
    _retrieve_with(s, [_row(s, "metadata", "W2")], {"oa.example": page})
    text = " ".join(p["text"] for p in _row(s, "retrieved", "W2")["meta"]["passages"])
    assert "github.com/leporini/classification" in text
    assert "ﬁ" not in text


def test_retrieve_applies_the_deposited_reference_count():
    s = _metadata_done()
    w2 = _row(s, "metadata", "W2")
    w2["meta"]["reference_count"] = 60  # the publisher deposited 60 references
    _retrieve_with(
        s,
        [w2],
        {
            "oa.example": ARTICLE_PAGE,
            "api.unpaywall.org": sources.PermanentError("HTTP 404"),
            # a truncated body is a reason to keep looking, so
            # the cascade reaches the arXiv title search
            "export.arxiv.org": FakeResponse(text="<feed></feed>"),
        },
    )
    meta = _row(s, "retrieved", "W2")["meta"]
    assert meta["text_complete"] is False, (
        "a body with no bibliography, against 60 deposited references, is truncated"
    )
    assert meta["retrieval"] == "partial_text"


def test_requeue_sends_a_retrieved_paper_back_through_retrieval():
    s = fresh_store()

    def go():
        for row in s.claim("enrolled", 10):
            P.stage_metadata_one(CFG, s, row)
        for row in s.claim("metadata", 10):
            P.stage_retrieve_one(CFG, s, row, metadata_only=False, pool=None)

    with_network(go)
    assert (
        s.requeue_for_retrieval(
            "https://openalex.org/W2", reason="preprint taken while the VoR was throttled"
        )
        == ""
    )
    w2 = _row(s, "metadata", "W2")
    assert not set(Store.RETRIEVAL_KEYS) & set(w2["meta"]), (
        "the stale retrieval must not ride along"
    )
    assert w2["meta"]["doi"] and w2["meta"]["oa_locations"], "the metadata facts must"
    with s._conn() as c:
        assert (
            c.execute(
                "SELECT error_type FROM failures WHERE openalex_id=?", ("https://openalex.org/W2",)
            ).fetchone()["error_type"]
            == "requeued"
        )
    _retrieve_with(s, [w2], {"oa.example": ARTICLE_PAGE})
    assert _row(s, "retrieved", "W2")["meta"]["retrieval"] == "full_text"


def test_requeue_refuses_a_paper_that_already_carries_a_classification():
    s, rows = _bare_retrieved()
    P._gate_unclassifiable(s, rows[0]["openalex_id"], rows[0]["meta"])
    why = s.requeue_for_retrieval(rows[0]["openalex_id"], reason="x")
    assert "classified" in why, why
    assert s.stage_counts().get("classified/pending") == 1, "a coded paper keeps its text"
    assert s.requeue_for_retrieval("https://openalex.org/W404", reason="x") == "not enrolled"


# ---------------------------------------------------------------------------
# recode: the one case where discarding a classification is right.
#
# requeue_for_retrieval refuses a classified paper because swapping its text
# detaches the codes from their evidence. That objection is to KEEPING the
# codes. When the stored document is the wrong one — a catalogue record, a
# preprint standing in for the version of record — the codes are a reading of
# something that is not the paper, and both go together.
# ---------------------------------------------------------------------------
def _classified_paper():
    s, rows = _bare_retrieved()
    P._gate_unclassifiable(s, rows[0]["openalex_id"], rows[0]["meta"])
    return s, rows[0]["openalex_id"]


def test_recode_discards_the_codes_and_the_text_together():
    s, oid = _classified_paper()
    assert s.stage_counts().get("classified/pending") == 1
    assert s.requeue_for_recode(oid, reason="classified from an OSTI catalogue page") == ""
    row = _row(s, "metadata", "W2")
    assert not set(Store.RETRIEVAL_KEYS) & set(row["meta"]), "stale text must not ride along"
    assert row["meta"]["doi"], "the metadata facts survive"
    with s._conn() as c:
        rec = c.execute(
            "SELECT record_json, attempts FROM papers WHERE openalex_id=?", (oid,)
        ).fetchone()
        assert rec["record_json"] is None, "the classification must be gone, not kept"
        assert rec["attempts"] == 0, "a fresh start, not a paper three attempts from dead"


def test_recode_keeps_what_it_discarded_in_the_audit_trail():
    """Destructive, so the previous record has to survive somewhere."""
    s, oid = _classified_paper()
    s.requeue_for_recode(oid, reason="wrong document")
    with s._conn() as c:
        f = c.execute(
            "SELECT error_type, message FROM failures WHERE openalex_id=? AND error_type='recoded'",
            (oid,),
        ).fetchone()
    assert f, "the recode must be logged"
    assert "wrong document" in f["message"]
    assert "discarded record" in f["message"], "the codes that were thrown away must be recoverable"


def test_recode_clears_evidence_rows_tied_to_the_old_text():
    s, oid = _classified_paper()
    with s._conn() as c:
        c.execute(
            "INSERT INTO evidence VALUES (?,?,?,?,?,?)",
            (oid, 1, "a quote from the wrong document", "body", "", "full_text"),
        )
    s.requeue_for_recode(oid, reason="wrong document")
    with s._conn() as c:
        assert (
            c.execute("SELECT COUNT(*) n FROM evidence WHERE openalex_id=?", (oid,)).fetchone()["n"]
            == 0
        ), "quotes from a discarded document must not outlive it"


def test_recode_refuses_a_paper_that_is_not_enrolled():
    s, _ = _classified_paper()
    assert s.requeue_for_recode("https://openalex.org/W404", reason="x") == "not enrolled"


def test_a_recoded_paper_can_be_retrieved_and_classified_again():
    s, oid = _classified_paper()
    s.requeue_for_recode(oid, reason="wrong document")
    _retrieve_with(s, [_row(s, "metadata", "W2")], {"oa.example": ARTICLE_PAGE})
    assert _row(s, "retrieved", "W2")["meta"]["retrieval"] == "full_text", (
        "the point of the recode is that the paper goes all the way round again"
    )


# ---------------------------------------------------------------------------
# exclude: out of the corpus, not out of the database.
#
# A duplicate enrolment or a correction notice must stop being counted. A
# DELETE would not hold — enrol is INSERT OR IGNORE, so the next enrol from the
# CSV re-adds the paper as pending — and would lose the codes besides.
# ---------------------------------------------------------------------------
def _done_paper():
    test_verify_stage_derives_unresolvable_from_dead_link()
    s = Store(DB)
    with s._conn() as c:
        oid = c.execute("SELECT openalex_id FROM papers WHERE status='done'").fetchone()[0]
    return s, oid


# ---------------------------------------------------------------------------
# Merging a separate run: `adopt` copies a paper with everything that explains
# its codes, and never overwrites one that is already here.
# ---------------------------------------------------------------------------
SRC_DB = "/tmp/_dap_adopt_src.sqlite"


def _adopt_setup():
    import shutil

    s, oid = _done_paper()
    with s._conn() as c:
        c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    shutil.copy(DB, SRC_DB)
    dst_path = "/tmp/_dap_adopt_dst.sqlite"
    for suf in ("", "-wal", "-shm"):
        Path(dst_path + suf).unlink(missing_ok=True)
    return Store(dst_path), oid


def _rows(s, sql, oid):
    with s._conn() as c:
        return [tuple(r) for r in c.execute(sql, (oid,))]


def test_adopt_copies_the_codes_and_what_they_rest_on():
    dst, oid = _adopt_setup()
    assert dst.adopt(SRC_DB, oid, reason="merge") == ""
    src = Store(SRC_DB)
    for sql in (
        "SELECT stage, status, scheme_version, record_json, meta_json FROM papers WHERE openalex_id=?",
        "SELECT * FROM evidence WHERE openalex_id=? ORDER BY ev_id",
        "SELECT * FROM verifications WHERE openalex_id=? ORDER BY axis, identifier",
    ):
        assert _rows(dst, sql, oid) == _rows(src, sql, oid), sql
    kinds = [r[0] for r in _rows(dst, "SELECT error_type FROM failures WHERE openalex_id=?", oid)]
    assert kinds[-1] == "adopted", "the merge must be on the audit trail"


def test_adopt_never_overwrites_a_paper_already_here():
    dst, oid = _adopt_setup()
    dst.enrol([{"openalex_id": oid, "doi": "", "arm": "quantum_computing"}])
    assert dst.adopt(SRC_DB, oid, reason="merge") == "already enrolled here"
    with dst._conn() as c:
        assert (
            c.execute("SELECT record_json FROM papers WHERE openalex_id=?", (oid,)).fetchone()[0]
            is None
        )


def test_adopt_refuses_a_row_coded_under_another_scheme():
    dst, oid = _adopt_setup()
    import sqlite3

    with sqlite3.connect(SRC_DB) as c:
        c.execute("UPDATE papers SET scheme_version='1.0.0' WHERE openalex_id=?", (oid,))
    assert "scheme" in dst.adopt(SRC_DB, oid, reason="merge")
    assert dst.stage_counts() == {}


def test_an_excluded_paper_is_not_exported():
    s, oid = _done_paper()
    assert s.exclude(oid, reason="duplicate of W9") == ""
    out = s.export_csv("/tmp/_dap_an.csv")
    assert out["analysis"] == 0, "an excluded paper must not reach the denominator"


def test_exclusion_keeps_the_codes_and_logs_the_reason():
    s, oid = _done_paper()
    with s._conn() as c:
        before = c.execute("SELECT record_json FROM papers WHERE openalex_id=?", (oid,)).fetchone()[
            0
        ]
    s.exclude(oid, reason="duplicate of W9")
    with s._conn() as c:
        r = c.execute(
            "SELECT status, record_json FROM papers WHERE openalex_id=?", (oid,)
        ).fetchone()
        f = c.execute(
            "SELECT message FROM failures WHERE openalex_id=? AND error_type='excluded'", (oid,)
        ).fetchone()
    assert r["status"] == "excluded"
    assert r["record_json"] == before, "exclusion is not a recode; the codes stay"
    assert f and "duplicate of W9" in f["message"] and "verified/done" in f["message"]


def test_re_enrolling_does_not_bring_an_excluded_paper_back():
    s, oid = _done_paper()
    s.exclude(oid, reason="duplicate")
    assert s.enrol([{"openalex_id": oid, "doi": "x"}]) == 0
    with s._conn() as c:
        assert (
            c.execute("SELECT status FROM papers WHERE openalex_id=?", (oid,)).fetchone()[0]
            == "excluded"
        )


def test_an_excluded_paper_is_never_claimed_by_a_stage():
    s = fresh_store()
    oid = s.claim("enrolled", 10)[0]["openalex_id"]
    s.exclude(oid, reason="correction notice")
    assert oid not in {r["openalex_id"] for r in s.claim("enrolled", 10)}
    assert oid not in {r["openalex_id"] for r in s.pending()}


def test_exclude_refuses_an_unknown_or_already_excluded_paper():
    s, oid = _done_paper()
    assert s.exclude("https://openalex.org/W404", reason="x") == "not enrolled"
    s.exclude(oid, reason="x")
    assert s.exclude(oid, reason="x") == "already excluded"


# ---------------------------------------------------------------------------
# `outdated`: out of the study window. Kept apart from `excluded` so the two
# reasons can be counted, but removed from the corpus in exactly the same way.
# ---------------------------------------------------------------------------
def test_an_outdated_paper_is_kept_but_not_counted_or_claimed():
    s, oid = _done_paper()
    assert s.exclude(oid, reason="pre-2021", as_status="outdated") == ""
    assert s.export_csv("/tmp/_dap_an.csv")["analysis"] == 0
    assert s.enrol([{"openalex_id": oid, "doi": "x"}]) == 0
    with s._conn() as c:
        r = c.execute(
            "SELECT status, record_json FROM papers WHERE openalex_id=?", (oid,)
        ).fetchone()
        f = c.execute(
            "SELECT message FROM failures WHERE openalex_id=? AND error_type='outdated'", (oid,)
        ).fetchone()
    assert r["status"] == "outdated" and r["record_json"], "the codes stay"
    assert f and "pre-2021" in f["message"]
    assert s.exclude(oid, reason="x") == "already outdated"


def test_exclude_refuses_an_unknown_removal_status():
    s, oid = _done_paper()
    with pytest.raises(ValueError):
        s.exclude(oid, reason="x", as_status="deleted")


# ---------------------------------------------------------------------------
# `adjudicate`: a human coder settles a finished paper field by field. It must
# pass the same schema gate as a classifier, log what it replaced, and leave
# `Actual` to verify.py.
# ---------------------------------------------------------------------------
def _adjudicate(oid, *sets, yes=True, reverify=False):

    P.cmd_adjudicate(
        argparse.Namespace(
            db=DB,
            id=oid,
            set=list(sets),
            coder="human:test",
            reason="merged duplicate",
            yes=yes,
            reverify=reverify,
        )
    )
    with Store(DB)._conn() as c:
        r = c.execute(
            "SELECT stage, status, record_json FROM papers WHERE openalex_id=?", (oid,)
        ).fetchone()
    return r["stage"], r["status"], json.loads(r["record_json"])


def test_adjudicate_rewrites_the_codes_and_logs_the_old_record():
    s, oid = _done_paper()
    with s._conn() as c:
        before = c.execute("SELECT record_json FROM papers WHERE openalex_id=?", (oid,)).fetchone()[
            0
        ]
    stage, status, flat = _adjudicate(oid, "das_present=false", "paper_type=theory")
    assert (stage, status) == ("verified", "done"), "no identifier changed, so no re-verify"
    assert flat["das_present"] is False and flat["paper_type"] == "theory"
    assert flat["model_id"] == "human:test"
    with s._conn() as c:
        f = c.execute(
            "SELECT message FROM failures WHERE openalex_id=? AND error_type='adjudicated'", (oid,)
        ).fetchone()[0]
    assert "merged duplicate" in f and before in f, "the replaced record must be recoverable"


def test_adjudicate_sends_a_changed_identifier_back_to_verify():
    s, oid = _done_paper()
    stage, status, flat = _adjudicate(
        oid, "b_identifiers=https://github.com/example/fixed-url", "b_case=openly_accessible"
    )
    assert (stage, status) == ("classified", "pending")
    assert flat["b_actual"] == "not_attempted", "Actual is re-derived, never carried over"
    assert flat["b_case"] == "openly_accessible" and flat["b_openness_rank"] == 5
    assert oid in {r["openalex_id"] for r in s.claim("classified", 10)}


def test_adjudicate_reverify_sends_an_unchanged_identifier_back_to_verify():
    """A settled case can change what verify.py derives even when the
    identifier is untouched.

    The case that forced this: a paper declared its data openly available and
    gave a location only for its code, which 404s. Settling the data case to
    `openly_accessible` must let `derive_case` promote it to
    `access_unresolvable` off that 404 — hard rule 6 reserves `Actual` and the
    promotion to verify.py, so neither can be written here. Without a way to
    ask for a re-verification the row would sit at `openly_accessible` with a
    stale `not_found` beside it, which is worse than either endpoint.
    """
    s, oid = _done_paper()
    stage, status, flat = _adjudicate(oid, "b_case=openly_accessible", reverify=True)
    assert (stage, status) == ("classified", "pending")
    assert flat["b_actual"] == "not_attempted", "Actual is re-derived, never carried over"
    assert flat["b_identifiers"], "the identifier itself is untouched"
    assert oid in {r["openalex_id"] for r in s.claim("classified", 10)}


def test_adjudicate_without_reverify_leaves_a_finished_paper_finished():
    s, oid = _done_paper()
    stage, status, flat = _adjudicate(oid, "das_present=false")
    assert (stage, status) == ("verified", "done")
    assert flat["b_actual"] != "not_attempted", "nothing asked for a re-verification"


def test_adjudicate_cannot_write_what_a_classifier_could_not():
    s, oid = _done_paper()
    with s._conn() as c:
        before = c.execute("SELECT record_json FROM papers WHERE openalex_id=?", (oid,)).fetchone()[
            0
        ]
    from dap.schema import SchemaError

    with pytest.raises(SchemaError):
        _adjudicate(oid, "b_case=fully_open")  # not a token
    with pytest.raises(SchemaError):
        _adjudicate(oid, "b_case=not_applicable")  # identifiers on an absent axis
    with pytest.raises(SystemExit):
        _adjudicate(oid, "b_actual=accessible")  # verify.py only
    with s._conn() as c:
        assert (
            c.execute("SELECT record_json FROM papers WHERE openalex_id=?", (oid,)).fetchone()[0]
            == before
        )


def _stamp(oid, version):
    with Store(DB)._conn() as c:
        rj = json.loads(
            c.execute("SELECT record_json FROM papers WHERE openalex_id=?", (oid,)).fetchone()[0]
        )
        rj["scheme_version"] = version
        c.execute(
            "UPDATE papers SET scheme_version=?, record_json=? WHERE openalex_id=?",
            (version, json.dumps(rj), oid),
        )


def test_adjudicate_restamps_an_older_patch_with_the_current_version():
    """A recode under a patch-level ruling must say so, in the column export
    groups on and in the record it writes out.
    """
    from dap.schema import SCHEME_VERSION

    major, minor, _ = SCHEME_VERSION.split(".")
    s, oid = _done_paper()
    _stamp(oid, f"{major}.{minor}.999")
    _, _, flat = _adjudicate(oid, "das_present=false")
    assert flat["scheme_version"] == SCHEME_VERSION
    with s._conn() as c:
        assert (
            c.execute("SELECT scheme_version FROM papers WHERE openalex_id=?", (oid,)).fetchone()[0]
            == SCHEME_VERSION
        )


def test_adjudicate_refuses_a_row_from_another_minor_version():
    from dap.schema import SCHEME_VERSION

    major, minor, _ = SCHEME_VERSION.split(".")
    s, oid = _done_paper()
    _stamp(oid, f"{major}.{int(minor) + 1}.0")
    with pytest.raises(SystemExit):
        _adjudicate(oid, "das_present=false")


def test_adjudicate_only_reports_without_yes():
    s, oid = _done_paper()
    with s._conn() as c:
        before = c.execute("SELECT record_json FROM papers WHERE openalex_id=?", (oid,)).fetchone()[
            0
        ]
    _, _, flat = _adjudicate(oid, "das_present=false", yes=False)
    assert json.dumps(flat) == before or flat == json.loads(before)


# ---------------------------------------------------------------------------
# Regression: the same class of bug as reference_count above, one field along.
#
# `arxiv_search` refuses any match it cannot check against the paper's authors,
# so a facts object rebuilt without them does not loosen the check — it
# disables the fallback outright, silently, and every paper with no recorded
# arXiv id goes back to "no retrievable text". sources.py collected the authors
# and the cascade asked for them, but nothing carried them across the stage
# boundary, so the feature was dead in the CLI while its own tests passed:
# those construct CrossrefFacts directly and never cross that boundary.
# ---------------------------------------------------------------------------
def test_the_authors_survive_the_metadata_to_retrieve_boundary():
    s = _metadata_done()
    w2 = _row(s, "metadata", "W2")
    assert w2["meta"]["authors"] == ["ueno", "tanaka"], (
        "the metadata stage must persist what the author check needs"
    )

    seen = {}
    orig = sources.arxiv_search
    sources.arxiv_search = lambda cfg, **kw: (seen.update(kw), (None, "no match"))[1]
    try:
        _retrieve_with(
            s,
            [w2],
            {
                "oa.example": sources.PermanentError("HTTP 404"),
                "api.unpaywall.org": sources.PermanentError("HTTP 404"),
            },
        )
    finally:
        sources.arxiv_search = orig

    assert seen, "the cascade never reached the arXiv search"
    assert seen["authors"] == ["ueno", "tanaka"], (
        "an empty author list would make arxiv_search refuse every match"
    )
    assert seen["year"] == 2022, "the year check needs the publisher's year, not today's"


# ---------------------------------------------------------------------------
# The same fallback, when no registry has authors to give.
#
# Quantum Information and Computation's DOIs (10.5555/...) are known to neither
# Crossref nor DataCite, so `authors` stayed empty and the arXiv title search
# refused every candidate: three QIC papers sat dead with no text. And 5,630
# venue-frame rows were fetched before `authors` was stored at all, so for
# every one of them the search was switched off. OpenAlex lists the authors.
# ---------------------------------------------------------------------------
OPENALEX_WITH_AUTHORS = dict(
    OPENALEX_WORK,
    authorships=[
        {"author": {"display_name": "Srinivasan Arunachalam"}},
        {"author": {"display_name": "Ronald de Wolf"}},
    ],
)

NOWHERE_TO_READ = {
    "oa.example": sources.PermanentError("HTTP 404"),
    "api.unpaywall.org": sources.PermanentError("HTTP 404"),
}


def _retrieve_spying_on_arxiv(s, rows, routes):
    seen = {}
    orig_search, orig_pget = sources.arxiv_search, P._get
    sources.arxiv_search = lambda cfg, **kw: (seen.update(kw), (None, "no match"))[1]
    P._get = fake_get_factory(routes)  # the stage's own OpenAlex lookup
    try:
        _retrieve_with(s, rows, routes)
    finally:
        sources.arxiv_search, P._get = orig_search, orig_pget
    return seen


def test_a_doi_no_registry_knows_takes_its_authors_from_openalex():
    s = fresh_store()
    routes = {
        "api.openalex.org": FakeResponse(OPENALEX_WITH_AUTHORS),
        "api.crossref.org": sources.PermanentError("HTTP 404"),
        "api.datacite.org": sources.PermanentError("HTTP 404"),
        **NOWHERE_TO_READ,
    }
    orig_get, orig_pget = sources._get, P._get
    sources._get = P._get = fake_get_factory(routes)
    try:
        for row in s.claim("enrolled", 10):
            P.stage_metadata_one(CFG, s, row)
    finally:
        sources._get, P._get = orig_get, orig_pget

    w2 = _row(s, "metadata", "W2")
    assert not w2["meta"]["crossref_found"]
    assert w2["meta"]["authors"] == ["arunachalam", "wolf"], (
        "with no registry record, the authors must come from OpenAlex"
    )
    seen = _retrieve_spying_on_arxiv(s, [w2], routes)
    assert seen.get("authors") == ["arunachalam", "wolf"], (
        "the title search must be able to verify its candidate"
    )


def test_a_row_stored_without_authors_looks_them_up_before_retrieving():
    s = _metadata_done()
    w2 = _row(s, "metadata", "W2")
    del w2["meta"]["authors"]  # as stored before authors were carried
    seen = _retrieve_spying_on_arxiv(
        s, [w2], {"api.openalex.org": FakeResponse(OPENALEX_WITH_AUTHORS), **NOWHERE_TO_READ}
    )
    assert seen.get("authors") == ["arunachalam", "wolf"]
    assert _row(s, "retrieved", "W2")["meta"]["authors"] == ["arunachalam", "wolf"], (
        "the looked-up authors must be persisted, not fetched again next time"
    )


def test_a_failed_author_lookup_does_not_fail_the_paper():
    s = _metadata_done()
    w2 = _row(s, "metadata", "W2")
    del w2["meta"]["authors"]
    _retrieve_spying_on_arxiv(
        s, [w2], {"api.openalex.org": sources.TransientError("HTTP 429"), **NOWHERE_TO_READ}
    )
    assert s.stage_counts().get("retrieved/pending", 0) >= 1, (
        "a missing author list must not cost the paper its retrieval"
    )


# ---------------------------------------------------------------------------
# A mirror that times out no longer parks the paper (Diana, 4 Oct 2026).
#
# The rule reads which location is the publisher's from the landing page and
# the source type, and the metadata stage trims every OpenAlex location to
# three fields before storing it. So this starts from `stage_metadata_one`: a
# rule that passes on a hand-built work could still be dead in the CLI.
# ---------------------------------------------------------------------------
MIRROR_TIMES_OUT = dict(
    OPENALEX_WITH_AUTHORS,
    best_oa_location={
        "landing_page_url": "https://doi.org/10.1145/1",
        "pdf_url": "https://publisher.example/1.pdf",
        "source": {"display_name": "Some Journal", "type": "journal"},
    },
    locations=[
        {
            "landing_page_url": "https://doi.org/10.1145/1",
            "pdf_url": "https://publisher.example/1.pdf",
            "source": {"display_name": "Some Journal", "type": "journal"},
        },
        {
            "landing_page_url": "https://repo.example/handle/1",
            "pdf_url": "https://mirror.example/1.pdf",
            "source": {"display_name": "A Repository", "type": "repository"},
        },
    ],
)


def test_a_mirror_timeout_falls_back_to_arxiv_end_to_end():
    s = fresh_store()
    routes = {
        "api.openalex.org": FakeResponse(MIRROR_TIMES_OUT),
        "api.crossref.org/works/10.1038": FakeResponse(CROSSREF_WITH_DAS),
        "api.crossref.org/works/10.1145": FakeResponse(CROSSREF_BARE),
        "api.datacite.org/dois/10.5061": FakeResponse(DATACITE_DATASET),
        "api.datacite.org": sources.PermanentError("HTTP 404"),
        "api.springer.com": sources.PermanentError("HTTP 403"),
        "publisher.example": sources.PermanentError("HTTP 403"),
        "doi.org": sources.PermanentError("HTTP 403"),
        "mirror.example": sources.TransientError("exhausted 4 attempts: ConnectTimeout"),
        "repo.example": sources.TransientError("exhausted 4 attempts: ConnectTimeout"),
        "api.unpaywall.org": sources.PermanentError("HTTP 404"),
        "arxiv.org/pdf/2401.99999": ARTICLE_PAGE,
    }
    orig_get, orig_pget, orig_search = sources._get, P._get, sources.arxiv_search
    sources._get = P._get = fake_get_factory(routes)
    sources.arxiv_search = lambda cfg, **kw: ("2401.99999", "title and author agree")
    try:
        for row in s.claim("enrolled", 10):
            P.stage_metadata_one(CFG, s, row)
        P.stage_retrieve_one(CFG, s, _row(s, "metadata", "W2"), metadata_only=False, pool=None)
    finally:
        sources._get, P._get, sources.arxiv_search = orig_get, orig_pget, orig_search

    w2 = _row(s, "retrieved", "W2")["meta"]
    assert w2["text_source"] == "arxiv", "the mirror's timeout must not park the paper"
    assert ("https://mirror.example/1.pdf", "transient: exhausted 4 attempts: ConnectTimeout") in [
        (a["url"], a["outcome"]) for a in w2["retrieval_attempts"]
    ]
    with s._conn() as c:
        assert (
            c.execute(
                "SELECT COUNT(*) FROM failures WHERE openalex_id=? AND stage='retrieve'",
                ("https://openalex.org/W2",),
            ).fetchone()[0]
            == 0
        )


# ---------------------------------------------------------------------------
# The batch window: newest first, nothing older than BATCH_YEAR_MIN.
#
# It governs what the NEXT batch picks up. It must never touch a paper that is
# already done, and it must not silently drop a paper whose year is not yet
# known -- the year arrives with the metadata stage, so an unknown year at
# `enrolled` is the normal case, not a reason to skip.
# ---------------------------------------------------------------------------
def _seed_years(s, pairs):
    with s._conn() as c:
        for suffix, year in pairs:
            c.execute("UPDATE papers SET year=? WHERE openalex_id LIKE ?", (year, f"%{suffix}"))


def test_the_batch_window_skips_papers_published_before_the_cutoff():
    s = fresh_store()
    _seed_years(s, [("W1", 2019), ("W2", 2024)])
    got = [r["openalex_id"] for r in s.claim("enrolled", 10, year_min=2021)]
    assert got == ["https://openalex.org/W2"], "the 2019 paper is out of the window"
    assert len(s.claim("enrolled", 10, year_min=None)) == 2, "and the window is optional"


def test_an_unknown_year_is_kept_not_skipped():
    """Every paper sits at `enrolled` with no year until metadata runs. A window
    that dropped them would stall the pipeline on its own first stage.
    """
    s = fresh_store()
    _seed_years(s, [("W1", 2019)])  # W2 deliberately left NULL
    got = [r["openalex_id"] for r in s.claim("enrolled", 10, year_min=2021)]
    assert got == ["https://openalex.org/W2"]


def test_newest_first_orders_the_batch():
    s = fresh_store()
    _seed_years(s, [("W1", 2019), ("W2", 2024)])
    newest = [r["openalex_id"] for r in s.claim("enrolled", 10, newest_first=True)]
    assert newest == ["https://openalex.org/W2", "https://openalex.org/W1"]
    oldest = [r["openalex_id"] for r in s.claim("enrolled", 10)]
    assert oldest == ["https://openalex.org/W1", "https://openalex.org/W2"]


def test_the_year_is_recorded_from_the_metadata_stage():
    s = _metadata_done()
    assert s.years() == {2024: 2}, "OpenAlex says 2024 for both fixtures"
    assert s.years(coded_only=True) == {}, "nothing is coded yet"


def test_backfilling_years_touches_nothing_else():
    """The column is additive. A database written before it existed must come
    back with every stored value intact and only `year` filled in.
    """
    s = _metadata_done()
    with s._conn() as c:
        before = {
            r["openalex_id"]: (r["stage"], r["status"], r["meta_json"])
            for r in c.execute("SELECT * FROM papers")
        }
        c.execute("UPDATE papers SET year=NULL")
    assert s.backfill_years() == 2
    with s._conn() as c:
        after = {
            r["openalex_id"]: (r["stage"], r["status"], r["meta_json"])
            for r in c.execute("SELECT * FROM papers")
        }
    assert after == before, "backfill must not rewrite anything but the year"
    assert s.years() == {2024: 2}


# ---------------------------------------------------------------------------
# Reclassifying a gate stub: keep the text, replace the placeholder codes.
# ---------------------------------------------------------------------------
def _stub_out(s, oid, *, model=""):
    """Write the record the gate writes for a paper it never read."""
    rec = {
        "openalex_id": oid,
        "a_case": "not_applicable",
        "b_case": "not_applicable",
        "needs_human_review": True,
        "model_id": model,
        "retrieval": "full_text",
    }
    with s._conn() as c:
        c.execute(
            "UPDATE papers SET stage='verified', status='done', record_json=? WHERE openalex_id=?",
            (json.dumps(rec), oid),
        )


def test_a_gate_stub_goes_back_to_the_classifier_with_its_text_intact():
    s = fresh_store()

    def go():
        for row in s.claim("enrolled", 10):
            P.stage_metadata_one(CFG, s, row)
        for row in s.claim("metadata", 10):
            P.stage_retrieve_one(CFG, s, row, metadata_only=False, pool=None)

    with_network(go)
    oid = "https://openalex.org/W2"
    before = _row(s, "retrieved", "W2")["meta"]
    _stub_out(s, oid)
    assert s.requeue_for_reclassification(oid, reason="stub, never read") == ""
    after = _row(s, "retrieved", "W2")["meta"]
    assert after["passages"] == before["passages"], "the text must survive untouched"
    with s._conn() as c:
        assert (
            c.execute("SELECT record_json FROM papers WHERE openalex_id=?", (oid,)).fetchone()[
                "record_json"
            ]
            is None
        )
        f = c.execute(
            "SELECT error_type, message FROM failures WHERE openalex_id=?", (oid,)
        ).fetchone()
    assert f["error_type"] == "reclassify"
    assert "discarded stub" in f["message"], "the placeholder must stay in the audit trail"


def test_it_refuses_a_paper_a_coder_actually_classified():
    """Replacing one coder's judgement with another's, silently, is how a
    two-coder study loses the thing it is measuring.
    """
    s = fresh_store()

    def go():
        for row in s.claim("enrolled", 10):
            P.stage_metadata_one(CFG, s, row)
        for row in s.claim("metadata", 10):
            P.stage_retrieve_one(CFG, s, row, metadata_only=False, pool=None)

    with_network(go)
    oid = "https://openalex.org/W2"
    _stub_out(s, oid, model="manual:claude-sonnet-5-frame")
    why = s.requeue_for_reclassification(oid, reason="nope")
    assert "carries a real classification" in why and "recode" in why
    with s._conn() as c:
        assert (
            c.execute("SELECT record_json FROM papers WHERE openalex_id=?", (oid,)).fetchone()[
                "record_json"
            ]
            is not None
        )


def test_it_refuses_a_stub_with_no_text_to_read():
    s = fresh_store()

    def go():
        for row in s.claim("enrolled", 10):
            P.stage_metadata_one(CFG, s, row)

    with_network(go)
    oid = "https://openalex.org/W2"
    _stub_out(s, oid)
    assert "no stored evidence" in s.requeue_for_reclassification(oid, reason="x")


def test_it_refuses_a_paper_marked_retrieved_that_stored_no_passages():
    """`retrieval` says text was fetched once, not that anything survived into
    the row. 140 papers are `partial_text` with zero passages; sending them
    here only burns a claim and re-stubs them.
    """
    s = fresh_store()

    def go():
        for row in s.claim("enrolled", 10):
            P.stage_metadata_one(CFG, s, row)
        for row in s.claim("metadata", 10):
            P.stage_retrieve_one(CFG, s, row, metadata_only=False, pool=None)

    with_network(go)
    oid = "https://openalex.org/W2"
    with s._conn() as c:
        meta = json.loads(
            c.execute("SELECT meta_json FROM papers WHERE openalex_id=?", (oid,)).fetchone()[
                "meta_json"
            ]
        )
        meta["passages"] = []
        meta["retrieval"] = "partial_text"
        c.execute("UPDATE papers SET meta_json=? WHERE openalex_id=?", (json.dumps(meta), oid))
    _stub_out(s, oid)
    why = s.requeue_for_reclassification(oid, reason="x")
    assert "no stored evidence" in why and "recode" in why


# ---------------------------------------------------------------------------
# A paper that ran out of attempts at a stage
# ---------------------------------------------------------------------------
def test_a_paper_out_of_attempts_can_be_given_another_try():
    """A schema rejection in manual mode spends an attempt exactly as a timeout
    does, so three malformed result lines retire a paper that has nothing wrong
    with it. `claim` skips it at MAX_ATTEMPTS and every requeue refuses it,
    because there is no classification to replace.
    """
    from dap.store import MAX_ATTEMPTS

    s = fresh_store()

    def go():
        for row in s.claim("enrolled", 10):
            P.stage_metadata_one(CFG, s, row)

    with_network(go)
    oid = "https://openalex.org/W2"
    for _ in range(MAX_ATTEMPTS):
        s.fail(oid, "classify_manual", "SchemaError", "paper_type: expected an object")
    assert oid not in {r["openalex_id"] for r in s.claim("metadata", 50)}, (
        "a paper at MAX_ATTEMPTS must be out of reach of claim"
    )

    assert s.clear_attempts(oid, reason="coder-side typo, not a retrieval failure") == ""
    assert oid in {r["openalex_id"] for r in s.claim("metadata", 50)}
    with s._conn() as c:
        r = c.execute(
            "SELECT status, attempts, last_error FROM papers WHERE openalex_id=?", (oid,)
        ).fetchone()
        f = c.execute(
            "SELECT stage, error_type, message FROM failures WHERE "
            "openalex_id=? AND error_type='attempts_cleared'",
            (oid,),
        ).fetchone()
    assert (r["status"], r["attempts"], r["last_error"]) == ("pending", 0, None)
    assert f["stage"] == "metadata", "the audit row records where it was stuck"
    assert "3 -> 0" in f["message"], "the trail must say what the counter was"


def test_it_will_not_clear_attempts_on_a_paper_that_is_not_stuck():
    s = fresh_store()

    def go():
        for row in s.claim("enrolled", 10):
            P.stage_metadata_one(CFG, s, row)

    with_network(go)
    oid = "https://openalex.org/W2"
    assert "not failed" in s.clear_attempts(oid, reason="nothing wrong with it")
    assert "not enrolled" in s.clear_attempts("https://openalex.org/W999", reason="x")


def test_it_will_not_clear_attempts_on_a_classified_paper():
    """Clearing the counter would send a coded paper back round the stage it
    already passed; `recode` is the command that is allowed to do that.
    """
    s = fresh_store()

    def go():
        for row in s.claim("enrolled", 10):
            P.stage_metadata_one(CFG, s, row)
        for row in s.claim("metadata", 10):
            P.stage_retrieve_one(CFG, s, row, metadata_only=False, pool=None)

    with_network(go)
    oid = "https://openalex.org/W2"
    _stub_out(s, oid)
    s.fail(oid, "verify", "TransientError", "timed out")
    assert "already carries a classification" in s.clear_attempts(oid, reason="x")


# ---------------------------------------------------------------------------
# Claiming only versions of record
# ---------------------------------------------------------------------------
def test_the_claim_can_skip_rows_whose_doi_is_a_preprint_registrars():
    """The same paper is routinely enrolled twice, once under its arXiv DOI and
    once under its publisher's. Coding both double-counts it, and the preprint
    copy cannot carry a publisher's availability statement, so the two disagree
    on `das_present` by construction.
    """
    from dap.store import PREPRINT_DOI_PREFIXES

    s = fresh_store()
    rows = [
        {
            "openalex_id": f"https://openalex.org/W{i}",
            "doi": doi,
            "title": f"paper {i}",
            "venue": "v",
            "year": 2024,
            "arm": "quantum_computing",
        }
        for i, doi in enumerate(
            [
                "10.1103/physrevlett.1.1",
                "10.48550/arxiv.2208.01063",
                "10.5281/zenodo.1",
                "10.21203/rs.3.rs-1",
                "10.2139/ssrn.1",
                "10.20944/preprints.1",
                "",
            ],
            start=100,
        )
    ]
    s.enrol(rows)
    mine = {r["openalex_id"] for r in rows}
    everything = {r["openalex_id"] for r in s.claim("enrolled", 500)} & mine
    assert everything == mine, "all nine are claimable without the filter"

    kept = {
        r["doi"]
        for r in s.claim("enrolled", 500, publisher_doi_only=True)
        if r["openalex_id"] in mine
    }
    # The venue half of the same filter. An arXiv row often carries no DOI at
    # all, which the prefix test cannot see; the venue arrives with metadata, so
    # this is what the filter looks at from the retrieved stage onward.
    oid = "https://openalex.org/W150"
    s.enrol([{"openalex_id": oid, "doi": "", "arm": "quantum_computing", "year": 2026}])
    with s._conn() as conn:
        conn.execute(
            "UPDATE papers SET stage='retrieved', meta_json=? WHERE openalex_id=?",
            (json.dumps({"venue": "arXiv (Cornell University)", "title": "no doi"}), oid),
        )
    assert [r["openalex_id"] for r in s.claim("retrieved", 500)] == [oid]
    assert s.claim("retrieved", 500, publisher_doi_only=True) == [], (
        "an arXiv row with no DOI is invisible to the prefix test; the venue "
        "test is what catches it"
    )
    assert kept == {"10.1103/physrevlett.1.1", ""}, (
        "only the publisher DOI and the row with no DOI at all may survive"
    )
    for pfx in PREPRINT_DOI_PREFIXES:
        assert not any(d.startswith(pfx) for d in kept)


def test_skipping_preprints_leaves_them_claimable():
    """The filter is a scope, not an exclusion: nothing is written, so the rows
    come straight back the moment it is off.
    """
    s = fresh_store()
    s.enrol(
        [
            {
                "openalex_id": "https://openalex.org/W200",
                "doi": "10.48550/arxiv.1",
                "title": "t",
                "venue": "v",
                "year": 2024,
                "arm": "quantum_computing",
            }
        ]
    )
    oid = "https://openalex.org/W200"
    assert not [
        r for r in s.claim("enrolled", 500, publisher_doi_only=True) if r["openalex_id"] == oid
    ]
    back = [r for r in s.claim("enrolled", 500) if r["openalex_id"] == oid]
    assert len(back) == 1 and back[0]["doi"] == "10.48550/arxiv.1"
    with s._conn() as c:
        r = c.execute(
            "SELECT status, attempts FROM papers WHERE openalex_id=?",
            ("https://openalex.org/W200",),
        ).fetchone()
    assert (r["status"], r["attempts"]) == ("pending", 0), "the filter must not touch the row"


def test_the_claim_can_skip_a_doi_that_is_already_coded():
    """OpenAlex mints two work ids for one DOI often enough to matter, and
    `is_classified` keys on the id, so the same paper gets coded twice under
    two ids unless the claim itself refuses the second copy.
    """
    s = fresh_store()
    doi = "10.1038/s41377-025-02085-5"
    # `_guard_unique_dois` now makes this state unreachable through `enrol`, so
    # the test builds it the way it can still arise: a database that predates
    # the guard, which gets a warning on open and keeps its duplicates. That is
    # exactly the database this filter exists for.
    with s._conn() as c:
        c.execute("DROP INDEX IF EXISTS ux_papers_live_doi")
        for w, stage, status, rec in (
            (301, "verified", "done", "{}"),
            (302, "enrolled", "pending", None),
        ):
            c.execute(
                "INSERT INTO papers (openalex_id, doi, status, stage, "
                "record_json, updated_at) VALUES (?,?,?,?,?,'now')",
                (f"https://openalex.org/W{w}", doi, status, stage, rec),
            )
    claimable = [r["openalex_id"] for r in s.claim("enrolled", 500) if r["doi"] == doi]
    assert claimable == ["https://openalex.org/W302"]
    assert not [r for r in s.claim("enrolled", 500, skip_coded_dois=True) if r["doi"] == doi], (
        "the twin of a coded paper must not be claimable"
    )
    with s._conn() as c:
        r = c.execute(
            "SELECT status FROM papers WHERE openalex_id=?", ("https://openalex.org/W302",)
        ).fetchone()
    assert r["status"] == "pending", "the filter must not touch the row"


def test_enrol_deduplicates_on_the_doi_as_well_as_the_id(tmp_path):
    """OpenAlex mints more than one work id for the same DOI, and a row enrolled
    twice is classified twice because `is_classified` keys on the id. Rows with
    no DOI must survive: an empty string is not an identity.
    """
    csv_path = tmp_path / "harvest.csv"
    csv_path.write_text(
        "openalex_id,doi,title,publication_year,venue\n"
        "https://openalex.org/W401,10.1103/physrevlett.1.1,a,2024,PRL\n"
        "https://openalex.org/W402,10.1103/PhysRevLett.1.1,a again,2024,PRL\n"
        "https://openalex.org/W403,,no doi one,2024,x\n"
        "https://openalex.org/W404,,no doi two,2024,x\n"
        "https://openalex.org/W405,10.1234/other,b,2024,y\n"
        "https://openalex.org/W406,10.1209/0295-5075/ae333e,c,2026,EPL\n"
        "https://openalex.org/W407,10.1209/0295-5075/ae333e/pdf,c again,2026,EPL\n"
    )

    db = tmp_path / "t.sqlite"
    P.cmd_enrol(
        argparse.Namespace(
            db=str(db), input=str(csv_path), arm_map=None, year_min=2017, year_max=2026, limit=None
        )
    )
    s = Store(str(db))
    got = {r["openalex_id"].rsplit("/", 1)[-1] for r in s.claim("enrolled", 50)}
    assert got == {"W401", "W403", "W404", "W405", "W406"}, (
        "the second copy of the DOI goes, in either case; both DOI-less rows "
        "stay; and a DOI carrying the PDF link's /pdf suffix is the same paper"
    )
    with s._conn() as c:
        kept = c.execute(
            "SELECT doi FROM papers WHERE openalex_id=?", ("https://openalex.org/W406",)
        ).fetchone()["doi"]
    assert kept == "10.1209/0295-5075/ae333e", "and the suffix is stripped on the way in"


def test_the_database_refuses_a_second_live_row_for_one_doi():
    """The guard is in the database, not in the care of whoever is writing."""
    import sqlite3

    s = fresh_store()
    doi = "10.1038/s41377-025-02085-5"
    s.enrol(
        [
            {
                "openalex_id": "https://openalex.org/W501",
                "doi": doi,
                "arm": "quantum_computing",
                "year": 2026,
            }
        ]
    )
    with s._conn() as c:
        try:
            c.execute(
                "INSERT INTO papers (openalex_id, doi, status, stage, updated_at) "
                "VALUES ('https://openalex.org/W502', ?, 'pending', 'enrolled', 'now')",
                (doi,),
            )
            raise AssertionError("a second live row for one DOI must be refused")
        except sqlite3.IntegrityError:
            pass


def test_the_doi_guard_still_allows_an_excluded_tombstone_and_rows_with_no_doi():
    """`exclude` keeps the losing copy of a resolved duplicate, with its codes,
    so the index has to let a tombstone sit on the same DOI. And 6.5% of the
    corpus has no DOI at all -- an empty string is not an identity.
    """
    s = fresh_store()
    doi = "10.1103/physrevlett.9.9"
    s.enrol(
        [
            {
                "openalex_id": "https://openalex.org/W601",
                "doi": doi,
                "arm": "quantum_computing",
                "year": 2026,
            },
            {
                "openalex_id": "https://openalex.org/W602",
                "doi": "",
                "arm": "quantum_computing",
                "year": 2026,
            },
            {
                "openalex_id": "https://openalex.org/W603",
                "doi": "",
                "arm": "quantum_computing",
                "year": 2026,
            },
        ]
    )
    assert s.exclude("https://openalex.org/W601", reason="dup") == ""
    s.enrol(
        [
            {
                "openalex_id": "https://openalex.org/W604",
                "doi": doi,
                "arm": "quantum_computing",
                "year": 2026,
            }
        ]
    )
    with s._conn() as c:
        live = c.execute(
            "SELECT COUNT(*) n FROM papers WHERE doi=? AND status!='excluded'", (doi,)
        ).fetchone()["n"]
        nodoi = c.execute(
            "SELECT COUNT(*) n FROM papers WHERE doi='' AND openalex_id LIKE '%W60%'"
        ).fetchone()["n"]
    assert live == 1, "the survivor goes in beside the tombstone"
    assert nodoi == 2, "both DOI-less rows must coexist"


def test_the_scope_rejects_repository_venues_named_in_the_middle():
    """ "DSpace@MIT" and "Research Repository (Delft University of Technology)"
    are not journals, and a prefix test cannot reach either. GLOB is used rather
    than LIKE because LIKE is case-insensitive in SQLite and these are checked
    case-sensitively, alongside the lowercase variants.
    """
    import json as _json

    s = fresh_store()
    cases = {
        "W701": "DSpace@MIT (Massachusetts Institute of Technology)",
        "W702": "Research Repository (Delft University of Technology)",
        "W703": "Qeios",
        "W704": "4TU.ResearchData",
        "W705": "Physical Review Letters",
    }
    s.enrol(
        [
            {
                "openalex_id": f"https://openalex.org/{w}",
                "doi": f"10.1/{w}",
                "arm": "quantum_computing",
                "year": 2026,
            }
            for w in cases
        ]
    )
    with s._conn() as c:
        for w, venue in cases.items():
            c.execute(
                "UPDATE papers SET stage='retrieved', meta_json=? WHERE openalex_id=?",
                (_json.dumps({"venue": venue}), f"https://openalex.org/{w}"),
            )
    kept = {
        r["openalex_id"].rsplit("/", 1)[-1]
        for r in s.claim("retrieved", 500, publisher_doi_only=True)
    }
    assert kept == {"W705"}, f"only the journal survives, got {kept}"
