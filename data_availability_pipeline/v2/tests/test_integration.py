"""
Layer tests for everything that talks to a network, a model, or a disk.

None of this needs an API key or an internet connection: the boundary is faked,
the logic above it is real. That is the line to hold — if a behaviour can only
be checked against the live internet, it will first be observed failing in
production, at scale, spending money.

    python3 tests/test_integration.py
"""
import copy
import csv
import json
import sqlite3
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import pytest
except ImportError:
    from test_guards import pytest  # reuse the shim

import dap.classify as C
import dap.sources as sources
import dap.verify as verify
from dap.classify import (classify, select_passages, should_escalate,
                          ClassificationError)
from dap.schema import (Actual, Arm, Case, RepoType, Retrieval, TextSource,
                        SCHEME_VERSION)
from dap.sources import (Config, CrossrefFacts, arxiv_id_from_openalex,
                         looks_like_article)
from dap.store import Store, MAX_ATTEMPTS

from fakes import (CROSSREF_BARE, CROSSREF_WITH_DAS, DATACITE_ARTICLE_ATTRS,
                   DATACITE_DATASET_ATTRS, DATACITE_PAPER, FakeAnthropic,
                   FakeBlock, FakeMessage, FakeResponse, FakeTextBlock,
                   OPENALEX_ARXIV, OPENALEX_NO_ARXIV, datacite_bulk,
                   fake_get_factory, good_payload)

CFG = Config(contact_email="researcher@institution.edu")


# ===========================================================================
# sources.crossref_facts — the layer that was at 0% coverage
# ===========================================================================
def _with_get(routes, fn):
    original = sources._get
    sources._get = fake_get_factory(routes)
    try:
        return fn()
    finally:
        sources._get = original


def test_das_found_under_an_unguessable_assertion_name():
    f = _with_get({"api.crossref.org": FakeResponse(CROSSREF_WITH_DAS)},
                  lambda: sources.crossref_facts(CFG, "10.1038/x"))
    assert f.found and f.has_das
    assert len(f.das_texts) == 2, "both data AND code availability should match"
    assert "Zenodo repository" in f.das_texts[0]


def test_non_das_assertions_are_not_mistaken_for_a_statement():
    f = _with_get({"api.crossref.org": FakeResponse(CROSSREF_WITH_DAS)},
                  lambda: sources.crossref_facts(CFG, "10.1038/x"))
    joined = " ".join(f.das_texts)
    assert "2023-08-01" not in joined and "2024-01-15" not in joined


def test_only_dataset_relations_are_collected():
    f = _with_get({"api.crossref.org": FakeResponse(CROSSREF_WITH_DAS)},
                  lambda: sources.crossref_facts(CFG, "10.1038/x"))
    assert f.dataset_ids == ["10.5281/zenodo.10499999"]
    assert f.has_structured_dataset_link
    # has-preprint is a relation but not a dataset one
    assert all("arXiv" not in d for d in f.dataset_ids)


def test_only_text_mining_links_are_kept():
    f = _with_get({"api.crossref.org": FakeResponse(CROSSREF_WITH_DAS)},
                  lambda: sources.crossref_facts(CFG, "10.1038/x"))
    assert len(f.tdm_links) == 1
    assert "springer" in f.tdm_links[0]["url"]


def test_references_are_captured_for_datacite_confirmation():
    f = _with_get({"api.crossref.org": FakeResponse(CROSSREF_WITH_DAS)},
                  lambda: sources.crossref_facts(CFG, "10.1038/x"))
    assert "10.5061/dryad.abc123" in f.dataset_refs
    assert "10.1103/physrevlett.100.100501" in f.dataset_refs  # lowercased


def test_bare_crossref_record_yields_nothing_but_does_not_crash():
    f = _with_get({"api.crossref.org": FakeResponse(CROSSREF_BARE)},
                  lambda: sources.crossref_facts(CFG, "10.1145/1"))
    assert f.found and not f.has_das and not f.has_structured_dataset_link
    assert f.container == "Proceedings of ISCA" and f.published_year == 2022


def test_missing_crossref_record_is_not_an_error():
    f = _with_get({"api.crossref.org": sources.PermanentError("HTTP 404")},
                  lambda: sources.crossref_facts(CFG, "10.0/nope"))
    assert f.found is False


def test_empty_doi_short_circuits():
    assert sources.crossref_facts(CFG, "").found is False


# ---------------------------------------------------------------------------
# Regression: a paper DOI registered with DataCite, not Crossref.
#
# The first live run killed https://openalex.org/W3098228168 (10.17169/refubium-4067)
# with "no full text and no Crossref record". Crossref really does 404 that DOI —
# it is a Freie Universitat Berlin repository deposit registered with DataCite,
# which holds a complete record. Asking only Crossref discarded a paper that was
# fully described at the other agency.
# ---------------------------------------------------------------------------
def test_datacite_registered_paper_is_not_treated_as_having_no_metadata():
    f = _with_get({"api.crossref.org": sources.PermanentError("HTTP 404"),
                   "api.datacite.org": FakeResponse(DATACITE_PAPER)},
                  lambda: sources.registry_facts(CFG, "10.17169/refubium-4067"))
    assert f.found is True, "a DataCite-registered paper still has metadata"
    assert f.container == "Freie Universitat Berlin"
    assert f.published_year == 2019


def test_crossref_is_preferred_and_datacite_not_called_when_it_answers():
    f = _with_get({"api.crossref.org": FakeResponse(CROSSREF_BARE)},
                  lambda: sources.registry_facts(CFG, "10.1145/1"))
    # No api.datacite.org route is registered, so fake_get_factory would raise
    # AssertionError if the fallback fired on a Crossref hit.
    assert f.found and f.container == "Proceedings of ISCA"


def test_datacite_fallback_reads_das_and_related_dois():
    f = _with_get({"api.crossref.org": sources.PermanentError("HTTP 404"),
                   "api.datacite.org": FakeResponse(DATACITE_PAPER)},
                  lambda: sources.registry_facts(CFG, "10.17169/refubium-4067"))
    assert f.has_das and "Zenodo" in f.das_texts[0]
    assert f.dataset_ids == ["10.5281/zenodo.1"], "IsSupplementedBy is a dataset link"
    assert f.dataset_refs == ["10.1103/physreva.1"], "Cites is a reference candidate"
    assert f.license_urls == ["http://creativecommons.org/licenses/by/4.0"]


def test_datacite_abstract_is_never_mistaken_for_a_das():
    """An abstract is not a data-availability statement. Promoting one would put
    a fabricated DAS in front of the classifier."""
    f = _with_get({"api.crossref.org": sources.PermanentError("HTTP 404"),
                   "api.datacite.org": FakeResponse(DATACITE_PAPER)},
                  lambda: sources.registry_facts(CFG, "10.17169/refubium-4067"))
    assert not any("lattice surgery" in t for t in f.das_texts)


def test_both_agencies_missing_still_reports_not_found():
    f = _with_get({"api.crossref.org": sources.PermanentError("HTTP 404"),
                   "api.datacite.org": sources.PermanentError("HTTP 404")},
                  lambda: sources.registry_facts(CFG, "10.0/nope"))
    assert f.found is False, "genuine absence must still be a failure"


def test_registry_facts_normalises_a_resolver_prefixed_doi():
    seen = {}

    def _spy(cfg, url, **kw):
        seen["url"] = url
        raise sources.PermanentError("HTTP 404")

    original = sources._get
    sources._get = _spy
    try:
        sources.registry_facts(CFG, "https://doi.org/10.1038/x")
    finally:
        sources._get = original
    assert "doi.org/10.1038" not in seen["url"].replace("api.crossref.org/works/", "")
    assert seen["url"].endswith("10.1038/x")


def _clear_dc_cache():
    with sources._dc_lock:
        sources._dc_cache.clear()


def test_datacite_distinguishes_dataset_from_article():
    _clear_dc_cache()
    got = _with_get({"api.datacite.org": FakeResponse(
                        datacite_bulk(("10.5061/dryad.abc123", DATACITE_DATASET_ATTRS)))},
                    lambda: sources.is_datacite_dataset(CFG, "10.5061/dryad.abc123"))
    assert got is not None
    _clear_dc_cache()
    got = _with_get({"api.datacite.org": FakeResponse(
                        datacite_bulk(("10.1103/x", DATACITE_ARTICLE_ATTRS)))},
                    lambda: sources.is_datacite_dataset(CFG, "10.1103/x"))
    assert got is None


# ---------------------------------------------------------------------------
# Throughput: the per-DOI DataCite loop was the dominant cost of the metadata
# stage. Measured on the first live run: 3,604 requests for 200 papers,
# serialised through a 5 qps per-host bucket -> ~12 minutes, 0 datasets found.
# These pin the properties that make the bulk form safe to substitute.
# ---------------------------------------------------------------------------
def test_many_dois_resolve_in_one_request():
    _clear_dc_cache()
    calls = []

    def counting(cfg, url, **kw):
        calls.append(kw.get("params", {}).get("query", ""))
        return FakeResponse(datacite_bulk(
            ("10.5281/zenodo.1", DATACITE_DATASET_ATTRS),
            ("10.1103/journal.1", DATACITE_ARTICLE_ATTRS)))

    original = sources._get
    sources._get = counting
    try:
        got = sources.datacite_datasets(
            CFG, ["10.5281/zenodo.1", "10.1103/journal.1", "10.1038/nope.1"])
    finally:
        sources._get = original

    assert len(calls) == 1, f"3 DOIs must cost 1 request, took {len(calls)}"
    assert "10.5281/zenodo.1" in calls[0] and "10.1038/nope.1" in calls[0]
    assert set(got) == {"10.5281/zenodo.1"}, "only the dataset-typed DOI counts"


def test_repeat_dois_are_not_refetched():
    _clear_dc_cache()
    calls = []

    def counting(cfg, url, **kw):
        calls.append(url)
        return FakeResponse(datacite_bulk(("10.5281/zenodo.1", DATACITE_DATASET_ATTRS)))

    original = sources._get
    sources._get = counting
    try:
        sources.datacite_datasets(CFG, ["10.5281/zenodo.1", "10.1103/j.1"])
        again = sources.datacite_datasets(CFG, ["10.5281/zenodo.1", "10.1103/j.1"])
    finally:
        sources._get = original

    assert len(calls) == 1, "the second lookup must be served from cache"
    assert set(again) == {"10.5281/zenodo.1"}, "cache returns the same verdict"


def test_duplicates_within_one_call_are_collapsed():
    _clear_dc_cache()
    seen = []

    def counting(cfg, url, **kw):
        seen.append(kw.get("params", {}).get("query", ""))
        return FakeResponse(datacite_bulk(("10.5281/zenodo.1", DATACITE_DATASET_ATTRS)))

    original = sources._get
    sources._get = counting
    try:
        sources.datacite_datasets(CFG, ["10.5281/zenodo.1"] * 25)
    finally:
        sources._get = original
    assert seen[0].count("10.5281/zenodo.1") == 1, "the DOI must appear once in the query"


def test_datacite_outage_is_not_cached_as_not_a_dataset():
    """A transient failure must leave the DOI unresolved so a rerun retries it.
    Caching an outage as a negative would silently lose Axis A evidence."""
    _clear_dc_cache()

    def failing(cfg, url, **kw):
        raise sources.TransientError("HTTP 503")

    original = sources._get
    sources._get = failing
    try:
        got = sources.datacite_datasets(CFG, ["10.5281/zenodo.1"])
    finally:
        sources._get = original
    assert got == {}
    with sources._dc_lock:
        assert "10.5281/zenodo.1" not in sources._dc_cache, "outage must not poison cache"


def test_batches_are_chunked():
    _clear_dc_cache()
    calls = []

    def counting(cfg, url, **kw):
        calls.append(kw.get("params", {}).get("query", ""))
        return FakeResponse(datacite_bulk())

    original = sources._get
    sources._get = counting
    try:
        sources.datacite_datasets(CFG, [f"10.1234/x{i}" for i in range(120)], chunk=50)
    finally:
        sources._get = original
    assert len(calls) == 3, f"120 DOIs at chunk=50 should be 3 requests, got {len(calls)}"


def test_doi_that_could_break_out_of_the_query_is_dropped():
    _clear_dc_cache()
    calls = []

    def counting(cfg, url, **kw):
        calls.append(kw.get("params", {}).get("query", ""))
        return FakeResponse(datacite_bulk())

    original = sources._get
    sources._get = counting
    try:
        sources.datacite_datasets(CFG, ['10.1234/a"b OR *', "10.1234/ok"])
    finally:
        sources._get = original
    assert len(calls) == 1
    assert '"10.1234/ok"' in calls[0]
    assert "OR *" not in calls[0], "a quote in a DOI must not escape the term"


# -- the wrong-paper bug from v1 --------------------------------------------
def test_arxiv_id_comes_from_metadata_not_a_title_search():
    assert arxiv_id_from_openalex(OPENALEX_ARXIV) == "2401.12345v2"
    assert arxiv_id_from_openalex(OPENALEX_NO_ARXIV) is None


# ===========================================================================
# verify — the layer that could never produce a failure in v1
# ===========================================================================
class _FakeRequests:
    # the real exception type, so `except requests.RequestException` still binds
    import requests as _real
    RequestException = _real.RequestException

    def __init__(self, head=None, get=None, routes=None):
        """`routes`, like fakes.fake_get_factory, maps a substring of the URL to
        a FakeResponse or an Exception, and serves both HEAD and GET; a URL no
        route matches gets `head`/`get`. Every URL asked for is in `urls`."""
        self._head, self._get = head, get
        self._routes = routes or {}
        self.head_calls = self.get_calls = 0
        self.urls: list[str] = []

    def _answer(self, url, default):
        self.urls.append(url)
        resp = next((r for needle, r in self._routes.items() if needle in url),
                    default)
        if isinstance(resp, Exception):
            raise resp
        return resp

    def head(self, url, **kw):
        self.head_calls += 1
        return self._answer(url, self._head)

    def get(self, url, **kw):
        self.get_calls += 1
        return self._answer(url, self._get)


def _with_requests(fake, fn):
    original = verify.requests
    verify.requests = fake
    try:
        return fn()
    finally:
        verify.requests = original


def test_clean_200_is_accessible():
    fake = _FakeRequests(head=FakeResponse(status=200, headers={"Content-Type": "application/zip"}))
    r = _with_requests(fake, lambda: verify.verify_identifier(CFG, "https://zenodo.org/records/1"))
    assert r.status is Actual.ACCESSIBLE and fake.get_calls == 0


def test_head_refusal_falls_back_to_get():
    """A repository that refuses HEAD is not a broken link. v1 called this
    requires_authentication and moved on."""
    fake = _FakeRequests(head=FakeResponse(status=405),
                         get=FakeResponse(status=200, headers={"Content-Type": "application/zip"}))
    r = _with_requests(fake, lambda: verify.verify_identifier(CFG, "https://osf.io/abc"))
    assert r.status is Actual.ACCESSIBLE and fake.get_calls == 1


def test_dead_doi_is_not_found():
    fake = _FakeRequests(head=FakeResponse(status=404))
    r = _with_requests(fake, lambda: verify.verify_identifier(CFG, "10.5281/zenodo.999999999"))
    assert r.status is Actual.NOT_FOUND and r.http_status == 404


def test_intermittent_403_host_does_not_establish_an_access_requirement():
    """gitlab.com answers automated requests with 403 during some periods and
    200 during others: two identifiers recorded `accessible` 200 in the corpus
    both returned 403 when re-checked, while a browser loaded the public
    project page with no login. A status that depends on the hour does not
    establish that the path gates access, so this is ERROR and the declared
    case survives (hard rule 6)."""
    fake = _FakeRequests(head=FakeResponse(status=403),
                         get=FakeResponse(status=403))
    r = _with_requests(fake, lambda: verify.verify_identifier(
        CFG, "https://gitlab.com/MartinBeseda/sa-oo-vqe-qiskit.git"))
    assert r.status is Actual.ERROR and r.http_status == 403
    assert verify.derive_case(Case.OPENLY_ACCESSIBLE, r.status)[0] is Case.OPENLY_ACCESSIBLE


def test_a_403_anywhere_else_still_gates_access():
    """The soft-403 list is narrow on purpose: a publisher's
    supplementary-material endpoint that answers 403 is a real credential wall
    and must keep promoting."""
    fake = _FakeRequests(head=FakeResponse(status=403),
                         get=FakeResponse(status=403))
    r = _with_requests(fake, lambda: verify.verify_identifier(
        CFG, "https://pubs.acs.org/doi/suppl/10.1021/acsomega.0c00000/suppl_file/ao0c00000_si_001.pdf"))
    assert r.status is Actual.AUTH_REQUIRED
    assert verify.derive_case(Case.OPENLY_ACCESSIBLE, r.status)[0] is Case.ACCESS_UNRESOLVABLE


def test_soft_404_host_does_not_establish_absence():
    """Kaggle answers an automated request with 404 whether or not the dataset
    exists -- all 15 kaggle.com identifiers in the corpus came back 404, among
    them datasets with thousands of upvotes that load fine in a browser. Only a
    check that establishes the path does not deliver may promote a case (hard
    rule 6), so this is ERROR, and the declared case survives."""
    fake = _FakeRequests(head=FakeResponse(status=404))
    r = _with_requests(fake, lambda: verify.verify_identifier(
        CFG, "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"))
    assert r.status is Actual.ERROR and r.http_status == 404
    assert verify.derive_case(Case.OPENLY_ACCESSIBLE, r.status)[0] is Case.OPENLY_ACCESSIBLE


def test_a_404_anywhere_else_still_establishes_absence():
    """The soft-404 list is narrow on purpose: a host only joins it once its
    404s are shown to be indiscriminate."""
    fake = _FakeRequests(head=FakeResponse(status=404))
    r = _with_requests(fake, lambda: verify.verify_identifier(
        CFG, "https://github.com/nobody/gone"))
    assert r.status is Actual.NOT_FOUND
    assert verify.derive_case(Case.OPENLY_ACCESSIBLE, r.status)[0] is Case.ACCESS_UNRESOLVABLE


def test_resolving_landing_page_with_no_data_is_caught():
    """The case v1 structurally could not see: HTTP 200, but nothing there."""
    html = "<html><body>This record has been withdrawn by the depositor.</body></html>"
    fake = _FakeRequests(head=FakeResponse(status=200, headers={"Content-Type": "text/html"},
                                           text=html))
    r = _with_requests(fake, lambda: verify.verify_identifier(CFG, "https://zenodo.org/records/2"))
    assert r.status is Actual.RESOLVES_NO_DATA


def test_login_wall_is_paywalled_not_accessible():
    html = "<html><body>Please sign in to continue. Purchase access.</body></html>"
    fake = _FakeRequests(head=FakeResponse(status=200, headers={"Content-Type": "text/html"},
                                           text=html))
    r = _with_requests(fake, lambda: verify.verify_identifier(CFG, "https://publisher.test/d"))
    assert r.status is Actual.PAYWALLED


def test_network_error_is_error_not_missing():
    import requests as _r
    fake = _FakeRequests(head=_r.RequestException("boom"))
    r = _with_requests(fake, lambda: verify.verify_identifier(CFG, "https://x.test/d"))
    assert r.status is Actual.ERROR


def test_non_resolvable_identifier_is_not_applicable():
    r = verify.verify_identifier(CFG, "Supplementary Table 3")
    assert r.status is Actual.NOT_APPLICABLE


def test_aggregate_takes_the_worst_case():
    mk = lambda s: verify.VerificationResult("i", s)
    assert verify.aggregate([mk(Actual.ACCESSIBLE), mk(Actual.NOT_FOUND)]) is Actual.NOT_FOUND
    assert verify.aggregate([mk(Actual.ACCESSIBLE), mk(Actual.ACCESSIBLE)]) is Actual.ACCESSIBLE
    assert verify.aggregate([]) is Actual.NOT_ATTEMPTED
    assert verify.aggregate([mk(Actual.NOT_APPLICABLE)]) is Actual.NOT_APPLICABLE


# ===========================================================================
# classify — constrained output, and the self-correction loop
# ===========================================================================
def _classify(client, payload_passages=None):
    return classify(
        client, model="fake-model", title="Transmon readout fidelity improvements",
        venue="PRApplied", year=2024, arm=Arm.QUANTUM_COMPUTING,
        openalex_id="W1", doi="10.1/x",
        crossref_summary={"das_texts": ["data are in Zenodo"], "dataset_ids": [],
                          "confirmed_datasets": []},
        passages=([{"keywords": ["dataset"], "offset": 0,
                    "text": ("Data availability: The datasets generated during the "
                             "current study are available in the Zenodo repository "
                             "under CC-BY. In the methods we simulate 10,000 random "
                             "circuit instances with seed 42.")}]
                   if payload_passages is None else payload_passages),
        retrieval=Retrieval.FULL_TEXT, text_source=TextSource.ARXIV, text_url="u")


def test_valid_payload_becomes_a_validated_record():
    client = FakeAnthropic([FakeMessage([FakeBlock(good_payload())])])
    rec, usage = _classify(client)
    assert rec.axis_b.case.value is Case.OPENLY_ACCESSIBLE
    assert rec.axis_a.case.value is Case.NOT_APPLICABLE
    assert not rec.is_purely_theoretical, "axis B is present, so not theory-only"
    assert usage.input_tokens == 1000 and usage.output_tokens == 400
    assert rec.axis_b.repo_types == [RepoType.ZENODO]
    # the model is never allowed to pre-empt verification
    assert rec.axis_b.actual.value is Actual.NOT_ATTEMPTED


def test_tool_use_is_forced():
    client = FakeAnthropic([FakeMessage([FakeBlock(good_payload())])])
    _classify(client)
    kw = client.calls[0]
    assert kw["tool_choice"] == {"type": "tool", "name": "record_classification"}
    assert kw["tools"][0]["input_schema"]["required"]


def test_schema_violation_is_handed_back_and_corrected():
    """The retry loop: an invalid first answer produces a tool_result error
    containing the validator's own complaint, and the second answer is kept."""
    bad = good_payload()
    bad["axis_b"]["identifiers"] = []          # openly_accessible with no path
    client = FakeAnthropic([
        FakeMessage([FakeBlock(bad, "tu_a")]),
        FakeMessage([FakeBlock(good_payload(), "tu_b")]),
    ])
    rec, usage = _classify(client)
    assert rec.axis_b.case.value is Case.OPENLY_ACCESSIBLE
    assert len(client.calls) == 2
    followup = client.calls[1]["messages"][-1]["content"][0]
    assert followup["is_error"] is True
    assert "identifier" in followup["content"]
    assert usage.input_tokens == 2000, "both attempts are billed and counted"


def test_persistently_invalid_output_raises_rather_than_writing_garbage():
    bad = good_payload()
    bad["axis_b"]["case"]["value"] = "fully_open"     # not in the vocabulary
    client = FakeAnthropic([FakeMessage([FakeBlock(bad, f"t{i}")]) for i in range(3)])
    with pytest.raises(ClassificationError, match="failed after 3 attempts"):
        _classify(client)


def test_truncated_output_is_an_error_not_a_partial_record():
    client = FakeAnthropic([FakeMessage([FakeBlock(good_payload())],
                                        stop_reason="max_tokens")])
    with pytest.raises(ClassificationError, match="max_tokens"):
        _classify(client)


def test_prose_only_answer_is_retried():
    client = FakeAnthropic([
        FakeMessage([FakeTextBlock()], stop_reason="end_turn"),
        FakeMessage([FakeBlock(good_payload())]),
    ])
    rec, _ = _classify(client)
    assert rec.axis_b.case.value is Case.OPENLY_ACCESSIBLE
    assert len(client.calls) == 2


def test_escalation_triggers():
    client = FakeAnthropic([FakeMessage([FakeBlock(good_payload())])])
    rec, _ = _classify(client)
    assert should_escalate(rec) is False

    flagged = good_payload(needs_human_review=True, review_reason="ambiguous")
    client = FakeAnthropic([FakeMessage([FakeBlock(flagged)])])
    rec, _ = _classify(client)
    assert should_escalate(rec) is True

    lowconf = good_payload()
    lowconf["axis_a"]["case"]["confidence"] = "medium"
    client = FakeAnthropic([FakeMessage([FakeBlock(lowconf)])])
    rec, _ = _classify(client)
    assert should_escalate(rec) is True


def test_crossref_das_reaches_the_prompt_as_authoritative():
    client = FakeAnthropic([FakeMessage([FakeBlock(good_payload())])])
    _classify(client)
    msg = client.calls[0]["messages"][0]["content"]
    assert "DATA AVAILABILITY STATEMENT" in msg and "data are in Zenodo" in msg
    assert "deposited by the publisher" in msg


def test_no_passages_prompt_says_the_text_is_missing():
    msg = C._build_user_message(title="T", venue="V", year=2024,
                                arm=Arm.QUANTUM_COMPUTING,
                                crossref_summary={}, passages=[])
    assert "No full text was retrievable" in msg


# ---------------------------------------------------------------------------
# Diana, 28 Oct 2026 (review_issues.md S219): a non-disclosure agreement makes
# the axis explicitly_restricted. Before this the close-pair rule said "an offer
# to supply on contact is on_request, however hedged", and P9 kept an on-request
# route on_request with the restriction recorded as a token, so identical NDA
# wording was stored at 6 on one paper and 7 on another.
# ---------------------------------------------------------------------------
def test_nda_statement_is_explicitly_restricted_and_passes_the_gate():
    prompt = " ".join(C.SYSTEM_PROMPT.split())          # the prompt wraps lines
    rule = (prompt.split("`on_request` vs `explicitly_restricted`")[1]
            .split("`openly_accessible` requires")[0])
    assert "non-disclosure agreement (NDA), the axis is explicitly_restricted" in rule
    assert "`confidential`" in rule
    assert C.PROMPT_VERSION >= "v2.1.0"                # introduced in v2.1.0

    nda = ("supporting data can only be made available to bona fide researchers "
           "subject to a non-disclosure agreement")
    payload = good_payload()
    payload["evidence"][0] = {"id": 1, "quote": nda, "section": "Data availability",
                              "locator": ""}
    payload["axis_b"].update(
        case={"value": "explicitly_restricted", "confidence": "high",
              "rationale": "access is conditioned on an NDA, though a contact is named",
              "evidence_ids": [1]},
        declared={"value": "restricted", "confidence": "high",
                  "rationale": "the statement restricts access", "evidence_ids": [1]},
        pinned="none", restriction="confidential",
        repo_types=[], identifiers=[], names=[])
    client = FakeAnthropic([FakeMessage([FakeBlock(payload)])])
    rec, _ = _classify(client, payload_passages=[{
        "keywords": ["data availability"], "offset": 0,
        "text": (f"Data availability: The datasets generated during the current "
                 f"study: {nda}. Details and how to request access are available "
                 f"from the corresponding author.")}])

    assert "non-disclosure agreement" in " ".join(client.calls[0]["system"].split())
    assert rec.axis_b.case.value is Case.EXPLICITLY_RESTRICTED
    assert rec.axis_b.restriction.value == "confidential"
    assert rec.prompt_version == C.PROMPT_VERSION


_NO_DATA = "No datasets were generated or analysed during the current study"
_BENCH = "We train and evaluate the hybrid classifier on the MNIST benchmark"


def _no_data_payload(axis_b):
    payload = good_payload()
    payload["evidence"] = [
        {"id": 1, "quote": _NO_DATA, "section": "Data availability", "locator": ""},
        {"id": 2, "quote": _BENCH, "section": "Methods", "locator": ""}]
    payload["paper_type"]["evidence_ids"] = [2]
    payload["axis_a"] = {
        "case": {"value": "access_not_provided", "confidence": "medium",
                 "rationale": "MNIST is named in the methods but given no route; the "
                              "statement does not settle Axis A", "evidence_ids": [2]},
        "data_type": {"value": "real", "confidence": "medium",
                      "rationale": "a public image benchmark", "evidence_ids": [2]},
        "declared": {"value": "none_stated", "confidence": "medium",
                     "rationale": "no route to the benchmark is stated", "evidence_ids": [2]},
        "granularity": {"value": "raw", "confidence": "low",
                        "rationale": "the benchmark as published", "evidence_ids": [2]},
        "pinned": "none", "restriction": "none_given",
        "repo_types": [], "identifiers": [], "names": ["MNIST"]}
    payload["axis_b"] = axis_b
    payload["das_present"] = True
    return payload


def _classify_no_data(payload):
    client = FakeAnthropic([FakeMessage([FakeBlock(payload)])])
    rec, _ = _classify(client, payload_passages=[{
        "keywords": ["data availability"], "offset": 0,
        "text": (f"{_BENCH}, reporting test accuracy for each circuit depth. "
                 f"Data availability: {_NO_DATA}.")}])
    return client, rec


def test_no_data_statement_codes_axis_b_only_and_passes_the_gate():
    """S219/S220 (28 Oct 2026): an explicit no-data statement makes Axis B
    not_applicable even when results are reported, and leaves Axis A to the
    body -- here a named benchmark the statement says nothing about."""
    prompt = " ".join(C.SYSTEM_PROMPT.split())
    rule = (prompt.split("One exception overrides the line above.")[1]
            .split("Likewise, citing or discussing")[0])
    assert "Axis B is `not_applicable`" in rule
    assert "governs Axis B only. It does NOT settle Axis A" in rule
    assert "set needs_human_review" in rule, "a denial beside a route stays open"
    assert "genuinely produce nothing" in prompt, "the general rule still stands"
    assert C.PROMPT_VERSION == "v2.3.0"

    axis_b = copy.deepcopy(good_payload()["axis_b"])
    axis_b.update(
        case={"value": "not_applicable", "confidence": "high",
              "rationale": "the statement denies any dataset was generated or analysed, "
                           "which governs Axis B under the no-data ruling",
              "evidence_ids": [1]},
        data_type={"value": "unclear", "confidence": "low",
                   "rationale": "no dataset is declared", "evidence_ids": [1]},
        declared={"value": "none_stated", "confidence": "medium",
                  "rationale": "declared that no datasets exist", "evidence_ids": [1]},
        granularity={"value": "not_applicable", "confidence": "medium",
                     "rationale": "no dataset is declared", "evidence_ids": [1]},
        pinned="none", repo_types=[], identifiers=[], names=[])
    client, rec = _classify_no_data(_no_data_payload(axis_b))
    assert "governs Axis B only" in " ".join(client.calls[0]["system"].split())
    assert rec.axis_b.case.value is Case.NOT_APPLICABLE
    assert rec.axis_a.case.value is Case.ACCESS_NOT_PROVIDED
    assert rec.prompt_version == "v2.3.0"


def test_h_no_data_statement_builder_passes_the_gate():
    """The batch builder for the ruling must produce what the gate accepts, so
    a manual batch cannot drift back to coding the output access_not_provided."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch"))
    import h
    axis_b = h.no_data_statement([1], "test accuracy for each circuit depth")
    axis_b.update(generation_code=False, generation_params=False,
                  seed_specified=False, environment_specified=False)
    assert "Axis A is coded from the body" in axis_b["case"]["rationale"]
    _, rec = _classify_no_data(_no_data_payload(axis_b))
    assert rec.axis_b.case.value is Case.NOT_APPLICABLE
    assert rec.axis_a.case.value is Case.ACCESS_NOT_PROVIDED


# ---------------------------------------------------------------------------
# Regression: evidence had to be verbatim, but nothing checked verbatim *from
# what*. A model shown only a title could quote the title back, satisfy
# "confidence=high requires an evidence_id", and land a confident code on a
# paper whose text was never retrieved. That is how v1 produced unusable output.
# ---------------------------------------------------------------------------
def test_quote_fabricated_from_the_title_is_rejected():
    payload = copy.deepcopy(good_payload())
    payload["evidence"] = [
        {"id": 1, "quote": "Transmon readout fidelity improvements",  # the title
         "section": "Title", "locator": ""},
        {"id": 2, "quote": "Transmon readout fidelity improvements",
         "section": "Title", "locator": ""}]
    client = FakeAnthropic([FakeMessage([FakeBlock(payload)]) for _ in range(3)])
    with pytest.raises(ClassificationError) as e:
        _classify(client, payload_passages=[])
    assert "not present in the supplied passages" in str(e.value)


def test_quote_absent_from_the_passages_is_rejected():
    payload = copy.deepcopy(good_payload())
    payload["evidence"][0]["quote"] = "all data are available on request from the authors"
    client = FakeAnthropic([FakeMessage([FakeBlock(payload)]) for _ in range(3)])
    with pytest.raises(ClassificationError):
        _classify(client)


def test_grounding_tolerates_whitespace_and_hyphenation():
    """PDF extraction wraps lines mid-word and collapses oddly; a real verbatim
    quote must not be rejected over layout."""
    g = C.grounding_text(crossref_summary={}, passages=[
        {"text": "the raw data are avail-\nable in the   Zenodo repository"}])
    ev = [type("E", (), {"id": 1, "quote": "data are available in the Zenodo repository"})()]
    C.check_evidence_grounded(ev, g)   # must not raise


def test_a_quote_from_the_das_alone_is_grounded():
    """The DAS is shown to the model, so quoting it is legitimate even with no
    passages at all."""
    payload = copy.deepcopy(good_payload())
    payload["evidence"] = [{"id": 1, "quote": "data are in Zenodo", "section": "DAS"},
                           {"id": 2, "quote": "data are in Zenodo", "section": "DAS"}]
    client = FakeAnthropic([FakeMessage([FakeBlock(payload)])])
    rec, _ = _classify(client, payload_passages=[])
    assert rec.evidence[0].quote == "data are in Zenodo"


# ===========================================================================
# store — resume, dedup, atomicity, version refusal
# ===========================================================================
def _store(tmp="/tmp/_dap_test.sqlite"):
    Path(tmp).unlink(missing_ok=True)
    Path(tmp + "-wal").unlink(missing_ok=True)
    return Store(tmp)


def test_enrol_is_idempotent():
    s = _store()
    rows = [{"openalex_id": "W1", "doi": "a", "arm": "quantum_computing"},
            {"openalex_id": "W2", "doi": "b", "arm": "machine_learning"}]
    assert s.enrol(rows) == 2
    assert s.enrol(rows) == 0, "re-enrolling must not duplicate"
    assert s.counts() == {"pending": 2}


def test_duplicate_openalex_id_is_structurally_impossible():
    s = _store()
    s.enrol([{"openalex_id": "W1", "doi": "a", "arm": "other_cs"}])
    with sqlite3.connect(s.path) as c:
        try:
            c.execute("INSERT INTO papers (openalex_id,status,updated_at) "
                      "VALUES ('W1','pending','now')")
            raise AssertionError("primary key did not prevent the duplicate")
        except sqlite3.IntegrityError:
            pass


def test_failed_papers_are_retried_then_parked():
    """v1 retried forever and grew failures.csv without bound."""
    s = _store()
    s.enrol([{"openalex_id": "W1", "doi": "a", "arm": "other_cs"}])
    for _ in range(MAX_ATTEMPTS):
        assert len(s.pending()) == 1
        s.fail("W1", "retrieval", "TransientError", "timeout")
    assert s.pending() == [], "should stop being offered after MAX_ATTEMPTS"


def test_permanent_failure_is_not_retried_at_all():
    s = _store()
    s.enrol([{"openalex_id": "W1", "doi": "a", "arm": "other_cs"}])
    s.fail("W1", "retrieval", "PermanentError", "404", permanent=True)
    assert s.pending() == []
    assert s.counts().get("dead") == 1


def test_export_refuses_to_pool_scheme_versions():
    """The single most dangerous silent corruption available to this project."""
    from fakes import good_payload  # noqa
    s = _store()
    s.enrol([{"openalex_id": "W1", "doi": "a", "arm": "other_cs"}])
    client = FakeAnthropic([FakeMessage([FakeBlock(good_payload())])])
    rec, _ = _classify(client)
    rec.openalex_id = "W1"
    s.save(rec)
    with sqlite3.connect(s.path) as c:       # simulate an un-migrated v1 row
        c.execute("INSERT INTO papers (openalex_id,status,scheme_version,"
                  "record_json,updated_at) VALUES ('W0','done','1.0.0','{}','now')")
    with pytest.raises(RuntimeError, match="scheme versions"):
        s.export_csv("/tmp/_x.csv")


def _store_with_second_version(version):
    from fakes import good_payload  # noqa
    s = _store()
    s.enrol([{"openalex_id": "W1", "doi": "a", "arm": "other_cs"},
             {"openalex_id": "W2", "doi": "b", "arm": "other_cs"}])
    for oid in ("W1", "W2"):
        client = FakeAnthropic([FakeMessage([FakeBlock(good_payload())])])
        rec, _ = _classify(client)
        rec.openalex_id = oid
        s.save(rec)
    with sqlite3.connect(s.path) as c:       # W2 recoded under another ruling
        rj = json.loads(c.execute("SELECT record_json FROM papers WHERE openalex_id='W2'")
                        .fetchone()[0])
        rj["scheme_version"] = version
        c.execute("UPDATE papers SET scheme_version=?, record_json=? WHERE openalex_id='W2'",
                  (version, json.dumps(rj)))
    return s


def test_export_pools_patch_versions_and_keeps_them_traceable():
    """A patch is a ruling on the same vocabulary: it pools, and each row says
    which version it was coded under."""
    major, minor, patch = SCHEME_VERSION.split(".")
    bumped = f"{major}.{minor}.{int(patch) + 1}"
    s = _store_with_second_version(bumped)
    out = "/tmp/_x_patch.csv"
    assert s.export_csv(out)["analysis"] == 2
    with open(out, newline="") as f:
        got = {r["openalex_id"]: r["scheme_version"] for r in csv.DictReader(f)}
    assert got == {"W1": SCHEME_VERSION, "W2": bumped}


def test_export_refuses_a_minor_version_difference():
    major, minor, _ = SCHEME_VERSION.split(".")
    s = _store_with_second_version(f"{major}.{int(minor) + 1}.0")
    with pytest.raises(RuntimeError, match="scheme versions"):
        s.export_csv("/tmp/_x_minor.csv")


def test_concurrent_writes_do_not_corrupt():
    s = _store()
    ids = [f"W{i}" for i in range(60)]
    s.enrol([{"openalex_id": i, "doi": "", "arm": "other_cs"} for i in ids])

    def work(i):
        client = FakeAnthropic([FakeMessage([FakeBlock(good_payload())])])
        rec, _ = _classify(client)
        rec.openalex_id = i
        s.save(rec)

    threads = [threading.Thread(target=work, args=(i,)) for i in ids]
    for t in threads: t.start()
    for t in threads: t.join()
    assert s.counts().get("done") == 60


def test_export_round_trips_tokens():
    import csv
    s = _store()
    s.enrol([{"openalex_id": "W1", "doi": "a", "arm": "quantum_computing"}])
    client = FakeAnthropic([FakeMessage([FakeBlock(good_payload())])])
    rec, _ = _classify(client)
    rec.openalex_id = "W1"
    s.save(rec)
    s.export_csv("/tmp/_an.csv", "/tmp/_ev.csv", "/tmp/_rq.csv")
    row = next(csv.DictReader(open("/tmp/_an.csv")))
    assert row["b_case"] == "openly_accessible"
    assert row["b_openness_rank"] == "5"
    assert row["b_repo_types"] == "zenodo"
    ev = list(csv.DictReader(open("/tmp/_ev.csv")))
    assert len(ev) == 2 and ev[0]["openalex_id"] == "W1"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ---------------------------------------------------------------------------
# Regression: the retrieval cascade only ever tried `best_oa_location`.
#
# OpenAlex also returns a `locations` list, which the pipeline fetches and
# stores but the cascade ignored. `best_oa_location` is frequently a publisher
# landing page — a doi.org stub that resolves to a JS shim with no extractable
# text — while the same paper sits readable in an institutional repository, PMC
# or OSTI. Measured on the first live run: trying the list recovered 22 of 79
# otherwise-unusable papers.
# ---------------------------------------------------------------------------
ARTICLE_HTML = ("<html><body><article>" +
                "We measured transmon readout fidelity across the device. " * 80 +
                "Data availability: the datasets are deposited in Zenodo. " +
                "</article></body></html>").encode()

#: A doi.org stub: HTTP 200, but the body carries no article text.
STUB_HTML = b"<html><body><div id='redirect'>Loading...</div></body></html>"


def _cascade(work, facts=None, routes=None):
    facts = facts or CrossrefFacts(found=True)
    original = sources._get
    sources._get = fake_get_factory(routes or {})
    try:
        return sources.retrieve_full_text(CFG, work=work, doi="10.1/x",
                                          title="Transmon readout fidelity", facts=facts)
    finally:
        sources._get = original


def test_repository_location_is_tried_when_best_oa_is_a_dead_stub():
    work = {"best_oa_location": {"landing_page_url": "https://doi.org/10.1/x"},
            "locations": [
                {"landing_page_url": "https://doi.org/10.1/x"},
                {"pdf_url": "https://repository.example/paper.pdf"}],
            "ids": {}}
    ft = _cascade(work, routes={
        "doi.org": FakeResponse(headers={"Content-Type": "text/html"}, content=STUB_HTML),
        "repository.example": FakeResponse(headers={"Content-Type": "text/html"},
                                           content=ARTICLE_HTML),
    })
    assert ft.ok, "a readable repository copy must not be discarded"
    assert ft.url == "https://repository.example/paper.pdf"


def test_a_url_is_never_fetched_twice_across_the_cascade():
    """best_oa_location is usually also present in `locations`; refetching it
    doubles the request count and the politeness cost for no new information."""
    calls = []

    def counting(cfg, url, **kw):
        calls.append(url)
        return FakeResponse(headers={"Content-Type": "text/html"}, content=STUB_HTML)

    work = {"best_oa_location": {"pdf_url": "https://pub.example/a.pdf"},
            "locations": [{"pdf_url": "https://pub.example/a.pdf"},
                          {"landing_page_url": "https://pub.example/a.pdf"}],
            "ids": {}}
    original = sources._get
    sources._get = counting
    try:
        sources.retrieve_full_text(CFG, work=work, doi="", title="t",
                                   facts=CrossrefFacts(found=True))
    finally:
        sources._get = original
    assert calls.count("https://pub.example/a.pdf") == 1, f"fetched {len(calls)} times"


def test_best_oa_location_still_wins_when_it_works():
    work = {"best_oa_location": {"pdf_url": "https://best.example/a.pdf"},
            "locations": [{"pdf_url": "https://other.example/b.pdf"}], "ids": {}}
    ft = _cascade(work, routes={
        "best.example": FakeResponse(headers={"Content-Type": "text/html"},
                                     content=ARTICLE_HTML),
        "other.example": sources.PermanentError("must not be reached"),
    })
    assert ft.ok and ft.url == "https://best.example/a.pdf"


# ---------------------------------------------------------------------------
# Regression: a scheme-less URL was treated as "not a resolvable identifier".
#
# Found on the first live verify run. Papers routinely write "github.com/org/repo"
# with no scheme, and `_normalise` returned "" for anything not starting with
# http(s):// or "10.". The paper was coded `openly_accessible`, verification was
# skipped entirely, and `Actual` came back `not_applicable` — the exact silent
# pass verify.py exists to prevent.
# ---------------------------------------------------------------------------
def test_scheme_less_url_is_verified_not_skipped():
    assert verify._normalise("github.com/BBN-Q/APS2-Comms") == \
        "https://github.com/BBN-Q/APS2-Comms"
    assert verify._normalise("zenodo.org/record/123") == "https://zenodo.org/record/123"
    assert verify._normalise("www.example.org/data") == "https://www.example.org/data"


def test_doi_forms_normalise():
    assert verify._normalise("10.5281/zenodo.51303") == "https://doi.org/10.5281/zenodo.51303"
    assert verify._normalise("doi:10.1038/nature23879") == "https://doi.org/10.1038/nature23879"
    assert verify._normalise("https://doi.org/10.1/x") == "https://doi.org/10.1/x"


def test_prose_is_not_mistaken_for_a_locator():
    """The looser rule must not start fetching sentences. An unresolvable
    identifier is a real finding; a fabricated URL is a wrong one."""
    for junk in ("Supplementary Material", "the data are available", "ImageNet",
                 "an internal corpus of support tickets", "", "   ", "see Methods"):
        assert verify._normalise(junk) == "", f"{junk!r} must not become a URL"


def test_trailing_punctuation_is_stripped():
    assert verify._normalise("github.com/a/b.") == "https://github.com/a/b"
    assert verify._normalise("(zenodo.org/record/1)") == ""  # leading paren is not a host


# ---------------------------------------------------------------------------
# Regression: an underscore lost to a space inside a repository path.
#
# Batch 210, W3111268576: the PDF's text layer gives
# `github.com/BOHRTECHNOLOGY/quantum tsp` and
# `github.com/EnekoOsaba/DWAVE4TSP Clustering Multitasking` for repositories
# named `quantum_tsp` and `DWAVE4TSP_ClusteringMultitasking`. In the second, one
# underscore became TWO spaces, so no substitution on the string is right; the
# PDF's own link annotation is wrong too, so the fix is on the retry side.
# Before this, a bare host with a space was NOT_APPLICABLE (never checked) and a
# scheme'd one was fetched with %20 and 404'd.
# ---------------------------------------------------------------------------
class _NoWait:
    def wait(self, url):
        pass


def _with_routes(fake, fn):
    """`_with_requests` without the per-host politeness delay: a repair makes
    up to nine requests to one host."""
    original = verify.LIMITER
    verify.LIMITER = _NoWait()
    try:
        return _with_requests(fake, fn)
    finally:
        verify.LIMITER = original


_TSP = "https://github.com/BOHRTECHNOLOGY/quantum tsp"
_TSP_FIXED = "https://github.com/BOHRTECHNOLOGY/quantum_tsp"
_DWAVE = "github.com/EnekoOsaba/DWAVE4TSP Clustering Multitasking"
_DWAVE_FIXED = "https://github.com/EnekoOsaba/DWAVE4TSP_ClusteringMultitasking"


def _ok(url):
    return FakeResponse(status=200, headers={"Content-Type": "text/html"},
                        text="<html>repository</html>", url=url)


def test_candidates_are_each_gap_underscore_or_nothing_all_underscore_first():
    got = [u for u, _ in verify._space_repairs("https://github.com/TNO- Quantum/qscore dwave")]
    assert got == ["https://github.com/TNO-_Quantum/qscore_dwave",
                   "https://github.com/TNO-_Quantum/qscoredwave",
                   "https://github.com/TNO-Quantum/qscore_dwave",   # de-hyphenated, no extra rule
                   "https://github.com/TNO-Quantum/qscoredwave"]


def test_candidates_touch_the_path_only_and_are_capped():
    assert verify._space_repairs("https://x.org/a?q=b c") == []        # query, not path
    assert verify._space_repairs("https://x.org/a  b")[0][0] == "https://x.org/a_b"  # one run, one gap
    assert len(verify._space_repairs("https://x.org/a b c d")) == 8
    assert verify._space_repairs("https://x.org/a b c d e") == []      # four gaps: noise
    assert verify._space_repairs("https://x.org/ab") == []


def test_one_space_repaired_to_underscore():
    fake = _FakeRequests(head=FakeResponse(status=404), routes={"quantum_tsp": _ok(_TSP_FIXED)})
    r = _with_routes(fake, lambda: verify.verify_identifier(CFG, _TSP))
    assert r.status is Actual.ACCESSIBLE and r.http_status == 200
    assert r.identifier == _TSP                      # verbatim, as the paper gave it
    assert r.final_url == _TSP_FIXED
    assert r.note == "repaired: space->underscore (1 of 2 candidates resolve)"
    assert fake.urls == [_TSP, _TSP_FIXED, "https://github.com/BOHRTECHNOLOGY/quantumtsp"]
    assert verify.derive_case(Case.OPENLY_ACCESSIBLE, r.status)[0] is Case.OPENLY_ACCESSIBLE


def test_one_underscore_that_became_two_spaces_is_found():
    """The DWAVE4TSP case: underscore, then nothing. A bare host with a space
    was NOT_APPLICABLE before; now it is checked."""
    fake = _FakeRequests(head=FakeResponse(status=404),
                         routes={"DWAVE4TSP_ClusteringMultitasking": _ok(_DWAVE_FIXED)})
    r = _with_routes(fake, lambda: verify.verify_identifier(CFG, _DWAVE))
    assert r.status is Actual.ACCESSIBLE
    assert r.identifier == _DWAVE and r.final_url == _DWAVE_FIXED
    assert r.note == "repaired: space->underscore, space->nothing (1 of 4 candidates resolve)"
    assert fake.urls[0] == "https://" + _DWAVE                 # the original, first
    assert fake.urls[1].endswith("DWAVE4TSP_Clustering_Multitasking")  # then all-underscore
    assert len(fake.urls) == 5                                 # every candidate, for ambiguity


def test_a_repair_that_redirects_names_the_candidate():
    """Live, 5 Oct: `TNO- Quantum/qscore dwave` repairs to
    `TNO-Quantum/qscore_dwave`, which GitHub forwards to `TNO-Quantum/qscore`.
    `final_url` is where the check landed; the note says how it got there, so a
    repair is not read as a rename."""
    fixed = "https://github.com/TNO-Quantum/qscore_dwave"
    fake = _FakeRequests(head=FakeResponse(status=404),
                         routes={"TNO-Quantum/qscore_dwave": _ok("https://github.com/TNO-Quantum/qscore")})
    r = _with_routes(fake, lambda: verify.verify_identifier(CFG, "github.com/TNO- Quantum/qscore dwave"))
    assert r.status is Actual.ACCESSIBLE
    assert r.final_url == "https://github.com/TNO-Quantum/qscore"
    assert r.note == ("repaired: space->nothing, space->underscore "
                      f"(1 of 4 candidates resolve), via {fixed}")


def test_two_candidates_resolving_is_ambiguous_and_nothing_is_picked():
    """BOHRTECHNOLOGY has two repositories starting with `quantum`: the shape
    that would otherwise be coded wrong without anyone seeing it."""
    fake = _FakeRequests(head=FakeResponse(status=404),
                         routes={"quantum_tsp": _ok(_TSP_FIXED),
                                 "quantumtsp": _ok("https://github.com/BOHRTECHNOLOGY/quantumtsp")})
    r = _with_routes(fake, lambda: verify.verify_identifier(CFG, _TSP))
    assert r.status is Actual.NOT_FOUND and r.http_status == 404
    assert r.final_url == "https://final.example/x"            # the original's, not a candidate's
    assert r.note.startswith("ambiguous repair, not picked: 2 of 2 candidates resolve")
    assert verify.derive_case(Case.OPENLY_ACCESSIBLE, r.status)[0] is Case.ACCESS_UNRESOLVABLE


def test_no_candidate_resolving_leaves_the_result_as_today():
    fake = _FakeRequests(head=FakeResponse(status=404))
    r = _with_routes(fake, lambda: verify.verify_identifier(CFG, _TSP))
    assert r == verify.VerificationResult(_TSP, Actual.NOT_FOUND, 404, "https://final.example/x")
    assert len(fake.urls) == 3
    assert verify.derive_case(Case.OPENLY_ACCESSIBLE, r.status)[0] is Case.ACCESS_UNRESOLVABLE


def test_a_403_generates_no_candidates():
    """A 403 means the path is probably right; retrying it wastes requests."""
    fake = _FakeRequests(head=FakeResponse(status=403), get=FakeResponse(status=403),
                         routes={"quantum_tsp": _ok(_TSP_FIXED)})
    r = _with_routes(fake, lambda: verify.verify_identifier(CFG, _TSP))
    assert r.status is Actual.AUTH_REQUIRED and r.note == "403 after GET fallback"
    assert fake.urls == [_TSP, _TSP]                           # HEAD, GET; no candidate


def test_a_soft_404_host_generates_no_candidates():
    fake = _FakeRequests(head=FakeResponse(status=404), routes={"b_c": _ok("x")})
    r = _with_routes(fake, lambda: verify.verify_identifier(
        CFG, "https://www.kaggle.com/datasets/a/b c"))
    assert r.status is Actual.ERROR and len(fake.urls) == 1


def test_too_many_gaps_leaves_the_result_unchanged():
    ident = "https://github.com/a/b c d e f"
    fake = _FakeRequests(head=FakeResponse(status=404))
    r = _with_routes(fake, lambda: verify.verify_identifier(CFG, ident))
    assert r == verify.VerificationResult(ident, Actual.NOT_FOUND, 404, "https://final.example/x")
    assert len(fake.urls) == 1


def test_one_match_beside_an_unfinished_check_is_not_a_repair():
    """A candidate that timed out could be a second match."""
    import requests as _r
    fake = _FakeRequests(head=FakeResponse(status=404),
                         routes={"quantum_tsp": _ok(_TSP_FIXED),
                                 "quantumtsp": _r.ConnectTimeout("slow")})
    r = _with_routes(fake, lambda: verify.verify_identifier(CFG, _TSP))
    assert r.status is Actual.NOT_FOUND
    assert r.note == "repair not established: 1 of 2 candidates resolve, 1 could not be checked"


def test_a_repaired_path_that_holds_nothing_still_promotes():
    """The repair finds the path; the check of that path still decides."""
    empty = FakeResponse(status=200, headers={"Content-Type": "text/html"},
                         text="<html>This record has been withdrawn</html>", url=_TSP_FIXED)
    fake = _FakeRequests(head=FakeResponse(status=404), routes={"quantum_tsp": empty})
    r = _with_routes(fake, lambda: verify.verify_identifier(CFG, _TSP))
    assert r.status is Actual.RESOLVES_NO_DATA
    assert r.note.startswith("repaired: space->underscore") and "no content" in r.note
    assert verify.derive_case(Case.OPENLY_ACCESSIBLE, r.status)[0] is Case.ACCESS_UNRESOLVABLE


def test_a_bare_host_with_a_space_is_checked_but_prose_still_is_not():
    assert verify._normalise("github.com/BOHRTECHNOLOGY/quantum tsp") == _TSP
    # the six space-bearing identifiers stored on 5 Oct are not locators
    # ...nor is prose with a host in it; the last one passes _BARE_HOST and is
    # refused only because the space falls before the first slash
    for stored in ("CCDC 2331560", "LMFDB (L-functions and Modular Forms Database), "
                   "cited as ref [20]", "zenodo.org and figshare",
                   "github.com: code in the repository/src"):
        assert verify._normalise(stored) == "", f"{stored!r} must not become a URL"
        assert verify.verify_identifier(CFG, stored).status is Actual.NOT_APPLICABLE


# ---------------------------------------------------------------------------
# Regression: a publisher landing page passed as article text, and reference
# lists reached the classifier as passages.
#
# Found in the first manual shakedown: of 18 papers, 3 had passages that were
# pure bibliography and 3 were publisher web-page chrome ("Download citation",
# "Google Scholar"). A landing page is long and carries the paper's own title,
# so the length floor and title-overlap check both pass it.
# ---------------------------------------------------------------------------
LANDING_PAGE = (
    "Transmon readout fidelity improvements\n"
    "Download citation\nShare this article\nGet shareable link\n"
    "Sorry, a shareable link is not currently available\n"
    "Explore related subjects\nGoogle Scholar\nAccess options\n"
    "Learn more about Institutional subscriptions\n"
    + "Discover the latest articles and news in related subjects. " * 80
)

BIBLIOGRAPHY = (
    "References\n"
    "1. Shor, P. W. et al. Polynomial-time algorithms. Phys. Rev. Lett. 41, 303 (1999). "
    "2. Preskill, J. et al. Quantum computing in the NISQ era. Quantum 2, 79 (2018). "
    "3. Ladd, T. D. et al. Quantum computers. Nat. 464, 45 (2010). "
    "4. Harrow, A. W. et al. Quantum supremacy. Nat. 549, 203 (2017). "
    "5. Lund, A. P. et al. Quantum sampling problems. Preprint at arXiv:1702.03061 (2017). "
) * 4

PROSE = ("We simulate the device and record the dataset for each run. " * 60)


def test_publisher_landing_page_is_not_article_text():
    assert looks_like_article(LANDING_PAGE, "Transmon readout fidelity improvements") is False


def test_an_article_mentioning_one_chrome_phrase_is_still_accepted():
    """The threshold must not reject a real paper that happens to say
    'advertisement' or cite Google Scholar once."""
    text = "Google Scholar was used to identify candidate papers. " + PROSE
    assert looks_like_article(text, "") is True


# A catalogue record clears every check the landing-page test relies on: it is
# long, it carries the exact title, it carries the abstract, and it carries the
# reference list, so `text_is_complete` passes too. Shaped on the real
# osti.gov/biblio/1777671, which 35 already-coded papers were classified from.
OSTI_RECORD = (
    "Skip to main content\n"
    "Materials challenges and opportunities for quantum computing hardware\n"
    "Journal Article - Science  DOI: https://doi.org/10.1126/science.abb2823 - OSTI ID: 1777671\n"
    "de Leon, Nathalie P.  Search OSTI.GOV for author \"de Leon, Nathalie P.\"\n"
    "Abstract: Quantum computing hardware technologies have advanced. "
    "Addressing these materials challenges will require scientists and engineers "
    "to work together across the current boundaries of the field. " * 12 +
    "\nView Journal Article  Cite  Citation Formats  MLA APA Chicago BibTeX\n"
    + BIBLIOGRAPHY
)

EPRINTS_RECORD = (
    "University of Southampton Institutional Repository\n"
    "A survey on quantum computing technology\n"
    "The power of quantum computing technologies is based on the fundamentals "
    "of quantum mechanics, such as superposition and entanglement. " * 20 +
    "\ne-pub ahead of print date: 23 January 2019  Published date: 1 February 2019\n"
    "Identifiers  Local EPrints ID: 430386  PURE UUID: 4e7ea2c0-783c-4ecd\n"
    "Actions (login required)  View Item\n"
)


def test_a_repository_record_page_is_not_article_text():
    """The abstract is not the paper. A record page answers 'is there a data
    statement?' with a confident no, because it contains no body to look in."""
    assert looks_like_article(
        OSTI_RECORD, "Materials challenges and opportunities for quantum computing") is False
    assert looks_like_article(EPRINTS_RECORD, "A survey on quantum computing technology") is False


# The journal's own index, served by Rinton Press at a Quantum Information and
# Computation article DOI. It was stored as W2202751465's text and queued for
# coding. Shaped on the real page.
QIC_CONTENTS = (
    "online Editorial Board Guidelines for Authors QIC Book Review Subscribers: "
    "to view the full text of a paper, click on the title of the paper. "
    "To subscribe to QIC, please click Here. Table of Contents "
    + " ".join(f"Vol.{v} ({1999 + v}) " + " ".join(str(i) for i in range(1, 17))
               + " Author Index" for v in range(24, 0, -1)) + " "
    + "Optimizing the Number of Gates in Quantum Search (pp1-20) doi: "
      "https://doi.org/10.26421/QIC17.3-4 " * 30
)


def test_a_journal_contents_page_is_not_article_text():
    assert len(QIC_CONTENTS.split()) > 400
    assert looks_like_article(QIC_CONTENTS, "Optimizing the Number of Gates in Quantum Search") is False


def test_one_mention_of_an_editorial_board_does_not_reject_a_paper():
    """Nine real papers in the corpus say "editorial board" once, in an
    acknowledgement or a conflict-of-interest note."""
    text = ("We measured the gate fidelity of the device across many runs. " * 60 +
            "The authors thank the editorial board for handling the manuscript.")
    assert looks_like_article(text, "gate fidelity of the device") is True


def test_the_record_check_passes_the_length_and_title_tests_it_backstops():
    """Without the marker check these documents are indistinguishable from an
    article — which is why they were accepted for months."""
    from dap.sources import _RECORD_MARKERS, text_is_complete
    body = " ".join(w for w in OSTI_RECORD.split()
                    if not any(m.split()[0] in w.lower() for m in _RECORD_MARKERS))
    assert len(OSTI_RECORD.split()) > 400, "long enough"
    assert looks_like_article(
        body, "Materials challenges and opportunities for quantum computing") is True, \
        "strip the catalogue furniture and every other check still passes it"
    assert text_is_complete(OSTI_RECORD, 20) is True, \
        "it even carries a bibliography, so the completeness check cannot catch it"


def test_an_article_whose_statement_names_a_repository_is_kept():
    """The papers this study most cares about say 'institutional repository' in
    their data statement. Rejecting on that phrase would discard them."""
    text = ("Data availability: the datasets are deposited in the University "
            "institutional repository and may be exported. ") + PROSE
    assert looks_like_article(text, "") is True


def test_bibliography_passages_are_dropped():
    from dap.classify import _is_bibliography
    assert _is_bibliography(BIBLIOGRAPHY) is True
    assert _is_bibliography(PROSE) is False


def test_a_paper_that_is_only_a_bibliography_yields_no_passages():
    """Better to gate it for a human than to classify a reference list."""
    assert select_passages(BIBLIOGRAPHY) == []


def test_article_prose_survives_the_bibliography_filter():
    out = select_passages(PROSE + BIBLIOGRAPHY)
    assert out, "the prose half must still produce passages"
    assert all(not _is_bib(p["text"]) for p in out)


def _is_bib(t):
    from dap.classify import _is_bibliography
    return _is_bibliography(t)


def test_a_short_citation_rich_snippet_is_not_called_a_bibliography():
    """Under 40 words there is not enough text to judge density."""
    from dap.classify import _is_bibliography
    assert _is_bibliography("See Shor et al. (1999) [1] and Preskill et al. (2018) [2].") is False


AUTHOR_YEAR_BIBLIOGRAPHY = (
    "Montanaro, A., & Pallister, S. (2016). Quantum algorithms and the finite element "
    "method. Physical Review A, 93, 032324. doi:10.1103/PhysRevA.93.032324. "
    "Landauer, R. (1961). Irreversibility and heat generation in the computing process. "
    "IBM Journal of Research and Development, 5(3), 183. "
    "Shor, P.W. (1994). Algorithms for quantum computation. In Proceedings of the 35th "
    "Annual Symposium on Foundations of Computer Science, pp. 124-134. "
    "Williams, C. P. (2011). Explorations in quantum computing. London: Springer. "
    "Retrieved from https://github.com/arrayfire/arrayfire. "
) * 2


def test_author_year_bibliography_is_detected():
    """The first patterns assumed a numeric/abbreviated physics style, so an
    entire Springer author-year bibliography scored 81-87 and slipped under the
    threshold — it reached the classifier as though it were the paper."""
    from dap.classify import _is_bibliography
    assert _is_bibliography(AUTHOR_YEAR_BIBLIOGRAPHY) is True
    assert select_passages(AUTHOR_YEAR_BIBLIOGRAPHY) == []


def test_prose_with_inline_author_year_citations_survives():
    """Body text cites too. 'as Landauer (1961) showed' must not be mistaken for
    a reference list, or the widened patterns would delete the paper."""
    from dap.classify import _is_bibliography
    prose = ("As Landauer (1961) first noted, any irreversible operation dissipates "
             "energy, and we simulate the dataset accordingly for each run. " * 12)
    assert _is_bibliography(prose) is False
    assert select_passages(prose), "prose citing sources must still yield passages"


# ---------------------------------------------------------------------------
# Regression: the availability statement was being deleted as a bibliography.
#
# In a two-column PDF the back matter is typeset beside the reference column, so
# PyMuPDF's reading order interleaves them and the window around "DATA
# AVAILABILITY" scores as a reference list. Measured on npj Quantum Information:
# the statement is present in the PDF of every paper, and reached the classifier
# for none of them — 243 papers were coded access_not_provided on its absence,
# against 0 for papers where it was captured.
# ---------------------------------------------------------------------------
INTERLEAVED_DAS = (
    "DATA AVAILABILITY  28. Miki, T., Okita, R. & Shirakashi, J. Variational ansatz "
    "The data that support the findings of this study are available from the "
    "corresponding  optimizations. AIP Adv. 12, 035247 (2022). "
    "authors upon reasonable request.  29. Duan, X. et al. Tuning a general purpose "
    "cache library. CCF Trans. 2, 164 (2020). "
    "30. Shor, P. W. Polynomial-time algorithms. Phys. Rev. Lett. 41, 303 (1999). "
    "31. Preskill, J. Quantum computing in the NISQ era. Quantum 2, 79 (2018). "
)


def test_availability_window_survives_the_bibliography_filter():
    from dap.classify import _is_bibliography
    assert _is_bibliography(INTERLEAVED_DAS) is True, \
        "precondition: interleaving makes this score as a reference list"
    got = select_passages(INTERLEAVED_DAS)
    assert got, "the statement must not be dropped along with the citations"
    assert any("upon reasonable request" in p["text"] for p in got)


def test_a_bibliography_without_an_availability_heading_is_still_dropped():
    """The rescue is narrow: only an explicit availability heading earns it.
    A reference list that merely cites a GitHub URL stays out."""
    assert select_passages(AUTHOR_YEAR_BIBLIOGRAPHY) == []
    assert select_passages(BIBLIOGRAPHY) == []


# ---------------------------------------------------------------------------
# Regression: a paper that matches no keyword is not a paper with no text.
#
# A pure-theory paper mentions no data, no repository and no availability, so it
# produced zero passages and was gated as "no retrievable text" — measured on
# the quantum frame with 4,700-16,500 cleanly extracted words in hand.
# ---------------------------------------------------------------------------
THEORY = ("We prove that the operator norm is bounded below by the spectral gap, "
          "and derive the corresponding inequality for the general case. " * 60)


def test_a_keyword_free_paper_still_yields_structural_passages():
    from dap.classify import structural_passages
    assert select_passages(THEORY) == [], "precondition: no keyword matches"
    got = structural_passages(THEORY)
    assert [p["section"] for p in got] == ["opening", "closing"]
    assert all(p["text"] for p in got)


def test_structural_passages_refuse_a_bibliography():
    """Otherwise this reintroduces exactly what the bibliography filter exists
    to stop: a reference list classified as though it were the paper."""
    from dap.classify import structural_passages
    assert structural_passages(AUTHOR_YEAR_BIBLIOGRAPHY * 3) == []


# ---------------------------------------------------------------------------
# The reference list is cut from the text before passage selection, rather
# than filtered passage by passage. Author-year lists score 70-90 citations per
# 1000 words, under the 120 bar, and reached the classifier as the paper's
# evidence; the old remedy loosened the bar for the whole paper instead.
# Measured on 13 real PDFs from the affected papers (25 Sep 2026): every
# availability heading and every appendix heading survived the cut.
# ---------------------------------------------------------------------------
BODY = "\n".join(["We prove that the operator norm is bounded below by the spectral gap."] * 80)
#: ACM style, full author names: one citation hit per ~20 words.
ACM_REFS = "\n".join(
    [f"Edward Aftandilian and Raluca Sauciuc. 20{10 + i % 9}. Building a simulation of useful program"
     f"\n   analysis tools. In Working Conference on Source Code Analysis, {i}–{i + 9}."
     for i in range(40)])


def test_a_headed_reference_list_is_cut_and_the_appendix_after_it_kept():
    from dap.classify import strip_bibliography
    text = BODY + "\nReferences\n" + ACM_REFS + "\nAppendix A: Proof of the bound\n" + BODY
    out = strip_bibliography(text)
    assert "Aftandilian" not in out
    assert "Appendix A: Proof of the bound" in out
    assert out.count("spectral gap") == 160, "no body line is lost"


def test_an_author_year_list_under_the_density_bar_is_still_cut():
    """The case the per-passage filter missed: a keyword ('simulation') in a
    cited title, in a list too sparse in citations to score as one."""
    from dap.classify import _is_bibliography, strip_bibliography
    assert not _is_bibliography(ACM_REFS), "precondition: under the 120 bar"
    assert select_passages(BODY + "\nREFERENCES\n" + ACM_REFS), "precondition: the keyword matches"
    assert select_passages(strip_bibliography(BODY + "\nREFERENCES\n" + ACM_REFS)) == []


def test_a_statement_sharing_lines_with_references_survives_the_cut():
    """npj's two-column layout: the statement's lines carry reference entries
    from the next column."""
    from dap.classify import strip_bibliography
    das = ("Data availability                                  21. Moghaddam, A. G. Phys. Rev. Lett. 131, 020401 (2023).\n"
           "The data that support the findings of this study are available upon   22. Ravindranath, V. Phys. Rev. B 108 (2023).\n"
           "reasonable request.                                 23. O'Dea, N. Phys. Rev. B 109, L020304 (2024).\n")
    text = BODY + "\nReferences\n" + BIBLIOGRAPHY + "\n" + das + BIBLIOGRAPHY
    out = strip_bibliography(text)
    assert "available upon" in out and "reasonable request" in out


def test_a_contents_line_naming_references_does_not_start_the_cut():
    from dap.classify import strip_bibliography
    toc = "   References                                  17             A. Measurement Heat\n"
    text = toc + BODY + "\n" + BODY
    assert strip_bibliography(text).count("spectral gap") == 160


def test_a_numbered_list_with_no_heading_is_cut_from_its_first_entry():
    from dap.classify import strip_bibliography
    refs = "\n".join(f"[{i}] A. Author, Title of a simulation paper, Phys. Rev. A {i}, 1 (2019)."
                     for i in range(1, 30))
    out = strip_bibliography(BODY + "\n" + refs)
    assert "[1] A. Author" not in out and out.count("spectral gap") == 80


def test_grobid_text_is_not_cut_again():
    """GROBID returns the body without its bibliography, and its section offsets
    index the text as given; removing lines would misplace every section."""
    from dap.classify import passage_text
    text = BODY + "\nReferences\n" + ACM_REFS
    assert passage_text(text, sections=[(0, "Introduction")]) == text
    assert "Aftandilian" not in passage_text(text, sections=[])


# ---------------------------------------------------------------------------
# A record page in a language the marker list does not know (review_issues S8).
# The structure gives it away: the paper's own citation, printed several times.
# ---------------------------------------------------------------------------
RECORD_TITLE = "Improving quantum state transfer with correcting pulses"
ABSTRACT = ("We study the transfer of a quantum state along a chain of coupled "
            "oscillators and show how shaped control pulses reduce the error. ") * 18
SPANISH_RECORD = (
    f"{RECORD_TITLE} Autor(es) Pérez, Juan; García, Ana Fecha de publicación 2023 "
    f"Citación Pérez, J. y García, A. (2023). {RECORD_TITLE}. Quantum Science and "
    f"Technology 8, 045012. Resumen {ABSTRACT} Aparece en las colecciones: Artículos "
    f"Ficheros en este ítem: articulo.pdf 944 kB Adobe PDF Exportar a: Pérez J, García A. "
    f"{RECORD_TITLE}. Quantum Sci Technol. 2023;8:045012. Licencia de uso CC BY")


def test_a_record_page_in_an_unknown_language_is_rejected():
    from dap.sources import _RECORD_MARKERS, repeats_title_like_a_record
    assert sum(m in SPANISH_RECORD.lower() for m in _RECORD_MARKERS) < 2, \
        "precondition: no marker phrase gives it away"
    assert repeats_title_like_a_record(SPANISH_RECORD, RECORD_TITLE)
    assert looks_like_article(SPANISH_RECORD, RECORD_TITLE) is False


def test_a_paper_behind_a_repository_cover_sheet_is_kept():
    """HAL prints the title twice on its cover sheet ("To cite this version")
    right above the paper's own title page — a record's shape, on a real paper."""
    cover = (f"{RECORD_TITLE} Juan Pérez To cite this version: Juan Pérez. {RECORD_TITLE}. "
             f"Quantum Science and Technology, 2023. HAL Id: hal-04396459 ")
    paper = cover + f"{RECORD_TITLE} Juan Pérez (2023) " + ABSTRACT * 30
    assert looks_like_article(paper, RECORD_TITLE) is True


def test_running_heads_do_not_make_a_paper_a_record():
    page = ABSTRACT * 2
    paper = "".join(f"{RECORD_TITLE} Page {i} of 9 (2023) " + page for i in range(9))
    from dap.sources import repeats_title_like_a_record
    assert repeats_title_like_a_record(paper, RECORD_TITLE) is False


def test_a_short_title_is_not_judged():
    from dap.sources import repeats_title_like_a_record
    assert repeats_title_like_a_record("Quantum walks (2020). Quantum walks (2021). " + ABSTRACT,
                                       "Quantum walks") is False


# ---------------------------------------------------------------------------
# Ligatures. A PDF text layer keeps "ﬁ" as one glyph, so a URL printed as
# github.com/leporini/classification was extracted one character short and
# could only ever verify as dead (review_issues S10; 58 coded papers).
# ---------------------------------------------------------------------------
def test_the_seven_latin_ligatures_are_folded_and_nothing_else():
    from dap.sources import fold_ligatures
    assert fold_ligatures("classiﬁcation ﬀ ﬂ ﬃ ﬄ ﬅ ﬆ") == "classification ff fl ffi ffl st st"
    assert fold_ligatures("10⁵ shots, T₁ = 3 µs") == "10⁵ shots, T₁ = 3 µs", \
        "full NFKC would turn 10⁵ into 105"


def test_pdf_text_arrives_with_ligatures_folded(monkeypatch):
    monkeypatch.setattr(sources, "_extract_pdf_fast",
                        lambda content, max_pages=60: "repository: github.com/leporini/classiﬁcation")
    text, _ = sources._extract_pdf(b"%PDF-1.4")
    assert "github.com/leporini/classification" in text


def test_a_quote_typed_without_the_ligature_matches_a_passage_stored_with_it():
    """Passages retrieved before the fold still carry the glyph."""
    from dap.classify import grounding_text, check_evidence_grounded
    from dap.schema import Evidence
    g = grounding_text(crossref_summary={}, passages=[{"text": "available at github.com/leporini/classiﬁcation."}])
    check_evidence_grounded([Evidence(id=1, quote="github.com/leporini/classification")], g)


def test_structural_passages_of_empty_text_are_empty():
    from dap.classify import structural_passages
    assert structural_passages("") == []
    assert structural_passages("   ") == []


def test_a_short_document_yields_only_an_opening():
    from dap.classify import structural_passages
    got = structural_passages("A short note on lattice gauge theory. " * 5)
    assert [p["section"] for p in got] == ["opening"]


# ---------------------------------------------------------------------------
# The deposited reference count as a free completeness check.
#
# A body-only TDM deposit, or a PDF whose extraction stopped early, looks like a
# perfectly good article: long, real prose, matching title. The only thing
# missing is the end of the document — which is exactly where data-availability
# statements live. Crossref tells us how many references should be there.
# ---------------------------------------------------------------------------
BIB_TAIL = "Shor, P. W. (1994). Phys. Rev. Lett. 41, 303. doi:10.1103/x. " * 40
BODY_ONLY = "We measured the device and recorded the dataset for each run. " * 300


def test_reference_count_is_collected_from_crossref():
    f = _with_get({"api.crossref.org": FakeResponse(
                      {"message": {"type": "journal-article", "reference-count": 74,
                                   "container-title": ["X"],
                                   "published": {"date-parts": [[2024]]}}})},
                  lambda: sources.crossref_facts(CFG, "10.1/x"))
    assert f.reference_count == 74


def test_text_without_a_bibliography_is_incomplete():
    assert sources.text_is_complete(BODY_ONLY, 40) is False
    assert sources.text_is_complete(BODY_ONLY + BIB_TAIL, 40) is True


def test_completeness_is_not_checked_when_few_references_are_deposited():
    """A paper with 3 deposited references tells us nothing; do not guess."""
    assert sources.text_is_complete(BODY_ONLY, 3) is True
    assert sources.text_is_complete("", 0) is True


def test_truncated_source_is_rejected_and_the_cascade_continues():
    facts = CrossrefFacts(found=True)
    facts.reference_count = 60
    work = {"best_oa_location": {"pdf_url": "https://truncated.example/a.pdf"},
            "locations": [{"pdf_url": "https://complete.example/b.pdf"}], "ids": {}}
    original = sources._get
    sources._get = fake_get_factory({
        "truncated.example": FakeResponse(headers={"Content-Type": "text/html"},
            content=f"<html><body><article>{BODY_ONLY}</article></body></html>".encode()),
        "complete.example": FakeResponse(headers={"Content-Type": "text/html"},
            content=f"<html><body><article>{BODY_ONLY}{BIB_TAIL}</article></body></html>".encode()),
    })
    try:
        ft = sources.retrieve_full_text(CFG, work=work, doi="", title="", facts=facts)
    finally:
        sources._get = original
    assert ft.ok and ft.complete
    assert ft.url == "https://complete.example/b.pdf", \
        "a truncated body must not end the cascade"


def test_a_truncated_body_is_still_kept_when_nothing_better_exists():
    """Incomplete text beats no text. It is returned, but marked."""
    facts = CrossrefFacts(found=True)
    facts.reference_count = 60
    work = {"best_oa_location": {"pdf_url": "https://truncated.example/a.pdf"},
            "locations": [], "ids": {}}
    original = sources._get
    sources._get = fake_get_factory({
        "truncated.example": FakeResponse(headers={"Content-Type": "text/html"},
            content=f"<html><body><article>{BODY_ONLY}</article></body></html>".encode()),
        "api.unpaywall.org": sources.PermanentError("HTTP 404"),
    })
    try:
        ft = sources.retrieve_full_text(CFG, work=work, doi="", title="", facts=facts)
    finally:
        sources._get = original
    assert ft.ok is True and ft.complete is False
    assert ft.text, "the partial body must not be discarded"


# ---------------------------------------------------------------------------
# Regression: a throttled location silently handed the paper to a worse copy.
#
# `_fetch_and_parse` caught TransientError alongside PermanentError and logged
# neither, so a 429 from the version of record read as "this copy is unusable"
# and the cascade moved on. Found investigating drift: re-running the same
# 200-paper slice into a fresh DB, 5 of 189 papers came back from a different
# source, 4 of them APS, and one coded paper went from the APS version of record
# to its arXiv preprint. Why was never recorded. The first re-retrieve after this
# fix logged APS answering 403, which still falls through by design (rule 8), so
# this closes the throttling path, not necessarily that drift.
# ---------------------------------------------------------------------------
PREFERRED_THEN_ARXIV = {"best_oa_location": {"pdf_url": "https://publisher.example/a.pdf"},
                        "locations": [{"landing_page_url": "https://arxiv.org/abs/2101.00001"}],
                        "ids": {}}
ARXIV_OK = FakeResponse(headers={"Content-Type": "text/html"}, content=ARTICLE_HTML)


def test_a_throttled_preferred_copy_is_retried_not_replaced_by_the_preprint():
    with pytest.raises(sources.TransientError, match="publisher.example"):
        _cascade(PREFERRED_THEN_ARXIV, routes={
            "publisher.example": sources.TransientError("exhausted 3 attempts: HTTP 429"),
            "arxiv.org": ARXIV_OK})


def test_a_throttled_copy_is_retried_when_nothing_else_is_readable():
    """Settling for no text is the same silent loss as settling for a preprint.
    The error must carry every location's outcome, or the failure row says
    nothing about what to retry."""
    work = {"best_oa_location": {"pdf_url": "https://publisher.example/a.pdf"},
            "locations": [{"landing_page_url": "https://doi.org/10.1/x"}], "ids": {}}
    with pytest.raises(sources.TransientError, match=r"doi\.org/10\.1/x -> not_article"):
        _cascade(work, routes={
            "publisher.example": sources.TransientError("HTTP 503"),
            "doi.org": FakeResponse(headers={"Content-Type": "text/html"}, content=STUB_HTML),
            "api.unpaywall.org": sources.PermanentError("HTTP 404")})


def test_a_refused_copy_still_falls_through_and_the_reason_is_kept():
    """A 403 is an answer, not weather. Rule 8: take the open copy instead."""
    ft = _cascade(PREFERRED_THEN_ARXIV, routes={
        "publisher.example": sources.PermanentError("HTTP 403"),
        "arxiv.org": ARXIV_OK})
    assert ft.ok and ft.source is TextSource.ARXIV
    assert [(a["url"], a["outcome"]) for a in ft.attempts] == [
        ("https://publisher.example/a.pdf", "permanent: HTTP 403"),
        ("https://arxiv.org/pdf/2101.00001", "ok")]


# ---------------------------------------------------------------------------
# PDF parsing: fast local path for the census, GROBID for the coded subset.
#
# pdfplumber reconstructs text from glyph positions with no notion of columns,
# so a two-column paper came back interleaved and word-joined — which is why
# stored evidence quotes read like "Forthetransmonqubitsthatwedesign".
# ---------------------------------------------------------------------------
TEI = b"""<?xml version="1.0"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body>
<div><head>INTRODUCTION</head><p>Quantum computing holds the promise of solving
problems untenable on conventional computers.</p></div>
<div><head>DATA AVAILABILITY</head><p>The datasets are deposited in Zenodo.</p></div>
</body></text></TEI>"""


def test_grobid_is_used_when_configured():
    cfg = Config(contact_email="a@b.edu", grobid_url="http://grobid.test")
    calls = []

    class R:
        status_code = 200
        content = TEI

    def fake_post(url, **kw):
        calls.append(url)
        return R()

    original = sources.requests.post
    sources.requests.post = fake_post
    try:
        text, sections = sources._extract_pdf(b"%PDF-1.4 fake", cfg=cfg)
    finally:
        sources.requests.post = original
    assert calls == ["http://grobid.test/api/processFulltextDocument"]
    assert "DATA AVAILABILITY" in text and "deposited in Zenodo" in text
    assert "Quantum computing holds the promise" in text
    assert [n for _, n in sections] == ["INTRODUCTION", "DATA AVAILABILITY"], \
        "section headings and their offsets must survive extraction"
    off = dict((n, o) for o, n in sections)
    assert text[off["DATA AVAILABILITY"]:].startswith("DATA AVAILABILITY")


def test_grobid_failure_falls_back_rather_than_losing_the_paper():
    """A service that is down, slow or returns junk must not cost us the text."""
    cfg = Config(contact_email="a@b.edu", grobid_url="http://grobid.test")

    def boom(url, **kw):
        raise sources.requests.RequestException("connection refused")

    original = sources.requests.post
    sources.requests.post = boom
    try:
        assert sources._extract_pdf_grobid(cfg, b"%PDF") is None
    finally:
        sources.requests.post = original


def test_no_grobid_configured_means_no_network_call():
    cfg = Config(contact_email="a@b.edu")          # grobid_url empty
    def boom(url, **kw):
        raise AssertionError("must not call GROBID when it is not configured")
    original = sources.requests.post
    sources.requests.post = boom
    try:
        assert sources._extract_pdf_grobid(cfg, b"%PDF") is None
    finally:
        sources.requests.post = original


def test_pdf_extraction_falls_back_when_pymupdf_cannot_parse():
    """A corrupt PDF must reach pdfplumber rather than raising."""
    assert isinstance(sources._extract_pdf_fast(b"not actually a pdf at all"), str)
    text, sections = sources._extract_pdf(b"not actually a pdf at all")
    assert isinstance(text, str) and sections == []


# ---------------------------------------------------------------------------
# Section structure through to evidence.locator.
#
# README listed `evidence.locator` as permanently empty for PDF sources because
# pdfplumber knows nothing about document structure. GROBID does, so a quote can
# be reported against the section it demonstrably came from. `section` is what
# the model claims; `locator` is derived by looking — the same rule as `Actual`.
# ---------------------------------------------------------------------------
def test_passages_are_tagged_with_their_section():
    text = ("INTRODUCTION\n" + "We study the device. " * 40 +
            "\n\nDATA AVAILABILITY\nThe dataset is deposited in Zenodo. " * 3)
    secs = [(0, "INTRODUCTION"), (text.index("DATA AVAILABILITY"), "DATA AVAILABILITY")]
    out = select_passages(text, sections=secs)
    assert out, "expected passages"
    assert any(p["section"] == "DATA AVAILABILITY" for p in out)


def test_passages_have_an_empty_section_without_structure():
    out = select_passages("the dataset is deposited in Zenodo. " * 40)
    assert out and all(p["section"] == "" for p in out), \
        "a parser with no structure must not invent one"


def test_locator_is_derived_from_where_the_quote_was_found():
    from dap.classify import locate_evidence
    passages = [{"text": "We study the device and record a dataset.", "section": "METHODS"},
                {"text": "The dataset is deposited in Zenodo.", "section": "DATA AVAILABILITY"}]
    ev = [sources_evidence(1, "deposited in Zenodo"),
          sources_evidence(2, "record a dataset")]
    locate_evidence(ev, passages)
    assert ev[0].locator == "section: DATA AVAILABILITY"
    assert ev[1].locator == "section: METHODS"


def test_locator_stays_empty_when_the_parser_had_no_structure():
    from dap.classify import locate_evidence
    ev = [sources_evidence(1, "deposited in Zenodo")]
    locate_evidence(ev, [{"text": "The dataset is deposited in Zenodo.", "section": ""}])
    assert ev[0].locator == "", "no structure means no locator, not a guessed one"


def sources_evidence(i, quote):
    from dap.schema import Evidence
    return Evidence(id=i, quote=quote)


def test_the_model_cannot_fake_a_locator():
    """A model-supplied locator is overwritten by the derived one."""
    from dap.classify import locate_evidence
    from dap.schema import Evidence
    ev = [Evidence(id=1, quote="deposited in Zenodo", locator="section: CONCLUSION")]
    locate_evidence(ev, [{"text": "The dataset is deposited in Zenodo.",
                          "section": "DATA AVAILABILITY"}])
    assert ev[0].locator == "section: DATA AVAILABILITY"


# ---------------------------------------------------------------------------
# Regression: das_source was hard-coded to crossref_assertion whenever a DAS was
# present. Found on the recent venue-frame shakedown: all 6 statements were read
# from full text (none of those papers had a Crossref DAS), yet all 6 were
# recorded as crossref_assertion.
# ---------------------------------------------------------------------------
def _das_payload():
    p = copy.deepcopy(good_payload()); p["das_present"] = True
    return p


def test_das_read_from_full_text_is_labelled_full_text():
    rec = C._assemble(_das_payload(), openalex_id="W1", doi="", title="t", venue="v",
                      year=2024, arm=Arm.QUANTUM_COMPUTING, model="m",
                      retrieval=Retrieval.FULL_TEXT, text_source=TextSource.ARXIV,
                      text_url="", usage=C.Usage(model="m"), crossref_das=False)
    assert rec.das_source == "full_text"


def test_das_deposited_to_crossref_is_labelled_crossref_assertion():
    rec = C._assemble(_das_payload(), openalex_id="W1", doi="", title="t", venue="v",
                      year=2024, arm=Arm.QUANTUM_COMPUTING, model="m",
                      retrieval=Retrieval.FULL_TEXT, text_source=TextSource.ARXIV,
                      text_url="", usage=C.Usage(model="m"), crossref_das=True)
    assert rec.das_source == "crossref_assertion"


def test_no_das_is_labelled_none_regardless_of_crossref():
    p = copy.deepcopy(good_payload()); p["das_present"] = False
    rec = C._assemble(p, openalex_id="W1", doi="", title="t", venue="v", year=2024,
                      arm=Arm.QUANTUM_COMPUTING, model="m", retrieval=Retrieval.FULL_TEXT,
                      text_source=TextSource.ARXIV, text_url="", usage=C.Usage(model="m"),
                      crossref_das=True)
    assert rec.das_source == "none"


def test_api_path_passes_the_crossref_flag_through():
    """The call site, not just _assemble. Asserting full_text here would prove
    nothing — that is also the default when a call site forgets the flag — so the
    input carries a Crossref statement and the record must say crossref_assertion."""
    client = FakeAnthropic([FakeMessage([FakeBlock(_das_payload())])])
    rec, _ = classify(
        client, model="fake-model", title="T", venue="V", year=2024,
        arm=Arm.QUANTUM_COMPUTING, openalex_id="W1", doi="10.1/x",
        crossref_summary={"das_texts": ["The datasets generated during the current study "
                                        "are available in the Zenodo repository"],
                          "dataset_ids": [], "confirmed_datasets": []},
        passages=[], retrieval=Retrieval.METADATA_ONLY,
        text_source=TextSource.NONE, text_url="")
    assert rec.das_source == "crossref_assertion"


# ---------------------------------------------------------------------------
# Regression: a catalogue page names the PDF it describes.
#
# OSTI, IRIS/Pure and EPrints landing pages are long enough and carry the
# paper's title, so they pass looks_like_article and were stored as the article.
# Measured on the quantum frame: 129 papers coded from a catalogue record, with
# review reasons reading "OSTI.gov citation metadata rather than article text".
# Both pages carry the PDF in a citation_pdf_url meta tag.
# ---------------------------------------------------------------------------
CATALOGUE_HTML = (
    "<html><head>"
    "<meta name='citation_pdf_url' content='https://osti.example/purl/1833031'>"
    "</head><body>"
    "<p>Ytterbium ion trap quantum computing: the current state-of-the-art</p>"
    "<p>Journal Article. OSTI ID: 1833031. Similar Records. Export citation.</p>"
    "</body></html>"
).encode()


def _cascade_over(locations, routes, *, doi="", title="Transmon readout fidelity"):
    """Drive the cascade over `locations` only. best_oa_location is read for
    pdf_url/url/oa_url and never for landing_page_url, so a catalogue page has
    to arrive through the locations list."""
    work = {"best_oa_location": {}, "locations": locations, "ids": {}}
    original = sources._get
    sources._get = fake_get_factory(routes) if isinstance(routes, dict) else routes
    try:
        return sources.retrieve_full_text(CFG, work=work, doi=doi, title=title,
                                          facts=CrossrefFacts(found=True))
    finally:
        sources._get = original


def test_a_catalogue_page_is_followed_to_the_pdf_it_names():
    ft = _cascade_over(
        [{"landing_page_url": "https://osti.example/biblio/1"}],
        {"osti.example/biblio": FakeResponse(headers={"Content-Type": "text/html"},
                                             content=CATALOGUE_HTML),
         "osti.example/purl": FakeResponse(headers={"Content-Type": "text/html"},
                                           content=ARTICLE_HTML)})
    assert ft.ok, "the article behind the catalogue record must be recovered"
    assert ft.url == "https://osti.example/purl/1833031", \
        "the stored URL must be the document actually parsed, not the catalogue"


def test_following_a_catalogue_page_is_not_recursive():
    """A catalogue page whose citation_pdf_url points at another catalogue page
    must stop, not walk a chain of them."""
    loop = ("<html><head>"
            "<meta name='citation_pdf_url' content='https://loop.example/b'>"
            "</head><body>Export citation. Similar Records.</body></html>").encode()
    calls = []

    def counting(cfg, url, **kw):
        calls.append(url)
        return FakeResponse(headers={"Content-Type": "text/html"}, content=loop)

    ft = _cascade_over([{"landing_page_url": "https://loop.example/a"}], counting)
    assert not ft.ok
    assert len(calls) == 2, f"expected exactly one follow, got {calls}"


def test_a_page_without_the_meta_tag_is_still_not_an_article():
    ft = _cascade_over([{"landing_page_url": "https://doi.org/10.1/x"}],
                       {"doi.org": FakeResponse(headers={"Content-Type": "text/html"},
                                                content=STUB_HTML)})
    assert not ft.ok
    assert [a["outcome"] for a in ft.attempts] == ["not_article"]


# ---------------------------------------------------------------------------
# Regression: DOAJ indexes open-access journals, it does not host them.
# 58 attempts across the frame, none yielding text, many collecting a 403.
# ---------------------------------------------------------------------------
def test_an_index_only_host_is_skipped_without_a_request():
    calls = []

    def counting(cfg, url, **kw):
        calls.append(url)
        return FakeResponse(headers={"Content-Type": "text/html"}, content=ARTICLE_HTML)

    ft = _cascade_over([{"landing_page_url": "https://doaj.org/article/abc"},
                        {"pdf_url": "https://repo.example/a.pdf"}], counting)
    assert "https://doaj.org/article/abc" not in calls, "DOAJ must not be fetched"
    assert ft.ok and ft.url == "https://repo.example/a.pdf"
    assert any(a["outcome"] == "skipped: index-only host" for a in ft.attempts), \
        "the skip must be recorded, so a fallback copy stays explicable"


def test_a_pubmed_abstract_page_is_never_fetched_but_pmc_still_is():
    """`pubmed.ncbi.nlm.nih.gov/<pmid>/` is always an abstract page; the full
    text lives on `pmc.ncbi.nlm.nih.gov`, a different host. 95 rows had a PubMed
    abstract stored as their text, 21 of them recorded as `full_text`, and three
    papers were coded off one before this guard went in."""
    calls = []

    def counting(cfg, url, **kw):
        calls.append(url)
        return FakeResponse(headers={"Content-Type": "text/html"}, content=ARTICLE_HTML)

    ft = _cascade_over([{"landing_page_url": "https://pubmed.ncbi.nlm.nih.gov/40123456/"},
                        {"pdf_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC123/pdf"}],
                       counting)
    assert not any("pubmed.ncbi" in u for u in calls), \
        "an abstract page must not even be requested"
    assert ft.ok and "pmc.ncbi" in ft.url, "PMC is a different host and must still work"
    assert any(a["outcome"] == "skipped: index-only host" for a in ft.attempts)


# ---------------------------------------------------------------------------
# Regression, forward-looking: searching arXiv by title is how v1 attached the
# WRONG paper's full text. It searched with max_results=1 and no author or year
# check, so a near-title match silently became the paper — a wrong
# classification, which nothing downstream can detect.
#
# The search is allowed back only as a last resort and only with the checks v1
# lacked. These tests exist to keep those checks in place.
# ---------------------------------------------------------------------------
def _arxiv_feed(*entries):
    body = "".join(
        f"<entry><id>http://arxiv.org/abs/{i}</id><title>{t}</title>"
        f"<published>{y}-01-01T00:00:00Z</published>"
        + "".join(f"<author><name>{a}</name></author>" for a in auth)
        + "</entry>"
        for i, t, y, auth in entries)
    return f"<feed xmlns='http://www.w3.org/2005/Atom'>{body}</feed>".encode()


REAL = ("2101.11390", "Quantum advantage in learning from experiments", 2021,
        ["Hsin-Yuan Huang", "Michael Broughton"])


def _search(routes, **kw):
    original = sources._get
    sources._get = fake_get_factory(routes)
    try:
        return sources.arxiv_search(CFG, **kw)
    finally:
        sources._get = original


def test_arxiv_search_accepts_a_title_and_author_match():
    got, why = _search({"arxiv.org": FakeResponse(content=_arxiv_feed(REAL))},
                       title="Quantum advantage in learning from experiments",
                       authors=["huang", "broughton"], year=2022)
    assert got == "2101.11390"
    assert "huang" in why or "broughton" in why, f"the reason must be auditable: {why}"


def test_arxiv_search_refuses_a_near_title_by_different_authors():
    """The v1 failure, exactly: a paper on the same topic with a similar title
    is not this paper. Without the author check this returns an id and the wrong
    text is classified as though it were the paper."""
    other = ("2202.00001", "Quantum advantage in learning from experiments",
             2022, ["Someone Else", "Another Person"])
    got, why = _search({"arxiv.org": FakeResponse(content=_arxiv_feed(other))},
                       title="Quantum advantage in learning from experiments",
                       authors=["huang", "broughton"], year=2022)
    assert got is None
    assert "author" in why


def test_arxiv_search_refuses_a_different_title_by_the_same_authors():
    """Prolific groups publish repeatedly. Sharing an author is not sharing a
    paper."""
    other = ("2202.00002", "Provably efficient machine learning for quantum "
             "many-body problems", 2022, ["Hsin-Yuan Huang"])
    got, why = _search({"arxiv.org": FakeResponse(content=_arxiv_feed(other))},
                       title="Quantum advantage in learning from experiments",
                       authors=["huang"], year=2022)
    assert got is None
    assert "overlap" in why


def test_arxiv_search_refuses_a_match_too_far_from_the_publication_year():
    old = ("0901.00003", "Quantum advantage in learning from experiments",
           2009, ["Hsin-Yuan Huang"])
    got, why = _search({"arxiv.org": FakeResponse(content=_arxiv_feed(old))},
                       title="Quantum advantage in learning from experiments",
                       authors=["huang"], year=2022)
    assert got is None
    assert "year" in why


def test_arxiv_search_refuses_when_we_have_no_authors_to_check():
    """Unverifiable is not a pass. With no authors on record the only available
    check is the title, which is exactly what went wrong in v1."""
    got, why = _search({"arxiv.org": FakeResponse(content=_arxiv_feed(REAL))},
                       title="Quantum advantage in learning from experiments",
                       authors=[], year=2022)
    assert got is None and "authors" in why


def test_arxiv_search_picks_the_matching_entry_out_of_several():
    decoy = ("2202.00009", "Quantum advantage in learning from noise", 2022,
             ["Someone Else"])
    got, _ = _search({"arxiv.org": FakeResponse(content=_arxiv_feed(decoy, REAL))},
                     title="Quantum advantage in learning from experiments",
                     authors=["huang"], year=2022)
    assert got == "2101.11390"


def test_the_cascade_only_searches_arxiv_after_everything_else_failed():
    """A search is a guess with checks on it. It must never pre-empt a location
    that would have given the version of record."""
    calls = []

    def counting(cfg, url, **kw):
        calls.append(url)
        if "export.arxiv.org" in url:
            raise AssertionError("searched arXiv while a real location worked")
        return FakeResponse(headers={"Content-Type": "text/html"}, content=ARTICLE_HTML)

    ft = _cascade_over([{"pdf_url": "https://repo.example/a.pdf"}], counting)
    assert ft.ok and ft.url == "https://repo.example/a.pdf"


def test_the_cascade_falls_back_to_a_verified_arxiv_search():
    body = ("<html><body><article>" +
            "We show a quantum advantage in learning from experiments on the "
            "device, and deposit the dataset. " * 60 +
            "</article></body></html>").encode()

    def routed(cfg, url, **kw):
        if "export.arxiv.org" in url:
            return FakeResponse(content=_arxiv_feed(REAL))
        if "arxiv.org/pdf/2101.11390" in url:
            return FakeResponse(headers={"Content-Type": "text/html"}, content=body)
        raise sources.PermanentError("HTTP 403")

    facts = CrossrefFacts(found=True, authors=["huang", "broughton"],
                          published_year=2022)
    original = sources._get
    sources._get = routed
    try:
        ft = sources.retrieve_full_text(
            CFG, work={"best_oa_location": {"pdf_url": "https://paywall.example/a.pdf"},
                       "locations": [], "ids": {}},
            doi="", title="Quantum advantage in learning from experiments",
            facts=facts)
    finally:
        sources._get = original
    assert ft.ok and ft.url == "https://arxiv.org/pdf/2101.11390"
    assert any("search:" in a["outcome"] for a in ft.attempts), \
        "the search and what verified it must be in the audit trail"


# ---------------------------------------------------------------------------
# A mirror that times out no longer holds the fallback back (Diana, 4 Oct 2026).
#
# On 25 Sep 18 frame papers sat parked behind institutional mirrors that kept
# timing out (orbilu.uni.lu, tudasportal.uni-nke.hu, hdl.handle.net) while the
# publisher had already answered 403 for good; 13 were still parked on 4 Oct.
# Waiting protects a version of record that might come back, and a repository
# copy is not one. The publisher, the Unpaywall lookup and arXiv are still
# waited for, and a mirror's timeout still means a retry if nothing is found.
# ---------------------------------------------------------------------------
MIRROR_DOI = "10.1103/prxquantum.4.020353"
PUBLISHER_THEN_MIRROR = {
    "best_oa_location": {"landing_page_url": f"https://doi.org/{MIRROR_DOI}",
                         "pdf_url": "https://journals.publisher.example/pdf/1",
                         "source": {"type": "journal"}},
    "locations": [
        {"landing_page_url": f"https://doi.org/{MIRROR_DOI}",
         "pdf_url": "https://journals.publisher.example/pdf/1",
         "source": {"display_name": "PRX Quantum"}},
        {"landing_page_url": "https://repo.example/handle/10993/59088",
         "pdf_url": "https://mirror.example/bitstream/1.pdf",
         "source": {"display_name": "ORBilu"}}],
    "ids": {}}
LEARNING_BODY = ("<html><body><article>" +
                 "We show a quantum advantage in learning from experiments on the "
                 "device, and deposit the dataset. " * 60 +
                 "</article></body></html>").encode()
LEARNING_PAGE = FakeResponse(headers={"Content-Type": "text/html"}, content=LEARNING_BODY)
ARXIV_FINDS_IT = {"export.arxiv.org": FakeResponse(content=_arxiv_feed(REAL)),
                  "arxiv.org/pdf/2101.11390": LEARNING_PAGE}
REFUSED = {"publisher.example": sources.PermanentError("HTTP 403"),
           "doi.org": sources.PermanentError("HTTP 403"),
           "api.unpaywall.org": sources.PermanentError("HTTP 404")}
TIMES_OUT = sources.TransientError("exhausted 4 attempts: ConnectTimeout")


def _mirror_cascade(routes, work=PUBLISHER_THEN_MIRROR):
    # Routes match by substring, first key first: a test's own routes go ahead
    # of the refusals, so `doi.org/<doi>` is seen before the bare `doi.org`.
    routes = {**routes, **{k: v for k, v in REFUSED.items() if k not in routes}}
    facts = CrossrefFacts(found=True, authors=["huang", "broughton"],
                          published_year=2021)
    original = sources._get
    sources._get = fake_get_factory(routes)
    try:
        return sources.retrieve_full_text(
            CFG, work=work, doi=MIRROR_DOI,
            title="Quantum advantage in learning from experiments", facts=facts)
    finally:
        sources._get = original


def test_a_mirror_that_times_out_falls_back_to_arxiv():
    ft = _mirror_cascade({"mirror.example": TIMES_OUT, "repo.example": TIMES_OUT,
                          **ARXIV_FINDS_IT})
    assert ft.ok and ft.source is TextSource.ARXIV
    assert ("https://mirror.example/bitstream/1.pdf",
            "transient: exhausted 4 attempts: ConnectTimeout") in \
        [(a["url"], a["outcome"]) for a in ft.attempts], \
        "the mirror's timeout must stay in the audit trail"


def test_a_later_repository_copy_is_taken_after_a_mirror_times_out():
    """W4386830972: the bitstream timed out and the same repository's handle
    page answered, and the paper was parked anyway."""
    ft = _mirror_cascade({"mirror.example": TIMES_OUT, "repo.example": LEARNING_PAGE})
    assert ft.ok and ft.url == "https://repo.example/handle/10993/59088"


def test_a_repository_doi_is_a_mirror_not_the_publisher():
    """ETH Zurich mints its own DOIs; only the paper's own DOI is the publisher's."""
    work = dict(PUBLISHER_THEN_MIRROR, locations=[
        {"landing_page_url": "https://doi.org/10.3929/ethz-b-000607273"}])
    ft = _mirror_cascade({"doi.org/10.3929": TIMES_OUT, **ARXIV_FINDS_IT},
                         work=work)
    assert ft.ok and ft.source is TextSource.ARXIV


def test_the_papers_own_doi_timing_out_is_still_waited_for():
    with pytest.raises(sources.TransientError, match=MIRROR_DOI):
        _mirror_cascade({f"doi.org/{MIRROR_DOI}": TIMES_OUT,
                         "mirror.example": LEARNING_PAGE, **ARXIV_FINDS_IT})


def test_the_publishers_host_timing_out_is_still_waited_for():
    with pytest.raises(sources.TransientError, match="publisher.example"):
        _mirror_cascade({"publisher.example": TIMES_OUT,
                         "mirror.example": LEARNING_PAGE, **ARXIV_FINDS_IT})


def test_an_unpaywall_lookup_timing_out_is_still_waited_for():
    """The lookup is not a copy: what it would have returned is unknown."""
    with pytest.raises(sources.TransientError, match="api.unpaywall.org"):
        _mirror_cascade({"mirror.example": TIMES_OUT, "repo.example": TIMES_OUT,
                         "api.unpaywall.org": TIMES_OUT,
                         "export.arxiv.org": AssertionError("searched arXiv while "
                                                            "Unpaywall was down")})


def test_a_mirror_timeout_with_nothing_else_found_is_still_retried():
    """Releasing the fallback must not turn a retry into a text-less record:
    the mirror may hold the only readable copy."""
    with pytest.raises(sources.TransientError, match="mirror.example"):
        _mirror_cascade({"mirror.example": TIMES_OUT, "repo.example": TIMES_OUT,
                         "export.arxiv.org": FakeResponse(content=_arxiv_feed())})


def test_a_repository_typed_preferred_location_is_a_mirror():
    work = {"best_oa_location": {"pdf_url": "https://mirror.example/bitstream/1.pdf",
                                 "landing_page_url": "https://repo.example/handle/1",
                                 "source": {"type": "repository"}},
            "locations": [], "ids": {}}
    ft = _mirror_cascade({"mirror.example": TIMES_OUT, **ARXIV_FINDS_IT},
                         work=work)
    assert ft.ok and ft.source is TextSource.ARXIV


def test_crossref_author_family_names_are_collected():
    payload = {"message": {"type": "journal-article", "container-title": ["X"],
                           "published": {"date-parts": [[2024]]},
                           "author": [{"family": "Huang", "given": "Hsin-Yuan"},
                                      {"family": "Broughton"}, {"given": "NoFamily"}]}}
    f = _with_get({"api.crossref.org": FakeResponse(payload)},
                  lambda: sources.crossref_facts(CFG, "10.1/x"))
    assert f.authors == ["huang", "broughton"]


# ---------------------------------------------------------------------------
# A hand-fetched PDF. Rule 8 forbids bypassing a publisher's bot wall; it does
# not forbid a person with a subscription downloading the paper. For ~640
# papers in this corpus — IOP and Springer above all — that is the only lawful
# route to the version of record.
# ---------------------------------------------------------------------------
def _pdf_bytes(text):
    """A one-page PDF carrying `text`, built with the parser we actually use."""
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(20, 20, 580, 800), text, fontsize=8)
    out = doc.tobytes()
    doc.close()
    return out


ARTICLE_WORDS = ("We measured transmon readout fidelity across the device and "
                 "deposited the dataset in Zenodo. " * 70)


def test_a_hand_fetched_pdf_is_used_and_wins_over_the_network(tmp_path):
    (tmp_path / "10.1088_x.y.pdf").write_bytes(_pdf_bytes(ARTICLE_WORDS))
    cfg = Config(contact_email="researcher@institution.edu", pdf_dir=str(tmp_path))

    def explode(c, url, **kw):
        raise AssertionError(f"went to the network with a local copy in hand: {url}")

    original = sources._get
    sources._get = explode
    try:
        ft = sources.retrieve_full_text(
            cfg, work={"best_oa_location": {"pdf_url": "https://paywall.example/a.pdf"},
                       "locations": [], "ids": {}},
            doi="10.1088/x.y", title="Transmon readout fidelity",
            facts=CrossrefFacts(found=True))
    finally:
        sources._get = original
    assert ft.ok and ft.source is TextSource.LOCAL_PDF
    assert ft.url.startswith("file://")
    assert "transmon" in ft.text.lower()


def test_a_local_pdf_can_be_named_by_openalex_id(tmp_path):
    (tmp_path / "W123.pdf").write_bytes(_pdf_bytes(ARTICLE_WORDS))
    cfg = Config(contact_email="researcher@institution.edu", pdf_dir=str(tmp_path))
    ft = sources.retrieve_full_text(
        cfg, work={"best_oa_location": {}, "locations": [],
                   "ids": {"openalex": "https://openalex.org/W123"}},
        doi="", title="Transmon readout fidelity", facts=CrossrefFacts(found=True))
    assert ft.ok and ft.source is TextSource.LOCAL_PDF


def test_a_local_pdf_that_is_not_the_article_is_refused(tmp_path):
    """A scanned page with no text layer, or the wrong paper filed under this
    name, is not usable just because it is local."""
    (tmp_path / "10.1088_x.y.pdf").write_bytes(_pdf_bytes("Download citation. "
                                                          "Google Scholar. Access options."))
    cfg = Config(contact_email="researcher@institution.edu", pdf_dir=str(tmp_path))
    ft = sources.retrieve_full_text(
        cfg, work={"best_oa_location": {}, "locations": [], "ids": {}},
        doi="10.1088/x.y", title="Transmon readout fidelity",
        facts=CrossrefFacts(found=True))
    assert not ft.ok, "a local file still has to look like the article"


def test_no_pdf_dir_configured_changes_nothing(tmp_path):
    cfg = Config(contact_email="researcher@institution.edu")
    original = sources._get
    sources._get = fake_get_factory({"repo.example": FakeResponse(
        headers={"Content-Type": "text/html"}, content=ARTICLE_HTML)})
    try:
        ft = sources.retrieve_full_text(
            cfg, work={"best_oa_location": {}, "locations": [{"pdf_url": "https://repo.example/a.pdf"}],
                       "ids": {}},
            doi="10.1088/x.y", title="Transmon readout fidelity",
            facts=CrossrefFacts(found=True))
    finally:
        sources._get = original
    assert ft.ok and ft.source is TextSource.OPENALEX_OA


# ---------------------------------------------------------------------------
# Regression: Jaccard treated a dropped subtitle as a different paper.
#
# The Nature Electronics version of "Scaling silicon-based quantum computing
# using CMOS technology" dropped the preprint's ": State of the art, challenges
# and perspectives". Jaccard scored that 0.53 against its own preprint - two
# shared authors, one year apart - and the cascade refused it. Containment
# recognises it; the distinctive-word floor keeps containment from matching a
# short generic title inside a longer one.
# ---------------------------------------------------------------------------
LONG_TITLE = ("Scaling silicon-based quantum computing using CMOS technology: "
              "State of the art, challenges and perspectives")
SHORT_TITLE = "Scaling silicon-based quantum computing using CMOS technology"


def test_a_dropped_subtitle_is_still_the_same_paper():
    from dap.sources import titles_agree, _title_overlap
    assert _title_overlap(SHORT_TITLE, LONG_TITLE) < 0.85, \
        "the case only arises because plain overlap fails"
    agreed, why = titles_agree(SHORT_TITLE, LONG_TITLE)
    assert agreed is True
    assert "contained" in why, "the reason must say which rule admitted it"


def test_a_generic_title_inside_a_longer_one_is_still_refused():
    """'Quantum Computing' is contained in almost every arXiv quantum paper,
    and a paper with exactly that title is in this corpus."""
    from dap.sources import titles_agree
    agreed, why = titles_agree(
        "Quantum Computing", "Quantum Computing for Finance: Overview and Prospects")
    assert agreed is False
    assert "too generic" in why


def test_containment_does_not_admit_a_merely_related_paper():
    from dap.sources import titles_agree
    agreed, _ = titles_agree(
        "Scaling silicon-based quantum computing using CMOS technology",
        "Scaling superconducting quantum processors with cryogenic control electronics")
    assert agreed is False


def test_the_author_check_still_decides_an_identically_titled_paper():
    """Containment cannot weaken this: the title route only gets you as far as
    the author check, which is what caught Saffman's identically-titled paper
    by a different group."""
    from dap.sources import titles_agree
    agreed, _ = titles_agree("Quantum computing with neutral atoms",
                             "Quantum computing with neutral atoms")
    assert agreed is True, "titles do agree - it is the authors that must not"


# ---------------------------------------------------------------------------
# A host that serves both a catalogue record and the real article.
# ---------------------------------------------------------------------------
def test_the_osti_record_path_is_skipped_but_the_text_path_is_not():
    from dap.sources import _is_catalogue_url
    assert _is_catalogue_url("https://www.osti.gov/biblio/1777671") is True
    assert _is_catalogue_url("https://www.osti.gov/servlets/purl/2427971") is False, \
        "37 papers' full text came from this path; blocking it discards them"
    assert _is_catalogue_url("https://example.org/biblio/1") is False, \
        "the rule is host-scoped, not a bare path match"


def test_a_denied_host_is_never_fetched():
    """DAP_DENY_HOSTS is the blunt instrument, for when a host must go entirely."""
    cfg = Config(contact_email="r@i.edu", deny_hosts=frozenset({"osti.gov"}))
    work = {"best_oa_location": {"pdf_url": "https://www.osti.gov/servlets/purl/1"},
            "open_access": {}, "locations": [], "ids": {}}
    orig = sources._get
    sources._get = fake_get_factory({"api.unpaywall.org": sources.PermanentError("HTTP 404")})
    try:
        ft = sources.retrieve_full_text(cfg, work=work, doi="10.1/x", title="A paper",
                                        facts=sources.CrossrefFacts(found=True))
    finally:
        sources._get = orig
    assert ft.ok is False
    assert any(a["outcome"] == "skipped: host denied by configuration"
               for a in ft.attempts), "the skip must be recorded, not silent"


PURE_RECORD = (
    "Landauer Versus Nernst: What is the True Cost of Cooling a Quantum System?\n"
    "Philip Taranto, Faraj Bakhshinezhad, Marcus Huber  Theoretical Physics Group\n"
    "Research output : Contribution to journal - Article - peer-review\n"
    "Original language English  Journal PRX QUANTUM  Publication status Published\n"
    "Access to Document  Other files and links  Cite this APA Author BIBTEX Harvard\n"
    "@article{731af3d1, title = \"Landauer Versus Nernst\", author = \"Philip Taranto\"}\n"
    + "We study the thermodynamic cost of cooling a quantum system to absolute zero. " * 40
)


def test_a_research_portal_record_is_not_article_text():
    """A Pure portal page prints the record in six citation styles and no body.
    It carries the title and an abstract, so nothing else catches it."""
    assert looks_like_article(
        PURE_RECORD, "Landauer Versus Nernst: What is the True Cost of Cooling a Quantum System?"
    ) is False


def test_a_paper_that_says_research_output_once_is_still_accepted():
    """The threshold protects a bibliometrics paper that uses the phrase."""
    text = ("We count the research output of each department by year. ") + PROSE
    assert looks_like_article(text, "") is True


def test_a_paper_that_tells_you_how_to_cite_it_is_not_a_record_page():
    """Found as a false positive: a Zenodo-deposited framework paper carrying a
    'How to Cite this Work' section and a BibTeX block for self-citation. That
    is article content, and the kind of article this study most wants to see."""
    text = ("C. How to Cite this Work. To reference this framework in academic "
            "research, use the following structured BibTeX configuration: "
            "@article{vlachos2026universal, title={Universal Lock-Free Spatial "
            "Framework}} ") + PROSE
    assert looks_like_article(text, "") is True


def test_a_paper_whose_only_passage_is_its_reference_list():
    """Found in a live batch: a two-column PDF interleaves the bibliography with
    the neighbouring column, which pads the word count with body text and pulls
    the density below the per-passage bar. The one surviving passage was a
    reference list, handed to the classifier as the paper's whole evidence."""
    from dap.classify import only_bibliography
    refs = [{"keywords": ["simulation"], "text": BIBLIOGRAPHY, "offset": 0, "section": ""}]
    assert only_bibliography(refs) is True
    prose = [{"keywords": ["dataset"], "text": PROSE, "offset": 0, "section": ""}]
    assert only_bibliography(prose) is False
    assert only_bibliography([]) is False, "no passages is a different condition"


def test_an_availability_heading_is_never_treated_as_bibliography():
    """The back matter of a two-column paper renders beside the reference column,
    so the one passage that answers the study's question looks like references."""
    from dap.classify import only_bibliography
    das = [{"keywords": ["data availability"], "text": BIBLIOGRAPHY,
            "offset": 0, "section": ""}]
    assert only_bibliography(das) is False


def test_a_pure_portal_in_danish_is_caught_too():
    """Pure is deployed in the local language; the English markers miss it."""
    text = ("Protected Hybrid Superconducting Qubit in an Array of Gate-Tunable "
            "Josephson Interferometers\n"
            "Originalsprog Engelsk  Tidsskrift PRX Quantum  Status Udgivet\n"
            "Adgang til dokumentet  Citationsformater APA Standard Harvard\n"
            + "We use an effective spin model to describe the system. " * 40)
    assert looks_like_article(
        text, "Protected Hybrid Superconducting Qubit in an Array of Gate-Tunable "
              "Josephson Interferometers") is False


def test_a_hal_record_page_is_caught_too():
    """HAL serves the record in French; two of these reached the classifier."""
    text = ("Article Dans Une Revue  PRX Quantum  Annee : 2022\n"
            "Experimentally Accessible Scheme for a Fractional Chern Insulator\n"
            "Sebastian Weber (1), Nicolai Lang (1)  Afficher plus de details\n"
            "Sebastian Weber  Fonction : Auteur  IdHAL : sebastian-weber\n"
            + "We propose a scheme for realising a fractional Chern insulator. " * 40)
    assert looks_like_article(
        text, "Experimentally Accessible Scheme for a Fractional Chern Insulator") is False


def test_a_transport_error_does_not_make_a_path_broken():
    """`access_unresolvable` means promised and broken. A connect timeout, a DNS
    failure or a 503 says the check did not complete -- an institutional GitLab
    behind a firewall and a decommissioned one produce the same error. 19
    verification rows across the three databases carry `error` and 8 have no
    HTTP status at all; promoting on those inflates the broken-link rate."""
    for actual in (Actual.NOT_FOUND, Actual.RESOLVES_NO_DATA):
        case, note = verify.derive_case(Case.OPENLY_ACCESSIBLE, actual)
        assert case is Case.ACCESS_UNRESOLVABLE, f"{actual} must still promote"
        assert actual.value in note

    case, note = verify.derive_case(Case.OPENLY_ACCESSIBLE, Actual.ERROR)
    assert case is Case.OPENLY_ACCESSIBLE, "an error must leave the case alone"
    assert note == "", "and must not claim a promotion in the note"

    # A path that resolves but gates access is a different matter: the check
    # completed and the answer was no.
    for actual in (Actual.PAYWALLED, Actual.AUTH_REQUIRED):
        case, _ = verify.derive_case(Case.OPENLY_ACCESSIBLE, actual)
        assert case is Case.ACCESS_UNRESOLVABLE
