"""
Regression tests written against the ACTUAL defects found in the v1 output.

Each test names the v1 symptom it prevents. If one of these ever fails, the
drift is back.

    python -m pytest tests/ -q          (or: python tests/test_guards.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import pytest
except ImportError:                                  # minimal stand-in so the
    import re as _re                                 # suite runs without pytest
    from contextlib import contextmanager

    class _P:
        @staticmethod
        @contextmanager
        def raises(exc, match=None):
            try:
                yield
            except exc as e:
                if match and not _re.search(match, str(e)):
                    raise AssertionError(
                        f"raised {exc.__name__} but message {str(e)!r} "
                        f"does not match {match!r}") from None
            else:
                raise AssertionError(f"{exc.__name__} was not raised")

        @staticmethod
        def main(_argv):
            # Introspect the module being RUN, not this shim's module — otherwise
            # importing the shim elsewhere silently re-runs this file's tests.
            g = dict(vars(sys.modules["__main__"]))
            tests = sorted(k for k in g if k.startswith("test_")
                           and callable(g[k]))
            failed = []
            for name in tests:
                try:
                    g[name]()
                    print(f"  PASS  {name}")
                except Exception as exc:
                    failed.append((name, exc))
                    print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
            print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
            return 1 if failed else 0

    pytest = _P()

from dap.schema import (Actual, AxisRecord, Case, Confidence, DataType,
                        Declared, Evidence, Granularity, Judgement, PaperRecord,
                        PaperType, Pinned, Restriction, RepoType, SchemaError,
                        Arm, Retrieval, TextSource)
from dap.classify import select_passages
from dap.verify import derive_case, classify_pinning
from dap.sources import looks_like_article, repo_type_for


def J(v, conf=Confidence.HIGH, ids=(1,)):
    return Judgement(v, conf, "because the evidence says so", list(ids))


def axis(case=Case.ACCESS_NOT_PROVIDED, *, identifiers=(), declared=Declared.NONE_STATED,
         actual=Actual.NOT_ATTEMPTED, restriction=Restriction.NONE_GIVEN):
    return AxisRecord(
        case=J(case), data_type=J(DataType.REAL), declared=J(declared),
        actual=Judgement(actual, Confidence.HIGH, "machine", []),
        granularity=J(Granularity.PROCESSED),
        restriction=restriction, identifiers=list(identifiers),
    )


def paper(**kw):
    base = dict(
        openalex_id="W1", doi="10.1/x", title="T", year=2024, venue="V",
        arm=Arm.QUANTUM_COMPUTING,
        paper_type=J(PaperType.EXPERIMENTAL),
        axis_a=axis(Case.NOT_APPLICABLE),
        axis_b=axis(Case.ACCESS_NOT_PROVIDED),
        evidence=[Evidence(1, "data are available on request")],
        das_present=True, das_source="crossref_assertion",
        needs_human_review=False, review_reason="",
        retrieval=Retrieval.FULL_TEXT, text_source=TextSource.ARXIV,
    )
    base.update(kw)
    return PaperRecord(**base)


# -- v1: `request`, `upon_request`, `available_on_request` all became values ---
def test_out_of_vocabulary_token_is_rejected():
    with pytest.raises(SchemaError, match="not a valid token"):
        Declared.coerce("upon_request")
    with pytest.raises(SchemaError):
        Declared.coerce("Public")            # no case-folding, deliberately
    assert Declared.coerce("on_request") is Declared.ON_REQUEST


# -- v1: A_case held '1' and '1.0' as distinct values -------------------------
def test_numeric_lookalikes_rejected():
    with pytest.raises(SchemaError):
        Case.coerce(2)
    with pytest.raises(SchemaError):
        Case.coerce("2.0")


# -- v1: 25 papers claiming "data available" with no link were coded Case 1 ----
def test_openly_accessible_requires_an_identifier():
    with pytest.raises(SchemaError, match="requires at least one identifier"):
        paper(axis_b=axis(Case.OPENLY_ACCESSIBLE, identifiers=())).validate()
    paper(axis_b=axis(Case.OPENLY_ACCESSIBLE,
                      identifiers=["10.5281/zenodo.123"])).validate()


# -- v1: `inaccessible` never appeared; the case was asserted, never derived ---
def test_unresolvable_cannot_be_asserted_without_failed_verification():
    with pytest.raises(SchemaError, match="derived from a failed verification"):
        paper(axis_b=axis(Case.ACCESS_UNRESOLVABLE,
                          identifiers=["https://x.test/d"],
                          actual=Actual.ACCESSIBLE)).validate()


def test_derive_case_promotes_on_broken_link():
    c, note = derive_case(Case.OPENLY_ACCESSIBLE, Actual.NOT_FOUND)
    assert c is Case.ACCESS_UNRESOLVABLE and "not_found" in note
    c, _ = derive_case(Case.OPENLY_ACCESSIBLE, Actual.ACCESSIBLE)
    assert c is Case.OPENLY_ACCESSIBLE
    # a landing page that gates access is not "openly" accessible
    c, _ = derive_case(Case.OPENLY_ACCESSIBLE, Actual.PAYWALLED)
    assert c is Case.ACCESS_UNRESOLVABLE


# -- v1: restriction reasons were free text, or absent entirely ----------------
def test_explicit_restriction_needs_a_reason():
    with pytest.raises(SchemaError, match="requires a restriction reason"):
        paper(axis_b=axis(Case.EXPLICITLY_RESTRICTED)).validate()
    paper(axis_b=axis(Case.EXPLICITLY_RESTRICTED,
                      restriction=Restriction.VENDOR_HARDWARE)).validate()


def test_on_request_must_match_declared():
    with pytest.raises(SchemaError, match="requires declared=on_request"):
        paper(axis_b=axis(Case.ON_REQUEST, declared=Declared.PUBLIC)).validate()


# -- v1: an axis coded "no data" while carrying repository fields --------------
def test_not_applicable_cannot_carry_identifiers():
    with pytest.raises(SchemaError, match="absent and present at once"):
        paper(axis_a=axis(Case.NOT_APPLICABLE,
                          identifiers=["https://zenodo.org/records/1"])).validate()


# -- v1: confidence was only ever high/medium across 1,290 judgement calls -----
def test_high_confidence_requires_evidence():
    with pytest.raises(SchemaError, match="requires at least one evidence_id"):
        Judgement(Case.OPENLY_ACCESSIBLE, Confidence.HIGH, "sure", []).validate(
            Case, field_name="axis_a.case")


def test_rationale_is_mandatory():
    with pytest.raises(SchemaError, match="rationale is required"):
        Judgement(Case.ON_REQUEST, Confidence.MEDIUM, "   ", [1]).validate(
            Case, field_name="axis_a.case")


# -- v1: 209 flagged in analysis.csv vs 195 in review_queue.csv ----------------
def test_low_confidence_forces_review():
    a = axis(Case.ACCESS_NOT_PROVIDED)
    a.case.confidence = Confidence.LOW
    with pytest.raises(SchemaError, match="must route to the review queue"):
        paper(axis_b=a, needs_human_review=False).validate()


def test_dangling_evidence_id_is_caught():
    a = axis(Case.ACCESS_NOT_PROVIDED)
    a.case.evidence_ids = [99]
    with pytest.raises(SchemaError, match=r"evidence_ids \[99\] do not exist"):
        paper(axis_b=a).validate()


def test_quote_length_is_capped():
    with pytest.raises(SchemaError, match="limit is 40"):
        Evidence(1, " ".join(["word"] * 41)).validate()


# -- v1: overlapping keyword windows dropped the second keyword's context ------
def test_overlapping_passages_merge_and_keep_all_keywords():
    text = ("Preamble. " * 40) + \
           "The data availability statement says the dataset is on zenodo. " + \
           ("Tail. " * 40)
    got = select_passages(text, window=200)
    assert len(got) == 1, "overlapping windows should merge into one span"
    kws = set(got[0]["keywords"])
    assert {"data availability", "dataset", "zenodo"} <= kws


def test_passages_returned_in_document_order():
    text = ("A " * 300) + "zenodo record here. " + ("B " * 300) + \
           "data availability statement here. " + ("C " * 300)
    got = select_passages(text, window=100)
    assert [p["offset"] for p in got] == sorted(p["offset"] for p in got)


# -- v1: a Cloudflare challenge page returning 200 was parsed as article text --
def test_bot_challenge_page_is_rejected():
    challenge = "Just a moment... Checking your browser before accessing. " * 50
    assert not looks_like_article(challenge, "Superconducting qubit coherence")


def test_wrong_paper_is_rejected_by_title_overlap():
    body = ("quantum annealing lattice gauge chromodynamics simulation " * 200)
    assert not looks_like_article(body, "Transmon readout fidelity improvements")
    matching = ("transmon readout fidelity improvements were measured " * 200)
    assert looks_like_article(matching, "Transmon readout fidelity improvements")


def test_short_stub_is_rejected():
    assert not looks_like_article("Abstract only. " * 5, "Some title")


# -- repository typing is deterministic, not left to the model ----------------
def test_repo_type_detection():
    assert repo_type_for("https://huggingface.co/datasets/x") is RepoType.HUGGINGFACE
    assert repo_type_for("10.5281/zenodo.7654321") is RepoType.ZENODO
    assert repo_type_for("https://github.com/o/r") is RepoType.GITHUB
    assert repo_type_for("https://ieee-dataport.org/x") is RepoType.IEEE_DATAPORT


def test_pinning_takes_the_most_specific():
    assert classify_pinning(["https://github.com/o/r"]) is Pinned.UNPINNED_URL
    assert classify_pinning(["https://github.com/o/r",
                             "https://github.com/o/r/tree/a1b2c3d"]) is Pinned.COMMIT_PINNED
    assert classify_pinning([]) is Pinned.NONE


# -- the theory/simulation trap that mis-coded 364 v1 papers ------------------
def test_purely_theoretical_requires_both_axes_absent():
    p = paper(axis_a=axis(Case.NOT_APPLICABLE), axis_b=axis(Case.NOT_APPLICABLE))
    p.validate()
    assert p.is_purely_theoretical
    # a simulation paper generates data: Axis B is present, so NOT theoretical
    p2 = paper(axis_a=axis(Case.NOT_APPLICABLE),
               axis_b=axis(Case.INSUFFICIENTLY_SPECIFIED))
    p2.validate()
    assert not p2.is_purely_theoretical


def test_openness_rank_is_ordinal_where_case_number_is_not():
    from dap.schema import CASE_NUMBER, OPENNESS_RANK
    assert CASE_NUMBER[Case.ON_REQUEST] > CASE_NUMBER[Case.ACCESS_NOT_PROVIDED]
    assert OPENNESS_RANK[Case.ON_REQUEST] > OPENNESS_RANK[Case.ACCESS_NOT_PROVIDED]
    assert OPENNESS_RANK[Case.NOT_APPLICABLE] is None


def test_flat_row_emits_tokens_not_labels():
    row = paper().flat()
    assert row["b_case"] == "access_not_provided"
    assert row["a_case_num"] == 1
    assert row["scheme_version"].startswith("2.")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
