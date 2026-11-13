"""Shared builders for manual classification batches.

Tracked, unlike the bNN.py files that import it: a batch builder is a record of
one afternoon's coding, but these builders are what keep successive batches on
the same reading of the scheme. The rule builders at the bottom of this file
exist so that a batch cannot quietly drift back to a pre-decision reading of
P1, P2, P3, P4, P5, P8 or P9. P3 was missing from this file until 3 Oct, and
batches 93-98 each restated it locally; S90 found that the local wording had
inverted it.

P4 was the next instance of the same gap: it had a rule string here from the
start but no builder, so batch 102 would have had to hand-roll it. `ref_pointer`
closes that on 3 Oct, matching the shape already coded by hand in September.
`named_no_route` grew a `decl` parameter the same day, because a named dataset
can be *described* as publicly available while still carrying no identifier, and
`declared` records what the paper says, not what it provides.

`NAMED_RULE`, `named_no_route` and `host_only` were added the same day for the
same reason, and they are the one pair here that guards a rule Diana has NOT
settled: the written rule (S18, and `classify.py`'s close-pair rule) says a
named external dataset with no route is `access_not_provided`, while her open
proposal would make it `openly_accessible`. The guard encodes the written rule
because that is the rule of record, not because the question is closed.

Used as `sys.path.insert(0, "scratch")` then `from h import ...`, with every
command run from v2/.
"""

import json


def J(v: str, c: str, r: str, ev: list[int] | None = None) -> dict:
    """Build one judgement object.

    Args:
        v: The value: a token of the field's vocabulary.
        c: The confidence token.
        r: The rationale, one sentence.
        ev: The ids of the evidence items that support the value.

    Returns:
        The judgement, in the shape the classification schema expects.
    """
    return {"value": v, "confidence": c, "rationale": r, "evidence_ids": ev or []}


def mkaxis(
    case: str,
    case_c: str,
    case_r: str,
    *,
    case_ev: list[int] | None = None,
    dt_ev: list[int] | None = None,
    gran_ev: list[int] | None = None,
    dt: str = "unclear",
    dt_c: str = "medium",
    dt_r: str = "data type not established from the retrieved passages",
    decl: str = "none_stated",
    decl_c: str = "medium",
    decl_r: str = "no availability statement located",
    decl_ev: list[int] | None = None,
    gran: str | None = None,
    gran_c: str = "low",
    gran_r: str = "not established from the retrieved passages",
    identifiers: list[str] | None = None,
    names: list[str] | None = None,
    repo_types: list[str] | None = None,
    restriction: str = "none_given",
    pinned: str = "none",
) -> dict:
    """Build one axis of a classification.

    Every other builder in this file ends in a call to this one.

    Args:
        case: The case token.
        case_c: The confidence of the case.
        case_r: The rationale of the case.
        case_ev: The evidence ids that support the case.
        dt_ev: The evidence ids that support the data type.
        gran_ev: The evidence ids that support the granularity.
        dt: The data type token.
        dt_c: The confidence of the data type.
        dt_r: The rationale of the data type.
        decl: The `declared` token: what the paper says about access.
        decl_c: The confidence of `declared`.
        decl_r: The rationale of `declared`.
        decl_ev: The evidence ids that support `declared`.
        gran: The granularity token. None chooses `not_applicable` for a
            `not_applicable` case and `unclear` otherwise.
        gran_c: The confidence of the granularity.
        gran_r: The rationale of the granularity.
        identifiers: The DOIs and URLs given, verbatim.
        names: The dataset or source names given.
        repo_types: The repository type tokens.
        restriction: The restriction token.
        pinned: The pinning token.

    Returns:
        The axis object, in the shape the classification schema expects.
    """
    if gran is None:
        gran = "not_applicable" if case == "not_applicable" else "unclear"
    return {
        "case": J(case, case_c, case_r, case_ev),
        "data_type": J(dt, dt_c, dt_r, dt_ev),
        "declared": J(decl, decl_c, decl_r, decl_ev),
        "granularity": J(gran, gran_c, gran_r, gran_ev),
        "pinned": pinned,
        "restriction": restriction,
        "repo_types": repo_types or [],
        "identifiers": identifiers or [],
        "names": names or [],
    }


def NA(reason: str, conf: str = "medium") -> dict:
    """Build an axis that does not apply.

    Args:
        reason: The rationale: why the axis does not apply.
        conf: The confidence of the case.

    Returns:
        A `not_applicable` axis.
    """
    return mkaxis("not_applicable", conf, reason)


def ev(*items: tuple[str, str]) -> list[dict]:
    """Build a record's evidence list.

    Args:
        *items: `(quote, section)` pairs, in the order their ids are to be
            assigned.

    Returns:
        The evidence items, numbered from 1, with `full_text` as source.
    """
    return [
        {"id": i + 1, "section": s, "locator": "", "source": "full_text", "quote": q}
        for i, (q, s) in enumerate(items)
    ]


P1 = "data declared to be included within the article with no separate location: policy question P1"
BOILER = (
    "on-request statement on a paper with no apparent dataset -- a human "
    "should confirm whether anything is actually available on request, or "
    "whether this is publisher boilerplate"
)
DENY_OUT = (
    "the statement denies any dataset while the paper reports generated "
    "computational output; whether that output counts is adjacent to "
    "policy question P3"
)
SUBJECT = (
    "the paper's own subject requires input data that the availability "
    "statement denies exists, and no source is named, so both axes rest "
    "on an absence the paper contradicts (policy question P3)"
)
NOAXA = (
    "the evaluation implies input data but names no source in the retrieved "
    "passages, so axis A could not be established"
)
CONTRIB = (
    "the availability statement denies any dataset while the author "
    "contributions block credits named authors with data curation, "
    "analysis or visualisation -- a human should determine which is accurate"
)
INTERLEAVE = (
    "the availability statement is interleaved with the neighbouring column "
    "in the source PDF, so only a contiguous fragment could be quoted"
)
NOSTAT_OUT = (
    "no availability statement of any kind appears, so the reported output has no declared route"
)


# Since 28 Oct 2026 (S220) `deny`, `denied_anyway` and `no_data_statement` are
# Axis-B builders only. A "no data" statement does not settle Axis A: code A
# from the body with `NA`, `needs_data`, `noaxa` or `named_no_route`, whatever
# the statement says. b76 once put `denied_anyway()` on axis A (W4406041920).
def deny(extra: str = "") -> dict:
    """Build Axis B for a "no data" statement on an analytical paper.

    Axis B only (S220): a "no data" statement does not settle Axis A.

    Args:
        extra: Text appended to the case rationale.

    Returns:
        A `not_applicable` axis that cites evidence 1, the statement.
    """
    return mkaxis(
        "not_applicable",
        "medium",
        "The statement declares no dataset was generated or analysed, which is "
        "consistent with a paper whose content is analytical throughout." + extra,
        case_ev=[1],
        dt="unclear",
        dt_c="medium",
        dt_r="no data of any kind is reported",
        decl="none_stated",
        decl_c="medium",
        decl_r="declared not applicable",
        decl_ev=[1],
        gran="not_applicable",
        gran_c="medium",
        gran_r="no dataset exists",
    )


def onreq(what: str = "the data") -> dict:
    """Build an axis for a formal statement that offers the data on request.

    Args:
        what: What the statement offers, as it reads in the rationale.

    Returns:
        An `on_request` axis that cites evidence 1, the statement.
    """
    return mkaxis(
        "on_request",
        "medium",
        f"A formal availability statement routes {what} through the "
        f"corresponding author on request, with no repository or identifier.",
        case_ev=[1],
        dt="unclear",
        dt_c="low",
        dt_r="no dataset is described in the retrieved text",
        decl="on_request",
        decl_c="medium",
        decl_r="access routed through the author on request",
        decl_ev=[1],
        gran="unclear",
        gran_c="low",
        gran_r="not characterised",
    )


def inarticle(
    dt: str = "unclear",
    dt_c: str = "low",
    dt_r: str = "what the declared data consists of is not established",
    dt_ev: list[int] | None = None,
) -> dict:
    """Build an axis for data declared to be included within the article (P1).

    Args:
        dt: The data type token.
        dt_c: The confidence of the data type.
        dt_r: The rationale of the data type.
        dt_ev: The evidence ids that support the data type.

    Returns:
        An `insufficiently_specified` axis, declared `public`, that cites
        evidence 1, the statement.
    """
    return mkaxis(
        "insufficiently_specified",
        "medium",
        "The statement asserts data exists and locates it only inside the "
        "article, giving no identifier a third party could resolve.",
        case_ev=[1],
        dt=dt,
        dt_c=dt_c,
        dt_r=dt_r,
        dt_ev=dt_ev,
        decl="public",
        decl_c="low",
        decl_r="declared as included within the article",
        decl_ev=[1],
        gran="unclear",
        gran_c="low",
        gran_r="not characterised",
    )


# External input that is never named is `insufficiently_specified`: the reader
# cannot identify the material at all, which is the classify.py close-pair rule
# ("an internal corpus of support tickets"). `access_not_provided` is for
# material that IS named but given no route ("We evaluate on ImageNet").
# Diana, 3 Oct 2026. From 24 Sep until then both builders below coded it
# `access_not_provided` -- needs_data here, and a copy named noaxa defined
# locally in b73-b94 -- with no decision behind it. Import these; do not
# redefine them in a batch builder, or the local copy shadows the rule.
def needs_data(what: str) -> dict:
    """Build Axis A for a task that needs external input the paper never names.

    Args:
        what: The task that requires the input.

    Returns:
        An `insufficiently_specified` axis at medium confidence (S83).
    """
    return mkaxis(
        "insufficiently_specified",
        "medium",
        f"{what} requires external input, but no source, corpus name or "
        f"access route appears in the retrieved passages, so a reader "
        f"cannot identify the material at all.",
        dt="real",
        dt_c="medium",
        dt_r=f"input data for {what.lower()}",
        decl="none_stated",
        decl_c="medium",
        decl_r="no source is named",
    )


def noaxa(what: str, conf: str = "low") -> dict:
    """Build Axis A for external input that the retrieved passages do not name.

    Args:
        what: The task that requires the input.
        conf: The confidence of the case and of the data type.

    Returns:
        An `insufficiently_specified` axis (S83).
    """
    return mkaxis(
        "insufficiently_specified",
        conf,
        f"{what} requires external input, but the retrieved passages name "
        f"no source, corpus or access route for it, so a reader cannot "
        f"identify the material at all.",
        dt="real",
        dt_c=conf,
        dt_r="input data the task requires",
        decl="none_stated",
        decl_c=conf,
        decl_r="no source is named",
    )


def denied_anyway() -> dict:
    """Build Axis B for a "no data" statement the paper's own task contradicts.

    Axis B only (S220).

    Returns:
        A `not_applicable` axis at low confidence that cites evidence 1.
    """
    return mkaxis(
        "not_applicable",
        "low",
        "The statement denies any dataset was generated or analysed, which "
        "cannot be reconciled with what the paper's task requires.",
        case_ev=[1],
        dt="unclear",
        dt_c="low",
        dt_r="the statement denies what the task requires",
        decl="none_stated",
        decl_c="medium",
        decl_r="declared that no datasets exist",
        decl_ev=[1],
        gran="unclear",
        gran_c="low",
        gran_r="not characterised",
    )


def no_data_statement(ev_ids: list[int], what: str = "results") -> dict:
    """Build Axis B under the no-data ruling.

    The statement governs B even when the paper reports `what`. Pass the
    evidence id of the statement itself. Only for Axis B; if the paper also
    gives an address for its code or data, do not use this -- code the route
    and flag it (the ruling leaves that open).

    Args:
        ev_ids: The evidence id of the statement itself.
        what: What the paper reports nonetheless.

    Returns:
        A `not_applicable` axis.
    """
    return mkaxis(
        "not_applicable",
        "medium",
        f"The paper states explicitly that no data was used, generated or "
        f"analysed, so Axis B is not_applicable although it reports {what}; "
        f"{NODATA_RULE}.",
        case_ev=ev_ids,
        dt="unclear",
        dt_c="low",
        dt_r="the statement declares that no dataset exists",
        decl="none_stated",
        decl_c="medium",
        decl_r="declared that no datasets exist",
        decl_ev=ev_ids,
        gran="not_applicable",
        gran_c="medium",
        gran_r="no dataset is declared",
    )


def nodata(kind: str = "analytical") -> dict:
    """Build an axis for a paper that reports no dataset and has no statement.

    Args:
        kind: What the paper's content is throughout, e.g. `"analytical"`.

    Returns:
        A `not_applicable` axis.
    """
    return mkaxis(
        "not_applicable",
        "medium",
        f"No dataset is reported: the paper's content is {kind} throughout, "
        f"and no availability statement exists to declare otherwise.",
        dt="unclear",
        dt_c="medium",
        dt_r="no data of any kind is reported",
        decl="none_stated",
        decl_c="medium",
        decl_r="no availability statement appears in the retrieved text",
        gran="not_applicable",
        gran_c="medium",
        gran_r="no dataset exists",
    )


def output_no_route(what: str) -> dict:
    """Build Axis B for computational output with no route to it.

    Args:
        what: The output the paper reports.

    Returns:
        An `access_not_provided` axis with data type `synthetic`.
    """
    return mkaxis(
        "access_not_provided",
        "medium",
        f"The paper reports {what}, so computational output exists, but no "
        f"availability statement, repository or route to it appears anywhere "
        f"in the retrieved text. Coded as data without a route rather than as "
        f"no data, because the paper says the output was produced.",
        dt="synthetic",
        dt_c="medium",
        dt_r=f"the {what}",
        decl="none_stated",
        decl_c="medium",
        decl_r="no availability statement appears in the retrieved text",
        gran="processed",
        gran_c="low",
        gran_r="results as reported in the text",
    )


def hw_no_route(ev_ids: list[int], what: str) -> dict:
    """Build Axis B for output measured on quantum hardware, with no route.

    `output_no_route`'s hardware variant: the output is measured on a real
    device, so the data type is `real`, not `synthetic`. Defined here from
    3 Oct; b99.py had it locally and b100.py needed it too, which is how the
    drift S83 and S90 found begins.

    Args:
        ev_ids: The evidence ids that support the case.
        what: The output the paper reports.

    Returns:
        An `access_not_provided` axis with data type `real`.
    """
    return mkaxis(
        "access_not_provided",
        "medium",
        f"The paper reports {what}, measured on named quantum hardware, so "
        f"data exists, and no availability statement, repository or route to "
        f"it appears anywhere in the retrieved text.",
        case_ev=ev_ids,
        dt="real",
        dt_c="medium",
        dt_r=what,
        dt_ev=ev_ids,
        decl="none_stated",
        decl_c="medium",
        decl_r="no availability statement appears in the retrieved text",
        gran="processed",
        gran_c="low",
        gran_r="results as reported in the text",
    )


def lab_no_route(ev_ids: list[int], what: str, gran: str = "processed") -> dict:
    """Build Axis B for output measured on laboratory apparatus, with no route.

    `output_no_route`'s laboratory variant: the measurement was taken on
    physical apparatus that is not a quantum processor, so the data type is
    `real` and the data is not "computational output".

    The third sibling, added 4 Oct for the reason `hw_no_route` was added on
    3 Oct and `human_no_route` on 4 Oct, and after measuring the drift it
    prevents. `output_no_route` hardcodes `dt="synthetic"` and the clause "so
    computational output exists". Applied to optics benches, dilution
    refrigerators, neutron beamlines and ion traps it has written
    **`data_type: synthetic` onto 77 rows whose `paper_type` is
    `experimental`**, and 61 of those 77 say "measured", "tomography",
    "coincidence", "detector" or "cryogenic" in their own case rationale --
    for example

      W4408034515  Nature Photonics  "the measured single-photon source
                   efficiencies, so computational output exists"
      W4400668424  Phys. Rev. Lett.  "the measured outcomes of the multiclient
                   blind computation, so computational output exists"
      W4406983171  IEEE TAS          "the measured sensor characteristics at
                   deep-cryogenic temperatures, so computational output exists"

    Reading the other 16 individually adds 7 more laboratory cases (trapped
    ions, a three-photon GHZ state, reconstructed qudit states, accelerometer
    traces), so about 68 of the 77 are measurement, not computation. The
    remainder -- W4221161984 "the paper's own numerical simulations",
    W4220810156 "gate-rescheduling algorithm implementation" -- are correctly
    `synthetic`, which is why the count is read and not quoted raw.

    `hw_no_route` stays the right builder when the apparatus IS the quantum
    processor; this one is for everything else on the bench. Those 77 rows are
    stored, so correcting them is Diana's call, not an append.

    Args:
        ev_ids: The evidence ids that support the case.
        what: The measurement the paper reports.
        gran: The granularity token.

    Returns:
        An `access_not_provided` axis with data type `real`.
    """
    return mkaxis(
        "access_not_provided",
        "medium",
        f"The paper reports {what}, measured on laboratory apparatus, so data "
        f"exists, and no availability statement, repository or route to it "
        f"appears anywhere in the retrieved text.",
        case_ev=ev_ids,
        dt="real",
        dt_c="medium",
        dt_r=what,
        dt_ev=ev_ids,
        decl="none_stated",
        decl_c="medium",
        decl_r="no availability statement appears in the retrieved text",
        gran=gran,
        gran_c="low",
        gran_r="results as reported in the text",
    )


def severed_statement(ev_ids: list[int], what: str = "the data") -> dict:
    """Build an axis for a canonical statement whose route clause is missing.

    The ROUTE clause is not in the retrieved text: the PDF's two-column layout
    interleaves the statement with its neighbour and the claim survives while
    "...from the corresponding author upon reasonable request" / "...at
    <doi>" does not.

    Added 4 Oct (S186) because the alternative is to infer the route from the
    publisher's usual template, and that inference is already in the store.
    Measured: **92 rows carry a canonical statement ("the data that support the
    findings of this study are available", "the datasets generated during
    and/or analysed during the current study are available") with no route word
    in the following 45 characters.** 48 of the 92 are classified, and those 48
    were given FIVE different cases on the identical severed sentence:

        on_request 30 | openly_accessible 9 | insufficiently_specified 6
        access_not_provided 2 | not_applicable 1

    They cannot all be readings of the text, because the text stops in the same
    place in every one of them. npj Quantum Information supplies about twenty,
    the AIP titles (APL, AIP Advances, J. Appl. Phys., Phys. Fluids, Phys.
    Plasmas, Rev. Sci. Instrum., AVS Quantum Science) most of the rest. The
    30 `on_request` rows are probably right about those publishers' templates
    -- and that is the problem: it is the template being coded, not the paper.

    So this builder records what the retrieved text establishes and nothing
    more: a claim is present, and no path a third party could follow is. It
    always flags, because the route is recoverable by a human with the PDF.

    Args:
        ev_ids: The evidence ids of the severed statement.
        what: What the statement says is available.

    Returns:
        An `insufficiently_specified` axis at low confidence. The record
        must be flagged for review.
    """
    return mkaxis(
        "insufficiently_specified",
        "low",
        f"A formal availability statement asserts that {what} supporting the "
        f"findings is available, and the clause naming the route is severed "
        f"from it by the column interleave, so no path is recoverable from the "
        f"retrieved text. Coded on what is present, not on what the "
        f"publisher's template usually says next.",
        case_ev=ev_ids,
        dt="unclear",
        dt_c="low",
        dt_r="not characterised in the surviving fragment",
        decl="public",
        decl_c="low",
        decl_r="declared available; the route clause is not in the retrieved text",
        decl_ev=ev_ids,
        gran="unclear",
        gran_c="low",
        gran_r="not characterised",
    )


def human_no_route(ev_ids: list[int], what: str) -> dict:
    """Build Axis B for human-subject material with no route to it.

    `output_no_route`'s human-subject variant: the data is what people said,
    so the data type is `real`, not `synthetic`, and it is not "computational
    output".

    Added 4 Oct for the same reason `hw_no_route` was added on 3 Oct, and after
    measuring the drift it prevents. `output_no_route` hardcodes
    `dt="synthetic"` and the clause "so computational output exists"; applied to
    interview and consultation studies it has written **`data_type: synthetic`
    onto three rows whose data is people**:

      W4399942646  "the practitioner evaluation ... 22 respondents against 17 criteria"
      W4400786164  "the consultation findings the abstract reports"
      W4417401724  "the coded interview responses from educators and experts"

    (A probe for this returns four; the fourth, W4399971555, says "participant
    numbers" about parties in a secret-sharing protocol, where `synthetic` is
    right. Read the matches.) Those three are stored rows, so correcting them is
    Diana's call, not an append -- S155. This builder stops the fourth.

    `restriction` stays `none_given` on purpose: whether human-subject material
    without a stated restriction should carry `privacy` by default is open, and
    S143 measured 8 rows where the authors said so themselves.

    Args:
        ev_ids: The evidence ids that support the case.
        what: The human-subject material the paper reports.

    Returns:
        An `access_not_provided` axis with data type `real`.
    """
    return mkaxis(
        "access_not_provided",
        "medium",
        f"The paper reports {what}, so human-subject material exists, and no "
        f"availability statement, repository or route to it appears anywhere "
        f"in the retrieved text.",
        case_ev=ev_ids,
        dt="real",
        dt_c="medium",
        dt_r=f"{what}, recorded from people",
        dt_ev=ev_ids,
        decl="none_stated",
        decl_c="medium",
        decl_r="no availability statement appears in the retrieved text",
        gran="processed",
        gran_c="low",
        gran_r="as reported in the text, not as collected",
    )


def openrepo(
    ids: list[str],
    repos: list[str],
    ev_ids: list[int],
    what: str,
    dt: str = "synthetic",
    gran: str = "both",
    gran_r: str = "data together with the code that produced it",
    pinned: str = "unpinned_url",
    names: list[str] | None = None,
    decl_c: str = "high",
) -> dict:
    """Build an axis for a statement that names the material AND its identifier.

    Defined here from 3 Oct; b98.py and b99.py each had a copy.

    Args:
        ids: The identifiers given, verbatim.
        repos: The repository type tokens.
        ev_ids: The evidence ids of the statement.
        what: The material the statement names.
        dt: The data type token.
        gran: The granularity token.
        gran_r: The rationale of the granularity.
        pinned: The pinning token.
        names: The dataset or source names given.
        decl_c: The confidence of `declared`.

    Returns:
        An `openly_accessible` axis, declared `public`.
    """
    return mkaxis(
        "openly_accessible",
        "medium",
        f"The statement names {what} and gives the identifier itself, so a "
        f"third party has an unconditional path to it.",
        case_ev=ev_ids,
        dt=dt,
        dt_c="medium",
        dt_r=what,
        dt_ev=ev_ids,
        decl="public",
        decl_c=decl_c,
        decl_r="declared openly available",
        decl_ev=ev_ids,
        gran=gran,
        gran_c="medium",
        gran_r=gran_r,
        identifiers=ids,
        repo_types=repos,
        pinned=pinned,
        names=names or [],
    )


def supp(ev_ids: list[int], what: str, gran: str = "processed", extra: str = "") -> dict:
    """Build an axis for data located only in the publisher's supplement.

    S76's shape, still open: the only route is the publisher's supplement,
    which is a real artefact with no identifier of its own.

    Args:
        ev_ids: The evidence ids of the statement.
        what: The material the statement locates in the supplement.
        gran: The granularity token.
        extra: Text appended to the case rationale.

    Returns:
        An `insufficiently_specified` axis.
    """
    return mkaxis(
        "insufficiently_specified",
        "medium",
        f"The statement locates {what} in the publisher's supplementary "
        f"material, which is a distinct artefact but has no identifier of its "
        f"own{extra}.",
        case_ev=ev_ids,
        dt="real",
        dt_c="medium",
        dt_r=what,
        dt_ev=ev_ids,
        decl="public",
        decl_c="medium",
        decl_r="declared provided with the article's supplement",
        decl_ev=ev_ids,
        gran=gran,
        gran_c="medium",
        gran_r="as deposited in the supplement",
        repo_types=["publisher_supplementary"],
    )


#: `paper_type` is a Judgement like any other field, and the gate rejects a bare
#: token. Only the value and its confidence are persisted -- the rationale is
#: validated and dropped -- so these state the definitional ground for the type
#: and each record's own reasoning lives in the axis rationales.
PT_WHY = {
    "theory": "the contribution is analytical: definitions, theorems and proofs",
    "simulation": "the contribution rests on computation the authors ran themselves",
    "experimental": "the contribution rests on measurements taken on physical apparatus",
    "benchmark_study": "the contribution is a systematic comparison across methods or a released benchmark",
    "survey_review": "the contribution is a survey of existing literature",
    "position": "the contribution is an argument or set of guidelines rather than a result",
    "systems_build": "the contribution is an engineered system or platform",
    "tool_paper": "the contribution is a released tool or library",
    #: Added 3 Oct, the last `PaperType` token with no entry here. `PT` raised
    #: KeyError on it, which would have pushed a batch into defining the
    #: rationale locally -- the drift shape S83, S90 and S91 each found.
    "dataset_paper": "the contribution IS the dataset: the paper describes a deposit rather than a result derived from one",
    "unclear": "the retrieved text does not establish what kind of contribution this is",
}


def PT(v: str, conf: str = "medium", why: str | None = None, ev: list[int] | None = None) -> dict:
    """Build the paper-type judgement.

    Args:
        v: The paper type token.
        conf: The confidence.
        why: The rationale. None uses the stock rationale in `PT_WHY`.
        ev: The evidence ids that support the type.

    Returns:
        The judgement object.
    """
    return J(v, conf, why or PT_WHY[v], ev)


def write(R: list[tuple], out: str) -> None:
    """Write a batch of records as a results file for `--import`.

    Args:
        R: The records, each a tuple `(custom_id, paper_type, axis_a,
            axis_b, das_present, needs_human_review, review_reason,
            evidence)`. `paper_type` may be a token, which is expanded with
            `PT`.
        out: The JSONL file to write.
    """
    with open(out, "w") as f:
        for cid, pt, a, b, das, rev, why, evs in R:
            if isinstance(pt, str):
                pt = PT(pt)
            f.write(
                json.dumps(
                    {
                        "custom_id": cid,
                        "classification": {
                            "evidence": evs,
                            "paper_type": pt,
                            "axis_a": a,
                            "axis_b": b,
                            "das_present": das,
                            "needs_human_review": rev,
                            "review_reason": why,
                        },
                    }
                )
                + "\n"
            )
    print(f"wrote {len(R)} records to {out}")


# ---------------------------------------------------------------------------
# The nine policy questions, as Diana settled them (25-27 Sep and 1 Oct 2026).
# review_issues.md S71-S72; memory review-issues-decisions-2026-09.
# Use these builders rather than hand-rolling a case, so a batch cannot drift
# back to the pre-decision reading.
# ---------------------------------------------------------------------------

P1_RULE = (
    "P1 as decided: data declared only as included in the article is "
    "not available -- a claim with no resolvable path"
)
P2_RULE = (
    "P2 as decided: the authors' own publicly released code discharges "
    "data availability; third-party software and code on request do not"
)
P3_RULE = (
    "P3 as decided (S71): a statement's denial GOVERNS the code -- the "
    "axis is not_applicable -- and the contradiction stays in the flag. "
    "Not 'code what the paper did and flag': S90 measured 80 rows that "
    "had drifted to that reading"
)
#: Not one of the nine, and not decided: the *written* rule for a named
#: external dataset with no route. On record since S18 ("the schema settles the
#: named-benchmark question") and in `classify.py`'s close-pair rule -- "'We
#: evaluate on ImageNet' names a thing you could go and find: access_not_provided".
#: Diana has an open proposal to make this `openly_accessible` instead, which
#: needs a SCHEME_VERSION bump because the gate requires an identifier. Until
#: she rules, code the written rule through `named_no_route` below. Do NOT reach
#: for `insufficiently_specified`: S91 measured 133 rows sitting there, 5 of them
#: coded after S18 was already on record.
NAMED_RULE = (
    "the written rule, S18 and the close-pair rule in classify.py: a "
    "named external dataset with no route is access_not_provided, "
    "because the reader can identify the material. Diana's proposal "
    "to make it openly_accessible is still open and needs a scheme bump"
)
#: Diana, 28 Oct 2026 (S219, clarified in S220; SCHEME_VERSION 2.0.1, prompt
#: v2.2.0). It narrows P3 to Axis B. S220 found 98 stored rows against it,
#: listed in nodata_ruling_candidates.csv; 40 were recoded and 42 are held.
NODATA_RULE = (
    "the no-data ruling (S219/S220, 28 Oct): an explicit statement that "
    "no data was used, created, generated or analysed codes Axis B "
    "not_applicable, even on a computational or experimental paper; Axis A "
    "is coded from the body, never from the statement. Not for a silent "
    "paper, a self-contradicting statement, or a denial beside a stated "
    "route, which is still open"
)
P4_RULE = (
    "P4 as decided: a reference-list or footnote pointer counts only "
    "where the reference itself resolves in the retrieved text"
)
P5_RULE = "P5 as decided: a future-tense promise is coded on the state at the version of record"
P8_RULE = (
    "P8 as decided: a tool paper's own released source is not axis-B "
    "data; axis B stays on results only"
)
P9_RULE = (
    "P9 as decided: an on-request route stays on_request even with a "
    "restriction attached; the restriction token records it"
)


def named_no_route(
    ev_ids: list[int],
    names: list[str],
    what: str,
    gran: str = "raw",
    dt: str = "real",
    decl: str | None = None,
) -> dict:
    """Build Axis A for a named dataset with no route to it.

    The written rule's other side from `noaxa`: the dataset IS identified and
    no route is given.

    Defined here from 3 Oct. Batches 93-99 each defined it locally, which is the
    same drift shape as the `noaxa` copies S83 found and the P3 wording S90
    found. The target case is the written one (`access_not_provided`); if Diana
    rules for `openly_accessible` this builder changes once, here.

    Args:
        ev_ids: The evidence ids that support the case.
        names: The dataset names the paper gives.
        what: The dataset, as it reads in the rationale.
        gran: The granularity token.
        dt: The data type token.
        decl: The `declared` token. None records that no availability
            statement was made.

    Returns:
        An `access_not_provided` axis.
    """
    return mkaxis(
        "access_not_provided",
        "medium",
        f"The paper identifies {what} by name, so a reader knows what was "
        f"used, and gives no repository, accession or route to the copy it "
        f"ran on.",
        case_ev=ev_ids,
        dt=dt,
        dt_c="medium",
        dt_r=what,
        dt_ev=ev_ids,
        decl=decl or "none_stated",
        decl_c="medium",
        decl_r=(
            "the dataset is named and described as publicly available, with no identifier"
            if decl == "public"
            else "the dataset is named, no route is given"
        ),
        decl_ev=ev_ids if decl else None,
        gran=gran,
        gran_c="medium",
        gran_r="as distributed",
        names=names,
    )


def ref_pointer(
    ev_ids: list[int],
    what: str = "the data",
    dt: str = "synthetic",
    dt_r: str = "computational output the paper reports",
    decl: str = "public",
    decl_r: str = "declared deposited, via a reference that does not resolve here",
) -> dict:
    """Build an axis whose only route is an unresolved numbered reference (P4).

    The only route is a numbered reference or footnote that the retrieved
    text does not resolve.

    P4 has been decided since 25 Sep and had a rule string here but no builder,
    which is the gap this file exists to close: batch 102 needed it for a Nature
    Communications row whose data sits in "the Zenodo repository44", and the
    shape is already in the store -- W4393260622 was coded this way by hand in
    September ("openly available at Ref. [79]", `insufficiently_specified`,
    declared `public`). Keep both of those properties: the claim is public, the
    path is not resolvable, so the case is `insufficiently_specified`.

    `decl` added 4 Oct, for the same reason `named_no_route` grew one: a dataset
    can be located by a bare numbered reference in body text with no
    availability statement anywhere ("a dataset provided by [20] was used"),
    and `declared` records what the paper says. Defaulting to `public` leaves
    every existing call unchanged.

    Args:
        ev_ids: The evidence ids of the pointer.
        what: The material behind the reference.
        dt: The data type token.
        dt_r: The rationale of the data type.
        decl: The `declared` token; `none_stated` when the pointer is in
            body text with no availability statement.
        decl_r: The rationale of `declared`.

    Returns:
        An `insufficiently_specified` axis.
    """
    where = (
        "The statement locates"
        if decl != "none_stated"
        else ("The paper, with no availability statement anywhere, locates")
    )
    return mkaxis(
        "insufficiently_specified",
        "medium",
        f"{where} {what} behind a numbered reference the "
        f"retrieved text does not resolve, so the claim is made but no path "
        f"a third party can follow is given.",
        case_ev=ev_ids,
        dt=dt,
        dt_c="medium",
        dt_r=dt_r,
        dt_ev=ev_ids,
        decl=decl,
        decl_c="medium",
        decl_r=decl_r,
        decl_ev=ev_ids,
        gran="unclear",
        gran_c="low",
        gran_r="not characterised in the retrieved text",
    )


def host_only(ev_ids: list[int], name: str, what: str, gran: str = "raw") -> dict:
    """Build Axis A for a named source with no dataset identified inside it.

    The boundary `named_no_route` must not swallow: a source is named and the
    material inside it is not, so the reader still cannot tell what was used.
    `insufficiently_specified` is correct here and S91's hand-read found 10
    rows of this shape in one batch window, so the distinction is not rare.

    Args:
        ev_ids: The evidence ids that support the case.
        name: The source the paper names.
        what: The material taken from it.
        gran: The granularity token.

    Returns:
        An `insufficiently_specified` axis.
    """
    return mkaxis(
        "insufficiently_specified",
        "medium",
        f"The paper names {name} as the source of {what} and identifies no "
        f"dataset, accession or record inside it, so a reader cannot tell "
        f"which material was used.",
        case_ev=ev_ids,
        dt="real",
        dt_c="medium",
        dt_r=what,
        dt_ev=ev_ids,
        decl="public",
        decl_c="medium",
        decl_r="the source is named as publicly available, with no record",
        decl_ev=ev_ids,
        gran=gran,
        gran_c="medium",
        gran_r="as distributed by the named source",
        names=[name],
    )


def own_code(
    ids: list[str],
    repos: list[str],
    ev_ids: list[int],
    what: str = "the code that generated the results",
    pinned: str = "unpinned_url",
) -> dict:
    """Build Axis B for the authors' own code, publicly released (P2).

    The code has an identifier a third party can resolve.

    Args:
        ids: The identifiers given, verbatim.
        repos: The repository type tokens.
        ev_ids: The evidence ids of the statement.
        what: What the statement offers.
        pinned: The pinning token.

    Returns:
        An `openly_accessible` axis.
    """
    return mkaxis(
        "openly_accessible",
        "medium",
        f"The statement offers {what}, released publicly by the authors with "
        f"an identifier a third party can follow, so the results are "
        f"reachable through their generator.",
        case_ev=ev_ids,
        dt="synthetic",
        dt_c="medium",
        dt_r="computational output, reachable through the released generator",
        decl="public",
        decl_c="high",
        decl_r="declared openly available",
        decl_ev=ev_ids,
        gran="unclear",
        gran_c="low",
        gran_r="not characterised",
        identifiers=ids,
        repo_types=repos,
        pinned=pinned,
    )


def code_unreachable(ev_ids: list[int], what: str = "the code that generated the results") -> dict:
    """Build Axis B for the authors' own code behind an unresolved pointer.

    P2 + P4: the authors' own code, but the pointer does not resolve.

    Args:
        ev_ids: The evidence ids of the statement.
        what: What the statement offers.

    Returns:
        An `insufficiently_specified` axis.
    """
    return mkaxis(
        "insufficiently_specified",
        "medium",
        f"The statement offers {what} and locates it only where the retrieved "
        f"text cannot resolve, so neither the results nor their generator has "
        f"a path.",
        case_ev=ev_ids,
        dt="synthetic",
        dt_c="medium",
        dt_r="computational output of the authors' own code",
        decl="public",
        decl_c="medium",
        decl_r="declared available",
        decl_ev=ev_ids,
        gran="unclear",
        gran_c="low",
        gran_r="not characterised",
    )


def third_party_code(ev_ids: list[int], name: str, what: str = "the results") -> dict:
    """Build Axis B for a statement that names somebody else's software.

    The P2 boundary: nothing of the authors' own is made available.

    Args:
        ev_ids: The evidence ids of the statement.
        name: The software the statement names.
        what: What is left without a route.

    Returns:
        An `access_not_provided` axis.
    """
    return mkaxis(
        "access_not_provided",
        "medium",
        f"The statement names only {name}, software the authors did not "
        f"release, so nothing of theirs is made available and {what} have no "
        f"route.",
        case_ev=ev_ids,
        dt="synthetic",
        dt_c="medium",
        dt_r="computational output the paper reports",
        decl="none_stated",
        decl_c="medium",
        decl_r="the declaration covers third-party software",
        decl_ev=ev_ids,
        gran="unclear",
        gran_c="low",
        gran_r="not characterised",
        names=[name],
    )


def promise(ev_ids: list[int], what: str = "the data", due: bool = True) -> dict:
    """Build an axis for a future-tense promise with no current route (P5).

    Args:
        ev_ids: The evidence ids of the statement.
        what: What is promised.
        due: Not used; kept so existing batch builders still call.

    Returns:
        An `insufficiently_specified` axis.
    """
    return mkaxis(
        "insufficiently_specified",
        "medium",
        f"The statement promises {what} in the future tense, so at the version "
        f"of record there is no location at all.",
        case_ev=ev_ids,
        dt="unclear",
        dt_c="low",
        dt_r="what is promised is not characterised",
        decl="public",
        decl_c="low",
        decl_r="declared available at a future date",
        decl_ev=ev_ids,
        gran="unclear",
        gran_c="low",
        gran_r="not characterised",
    )


def gated(
    ev_ids: list[int],
    what: str = "the data",
    restriction: str = "none_given",
    dt: str | None = None,
    dt_r: str | None = None,
    gran: str | None = None,
    gran_r: str | None = None,
) -> dict:
    """Build an axis for an on-request route, with or without a restriction (P9).

    `dt`/`gran` added 4 Oct (S164) and both default to the old behaviour, so
    every existing call is unchanged. They exist because the hard-coded
    `data_type: unclear` is the same defect `human_no_route` was written for:
    an on-request statement says nothing about what the data IS, but the body
    of the paper often does, and "unclear" then throws away a fact the text
    establishes. `W4404757272` states its input is "a Chinese commercial
    bank's fraud detection scenario" -- real transaction records of real
    accounts -- and that is `real`, not `unclear`.

    Pass `dt` only where the retrieved text says what the material is. Leaving
    it off is still the right answer whenever the statement is all there is.

    Args:
        ev_ids: The evidence ids of the statement.
        what: What is offered on request.
        restriction: The restriction token attached to the offer.
        dt: The data type token, where the text says what the material is;
            None records `unclear`.
        dt_r: The rationale of the data type.
        gran: The granularity token; None records `unclear`.
        gran_r: The rationale of the granularity.

    Returns:
        An `on_request` axis.
    """
    return mkaxis(
        "on_request",
        "medium",
        f"A statement routes {what} through the authors on request; any "
        f"restriction attached is recorded separately rather than taken to "
        f"close the route.",
        case_ev=ev_ids,
        dt=dt or "unclear",
        dt_c="medium" if dt else "low",
        dt_r=dt_r or "not characterised in the retrieved text",
        dt_ev=ev_ids if dt else None,
        decl="on_request",
        decl_c="medium",
        decl_r="access routed through the authors on request",
        decl_ev=ev_ids,
        gran=gran or "unclear",
        gran_c="medium" if gran else "low",
        gran_r=gran_r or "not characterised",
        restriction=restriction,
    )


def restricted(
    ev_ids: list[int],
    what: str,
    restriction: str,
    why: str,
    dt: str = "real",
    gran: str = "unclear",
    gran_r: str = "not characterised in the retrieved text",
) -> dict:
    """Build an axis for a statement that refuses a deposit and gives its reason.

    Added 4 Oct (S159). `explicitly_restricted` was the last case in
    `schema.py` with no builder here, and 18 stored row-axes had been
    hand-rolled through `mkaxis` by four different coders. Measured before
    writing it, the hand-rolling had NOT drifted -- `declared` is `restricted`
    on all 18, and `data_type`/`granularity` vary with the subject as they
    should -- so this builder removes a risk rather than fixing a fault, and it
    pins the one field that must not vary.

    `restriction` is the author's stated ground, from `Restriction`: the
    18 use privacy 5, third_party 3, confidential 3, commercial 2,
    other_stated 2, and one each of proprietary, intellectual_property and
    size_technical. Pass `why` as the paper's own reason, in its own terms.

    Args:
        ev_ids: The evidence ids of the statement.
        what: What the statement refuses to share.
        restriction: The author's stated ground, a `Restriction` token.
        why: The paper's own reason, in its own terms.
        dt: The data type token.
        gran: The granularity token.
        gran_r: The rationale of the granularity.

    Returns:
        An `explicitly_restricted` axis, declared `restricted`.
    """
    return mkaxis(
        "explicitly_restricted",
        "medium",
        f"The statement refuses {what} outright and gives its reason: {why}.",
        case_ev=ev_ids,
        dt=dt,
        dt_c="medium",
        dt_r=what,
        dt_ev=ev_ids,
        decl="restricted",
        decl_c="high",
        decl_r="declared withheld, with a stated ground",
        decl_ev=ev_ids,
        gran=gran,
        gran_c="low",
        gran_r=gran_r,
        restriction=restriction,
    )


def tool_no_results(name: str = "the released package") -> dict:
    """Build Axis B for a tool paper that reports no results (P8).

    The contribution is the tool; no results are reported alongside.

    Args:
        name: The tool, as it reads in the rationale.

    Returns:
        A `not_applicable` axis.
    """
    return mkaxis(
        "not_applicable",
        "medium",
        f"The contribution is {name} itself and the paper reports no dataset "
        f"of results alongside it; a released tool is software, not axis-B "
        f"data.",
        dt="unclear",
        dt_c="medium",
        dt_r="no dataset of results is reported",
        decl="none_stated",
        decl_c="medium",
        decl_r="no availability statement for data of its own",
        gran="not_applicable",
        gran_c="medium",
        gran_r="no dataset exists",
    )
