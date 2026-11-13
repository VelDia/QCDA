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


def J(v, c, r, ev=None):
    return {"value": v, "confidence": c, "rationale": r, "evidence_ids": ev or []}


def mkaxis(
    case,
    case_c,
    case_r,
    *,
    case_ev=None,
    dt_ev=None,
    gran_ev=None,
    dt="unclear",
    dt_c="medium",
    dt_r="data type not established from the retrieved passages",
    decl="none_stated",
    decl_c="medium",
    decl_r="no availability statement located",
    decl_ev=None,
    gran=None,
    gran_c="low",
    gran_r="not established from the retrieved passages",
    identifiers=None,
    names=None,
    repo_types=None,
    restriction="none_given",
    pinned="none",
):
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


def NA(reason, conf="medium"):
    return mkaxis("not_applicable", conf, reason)


def ev(*items):
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
def deny(extra=""):
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


def onreq(what="the data"):
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
    dt="unclear",
    dt_c="low",
    dt_r="what the declared data consists of is not established",
    dt_ev=None,
):
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
def needs_data(what):
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


def noaxa(what, conf="low"):
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


def denied_anyway():
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


def no_data_statement(ev_ids, what="results"):
    """Axis B under the no-data ruling: the statement governs B even when the
    paper reports `what`. Pass the evidence id of the statement itself. Only
    for Axis B; if the paper also gives an address for its code or data, do
    not use this -- code the route and flag it (the ruling leaves that open).
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


def nodata(kind="analytical"):
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


def output_no_route(what):
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


def hw_no_route(ev_ids, what):
    """`output_no_route`'s hardware variant: the output is measured on a real
    device, so the data type is `real`, not `synthetic`. Defined here from
    3 Oct; b99.py had it locally and b100.py needed it too, which is how the
    drift S83 and S90 found begins.
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


def lab_no_route(ev_ids, what, gran="processed"):
    """`output_no_route`'s laboratory variant: the measurement was taken on
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


def severed_statement(ev_ids, what="the data"):
    """A canonical availability statement whose ROUTE clause is not in the
    retrieved text: the PDF's two-column layout interleaves the statement with
    its neighbour and the claim survives while "...from the corresponding
    author upon reasonable request" / "...at <doi>" does not.

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


def human_no_route(ev_ids, what):
    """`output_no_route`'s human-subject variant: the data is what people said,
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
    ids,
    repos,
    ev_ids,
    what,
    dt="synthetic",
    gran="both",
    gran_r="data together with the code that produced it",
    pinned="unpinned_url",
    names=None,
    decl_c="high",
):
    """A statement that names the material AND gives its identifier. Defined
    here from 3 Oct; b98.py and b99.py each had a copy.
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


def supp(ev_ids, what, gran="processed", extra=""):
    """S76's shape, still open: the only route is the publisher's supplement,
    which is a real artefact with no identifier of its own.
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


def PT(v, conf="medium", why=None, ev=None):
    return J(v, conf, why or PT_WHY[v], ev)


def write(R, out):
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


def named_no_route(ev_ids, names, what, gran="raw", dt="real", decl=None):
    """The written rule's other side from `noaxa`: the dataset IS identified and
    no route is given.

    Defined here from 3 Oct. Batches 93-99 each defined it locally, which is the
    same drift shape as the `noaxa` copies S83 found and the P3 wording S90
    found. The target case is the written one (`access_not_provided`); if Diana
    rules for `openly_accessible` this builder changes once, here.
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
    ev_ids,
    what="the data",
    dt="synthetic",
    dt_r="computational output the paper reports",
    decl="public",
    decl_r="declared deposited, via a reference that does not resolve here",
):
    """P4: the only route is a numbered reference or footnote that the retrieved
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


def host_only(ev_ids, name, what, gran="raw"):
    """The boundary `named_no_route` must not swallow: a source is named and the
    material inside it is not, so the reader still cannot tell what was used.
    `insufficiently_specified` is correct here and S91's hand-read found 10 rows
    of this shape in one batch window, so the distinction is not rare.
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


def own_code(ids, repos, ev_ids, what="the code that generated the results", pinned="unpinned_url"):
    """P2: the authors' own code, publicly released, with a resolvable id."""
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


def code_unreachable(ev_ids, what="the code that generated the results"):
    """P2 + P4: the authors' own code, but the pointer does not resolve."""
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


def third_party_code(ev_ids, name, what="the results"):
    """P2 boundary: the statement names somebody else's software."""
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


def promise(ev_ids, what="the data", due=True):
    """P5: a future-tense promise with no current route and no identifier."""
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
    ev_ids, what="the data", restriction="none_given", dt=None, dt_r=None, gran=None, gran_r=None
):
    """P9: an on-request route, with or without a restriction attached.

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
    ev_ids,
    what,
    restriction,
    why,
    dt="real",
    gran="unclear",
    gran_r="not characterised in the retrieved text",
):
    """A statement that refuses a deposit outright and gives its reason.

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


def tool_no_results(name="the released package"):
    """P8: the contribution is the tool; no results are reported alongside."""
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
