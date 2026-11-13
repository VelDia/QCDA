"""dap.verify — does the declared access path actually work?

This module is the only thing permitted to set `Actual`, and the only thing
that can promote a case to `access_unresolvable`. The classifier is explicitly
forbidden from asserting verification it did not perform; that separation is
what makes declared-vs-actual a real finding rather than a restatement.

Why v1 found nothing here: it sent a bare HEAD with no User-Agent, mapped 403
straight to "requires_authentication", and treated any 200 as success. So
`inaccessible` occurred zero times across 1,290 papers and the access-failure
column was populated in four rows — the pipeline structurally could not observe
the failure mode the study is about.

Three fixes:
  * HEAD, then GET on anything that isn't a clean 200 (many repositories refuse
    HEAD outright, and a refused HEAD is not a broken link).
  * A real browser-ish User-Agent, because some repository CDNs 403 unknown
    agents and that is not the same as authentication.
  * Landing-page inspection: a DOI that resolves to a "this dataset has been
    withdrawn" or an empty record still returns 200. `resolves_no_data`
    separates "the link works" from "the data is there", which is precisely the
    distinction Case 5 rests on.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, replace
from typing import Iterable, Optional
from urllib.parse import urlparse, urlsplit, urlunsplit

import requests

from .schema import Actual, Case, Pinned
from .sources import LIMITER, Config

#: No contact address: this goes to every repository and dataset host a paper
#: cites, none of which asks for one.
_UA = "Mozilla/5.0 (compatible; data-availability-pipeline/2.3) requests"

#: Text that means "resolved, but there is no data here".
_EMPTY_RECORD_MARKERS = (
    "has been withdrawn",
    "has been retracted",
    "record not found",
    "this item has been removed",
    "no files",
    "0 files",
    "dataset is no longer available",
    "embargoed until",
    "page not found",
    "does not exist",
    "restricted access",
)

_AUTH_MARKERS = (
    "sign in",
    "log in to continue",
    "institutional login",
    "purchase access",
    "get access",
    "subscribe to view",
)

#: Hosts that answer an automated request with 404 whether or not the record
#: exists. Measured, not guessed: all 15 `kaggle.com` identifiers in the corpus
#: came back 404, including `mlg-ulb/creditcardfraud` (13,517 upvotes),
#: `blastchar/telco-customer-churn` (3,896) and `Cornell-University/arxiv`,
#: every one of which loads in a browser. A 404 from such a host does not
#: establish that the path fails to deliver, which is the only thing that may
#: promote a case to `access_unresolvable` (hard rule 6), so it is recorded as
#: ERROR -- a check that did not complete -- and the declared case stands.
#:
#: This is NOT a bot-detection bypass (hard rule 8): the request is unchanged
#: and the failure is logged. It only stops us reading a soft block as proof of
#: absence. A host belongs here when its 404s have been shown to be
#: indiscriminate, never on suspicion.
_SOFT_404_HOSTS = frozenset({"kaggle.com", "www.kaggle.com"})

#: Hosts whose 403 is an intermittent block on automated requests rather than a
#: credential requirement. Measured, not guessed: of 30 `gitlab` identifiers in
#: the two databases, 26 are recorded `accessible` 200 and one
#: `auth_required` 403. Re-checking three of them on 2026-10-03 -- including
#: two of the recorded 200s, `gitlab.com/plut0n/bcert` and
#: `gitlab.com/qjornet/ted-qc.git` -- returned 403 for every one, while a
#: browser loaded the public project page with no login. So the same URL is
#: 200 or 403 depending on when it is asked, and neither the `.git` suffix nor
#: the particular project decides it.
#:
#: A status that varies with the hour does not establish that the path fails to
#: deliver, which is the only thing that may promote a case (hard rule 6), so a
#: 403 from such a host is recorded as ERROR -- a check that did not complete --
#: and the declared case stands. Without this, a reverify that happens to run
#: during a block demotes every gitlab row to `access_unresolvable` at once.
#:
#: This is NOT a bot-detection bypass (hard rule 8): the request is unchanged
#: and the failure is logged. A host belongs here only once its 403s have been
#: shown to come and go on URLs known to be public; a genuine credential wall,
#: such as a publisher's supplementary-material endpoint, must keep promoting.
_SOFT_403_HOSTS = frozenset({"gitlab.com", "www.gitlab.com"})


def _soft_404(host: str) -> bool:
    return host in _SOFT_404_HOSTS


def _soft_403(host: str) -> bool:
    return host in _SOFT_403_HOSTS


@dataclass
class VerificationResult:
    identifier: str
    status: Actual
    http_status: Optional[int] = None
    final_url: str = ""
    note: str = ""


def _pinning_of(identifier: str) -> Pinned:
    s = identifier.lower()
    if re.search(r"github\.com/.+/(commit|tree|releases/tag)/[0-9a-z._-]+", s):
        return Pinned.COMMIT_PINNED
    if re.search(r"zenodo\.org/record(s)?/\d+", s) or re.match(r"10\.5281/zenodo\.\d+", s):
        return Pinned.DOI_VERSIONED
    if s.startswith("10.") or "doi.org/" in s:
        return Pinned.DOI_VERSIONED
    if s.startswith("http"):
        return Pinned.UNPINNED_URL
    return Pinned.NONE


def classify_pinning(identifiers: Iterable[str]) -> Pinned:
    """Most specific pinning across all identifiers for an axis."""
    best = Pinned.NONE
    order = {
        Pinned.NONE: 0,
        Pinned.UNPINNED_URL: 1,
        Pinned.DOI_VERSIONED: 2,
        Pinned.COMMIT_PINNED: 3,
    }
    for ident in identifiers:
        p = _pinning_of(ident)
        if order[p] > order[best]:
            best = p
    return best


#: A bare host, as papers routinely write them: "github.com/org/repo",
#: "zenodo.org/record/123", "www.example.org/data". Requires a dot-separated
#: domain with a real TLD, so prose is not mistaken for a URL.
_BARE_HOST = re.compile(r"^[a-z0-9][a-z0-9.\-]*\.[a-z]{2,}(?:[/:?#]|$)", re.I)


def _normalise(identifier: str) -> str:
    """Turn what a paper wrote into something fetchable, or "" if it is not a
    locator at all.

    A scheme-less URL is not an unverifiable identifier — it is the *normal* way
    papers cite a repository. Treating it as unresolvable meant a paper coded
    `openly_accessible` was never checked at all, which is precisely the silent
    pass this module exists to prevent.

    A space in the PATH is kept, for `_space_repairs` to work on after a 404:
    some PDFs' text layers render `_` as a space, so
    `github.com/BOHRTECHNOLOGY/quantum tsp` is a repository address, not prose.
    Refusing it returned "" here, which made it NOT_APPLICABLE -- the same
    silent pass, one level down. A space before the first slash still refuses,
    so "github.com: code in the repository/src" does not become a URL.
    """
    s = identifier.strip().rstrip(".,;)")
    if not s:
        return ""
    if s.startswith("10."):
        return f"https://doi.org/{s}"
    low = s.lower()
    if low.startswith("doi:"):
        rest = s[4:].strip()
        return f"https://doi.org/{rest}" if rest.startswith("10.") else ""
    if s.startswith(("http://", "https://")):
        return s
    if _BARE_HOST.match(s) and " " not in s.split("/", 1)[0]:
        return f"https://{s}"
    return ""


#: A repair is attempted for at most this many gaps in a path. Each gap doubles
#: the requests, and past three the candidates stop being a reading of the
#: text and become a search.
_MAX_REPAIR_GAPS = 3


def _space_repairs(url: str) -> list[tuple[str, tuple[str, ...]]]:
    """Candidate repairs of a URL whose path lost characters to spaces, as
    (candidate_url, fills), all-underscore first.

    Measured in the batch 210 export: PDF text extraction rendered
    `DWAVE4TSP_ClusteringMultitasking` as `DWAVE4TSP Clustering Multitasking`.
    One underscore produced TWO spaces, and `github.com/cda-tum/qcec and` is a
    space that genuinely ends the URL. Three different originals behind one
    character in the text, so no substitution rule on the string is correct:
    each gap is independently `_` or nothing, and only a fetch can tell which.
    The "nothing" option also de-hyphenates (`TNO- Quantum` -> `TNO-Quantum`).
    Cutting at the space is not a candidate: a cut inside `owner/repo` lands
    on the owner's page, which answers 200 for any repository name.

    A run of spaces is one gap. Only the path is touched, never the host, the
    query or the fragment. Above `_MAX_REPAIR_GAPS` gaps nothing is generated.
    """
    parts = urlsplit(url)
    pieces = re.split(r" +", parts.path)
    gaps = len(pieces) - 1
    if not gaps or gaps > _MAX_REPAIR_GAPS:
        return []
    out = []
    for fills in itertools.product(("_", ""), repeat=gaps):
        path = pieces[0] + "".join(f + p for f, p in zip(fills, pieces[1:]))
        out.append((urlunsplit(parts._replace(path=path)), fills))
    return out


def _repair(
    cfg: Config, identifier: str, url: str, original: VerificationResult
) -> VerificationResult:
    """After a hard 404 on a path with spaces, try each `_space_repairs`
    candidate and use one only if it is the only one that exists.

    Three outcomes, and the middle one is the reason this function exists
    rather than a substitution rule: BOHRTECHNOLOGY has two repositories whose
    names start with `quantum`, so a path the text leaves open can resolve more
    than one way.
      * exactly one candidate exists -> its own result, with `final_url` the
        repaired address and `note` the audit string. `identifier` stays the
        paper's verbatim string (schema.py: "DOIs / URLs, verbatim"); a repair
        is morally a redirect, which is what `final_url` is for.
      * more than one exists -> AMBIGUOUS: the original NOT_FOUND stands and
        the note says so. Nothing is picked.
      * none exists -> the original result, untouched.
    A candidate whose check did not complete (ERROR) could be a second match,
    so one success beside it is not "exactly one" either.

    Only on NOT_FOUND: a 403, a paywall or an error means the path is probably
    right. Never on a soft-404 host, where a 404 is not evidence (and becomes
    ERROR before it reaches here).
    """
    if original.status is not Actual.NOT_FOUND:
        return original
    if _soft_404((urlsplit(url).hostname or "").lower()):
        return original
    candidates = _space_repairs(url)
    if not candidates:
        return original

    found, incomplete = [], 0
    for cand, fills in candidates:
        res = _check(cfg, identifier, cand)
        if res.status is Actual.NOT_FOUND:
            continue
        if res.status is Actual.ERROR:
            incomplete += 1
            continue
        found.append((cand, fills, res))

    tried = f"{len(found)} of {len(candidates)} candidates resolve"
    if len(found) == 1 and not incomplete:
        cand, fills, res = found[0]
        how = ", ".join("space->underscore" if f else "space->nothing" for f in fills)
        note = f"repaired: {how} ({tried})"
        if res.final_url and res.final_url != cand:
            # GitHub forwards renamed repositories; without the candidate, a
            # repair reads as a rename (`TNO-Quantum/qscore_dwave` -> `/qscore`)
            note += f", via {cand}"
        return replace(
            res, final_url=res.final_url or cand, note=f"{note}; {res.note}" if res.note else note
        )
    if len(found) > 1:
        return replace(
            original,
            note=(f"ambiguous repair, not picked: {tried}: " + ", ".join(c for c, _, _ in found)),
        )
    if incomplete:
        return replace(
            original, note=(f"repair not established: {tried}, {incomplete} could not be checked")
        )
    return original


def verify_identifier(cfg: Config, identifier: str) -> VerificationResult:
    url = _normalise(identifier)
    if not url:
        return VerificationResult(
            identifier, Actual.NOT_APPLICABLE, note="not a resolvable identifier"
        )
    return _repair(cfg, identifier, url, _check(cfg, identifier, url))


def _check(cfg: Config, identifier: str, url: str) -> VerificationResult:
    """One fetch of `url`, reported against the paper's `identifier`."""
    headers = {"User-Agent": _UA, "Accept": "text/html,application/xhtml+xml,*/*"}
    host = (urlparse(url).hostname or "").lower()
    if host in cfg.tdm_headers:
        headers.update(cfg.tdm_headers[host])

    try:
        LIMITER.wait(url)
        r = requests.head(url, headers=headers, timeout=cfg.timeout, allow_redirects=True)
        # Repositories frequently refuse HEAD; that is not evidence of anything.
        if r.status_code in (403, 405, 501) or r.status_code >= 500:
            LIMITER.wait(url)
            r = requests.get(
                url, headers=headers, timeout=cfg.timeout, allow_redirects=True, stream=True
            )
    except requests.RequestException as e:
        return VerificationResult(identifier, Actual.ERROR, note=type(e).__name__)

    code, final = r.status_code, r.url

    if code == 404:
        if _soft_404(host):
            return VerificationResult(
                identifier,
                Actual.ERROR,
                code,
                final,
                note="404 from a host that refuses automated requests "
                "indiscriminately; existence not established",
            )
        return VerificationResult(identifier, Actual.NOT_FOUND, code, final)
    if code in (401, 407):
        return VerificationResult(identifier, Actual.AUTH_REQUIRED, code, final)
    if code == 402:
        return VerificationResult(identifier, Actual.PAYWALLED, code, final)
    if code == 403:
        if _soft_403(host):
            return VerificationResult(
                identifier,
                Actual.ERROR,
                code,
                final,
                note="403 from a host that blocks automated requests "
                "intermittently; access requirement not established",
            )
        return VerificationResult(
            identifier, Actual.AUTH_REQUIRED, code, final, note="403 after GET fallback"
        )
    if code >= 400:
        return VerificationResult(identifier, Actual.ERROR, code, final)

    # 200 — but does the page actually carry the data?
    body = ""
    try:
        ctype = (r.headers.get("Content-Type") or "").lower()
        if "html" in ctype:
            body = (r.text if hasattr(r, "text") else "")[:20000].lower()
    except Exception:
        body = ""

    if body:
        if any(m in body for m in _EMPTY_RECORD_MARKERS):
            return VerificationResult(
                identifier,
                Actual.RESOLVES_NO_DATA,
                code,
                final,
                note="landing page reports no content",
            )
        if any(m in body for m in _AUTH_MARKERS) and "download" not in body:
            return VerificationResult(
                identifier, Actual.PAYWALLED, code, final, note="landing page gates access"
            )

    return VerificationResult(identifier, Actual.ACCESSIBLE, code, final)


def aggregate(results: list[VerificationResult]) -> Actual:
    """Worst-case wins: one broken link in a set makes the claim unreliable."""
    if not results:
        return Actual.NOT_ATTEMPTED
    severity = [
        Actual.NOT_FOUND,
        Actual.RESOLVES_NO_DATA,
        Actual.PAYWALLED,
        Actual.AUTH_REQUIRED,
        Actual.ERROR,
        Actual.ACCESSIBLE,
        Actual.NOT_APPLICABLE,
        Actual.NOT_ATTEMPTED,
    ]
    considered = [r.status for r in results if r.status is not Actual.NOT_APPLICABLE]
    if not considered:
        return Actual.NOT_APPLICABLE
    for s in severity:
        if s in considered:
            return s
    return Actual.NOT_ATTEMPTED


def derive_case(declared_case: Case, actual: Actual) -> tuple[Case, str]:
    """Apply the one rule the model is not allowed to apply itself.

    `access_unresolvable` means a specific path was promised and did not hold.
    That is a fact about the world, established here, not a judgement the
    classifier may make from the text. Returns (case, note).

    `Actual.ERROR` is deliberately NOT a promotion. A connect timeout, a DNS
    failure or a 503 says the check did not complete, not that the path is
    broken: an institutional GitLab behind a firewall and a decommissioned one
    produce the same error. Promoting on it asserts "promised and broken" from
    evidence that only supports "we could not reach it", and inflates the
    broken-link rate in one direction. 19 verification rows in the three
    databases carry `error`; 8 of them have no HTTP status at all. The outcome
    is still recorded in `actual`, which is what the separate token is for.
    """
    broken = (Actual.NOT_FOUND, Actual.RESOLVES_NO_DATA)
    if declared_case is Case.OPENLY_ACCESSIBLE and actual in broken:
        return Case.ACCESS_UNRESOLVABLE, (
            f"promoted from openly_accessible: verification returned {actual.value}"
        )
    if declared_case is Case.OPENLY_ACCESSIBLE and actual in (
        Actual.PAYWALLED,
        Actual.AUTH_REQUIRED,
    ):
        return Case.ACCESS_UNRESOLVABLE, (
            f"promoted from openly_accessible: path resolves but gates access "
            f"({actual.value}); 'openly accessible' requires access without a "
            f"credential"
        )
    return declared_case, ""


__all__ = [
    "VerificationResult",
    "verify_identifier",
    "aggregate",
    "derive_case",
    "classify_pinning",
]
