"""dap.sources — metadata-first acquisition.

Order matters here, and it is the opposite of the v1 script's order.

v1 went straight for full text, which meant every paper cost a PDF download and
a parse, publisher bot-walls became hard failures (69 of its 124 logged
failures), and a Cloudflare challenge page returning HTTP 200 was parsed as
article text and classified.

This module asks the metadata first. For a data-availability study a large share
of papers can be settled from Crossref alone:

  * `assertion[]`  — free-form publisher-deposited statements. Several
                     publishers deposit the Data Availability Statement here.
                     There is no fixed key for it, so we pattern-match on the
                     assertion's name/label rather than assuming one.
  * `relation`     — structural dataset links (`references`,
                     `is-supplemented-by`). These are asserted by the publisher,
                     not parsed out of prose, so they are the highest-quality
                     signal available.
  * `reference[]`  — cited DOIs; those registered with DataCite rather than
                     Crossref are datasets/software, which is a reliable way to
                     find Axis A material.
  * `link[]`       — full text deposited for TDM, flagged
                     `intended-application: "text-mining"`. This is the
                     *sanctioned* full-text route. Presence does not guarantee
                     access: for subscription content it needs the
                     institution's TDM credentials, which slot into
                     TDM_HEADERS below.

Only when metadata is silent do we pay for full text.
"""

from __future__ import annotations

import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import unquote, urlparse

import requests

from .redact import redact
from .schema import RepoType, TextSource


# ---------------------------------------------------------------------------
# Polite, per-host rate limiting. One bucket per hostname, shared across threads.
# ---------------------------------------------------------------------------
class _HostLimiter:
    """Polite rate limiting: one bucket per hostname, shared across threads."""

    def __init__(self, default_qps: float = 3.0) -> None:
        """Set up the per-host rates.

        Args:
            default_qps: The requests per second allowed to a host that has no
                rate of its own.
        """
        self._default = default_qps
        self._rates = {
            "api.crossref.org": 20.0,  # generous with a mailto (polite pool)
            "api.openalex.org": 8.0,
            "api.unpaywall.org": 8.0,
            "api.datacite.org": 5.0,
            "export.arxiv.org": 0.3,  # arXiv asks for ~1 req / 3 s
            "arxiv.org": 0.3,
        }
        self._next: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, url: str) -> None:
        """Block until the next request to a URL's host is due.

        Args:
            url: The URL about to be requested; only its hostname matters.
        """
        host = (urlparse(url).hostname or "").lower()
        interval = 1.0 / self._rates.get(host, self._default)
        with self._lock:
            now = time.monotonic()
            due = self._next.get(host, 0.0)
            sleep_for = max(0.0, due - now)
            self._next[host] = max(now, due) + interval
        if sleep_for:
            time.sleep(sleep_for)


LIMITER = _HostLimiter()


class TransientError(RuntimeError):
    """Retryable: timeout, 429, 5xx. The caller should back off, not give up."""


class PermanentError(RuntimeError):
    """Not retryable: 404, malformed response, unsupported content."""


@dataclass
class Config:
    """The settings of one pipeline run.

    Attributes:
        contact_email: Optional contact address. Sent ONLY to Crossref
            (`mailto=`, its polite pool) and to Unpaywall (`email=`, which
            its terms require). Never in the User-Agent, never to any other
            host, never stored (see redact.py). Without it, Crossref is
            queried in its public pool and Unpaywall is skipped.
        openalex_key: The OpenAlex API key; empty to send none.
        core_key: The CORE API key; empty to send none.
        tdm_headers: Publisher TDM credentials, keyed by hostname, e.g.
            `{"api.elsevier.com": {"X-ELS-APIKey": "..."}}`.
        timeout: The request timeout in seconds.
        max_retries: How many times a request is attempted.
        grobid_url: Optional GROBID service. When set, PDFs are parsed there
            instead of locally: far better text and real section structure,
            but a network call and an order of magnitude slower.
        grobid_timeout: The timeout of a GROBID request in seconds.
        pdf_dir: A directory of PDFs fetched by hand, for papers behind a bot
            wall the OA cascade is not allowed to work around. Files are
            named after the DOI with "/" written as "_", or after the
            OpenAlex id.
        deny_hosts: Hosts never to fetch from, lowercased.
    """

    #: Optional. Sent ONLY to Crossref (`mailto=`, its polite pool) and to
    #: Unpaywall (`email=`, which its terms require). Never in the User-Agent,
    #: never to any other host, never stored (see redact.py). Without it,
    #: Crossref is queried in its public pool and Unpaywall is skipped.
    contact_email: str = ""
    openalex_key: str = ""
    core_key: str = ""
    #: Publisher TDM credentials, keyed by hostname, e.g.
    #: {"api.elsevier.com": {"X-ELS-APIKey": "..."}}
    tdm_headers: dict[str, dict[str, str]] = field(default_factory=dict)
    timeout: int = 30
    max_retries: int = 4
    #: Optional GROBID service. When set, PDFs are parsed there instead of
    #: locally: far better text and real section structure, but a network call
    #: and an order of magnitude slower. Use it for the subset that will be
    #: classified, not for a census pass.
    grobid_url: str = ""
    grobid_timeout: int = 300
    #: A directory of PDFs fetched by hand, for papers behind a bot wall the OA
    #: cascade is not allowed to work around. Rule 8 forbids bypassing those
    #: walls; it does not forbid a person with a library subscription
    #: downloading the paper and putting it here. Files are named after the DOI
    #: with "/" written as "_", or after the OpenAlex id: 10.1088/x.y ->
    #: `10.1088_x.y.pdf`, or `W2911423727.pdf`.
    pdf_dir: str = ""
    #: Hosts never to fetch from, lowercased, from DAP_DENY_HOSTS. A deliberate
    #: blunt instrument: it discards every copy a host holds, good and bad.
    #: Prefer `_CATALOGUE_PATHS` when a host serves both.
    deny_hosts: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        """Normalise the contact address and reject one that is not an address.

        Raises:
            ValueError: If `contact_email` is set but contains no `@`.
        """
        self.contact_email = (self.contact_email or "").strip()
        if self.contact_email and "@" not in self.contact_email:
            raise ValueError("contact_email must be an email address, or empty")

    @property
    def user_agent(self) -> str:
        """The User-Agent sent to every host; it carries no contact address."""
        # No address here: the User-Agent goes to every host the pipeline
        # touches, including every publisher and repository a paper cites.
        return "data-availability-pipeline/2.3 (bibliometric research)"


def _get(
    cfg: Config,
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    stream: bool = False,
) -> requests.Response:
    """GET a URL with per-host rate limiting and bounded exponential backoff.

    v1 had no retry at all, so a single transient 429 from OpenAlex turned into
    a permanent failure row that was re-attempted on every subsequent run and
    re-logged every time, growing failures.csv without bound.

    Args:
        cfg: The run configuration: timeout, retries and per-host TDM headers.
        url: The URL to fetch.
        params: The query parameters, or None.
        headers: Extra request headers; they override the defaults.
        stream: Whether to stream the response body.

    Returns:
        The response, whose status is always 200.

    Raises:
        PermanentError: If the server answers with a status that a retry
            cannot change, such as 404.
        TransientError: If every attempt timed out, was throttled or met a
            server error. The message is redacted.
    """
    h = {"User-Agent": cfg.user_agent, "Accept": "application/json"}
    host = (urlparse(url).hostname or "").lower()
    if host in cfg.tdm_headers:
        h.update(cfg.tdm_headers[host])
    if headers:
        h.update(headers)

    last: Exception | None = None
    for attempt in range(cfg.max_retries):
        LIMITER.wait(url)
        try:
            r = requests.get(
                url,
                params=params,
                headers=h,
                timeout=cfg.timeout,
                stream=stream,
                allow_redirects=True,
            )
        except requests.RequestException as e:
            # The exception text carries the URL and its query string.
            last = TransientError(redact(f"{type(e).__name__}: {e}"))
        else:
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                last = TransientError(f"HTTP {r.status_code}")
                retry_after = r.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    time.sleep(min(int(retry_after), 60))
                    continue
            elif r.status_code == 404:
                raise PermanentError("HTTP 404")
            else:
                raise PermanentError(f"HTTP {r.status_code}")
        time.sleep(min(2**attempt, 30))
    raise TransientError(f"exhausted {cfg.max_retries} attempts: {last}")


# ---------------------------------------------------------------------------
# Crossref
# ---------------------------------------------------------------------------
CROSSREF_WORKS = "https://api.crossref.org/works/{doi}"

#: Patterns matched against an assertion's `name` and `label`. Publishers use
#: wildly inconsistent wording here, which is exactly why this is a pattern list
#: and not a key lookup.
_DAS_NAME_PATTERNS = re.compile(
    r"data[\s_-]*(availability|access|sharing|statement)"
    r"|availability[\s_-]*of[\s_-]*data"
    r"|code[\s_-]*availability"
    r"|supporting[\s_-]*information"
    r"|research[\s_-]*data[\s_-]*policy",
    re.I,
)

_DATASET_RELATION_TYPES = {
    "references",
    "is-supplemented-by",
    "issupplementedby",
    "is-supplement-to",
    "issupplementto",
    "has-related-material",
    "is-derived-from",
    "isderivedfrom",
    "is-source-of",
}

#: DataCite spells relation types in CamelCase ("IsSupplementTo") where Crossref
#: uses dashes ("is-supplement-to"). Split on the case boundary so one set of
#: relation names serves both agencies.
_CAMEL_TO_DASH = re.compile(r"([a-z0-9])([A-Z])")


def _normalise_doi(doi: str) -> str:
    """Strip resolver prefixes so a bare DOI reaches the API path.

    Args:
        doi: A DOI, bare or as a `doi.org` URL. None is read as empty.

    Returns:
        The bare DOI.
    """
    return (
        (doi or "")
        .strip()
        .replace("https://doi.org/", "")
        .replace("http://dx.doi.org/", "")
        .replace("https://dx.doi.org/", "")
        .replace("http://doi.org/", "")
    )


@dataclass
class CrossrefFacts:
    """Everything the metadata layer could establish without reading the paper.

    Attributes:
        found: Whether a registry knows the DOI.
        das_texts: The data-availability statements the publisher deposited.
        dataset_ids: Dataset DOIs from structural relations.
        dataset_refs: Cited DOIs from the reference list, lowercased;
            dataset candidates until confirmed against DataCite.
        tdm_links: Full-text links deposited for text mining, each with
            `url`, `content_type` and `version`.
        license_urls: The licence URLs deposited.
        container: The journal or other container title.
        published_year: The publication year, or None if unknown.
        type: The registry's work type.
        authors: Author family names, lowercased, in deposit order. Collected
            so a title-based search can be checked against them.
        reference_count: How many references the publisher deposited. The
            retrieved text should contain a bibliography of roughly this size.
    """

    found: bool = False
    das_texts: list[str] = field(default_factory=list)
    dataset_ids: list[str] = field(default_factory=list)  # DOIs from relations
    dataset_refs: list[str] = field(default_factory=list)  # DOIs from references
    tdm_links: list[dict[str, str]] = field(default_factory=list)
    license_urls: list[str] = field(default_factory=list)
    container: str = ""
    published_year: Optional[int] = None
    type: str = ""
    #: Author family names, lowercased, in deposit order. Collected so a
    #: title-based search can be checked against them — a title alone is not
    #: enough to identify a paper (see `arxiv_search`).
    authors: list[str] = field(default_factory=list)
    #: How many references the publisher deposited. The retrieved text should
    #: contain a bibliography of roughly this size; if it contains none, the
    #: extraction stopped early and another source should be tried.
    reference_count: int = 0

    @property
    def has_das(self) -> bool:
        """Whether any deposited data-availability statement is non-empty."""
        return any(t.strip() for t in self.das_texts)

    @property
    def has_structured_dataset_link(self) -> bool:
        """Whether the publisher asserted a structural link to a dataset."""
        return bool(self.dataset_ids)


def crossref_facts(cfg: Config, doi: str) -> CrossrefFacts:
    """Collect what Crossref knows about a paper.

    `TransientError` propagates when Crossref cannot be reached.

    Args:
        cfg: The run configuration.
        doi: The paper's DOI, bare or as a URL.

    Returns:
        The facts found. `found` is False if the DOI is empty or Crossref
        does not know it.
    """
    if not doi:
        return CrossrefFacts(found=False)
    doi = _normalise_doi(doi)
    try:
        r = _get(
            cfg,
            CROSSREF_WORKS.format(doi=doi),
            params={"mailto": cfg.contact_email} if cfg.contact_email else None,
        )
    except PermanentError:
        return CrossrefFacts(found=False)
    msg = (r.json() or {}).get("message") or {}

    facts = CrossrefFacts(found=True)
    facts.reference_count = int(msg.get("reference-count") or 0)
    facts.type = msg.get("type", "")
    facts.container = (msg.get("container-title") or [""])[0]
    dp = (msg.get("published") or {}).get("date-parts") or [[None]]
    facts.published_year = dp[0][0] if dp and dp[0] else None
    for a in msg.get("author") or []:
        fam = str(a.get("family") or "").strip().lower()
        if fam:
            facts.authors.append(fam)

    # --- DAS text from assertions -------------------------------------------
    for a in msg.get("assertion") or []:
        name = str(a.get("name", ""))
        label = str(a.get("label", ""))
        group = a.get("group") or {}
        gname = str(group.get("name", "")) + " " + str(group.get("label", ""))
        if _DAS_NAME_PATTERNS.search(f"{name} {label} {gname}"):
            val = str(a.get("value", "")).strip()
            if val:
                facts.das_texts.append(val)

    # --- structural dataset links from `relation` ---------------------------
    # relation is a hashmap: {relation-name: [{id-type, id, asserted-by}, ...]}
    for rel_name, entries in (msg.get("relation") or {}).items():
        if rel_name.lower().replace("_", "-") not in _DATASET_RELATION_TYPES:
            continue
        for e in entries or []:
            ident = str(e.get("id", "")).strip()
            if ident:
                facts.dataset_ids.append(ident)

    # --- cited DOIs (dataset candidates, confirmed later against DataCite) ---
    for ref in msg.get("reference") or []:
        d = (ref.get("DOI") or "").strip()
        if d:
            facts.dataset_refs.append(d.lower())

    # --- TDM full-text links -------------------------------------------------
    for ln in msg.get("link") or []:
        if str(ln.get("intended-application", "")).lower() == "text-mining":
            facts.tdm_links.append(
                {
                    "url": ln.get("URL", ""),
                    "content_type": ln.get("content-type", ""),
                    "version": ln.get("content-version", ""),
                }
            )

    for lic in msg.get("license") or []:
        if lic.get("URL"):
            facts.license_urls.append(lic["URL"])

    return facts


# ---------------------------------------------------------------------------
# DataCite — confirms which cited DOIs are actually datasets/software
# ---------------------------------------------------------------------------
_DATACITE = "https://api.datacite.org/dois/{doi}"

_REPO_BY_HOST = [
    ("zenodo.org", RepoType.ZENODO),
    ("figshare.com", RepoType.FIGSHARE),
    ("datadryad.org", RepoType.DRYAD),
    ("dryad", RepoType.DRYAD),
    ("osf.io", RepoType.OSF),
    ("github.com", RepoType.GITHUB),
    ("gitlab.", RepoType.GITLAB),
    ("huggingface.co", RepoType.HUGGINGFACE),
    ("kaggle.com", RepoType.KAGGLE),
    ("openml.org", RepoType.OPENML),
    ("paperswithcode.com", RepoType.PAPERSWITHCODE),
    ("ieee-dataport.org", RepoType.IEEE_DATAPORT),
    ("materialscloud.org", RepoType.MATERIALS_CLOUD),
    ("s3.amazonaws.com", RepoType.CLOUD_BUCKET),
    ("storage.googleapis.com", RepoType.CLOUD_BUCKET),
    ("blob.core.windows.net", RepoType.CLOUD_BUCKET),
]


def repo_type_for(url_or_doi: str) -> RepoType:
    """Classify a repository from its URL or DOI.

    Deterministic: the model is never asked to guess a repository type it can
    get wrong.

    Args:
        url_or_doi: A URL or a DOI. None is read as empty.

    Returns:
        The repository type; OTHER for an unrecognised DOI and UNKNOWN for
        anything else unrecognised.
    """
    s = (url_or_doi or "").lower()
    for needle, rt in _REPO_BY_HOST:
        if needle in s:
            return rt
    if s.startswith("10.5281/") or "zenodo" in s:
        return RepoType.ZENODO
    if s.startswith("10.6084/"):
        return RepoType.FIGSHARE
    if s.startswith("10.5061/"):
        return RepoType.DRYAD
    if s.startswith("10."):
        return RepoType.OTHER
    return RepoType.UNKNOWN


def datacite_facts(cfg: Config, doi: str) -> CrossrefFacts:
    """Collect the same facts as `crossref_facts` for a DataCite DOI.

    Not every paper DOI is a Crossref DOI. Institutional repositories, theses
    and preprint servers register with DataCite instead, and Crossref returns
    404 for them. Treating that 404 as "this paper has no metadata" discards a
    paper that is fully described at the other agency.

    Args:
        cfg: The run configuration.
        doi: The paper's DOI, bare or as a URL.

    Returns:
        The facts found. `found` is False if the DOI is empty, DataCite does
        not know it, or DataCite could not be reached.
    """
    if not doi:
        return CrossrefFacts(found=False)
    doi = _normalise_doi(doi)
    try:
        r = _get(cfg, _DATACITE.format(doi=doi.lower()))
    except (PermanentError, TransientError):
        return CrossrefFacts(found=False)
    attrs = ((r.json() or {}).get("data") or {}).get("attributes") or {}
    if not attrs:
        return CrossrefFacts(found=False)

    facts = CrossrefFacts(found=True)
    facts.reference_count = sum(
        1
        for r in (attrs.get("relatedIdentifiers") or [])
        if str(r.get("relationType", "")).lower() in ("cites", "references")
    )
    facts.type = (attrs.get("types") or {}).get("resourceTypeGeneral") or ""
    facts.container = attrs.get("publisher") or ""
    year = attrs.get("publicationYear")
    facts.published_year = int(year) if str(year).isdigit() else None
    for c in attrs.get("creators") or []:
        fam = str(c.get("familyName") or "").strip().lower()
        if not fam and c.get("name"):
            # DataCite often carries only "Family, Given" in a single field.
            fam = str(c["name"]).split(",")[0].strip().lower()
        if fam:
            facts.authors.append(fam)

    # --- DAS text from descriptions -----------------------------------------
    # Only descriptions whose *type* names a data statement. An abstract is not
    # a data-availability statement, and feeding one in as though it were would
    # put a fabricated DAS in front of the classifier.
    for d in attrs.get("descriptions") or []:
        dtype = str(d.get("descriptionType", ""))
        if _DAS_NAME_PATTERNS.search(dtype):
            val = str(d.get("description", "")).strip()
            if val:
                facts.das_texts.append(val)

    # --- related identifiers: structural dataset links and cited DOIs -------
    for rel in attrs.get("relatedIdentifiers") or []:
        if str(rel.get("relatedIdentifierType", "")).upper() != "DOI":
            continue
        ident = str(rel.get("relatedIdentifier", "")).strip()
        if not ident:
            continue
        rtype = str(rel.get("relationType", ""))
        norm = _CAMEL_TO_DASH.sub(r"\1-\2", rtype).lower()
        if norm in _DATASET_RELATION_TYPES or rtype.lower() in _DATASET_RELATION_TYPES:
            facts.dataset_ids.append(ident)
        elif norm in ("cites", "is-cited-by"):
            facts.dataset_refs.append(ident.lower())

    for rights in attrs.get("rightsList") or []:
        if rights.get("rightsUri"):
            facts.license_urls.append(rights["rightsUri"])

    return facts


def registry_facts(cfg: Config, doi: str) -> CrossrefFacts:
    """Collect a paper's metadata facts: Crossref first, DataCite as fallback.

    DOIs live at one agency or the other; asking only Crossref silently drops
    everything registered elsewhere.

    Args:
        cfg: The run configuration.
        doi: The paper's DOI, bare or as a URL.

    Returns:
        The facts from whichever registry knows the DOI; `found` is False if
        neither does.
    """
    facts = crossref_facts(cfg, doi)
    if facts.found or not doi:
        return facts
    return datacite_facts(cfg, doi)


_DATACITE_BULK = "https://api.datacite.org/dois"

#: DataCite resource types that count as a data/software artefact.
_DATASET_TYPES = ("dataset", "software", "collection", "model")

#: Resolved DOIs, shared across threads and papers. Value is the attributes dict
#: for a dataset, or None for "looked it up, not a dataset". References repeat
#: heavily across a field's corpus, so the hit rate is high; the cap keeps a
#: census-sized run from growing this without bound.
_DC_CACHE_MAX = 200_000
_dc_cache: dict[str, Optional[dict[str, Any]]] = {}
_dc_lock = threading.Lock()


def _datacite_chunk(cfg: Config, dois: list[str]) -> dict[str, dict[str, Any]]:
    """Look up several DOIs at DataCite in one bulk query.

    DataCite's Lucene `doi:` filter accepts a disjunction, so N DOIs cost one
    request instead of N.

    Args:
        cfg: The run configuration.
        dois: The bare, lowercased DOIs to look up.

    Returns:
        The DataCite attributes of each DOI found, by lowercased DOI.
    """
    q = " OR ".join(f'"{d}"' for d in dois)
    r = _get(cfg, _DATACITE_BULK, params={"query": f"doi:({q})", "page[size]": 1000})
    out: dict[str, dict[str, Any]] = {}
    for item in (r.json() or {}).get("data") or []:
        did = str(item.get("id", "")).strip().lower()
        if did:
            out[did] = item.get("attributes") or {}
    return out


def datacite_datasets(
    cfg: Config, dois: list[str], *, chunk: int = 50
) -> dict[str, dict[str, Any]]:
    """Find which DOIs are DataCite dataset or software records.

    Bulk and cached. The per-DOI endpoint made this the dominant cost of the
    metadata stage: one request per cited DOI, serialised through a 5 qps
    per-host bucket. Same answers, ~two orders of magnitude fewer requests.

    Args:
        cfg: The run configuration.
        dois: The DOIs to check, bare or as URLs. Duplicates, empty values
            and DOIs holding a quote or backslash are skipped.
        chunk: The number of DOIs per bulk query.

    Returns:
        The DataCite attributes of each DOI that is a dataset, software,
        collection or model record, by lowercased DOI. A DOI that could not
        be resolved because DataCite was unreachable is left out and not
        cached.
    """
    result: dict[str, dict[str, Any]] = {}
    want: list[str] = []
    seen: set[str] = set()
    for d in dois:
        k = _normalise_doi(d).lower()
        # A quote or backslash would break out of the quoted Lucene term.
        if not k or k in seen or '"' in k or "\\" in k:
            continue
        seen.add(k)
        with _dc_lock:
            if k in _dc_cache:
                hit = _dc_cache[k]
                if hit is not None:
                    result[k] = hit
                continue
        want.append(k)

    for i in range(0, len(want), chunk):
        part = want[i : i + chunk]
        try:
            found = _datacite_chunk(cfg, part)
        except (PermanentError, TransientError):
            # A DataCite outage must not kill the paper, and must not be cached
            # as "not a dataset" — leave these unresolved so a rerun retries.
            continue
        with _dc_lock:
            if len(_dc_cache) > _DC_CACHE_MAX:
                _dc_cache.clear()
            for k in part:
                attrs = found.get(k)
                rtype = ((attrs or {}).get("types") or {}).get("resourceTypeGeneral") or ""
                if attrs is not None and rtype.lower() in _DATASET_TYPES:
                    _dc_cache[k] = attrs
                    result[k] = attrs
                else:
                    _dc_cache[k] = None
    return result


def is_datacite_dataset(cfg: Config, doi: str) -> Optional[dict[str, Any]]:
    """Return the DataCite attributes of a DOI if it is a dataset record.

    A DOI registered with DataCite and typed `Dataset`/`Software` is a far more
    reliable Axis A signal than any sentence in a paper. Single-DOI form of
    `datacite_datasets`, which is what the pipeline uses.

    Args:
        cfg: The run configuration.
        doi: The DOI to check, bare or as a URL.

    Returns:
        The DataCite attributes if the DOI is a dataset, software,
        collection or model record, otherwise None.
    """
    if not doi:
        return None
    got = datacite_datasets(cfg, [doi])
    return got.get(_normalise_doi(doi).lower())


# ---------------------------------------------------------------------------
# Full text — only reached when metadata is inconclusive
# ---------------------------------------------------------------------------
_CHALLENGE_MARKERS = (
    "enable javascript and cookies",
    "checking your browser",
    "cf-browser-verification",
    "captcha",
    "are you a robot",
    "unusual traffic",
    "access denied",
    "just a moment",
    "verify you are human",
    "ddos protection",
)


#: Publisher landing-page furniture. A landing page is long and carries the
#: paper's own title, so length and title-overlap cannot tell it from the
#: article — but no article body says "Download citation" or "Google Scholar".
#: Measured over a 12-paper calibration set: landing pages carried 10-12 of
#: these, every real article body carried none.
_CHROME_MARKERS = (
    "download citation",
    "share this article",
    "get shareable link",
    "explore related subjects",
    "google scholar",
    "sign in to",
    "subscribe to journal",
    "rent or buy",
    "access options",
    "institutional subscriptions",
    "buy this article",
    "article purchase",
    "cookies we use",
    "skip to main content",
    "advertisement",
    "sorry, a shareable link",
    "discover the latest articles",
    "content being viewed by others",
    "learn more about institutional",
    "use our pre-submission checklist",
)

#: How many distinct chrome markers make it a landing page rather than an
#: article that happens to mention one. Three is deliberately well clear of the
#: highest count seen on a real article body (zero).
_CHROME_LIMIT = 3


#: Repository *record* furniture. A catalogue page — OSTI.GOV's /biblio/ view,
#: an EPrints or DSpace item record, a journal's navigation shell — is a
#: different animal from a publisher landing page: it carries the real title,
#: the real abstract and often the real reference list, so it clears the
#: length, title-overlap and bibliography checks, and it carries too little
#: marketing chrome to trip `_CHROME_MARKERS`. What it does not carry is the
#: paper's body, which is where a data-availability statement lives.
#:
#: Measured: 35 already-coded papers were classified from an OSTI /biblio/ page.
#: Every one of them was flagged by its coder, and 28 of the 35 came out
#: `not_applicable` on axis B with no statement — precisely the reading an
#: abstract-only document produces, and precisely the direction that would
#: inflate this study's headline claim. OSTI redirects /biblio/ to
#: /servlets/purl/ by itself when it holds the full text, so rejecting the
#: record page costs nothing where real text exists.
#:
#: "institutional repository" is deliberately absent: a real availability
#: statement may name one, and discarding those papers would throw away the
#: very documents this study is about.
_RECORD_MARKERS = (
    # OSTI.GOV, EPrints, DSpace
    "citation formats",
    "search osti.gov",
    "osti id:",
    "local eprints id",
    "pure uuid",
    "actions (login required)",
    "show full item record",
    "files in this item",
    "this item appears in the following",
    "e-pub ahead of print date",
    "share the qr code",
    "add to citation manager",
    "export citation",
    # Pure research portals: the page dumps the record in six citation styles.
    # "cite this" and "@article{" are deliberately NOT here: they caught a real
    # paper that carries its own "How to Cite this Work" section and a BibTeX
    # block for self-citation, which is exactly the kind of paper this study is
    # about. Portal furniture only.
    "research output",
    "other files and links",
    "access to document",
    "publication status",
    # German, Italian and Danish institutional catalogues. Pure is deployed in
    # the local language, so the English markers above miss a Danish or Dutch
    # portal serving exactly the same record page.
    "dokumenttyp",
    "erweiterte suche",
    "zeitschriftenaufsatz",
    "scheda breve",
    "scheda completa",
    "adgang til dokumentet",
    "citationsformater",
    "originalsprog",
    "article dans une revue",
    "fonction : auteur",
    "afficher plus de d",
    "r\u00e9duire la vue d\u00e9taill\u00e9e",
    "idhal",
    # A journal's own index page. Rinton Press serves Quantum Information and
    # Computation's table of contents at the article DOI, and it was stored and
    # queued for coding as the paper. Audited 25 Sep 2026 over every stored
    # passage in the three databases: these catch that page and nothing else;
    # "editorial board" alone appears in 9 real papers, which one marker cannot
    # reject.
    "guidelines for authors",
    "editorial board",
    "author index",
)

#: Two, because each of these phrases is specific enough that a pair is
#: conclusive, and a record page always carries several.
_RECORD_LIMIT = 2


def _words(text: str) -> list[str]:
    """Split text into lowercase alphanumeric words.

    Args:
        text: The text to split. None is read as empty.

    Returns:
        The words, in order.
    """
    return re.findall(r"[a-z0-9]+", (text or "").lower())


#: Longer than any record page measured (3,420 words), shorter than a paper
#: with a repository cover sheet (8,000+).
_RECORD_MAX_WORDS = 4000
_YEAR = re.compile(r"(?:19|20)\d\d$")


def repeats_title_like_a_record(text: str, title: str) -> bool:
    """Tell whether a page lists its own paper's citation several times over.

    A catalogue or portal record prints the reference in APA, Harvard, BibTeX,
    Vancouver and so on, so the title recurs back to back. That is a property
    of the page, not of its language — the marker list above has to learn every
    language Pure and DSpace are deployed in, one country at a time (Danish,
    French, then Spanish reached the classifier). A paper's own title recurs
    only as a running head, a page apart.

    Measured 27 Sep 2026: 9 record pages (OSTI, EPrints, IRIS, DSpace, Pure in
    English, Danish and Spanish, 420-3,420 words) against 38 real papers from
    21 hosts. Every record page had two title occurrences within 44 words, each
    printed as a citation — with a year beside it (51 of 57 repetitions had one
    within 25 words). So did three real papers — HAL and Chalmers PDFs whose
    repository cover sheet ("To cite this version: ...") prints the title twice
    above the paper's own title page — but those ran to 8,000-16,000 words. A
    record page is short, so nothing longer than `_RECORD_MAX_WORDS` is judged
    at all. Only the first six title words are matched, since pages truncate
    long titles; a title shorter than four words is too common a phrase to count.

    Args:
        text: The text of the page.
        title: The paper's title.

    Returns:
        True if the page is shorter than `_RECORD_MAX_WORDS` and repeats the
        title, with a year beside it, closely enough to be a list of citation
        styles.
    """
    key = _words(title)[:6]
    words = _words(text)
    if len(key) < 4 or len(words) >= _RECORD_MAX_WORDS:
        return False
    k, span = len(key), len(_words(title)) + 25
    pos = [
        i
        for i in range(len(words) - k + 1)
        if words[i] == key[0]
        and words[i : i + k] == key
        and any(_YEAR.match(w) for w in words[max(0, i - 25) : i + span])
    ]
    if any(pos[j + 2] - pos[j] < 400 for j in range(len(pos) - 2)):
        return True  # three styles in one block
    return any(b - a < 60 for a, b in zip(pos, pos[1:]))


#: Citation furniture: "et al", "(2017)", journal abbreviations, "[12]", arXiv
#: ids, reference DOIs, author-initial forms. One reference produces several of
#: these, so a real bibliography scores well above one hit per deposited ref.
CITATION = re.compile(
    r"(?:\bet al\b)"
    r"|(?:\b(?:19|20)\d{2}\b\s*[\).,])"
    r"|(?:\bPhys\.|\bRev\.|\bNat\.|\bLett\.|\bProc\.|\bPreprint at\b|\barXiv:)"
    r"|(?:\[\d{1,3}(?:[,\u2013-]\d{1,3})*\])"
    r"|(?:doi\.org/10\.)"
    r"|(?:doi:\s?10\.)"
    r"|(?:\bRetrieved from\b)"
    r"|(?:\bpp\.\s?\d)"
    r"|(?:\b[A-Z][a-z]{2,},\s+[A-Z]\.(?:\s?[A-Z]\.)*)",
    re.I,
)

#: Below this many deposited references there is nothing to check against.
_MIN_REFS_TO_CHECK = 10

#: Citation hits per deposited reference. Measured across 20 real papers the
#: healthy range is 2.3 - 16.9, because one reference yields several hits. 0.5
#: therefore fires only when the bibliography is essentially absent, with a
#: 4.6x margin below the lowest healthy paper observed.
_COMPLETENESS_RATIO = 0.5


def text_is_complete(text: str, reference_count: int) -> bool:
    """Check that a text contains the bibliography the publisher says exists.

    A body-only TDM deposit or a PDF whose extraction stopped early looks like a
    perfectly good article — it is long, it matches the title, it carries real
    prose. The only thing missing is the end of the document, which is exactly
    where data-availability statements live. The deposited reference count is a
    free, publisher-asserted check on that.

    Args:
        text: The retrieved text.
        reference_count: How many references the publisher deposited.

    Returns:
        False if the publisher deposited at least `_MIN_REFS_TO_CHECK`
        references and the text carries too few citation marks to hold them;
        True otherwise.
    """
    if reference_count < _MIN_REFS_TO_CHECK:
        return True
    return len(CITATION.findall(text)) >= reference_count * _COMPLETENESS_RATIO


def looks_like_article(text: str, title: str, *, min_words: int = 400) -> bool:
    """Tell whether a retrieved text is the article and not a page about it.

    Rejects bot-challenge pages, paywall stubs, publisher landing pages and
    catalogue records that return HTTP 200.

    This is the check v1 lacked. There, any response longer than 200 characters
    was accepted as article text, so a Cloudflare interstitial would be
    keyword-searched and classified as though it were the paper. The landing-page
    case is subtler and was found on the first live run: a Nature article page is
    long, carries the real title, and passes both of those tests while containing
    no article body at all.

    Args:
        text: The retrieved text.
        title: The paper's title; the text must share enough of its
            distinctive words.
        min_words: The fewest words an article may have.

    Returns:
        True if the text passes every check.
    """
    if not text:
        return False
    lowered = text[:4000].lower()
    if any(m in lowered for m in _CHALLENGE_MARKERS):
        return False
    whole = text.lower()
    if sum(1 for m in _CHROME_MARKERS if m in whole) >= _CHROME_LIMIT:
        return False
    if sum(1 for m in _RECORD_MARKERS if m in whole) >= _RECORD_LIMIT:
        return False
    if repeats_title_like_a_record(text, title):
        return False
    words = text.split()
    if len(words) < min_words:
        return False
    # The retrieved document should contain a decent share of the title's
    # distinctive words; if it does not, we probably fetched the wrong thing.
    title_words = {w for w in re.findall(r"[a-z]{5,}", (title or "").lower())}
    if title_words:
        body = set(re.findall(r"[a-z]{5,}", text[:20000].lower()))
        overlap = len(title_words & body) / len(title_words)
        if overlap < 0.4:
            return False
    return True


@dataclass
class FullText:
    """The text retrieved for one paper, and where it came from.

    Attributes:
        text: The extracted text; empty if nothing was retrieved.
        source: The kind of location the text came from.
        url: The URL the text was fetched from.
        ok: Whether usable article text was retrieved.
        complete: False when the publisher deposited references but the
            retrieved text has no bibliography: the document was cut short.
        sections: `(character offset, section heading)` pairs in document
            order, when the parser knew the document's structure. Empty for
            parsers that do not.
        attempts: Every location tried, in cascade order, with what it
            yielded: "ok", "incomplete", "not_article", "parse_error: ...",
            "permanent: HTTP 403", "transient: ...". Without it a fallback
            copy is indistinguishable from a first choice.
    """

    text: str
    source: TextSource
    url: str
    ok: bool
    #: False when the publisher deposited references but the retrieved text has
    #: no bibliography — the document was cut short.
    complete: bool = True
    #: (character offset, section heading) pairs in document order, when the
    #: parser knew the document's structure. Empty for parsers that do not.
    sections: list[tuple[int, str]] = field(default_factory=list)
    #: Every location tried, in cascade order, with what it yielded: "ok",
    #: "incomplete", "not_article", "parse_error: ...", "permanent: HTTP 403",
    #: "transient: ...". Without it a fallback copy is indistinguishable from a
    #: first choice.
    attempts: list[dict] = field(default_factory=list)


#: The Latin typographic ligatures, U+FB00-FB06, folded to their letters.
#: PDF text layers keep them as single glyphs, so "classiﬁcation" in a URL is
#: one character short of the address the paper prints and verify.py reports
#: it dead; 58 coded papers had one inside an identifier. Only these seven:
#: full NFKC would also flatten superscripts, turning 10⁵ into 105.
_LIGATURES = str.maketrans(
    {chr(c): unicodedata.normalize("NFKC", chr(c)) for c in range(0xFB00, 0xFB07)}
)


def fold_ligatures(text: str) -> str:
    """Replace the Latin typographic ligatures with their letters.

    Args:
        text: The text to fold. An empty value is returned unchanged.

    Returns:
        The text with U+FB00 to U+FB06 written out as plain letters.
    """
    return text.translate(_LIGATURES) if text else text


def _extract_pdf_pdfplumber(content: bytes, max_pages: int = 60) -> str:
    """Extract a PDF's text with pdfplumber; the last resort.

    Reconstructs text from glyph positions with no notion of columns, so a
    two-column paper comes back interleaved and word-joined ("Thetechnological
    world is inthemidstof"). Kept only as a fallback.

    Args:
        content: The bytes of the PDF.
        max_pages: The most pages to read.

    Returns:
        The pages' text, joined by blank lines.
    """
    import io

    import pdfplumber

    parts: list[str] = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages[:max_pages]:
            parts.append(page.extract_text() or "")
    return "\n\n".join(parts)


def _extract_pdf_fast(content: bytes, max_pages: int = 60) -> str:
    """Extract a PDF's text with PyMuPDF, falling back to pdfplumber.

    PyMuPDF with `sort=True` recovers reading order on multi-column layouts.
    Measured on a Nature two-column PDF: 7,001 words in 0.33s against
    pdfplumber's 3,918 garbled words in 0.80s. No document structure.

    Args:
        content: The bytes of the PDF.
        max_pages: The most pages to read.

    Returns:
        The pages' text, joined by blank lines; an empty string if both
        parsers refuse the file.
    """
    try:
        import pymupdf
    except ImportError:
        return _extract_pdf_pdfplumber(content, max_pages)
    try:
        with pymupdf.open(stream=content, filetype="pdf") as doc:
            pages = [doc[i].get_text("text", sort=True) for i in range(min(len(doc), max_pages))]
        return "\n\n".join(pages)
    except Exception:
        pass
    try:
        return _extract_pdf_pdfplumber(content, max_pages)
    except Exception:
        # Both parsers refused it. Empty text fails looks_like_article, so the
        # cascade moves on to the next source instead of the paper dying here.
        return ""


def _extract_pdf_grobid(cfg: Config, content: bytes) -> Optional[tuple[str, list[tuple[int, str]]]]:
    """Extract a PDF's text and section structure through a GROBID service.

    GROBID gives sections, the body separated from the bibliography, and
    figure captions.

    Far better text than any local parser, but it is a network service and an
    order of magnitude slower, so it is opt-in via DAP_GROBID_URL.

    Args:
        cfg: The run configuration; `grobid_url` names the service.
        content: The bytes of the PDF.

    Returns:
        A tuple `(text, sections)`, where `sections` holds `(offset,
        heading)` pairs; or None if no service is configured, or it is
        unreachable, slow or returns nothing usable, so the caller can fall
        back rather than lose the paper.
    """
    if not cfg.grobid_url:
        return None
    try:
        r = requests.post(
            cfg.grobid_url.rstrip("/") + "/api/processFulltextDocument",
            files={"input": ("paper.pdf", content, "application/pdf")},
            data={"consolidateHeader": "0", "consolidateCitations": "0"},
            timeout=cfg.grobid_timeout,
        )
        if r.status_code != 200 or not r.content:
            return None
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(r.content, "lxml-xml")
        body = soup.find("body")
        if body is None:
            return None
        # Walk the divs so the character offset of each section heading is known.
        # That offset is what lets a passage — and then a quote — be reported
        # against the section it actually came from.
        parts: list[str] = []
        sections: list[tuple[int, str]] = []
        pos = 0
        for div in body.find_all("div", recursive=False) or []:
            head = div.find("head")
            name = fold_ligatures(head.get_text(" ", strip=True)) if head else ""
            # Folded before its offset is counted, so the sections still line up.
            chunk = fold_ligatures(div.get_text("\n", strip=True))
            if not chunk:
                continue
            if name:
                sections.append((pos, name))
            parts.append(chunk)
            pos += len(chunk) + 2
        text = "\n\n".join(parts) if parts else fold_ligatures(body.get_text("\n", strip=True))
        if not text:
            return None
        return text, sections
    except Exception:
        return None


def _extract_pdf(
    content: bytes, max_pages: int = 60, cfg: Optional[Config] = None
) -> tuple[str, list[tuple[int, str]]]:
    """Extract a PDF's text, through GROBID when it is configured.

    Args:
        content: The bytes of the PDF.
        max_pages: The most pages the local parsers read.
        cfg: The run configuration, or None to use the local parsers only.

    Returns:
        A tuple `(text, sections)`. `sections` is empty for parsers with no
        structure, which is every path except GROBID.
    """
    if cfg is not None:
        got = _extract_pdf_grobid(cfg, content)
        if got:
            return got
    return fold_ligatures(_extract_pdf_fast(content, max_pages)), []


def _extract_html(content: bytes) -> str:
    """Extract the article text of an HTML page.

    Scripts, styles and page furniture are dropped, and the `article` or
    `main` element is preferred over the whole body.

    Args:
        content: The bytes of the page.

    Returns:
        The text, with ligatures folded.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(content, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    node = soup.find("article") or soup.find("main") or soup.body
    return fold_ligatures(
        node.get_text("\n", strip=True) if node else soup.get_text("\n", strip=True)
    )


def _citation_pdf_url(content: bytes) -> Optional[str]:
    """Find the PDF a landing page points at, via `citation_pdf_url`.

    OSTI, institutional repositories (IRIS, Pure, EPrints) and most publishers
    emit this tag on the catalogue page. Without following it the cascade stores
    the catalogue record — author list, BibTeX, "Similar Records" — and the
    paper is coded from a page that contains no article text at all.

    Args:
        content: The bytes of the landing page.

    Returns:
        The absolute URL in the page's `citation_pdf_url` meta tag, or None
        if there is none or the page cannot be parsed.
    """
    try:
        from bs4 import BeautifulSoup

        tag = BeautifulSoup(content, "lxml").find("meta", attrs={"name": "citation_pdf_url"})
    except Exception:
        return None
    href = tag.get("content") if tag else None
    href = (href or "").strip()
    return href if href.startswith(("http://", "https://")) else None


def _fetch_and_parse(
    cfg: Config, url: str, source: TextSource, title: str, *, _followed: bool = False
) -> tuple[Optional[FullText], str]:
    """Fetch and parse one location.

    TransientError is deliberately not caught. It used to be, alongside
    PermanentError, so a throttled version of record read exactly like a missing
    one and the cascade quietly settled for a worse copy. Whether a fallback is
    acceptable is the cascade's decision, and it has to know this happened.

    Args:
        cfg: The run configuration.
        url: The location to fetch.
        source: The kind of location, recorded in the result.
        title: The paper's title, used to check that the text is the article.
        _followed: Whether this call is already following a page's
            `citation_pdf_url`, so it is not followed again.

    Returns:
        A tuple `(full_text, outcome)`. `full_text` is None unless article
        text was found; `outcome` is "ok", "not_article", "permanent: ..."
        or "parse_error: ...".
    """
    try:
        r = _get(cfg, url, headers={"Accept": "*/*"})
    except PermanentError as e:
        return None, f"permanent: {e}"
    ctype = (r.headers.get("Content-Type") or "").lower()
    # Trust the bytes, not the URL or the header. A ".pdf" URL that serves an
    # HTML landing page is routine; so is a "text/html" header on a real PDF.
    head = r.content[:8] if r.content else b""
    is_pdf = head.startswith(b"%PDF")
    sections: list[tuple[int, str]] = []
    try:
        if is_pdf:
            text, sections = _extract_pdf(r.content, cfg=cfg)
        elif (
            "xml" in ctype
            and not head.lstrip().startswith(b"<!")
            and b"<html" not in r.content[:2000].lower()
        ):
            from bs4 import BeautifulSoup

            text = fold_ligatures(BeautifulSoup(r.content, "lxml-xml").get_text("\n", strip=True))
        else:
            text = _extract_html(r.content)
    except Exception as e:
        return None, f"parse_error: {type(e).__name__}"
    html: Optional[FullText] = None
    if looks_like_article(text, title):
        html = FullText(text=text, source=source, url=url, ok=True, sections=sections)
    # An HTML page that names its own PDF is at best a worse copy of it, and at
    # worst a catalogue record that merely describes it. Prefer the PDF.
    #
    # The length floor cannot separate the two: an OSTI record runs to 1,900
    # words and an institutional-repository page to 700, both carrying the real
    # title, so both are accepted as article text and neither contains any. The
    # PDF is the version of record either way, and it is the copy that reaches
    # the back matter where availability statements live. Falls back to the HTML
    # when the PDF does not pan out, so this can only improve on what we had.
    if not (_followed or is_pdf):
        pdf_url = _citation_pdf_url(r.content)
        if pdf_url and pdf_url != url:
            better, _ = _fetch_and_parse(cfg, pdf_url, source, title, _followed=True)
            if better is not None:
                return better, "ok"
    if html is not None:
        return html, "ok"
    return None, "not_article"


def openalex_family_names(work: dict) -> list[str]:
    """List author family names from an OpenAlex work.

    Used when no registry lists any authors.

    `arxiv_search` refuses a match it cannot check against authors, and
    `authors` came only from Crossref or DataCite. A DOI neither registry knows
    (Quantum Information and Computation's `10.5555/...`) therefore never got a
    title search, and neither did the 5,630 venue-frame rows whose metadata
    predates `authors` being stored. OpenAlex lists the authors for both.

    The family name is the last token of `display_name`, lowercased: the same
    rule `arxiv_search` applies to arXiv's side, so the two compare like for like.

    Args:
        work: The OpenAlex work object.

    Returns:
        The lowercased family names, in authorship order.
    """
    out = []
    for a in work.get("authorships") or []:
        name = (
            (a.get("author") or {}).get("display_name") or a.get("raw_author_name") or ""
        ).strip()
        if name:
            out.append(name.split()[-1].lower())
    return out


def arxiv_id_from_openalex(work: dict) -> Optional[str]:
    """Pull the arXiv id out of an OpenAlex work rather than searching by title.

    v1 searched arXiv by title with max_results=1 and no author or year check,
    so a near-title match silently attached a *different paper's* full text —
    a wrong classification rather than a missing one. The id is usually already
    in the metadata; use it.

    Args:
        work: The OpenAlex work object.

    Returns:
        The arXiv id, e.g. `2101.01234`, or None if the work records none.
    """
    for loc in work.get("locations") or []:
        src = loc.get("source") or {}
        landing = loc.get("landing_page_url") or ""
        pdf = loc.get("pdf_url") or ""
        if "arxiv.org" in (landing + pdf).lower():
            m = re.search(
                r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)", landing + " " + pdf, re.I
            )
            if m:
                return m.group(1)
        if (src.get("display_name") or "").lower().startswith("arxiv"):
            m = re.search(r"([0-9]{4}\.[0-9]{4,5})", landing + " " + pdf)
            if m:
                return m.group(1)
    ids = work.get("ids") or {}
    for v in ids.values():
        if isinstance(v, str) and "arxiv" in v.lower():
            m = re.search(r"([0-9]{4}\.[0-9]{4,5})", v)
            if m:
                return m.group(1)
    return None


#: Hosts that only ever serve a catalogue entry, never the article. Trying them
#: spends a request and usually collects a bot-wall 403 on the way to the same
#: answer — measured: 58 such attempts across the quantum frame, none yielding
#: text. DOAJ indexes open-access journals; it does not host their content.
#: Hosts that never serve an article body, only a record about one. PubMed is
#: the case that matters here: `pubmed.ncbi.nlm.nih.gov/<pmid>/` is always an
#: abstract page -- the full text lives on `pmc.ncbi.nlm.nih.gov`, which is a
#: different host and stays allowed. 95 rows in dap_frame had a PubMed abstract
#: page stored as their text, 21 of them recorded as `full_text`, and three
#: papers were coded off one before this went in. Marker phrases would have
#: caught about half; the host is deterministic.
_INDEX_ONLY_HOSTS = frozenset(
    {
        "doaj.org",
        "www.doaj.org",
        "pubmed.ncbi.nlm.nih.gov",
    }
)

#: Paths that serve a catalogue record on a host that ALSO serves real text, so
#: the host itself must not be blocked. OSTI.GOV is the case that forced this:
#: `/biblio/<id>` is the record page (35 papers were coded from one), while
#: `/servlets/purl/<id>` is the full text (37 papers, 20 of them carrying a data
#: availability statement - a better rate than the corpus average). Blocking the
#: host would discard the good half to be rid of the bad half. OSTI redirects
#: /biblio/ to /servlets/purl/ by itself when it holds the text, so skipping the
#: record path costs nothing.
_CATALOGUE_PATHS = (
    ("osti.gov", "/biblio/"),
    ("www.osti.gov", "/biblio/"),
)


def _is_catalogue_url(url: str) -> bool:
    """Tell whether a URL is a catalogue record on a host that also serves text.

    Args:
        url: The URL to test.

    Returns:
        True if the host and path prefix are listed in `_CATALOGUE_PATHS`.
    """
    u = urlparse(url)
    host = (u.hostname or "").lower()
    return any(host == h and u.path.startswith(pre) for h, pre in _CATALOGUE_PATHS)


def _local_pdf(cfg: Config, *, doi: str, work: dict, title: str) -> Optional[FullText]:
    """Read a PDF someone downloaded by hand, if one is there for this paper.

    Rule 8 forbids bypassing a publisher's bot wall. It does not forbid a
    person with a subscription fetching the paper themselves — which is the
    only lawful route to several hundred papers in this corpus, IOP and
    Springer above all. This is where those land.

    Args:
        cfg: The run configuration; `pdf_dir` names the directory.
        doi: The paper's DOI, which names the file with "/" written as "_".
        work: The OpenAlex work object; its id is the alternative file name.
        title: The paper's title, used to check that the file is the article.

    Returns:
        The text of the local PDF, or None if no directory is configured, no
        file is there, or the file does not read as the article.
    """
    if not cfg.pdf_dir:
        return None
    import os

    names = []
    if doi:
        d = _normalise_doi(doi)
        names += [d.replace("/", "_"), d.replace("/", "_").lower()]
    oid = ((work.get("ids") or {}).get("openalex") or "").rsplit("/", 1)[-1]
    if oid:
        names.append(oid)
    for stem in names:
        path = os.path.join(cfg.pdf_dir, stem + ".pdf")
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "rb") as fh:
                content = fh.read()
            text, sections = _extract_pdf(content, cfg=cfg)
        except Exception:
            continue
        # Held to the same bar as anything off the network: a scanned page with
        # no text layer, or the wrong paper filed under this name, is not usable
        # just because it is local.
        if looks_like_article(text, title):
            return FullText(
                text=text,
                source=TextSource.LOCAL_PDF,
                url=f"file://{path}",
                ok=True,
                sections=sections,
            )
    return None


ARXIV_API = "http://export.arxiv.org/api/query"

#: How much of the title must survive normalisation and still match, as a
#: Jaccard overlap of word sets. 0.85 admits a corrected typo between preprint
#: and version of record, and rejects a paper that merely shares a topic.
_TITLE_MATCH = 0.85

#: Jaccard punishes a *dropped subtitle* as hard as a different paper, because
#: the extra words count against the union. Measured: the Nature Electronics
#: version of "Scaling silicon-based quantum computing using CMOS technology"
#: dropped the preprint's ": State of the art, challenges and perspectives",
#: scoring 0.53 against its own preprint - two shared authors, one year apart,
#: and refused.
#:
#: So a second route: if essentially every word of the SHORTER title appears in
#: the longer one, that is a subtitle difference, not a different paper. This
#: is asymmetric and therefore looser, so it is gated on the shorter title
#: carrying enough distinctive words to identify anything at all. Without that
#: floor, "Quantum Computing" is contained in every quantum paper on arXiv -
#: and a paper with exactly that title is in this corpus.
_TITLE_CONTAIN = 0.90
_MIN_DISTINCTIVE = 6

#: Words that identify no paper. Kept small and generic on purpose: this is a
#: floor on how much signal the shorter title carries, not a topic filter.
_TITLE_STOP = frozenset(
    (
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "into",
        "of",
        "on",
        "or",
        "the",
        "to",
        "via",
        "with",
        "using",
        "toward",
        "towards",
        "its",
        "new",
        "novel",
        "based",
        "approach",
        "study",
        "review",
        "survey",
    )
)

#: A preprint precedes its version of record, sometimes by years. Beyond this
#: the match is more likely a different paper by the same group.
_YEAR_SLACK = 3


def _norm_title(t: str) -> set[str]:
    """Reduce a title to its set of lowercase alphanumeric words.

    Args:
        t: The title. None is read as empty.

    Returns:
        The set of words.
    """
    return set(re.sub(r"[^a-z0-9 ]+", " ", (t or "").lower()).split())


def _title_overlap(a: str, b: str) -> float:
    """Compute the Jaccard overlap of two titles' word sets.

    Args:
        a: One title.
        b: The other title.

    Returns:
        The overlap, from 0.0 to 1.0; 0.0 if either title has no words.
    """
    x, y = _norm_title(a), _norm_title(b)
    if not x or not y:
        return 0.0
    return len(x & y) / len(x | y)


def _title_contains(a: str, b: str) -> tuple[float, int]:
    """Measure how much of the shorter title the longer one covers.

    Also counts the distinctive words the shorter one has to offer. Both are
    needed: coverage alone would match any short generic title inside a
    longer one.

    Args:
        a: One title.
        b: The other title.

    Returns:
        A tuple `(coverage, distinctive)`: the share of the shorter title's
        words found in the other, and the number of its words that are not
        in `_TITLE_STOP`.
    """
    x, y = _norm_title(a), _norm_title(b)
    if not x or not y:
        return 0.0, 0
    short = x if len(x) <= len(y) else y
    return len(x & y) / len(short), len(short - _TITLE_STOP)


def titles_agree(ours: str, theirs: str) -> tuple[bool, str]:
    """Decide whether two titles name the same paper.

    Args:
        ours: The title on record.
        theirs: The candidate's title.

    Returns:
        A tuple `(agreed, why)`, where `why` says which rule decided.
    """
    ov = _title_overlap(ours, theirs)
    if ov >= _TITLE_MATCH:
        return True, f"title {ov:.2f}"
    cov, distinctive = _title_contains(ours, theirs)
    if cov >= _TITLE_CONTAIN and distinctive >= _MIN_DISTINCTIVE:
        return True, f"title {ov:.2f} but {cov:.2f} contained on {distinctive} words"
    if cov >= _TITLE_CONTAIN:
        return False, (
            f"title contained {cov:.2f} but only {distinctive} distinctive "
            f"words, too generic to identify a paper"
        )
    return False, f"closest title overlap {ov:.2f} < {_TITLE_MATCH}"


def arxiv_search(
    cfg: Config, *, title: str, authors: list[str], year: Optional[int] = None, max_results: int = 8
) -> tuple[Optional[str], str]:
    """Find a paper's arXiv id by title, refusing a match the authors do not confirm.

    v1 did the title half of this with `max_results=1` and no author or year
    check, and it silently attached a *different paper's* full text — a wrong
    classification, which is worse than a missing one, because nothing
    downstream can tell it is wrong. That failure is the reason
    `arxiv_id_from_openalex` exists and is tried first: the id is usually
    already in the metadata, and an id needs no guessing.

    This is the fallback for papers where no id is recorded and no location
    yielded text. Three independent checks must all pass, and the reason string
    records which one decided, so a match can be audited rather than trusted:

      * the titles must overlap by `_TITLE_MATCH` after normalisation,
      * at least one author family name must appear on both sides,
      * the years must be within `_YEAR_SLACK`.

    A candidate with no authors on our side fails: unverifiable is not a pass.

    Args:
        cfg: The run configuration.
        title: The paper's title.
        authors: The paper's author family names, lowercased.
        year: The publication year, or None to skip the year check.
        max_results: The most arXiv entries to consider.

    Returns:
        A tuple `(arxiv_id, why)`. `arxiv_id` is None when no entry passed
        every check; `why` records which check decided.
    """
    if not title:
        return None, "no title to search on"
    if not authors:
        return None, "no authors on record to verify against"
    try:
        r = _get(
            cfg,
            ARXIV_API,
            params={"search_query": f'ti:"{title}"', "max_results": max_results, "start": 0},
        )
    except (PermanentError, TransientError) as e:
        return None, f"arxiv query failed: {e}"

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(r.content, "lxml-xml")
    ours = {a for a in authors if a}
    best = ""
    for entry in soup.find_all("entry"):
        cand_title = entry.find("title").get_text(" ", strip=True) if entry.find("title") else ""
        agreed, why_title = titles_agree(title, cand_title)
        if not agreed:
            best = best or why_title
            continue
        theirs = set()
        for a in entry.find_all("author"):
            nm = a.find("name")
            if nm:
                theirs.add(nm.get_text(" ", strip=True).split()[-1].lower())
        shared = ours & theirs
        if not shared:
            best = "title matched but no author in common"
            continue
        if year:
            pub = entry.find("published")
            ytxt = pub.get_text(strip=True)[:4] if pub else ""
            if ytxt.isdigit() and abs(int(ytxt) - int(year)) > _YEAR_SLACK:
                best = f"title and author matched but year {ytxt} vs {year}"
                continue
        idtag = entry.find("id")
        raw = idtag.get_text(strip=True) if idtag else ""
        m = re.search(r"arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5})", raw)
        if not m:
            m = re.search(r"arxiv\.org/abs/([a-z-]+/[0-9]{7})", raw)
        if m:
            # `why_title` says which rule admitted it: a plain overlap, or
            # containment despite a dropped subtitle. An auditor needs to know.
            return m.group(1), (
                f"{why_title}, author '{sorted(shared)[0]}'{', year ' + str(year) if year else ''}"
            )
    return None, best or "no arxiv entry matched"


#: Hosts that resolve a DOI rather than serve a copy. A resolver URL is the
#: publisher's only when the DOI is the paper's own: repositories mint DOIs
#: too, and ETH Zurich's `10.3929/ethz-b-...` timed out for a PRX Quantum paper.
_DOI_RESOLVERS = frozenset({"doi.org", "dx.doi.org", "www.doi.org"})

#: arXiv is what the cascade falls back to, not a mirror it falls back from.
_ARXIV_HOSTS = frozenset({"arxiv.org", "www.arxiv.org", "export.arxiv.org"})


def _resolver_doi(url: str) -> str:
    """Extract the DOI a resolver URL points at.

    Args:
        url: Any URL. None is read as empty.

    Returns:
        The DOI, lowercased; an empty string for any URL that is not on a
        DOI resolver host.
    """
    u = urlparse(url or "")
    if (u.hostname or "").lower() not in _DOI_RESOLVERS:
        return ""
    return unquote(u.path).strip("/").lower()


def _publisher_hosts(work: dict, doi: str, facts: CrossrefFacts) -> frozenset[str]:
    """List the hosts that serve the publisher's own copy of a paper.

    Crossref TDM links are deposited by the publisher. So is any OpenAlex
    location whose landing page is the paper's own DOI. The preferred OA
    location counts as the publisher's unless OpenAlex types its source a
    repository: when nothing says otherwise, a host keeps the old rule and is
    waited for. Resolver hosts are left out; `_resolver_doi` decides those.

    Args:
        work: The OpenAlex work object.
        doi: The paper's own DOI.
        facts: The paper's metadata facts; its TDM links are the publisher's.

    Returns:
        The lowercased hostnames, without DOI resolver hosts.
    """
    own = (doi or "").strip().lower()
    oa = work.get("best_oa_location") or work.get("open_access") or {}
    locs = [
        loc
        for loc in (work.get("locations") or [])
        if own and _resolver_doi(loc.get("landing_page_url")) == own
    ]
    if (oa.get("source") or {}).get("type") != "repository" or (
        own and _resolver_doi(oa.get("landing_page_url")) == own
    ):
        locs.append(oa)
    urls = [ln.get("url") for ln in facts.tdm_links]
    for loc in locs:
        urls += [loc.get(k) for k in ("pdf_url", "landing_page_url", "url", "oa_url")]
    hosts = {(urlparse(u).hostname or "").lower() for u in urls if u}
    return frozenset(hosts - {""} - _DOI_RESOLVERS)


def retrieve_full_text(
    cfg: Config, *, work: dict, doi: str, title: str, facts: CrossrefFacts
) -> FullText:
    """Retrieve a paper's full text: a cascade, cheapest and most reliable first.

    Crossref TDM links come first because they are what the publisher itself
    deposited for this purpose; with institutional TDM headers configured they
    also reach subscription content legitimately.

    Args:
        cfg: The run configuration.
        work: The OpenAlex work object, which lists the paper's locations.
        doi: The paper's DOI.
        title: The paper's title, used to check that a text is the article.
        facts: The paper's metadata facts: TDM links, authors and the
            deposited reference count.

    Returns:
        The text retrieved. `ok` is False if no location yielded article
        text; `attempts` lists every location tried.

    Raises:
        TransientError: If a preferred location failed only temporarily, so
            the paper should be retried rather than settled with a worse copy.
    """  # noqa: DOC503 -- pydoclint reads `raise retry_later()` as the exception's name
    seen: set[str] = set()
    partial: Optional[FullText] = None
    attempts: list[dict] = []

    def note(url: str, source: TextSource, outcome: str) -> None:
        attempts.append({"url": url, "source": source.value, "outcome": outcome})

    # A hand-fetched copy wins outright. It is the version of record, obtained
    # legitimately by someone with access, for a paper the cascade is forbidden
    # to reach — so there is nothing better to look for.
    local = _local_pdf(cfg, doi=doi, work=work, title=title)
    if local is not None:
        note(local.url, TextSource.LOCAL_PDF, "ok")
        local.attempts = attempts
        return local

    # A mirror that times out is not waited for (Diana, 4 Oct 2026). Waiting
    # protects a version of record that might come back; a repository copy is
    # not one, and an institutional host that is down tends to stay down. On
    # 25 Sep 18 papers sat parked behind mirrors that kept timing out while the
    # publisher had already answered 403 for good. A publisher, a DOI lookup
    # and arXiv are still waited for.
    publisher = _publisher_hosts(work, doi, facts)
    own_doi = (doi or "").strip().lower()
    mirrors: set[str] = set()  # URLs of mirrors that failed transiently

    def is_mirror(url: str, source: TextSource) -> bool:
        if source not in (TextSource.OPENALEX_OA, TextSource.UNPAYWALL):
            return False
        host = (urlparse(url).hostname or "").lower()
        if host in _DOI_RESOLVERS:
            # With no DOI of its own to compare, a resolver URL may be the
            # publisher's; it is waited for, as before.
            return bool(own_doi) and _resolver_doi(url) != own_doi
        return host not in publisher and host not in _ARXIV_HOSTS

    def throttled(*, counting_mirrors: bool = False) -> bool:
        return any(
            a["outcome"].startswith("transient") and (counting_mirrors or a["url"] not in mirrors)
            for a in attempts
        )

    def retry_later() -> TransientError:
        # Every outcome goes into the message, so the failure row says what to retry.
        return TransientError(
            "a preferred location failed only temporarily; retry rather than settle "
            "for what the cascade found after it: "
            + "; ".join(f"{a['source']} {a['url']} -> {a['outcome']}" for a in attempts)
        )

    def attempt(
        url: str, source: TextSource, *, publisher_copy: bool = False
    ) -> Optional[FullText]:
        nonlocal partial
        if not url or url in seen:
            return None
        seen.add(url)
        host = (urlparse(url).hostname or "").lower()
        if host in _INDEX_ONLY_HOSTS:
            note(url, source, "skipped: index-only host")
            return None
        if host in cfg.deny_hosts or host.rpartition("www.")[2] in cfg.deny_hosts:
            note(url, source, "skipped: host denied by configuration")
            return None
        if _is_catalogue_url(url):
            note(url, source, "skipped: catalogue record path, not the article")
            return None
        try:
            ft, outcome = _fetch_and_parse(cfg, url, source, title)
        except TransientError as e:
            note(url, source, f"transient: {e}")
            if not publisher_copy and is_mirror(url, source):
                mirrors.add(url)
            return None
        if ft and not text_is_complete(ft.text, facts.reference_count):
            # Truncated: the publisher deposited references and this copy has no
            # bibliography. Keep looking, but do not throw away what we have —
            # an incomplete body still beats no text at all.
            note(url, source, "incomplete")
            if partial is None:
                partial = FullText(
                    text=ft.text,
                    source=ft.source,
                    url=ft.url,
                    ok=True,
                    complete=False,
                    sections=ft.sections,
                )
            return None
        note(url, source, outcome)
        if ft is None:
            return None
        # The cascade is in order of preference. If an earlier location was only
        # throttled, this copy is not the one a healthy network would have given —
        # the preprint in place of the version of record, typically. A mirror's
        # timeout is the exception: what it held was not the version of record.
        if throttled():
            raise retry_later()
        ft.attempts = attempts
        return ft

    for ln in facts.tdm_links:
        ft = attempt(ln["url"], TextSource.CROSSREF_TDM)
        if ft:
            return ft

    oa = work.get("best_oa_location") or work.get("open_access") or {}
    for url in filter(None, [oa.get("pdf_url"), oa.get("url"), oa.get("oa_url")]):
        ft = attempt(url, TextSource.OPENALEX_OA)
        if ft:
            return ft

    arxiv_id = arxiv_id_from_openalex(work)
    if arxiv_id:
        ft = attempt(f"https://arxiv.org/pdf/{arxiv_id}", TextSource.ARXIV)
        if ft:
            return ft

    # Every other location OpenAlex knows about. `best_oa_location` is often the
    # publisher's landing page — a doi.org stub or a JS shim with no text — while
    # the same paper sits readable in an institutional repository, PMC or OSTI.
    # Measured on the first live run: this recovered 22 of 79 otherwise-unusable
    # papers, so skipping the list discarded a quarter of the review queue.
    for loc in work.get("locations") or []:
        for url in (loc.get("pdf_url"), loc.get("landing_page_url")):
            ft = attempt(url, TextSource.OPENALEX_OA)
            if ft:
                return ft

    if doi and not cfg.contact_email:
        note(
            f"https://api.unpaywall.org/v2/{doi}",
            TextSource.UNPAYWALL,
            "skipped: no DAP_CONTACT_EMAIL (Unpaywall requires one)",
        )
    if doi and cfg.contact_email:
        # Only the lookup is guarded here. The attempts below must be able to
        # raise: a blanket except around them would swallow retry_later().
        api = f"https://api.unpaywall.org/v2/{doi}"
        best: dict = {}
        try:
            r = _get(cfg, api, params={"email": cfg.contact_email})
            best = (r.json() or {}).get("best_oa_location") or {}
        except TransientError as e:
            note(api, TextSource.UNPAYWALL, f"transient: {e}")
        except PermanentError as e:
            note(api, TextSource.UNPAYWALL, f"permanent: {e}")
        for url in filter(None, [best.get("url_for_pdf"), best.get("url")]):
            ft = attempt(
                url, TextSource.UNPAYWALL, publisher_copy=best.get("host_type") == "publisher"
            )
            if ft:
                return ft

    # Before guessing, rule out a retry. If a preferred location was only
    # throttled, this paper's copy is not missing — it is temporarily
    # unavailable, and searching arXiv now would attach a preprint in place of
    # a version of record that a later run would have read. A throttle must
    # never be the reason a paper ends up coded from a preprint -- unless every
    # throttled location was a mirror, whose copy was not the version of record
    # either.
    if throttled():
        raise retry_later()

    # Last resort: no location we know of yielded text, and no arXiv id was
    # recorded in the metadata. Search for one by title and refuse it unless the
    # authors agree — see `arxiv_search` for why that check is not optional.
    if not arxiv_id and title and facts.authors:
        found, why = arxiv_search(
            cfg, title=title, authors=facts.authors, year=facts.published_year
        )
        note(ARXIV_API, TextSource.ARXIV, f"search: {why}" if found else f"search failed: {why}")
        if found:
            ft = attempt(f"https://arxiv.org/pdf/{found}", TextSource.ARXIV)
            if ft:
                return ft

    # Nothing complete was found. A mirror's timeout did not stop the fallback,
    # but with the fallback gone too it is a reason to retry, not to settle for
    # a truncated body or for none: the mirror may hold the only readable copy.
    if throttled(counting_mirrors=True):
        raise retry_later()
    if partial is not None:
        partial.attempts = attempts
        return partial
    return FullText(text="", source=TextSource.NONE, url="", ok=False, attempts=attempts)


__all__ = [
    "Config",
    "CrossrefFacts",
    "crossref_facts",
    "is_datacite_dataset",
    "repo_type_for",
    "retrieve_full_text",
    "looks_like_article",
    "arxiv_id_from_openalex",
    "openalex_family_names",
    "arxiv_search",
    "FullText",
    "TransientError",
    "PermanentError",
]
