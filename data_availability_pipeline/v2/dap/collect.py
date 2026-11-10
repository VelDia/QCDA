"""Stage 0: collect the corpus for a subject of interest from OpenAlex.

    python -m dap.pipeline collect --subject "quantum computing" --out papers.csv

The subject is the only thing that ties a run to a field. Everything after
`enrol` is subject-agnostic: the coding scheme, retrieval and verification are
the same for any discipline.

Where the subject is matched (`--search-in`):

  title                  title only: the narrowest, highest-precision corpus
  title_and_abstract     title or abstract (the default)
  fulltext               OpenAlex's default `search=`, which also reads the full
                         text where OpenAlex has it. The quantum case study was
                         harvested this way, with `--filter concepts.id:C121332964`
                         (544,156 rows for "quantum computing"), and it pulls in a great deal of off-topic
                         physics, so it was then narrowed by title keywords
                         (`keyword_arms.py`). Prefer `title_and_abstract`.

`--filter` passes any extra OpenAlex filter through unchanged, for example a
topic (`topics.id:T10682`) or a work type (`type:article`). Filters are ANDed.

The harvest is resumable: the cursor is saved beside `--out` every page, so an
interrupted run continues exactly where it stopped. It never prompts; a quota
exhaustion saves and exits with a message.

Output columns are what `enrol` reads (`openalex_id`, `doi`,
`publication_year`) plus `title`, `venue`, `is_oa`, `oa_status`,
`cited_by_count`, `type`. A `<out>.query.json` records the exact query, so the
corpus can be re-drawn.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import requests

OPENALEX_WORKS = "https://api.openalex.org/works"
PER_PAGE = 200                      # OpenAlex's maximum
COLUMNS = ["openalex_id", "doi", "title", "publication_year", "venue",
           "is_oa", "oa_status", "cited_by_count", "type"]
SEARCH_FIELDS = ("title", "title_and_abstract", "fulltext")


class QuotaExhausted(RuntimeError):
    """The server asked for a wait that is a quota reset, not a throttle."""


@dataclass
class Query:
    subject: str
    search_in: str = "title_and_abstract"
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    extra_filter: str = ""
    max_papers: Optional[int] = None

    def __post_init__(self):
        self.subject = (self.subject or "").strip()
        if not self.subject:
            raise ValueError("--subject is required and may not be empty")
        if self.search_in not in SEARCH_FIELDS:
            raise ValueError(f"--search-in must be one of {SEARCH_FIELDS}")
        if "," in self.subject and self.search_in != "fulltext":
            # A comma separates OpenAlex filters, so it would split the subject.
            raise ValueError("a subject used in a title/abstract filter may not "
                             "contain a comma; use OR between terms instead")

    def filters(self) -> list[str]:
        out = []
        if self.search_in == "title":
            out.append(f"title.search:{self.subject}")
        elif self.search_in == "title_and_abstract":
            out.append(f"title_and_abstract.search:{self.subject}")
        if self.year_min and self.year_max:
            out.append(f"publication_year:{self.year_min}-{self.year_max}")
        elif self.year_min:
            out.append(f"publication_year:>{self.year_min - 1}")
        elif self.year_max:
            out.append(f"publication_year:<{self.year_max + 1}")
        if self.extra_filter:
            out.append(self.extra_filter)
        return out

    def params(self) -> dict:
        p: dict = {"per-page": PER_PAGE}
        if self.search_in == "fulltext":
            p["search"] = self.subject
        f = self.filters()
        if f:
            p["filter"] = ",".join(f)
        return p


@dataclass
class Http:
    """The one network boundary, so tests can fake it."""
    api_key: str = ""
    get: Callable = field(default=requests.get)
    sleep: Callable = field(default=time.sleep)
    max_wait: float = 120.0
    retries: int = 6

    def works(self, params: dict) -> dict:
        params = dict(params)
        if self.api_key:
            params["api_key"] = self.api_key
        for attempt in range(self.retries):
            r = self.get(OPENALEX_WORKS, params=params, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                ra = r.headers.get("Retry-After")
                wait = float(ra) if ra else float(2 ** attempt)
                if wait > self.max_wait:
                    raise QuotaExhausted(
                        f"OpenAlex asked for a {wait:.0f}s wait: the daily quota is "
                        f"spent. Rerun later (or with another OPENALEX_API_KEY); the "
                        f"harvest resumes from the saved cursor.")
                self.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()
        return r.json()


def _row(w: dict) -> dict:
    loc = w.get("primary_location") or {}
    src = loc.get("source") or {}
    oa = w.get("open_access") or {}
    return {
        "openalex_id": w.get("id", ""),
        "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
        "title": w.get("title") or w.get("display_name") or "",
        "publication_year": w.get("publication_year"),
        "venue": src.get("display_name") or "",
        "is_oa": oa.get("is_oa"),
        "oa_status": oa.get("oa_status") or "",
        "cited_by_count": w.get("cited_by_count"),
        "type": w.get("type") or "",
    }


def _state_path(out: Path) -> Path:
    return out.with_name(out.name + ".cursor.json")


def collect(q: Query, out: Path, http: Http, *, log=print) -> int:
    """Harvest into `out`, appending and resuming. Returns rows on disk."""
    out = Path(out)
    state = _state_path(out)
    seen: set[str] = set()
    if out.exists():
        with open(out, newline="", encoding="utf-8") as f:
            seen = {r["openalex_id"] for r in csv.DictReader(f)}
    cursor = "*"
    if state.exists():
        saved = json.loads(state.read_text())
        if saved.get("params") != q.params():
            sys.exit(f"{state} belongs to a different query; delete it or use "
                     f"another --out")
        cursor = saved.get("cursor") or "*"
        log(f"resuming: {len(seen)} rows on disk")
    elif seen:
        sys.exit(f"{out} exists with no cursor file, so it is a finished harvest "
                 f"or another query's. Use another --out.")

    Path(str(out) + ".query.json").write_text(json.dumps({
        "subject": q.subject, "search_in": q.search_in, "year_min": q.year_min,
        "year_max": q.year_max, "extra_filter": q.extra_filter,
        "openalex_params": q.params(),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))

    new_file = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            w.writeheader()
        while True:
            if q.max_papers and len(seen) >= q.max_papers:
                break
            params = dict(q.params(), cursor=cursor)
            try:
                data = http.works(params)
            except QuotaExhausted as e:
                f.flush()
                state.write_text(json.dumps({"cursor": cursor, "params": q.params()}))
                log(str(e))
                return len(seen)
            results = data.get("results") or []
            for work in results:
                r = _row(work)
                if not r["openalex_id"] or r["openalex_id"] in seen:
                    continue
                if q.max_papers and len(seen) >= q.max_papers:
                    break
                seen.add(r["openalex_id"])
                w.writerow(r)
            f.flush()
            total = (data.get("meta") or {}).get("count")
            nxt = (data.get("meta") or {}).get("next_cursor")
            log(f"  {len(seen)} / {total} works")
            if not results or not nxt or nxt == cursor:
                break
            cursor = nxt
            state.write_text(json.dumps({"cursor": cursor, "params": q.params()}))
            http.sleep(0.1)
    if state.exists():
        state.unlink()                  # finished: nothing left to resume
    return len(seen)


def cmd_collect(a) -> None:
    q = Query(subject=a.subject, search_in=a.search_in, year_min=a.year_min,
              year_max=a.year_max, extra_filter=a.filter or "",
              max_papers=a.max_papers)
    http = Http(api_key=os.environ.get("OPENALEX_API_KEY", ""))
    print(f"OpenAlex query: {json.dumps(q.params())}", file=sys.stderr)
    n = collect(q, Path(a.out), http, log=lambda m: print(m, file=sys.stderr))
    print(f"{n} works in {a.out}", file=sys.stderr)


def add_parser(sub) -> None:
    c = sub.add_parser("collect", help="harvest a subject's papers from OpenAlex")
    c.set_defaults(fn=cmd_collect)
    c.add_argument("--subject", required=True,
                   help='the field of interest, e.g. "quantum computing"; OpenAlex '
                        'search syntax (AND, OR, NOT, "quoted phrases") works')
    c.add_argument("--search-in", choices=SEARCH_FIELDS, default="title_and_abstract")
    c.add_argument("--year-min", type=int)
    c.add_argument("--year-max", type=int)
    c.add_argument("--filter", help="extra OpenAlex filter, e.g. type:article")
    c.add_argument("--max-papers", type=int)
    c.add_argument("--out", default="papers.csv")
