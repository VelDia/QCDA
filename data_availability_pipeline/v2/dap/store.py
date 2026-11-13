"""dap.store — durable state in SQLite instead of append-only CSV.

v1 resumed by reading analysis.csv with pandas inside a bare `except` that
returned an empty set on any failure. One malformed row — an unescaped newline
inside a quoted field, which hand-edited CSVs acquire routinely — and the next
run silently reprocessed the entire corpus, duplicating every record and
re-spending the API budget. Nothing surfaced the problem.

SQLite fixes the whole class of problem: openalex_id is a PRIMARY KEY so a
duplicate is impossible rather than merely unlikely, writes are atomic so a
kill -9 mid-paper cannot leave a half-row, and `status` makes the difference
between "not yet attempted", "failed transiently, retry" and "failed
permanently, stop trying" explicit instead of inferred.

CSV is still what you analyse in — `export_csv()` writes it on demand — but it
is a view of the database, never the source of truth.
"""

from __future__ import annotations

import csv
import json
import sqlite3
import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional

from .redact import redact
from .schema import FLAT_COLUMNS, SCHEME_VERSION, PaperRecord

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS papers (
    openalex_id     TEXT PRIMARY KEY,
    doi             TEXT,
    status          TEXT NOT NULL,          -- pending|done|failed|dead|skipped|excluded|outdated
    stage           TEXT NOT NULL DEFAULT 'enrolled',
    batch_id        TEXT,
    attempts        INTEGER NOT NULL DEFAULT 0,
    scheme_version  TEXT,
    arm             TEXT,
    meta_json       TEXT,                   -- carried between stages
    record_json     TEXT,                   -- full PaperRecord.flat() payload
    last_error      TEXT,
    updated_at      TEXT NOT NULL,
    -- Publication year, denormalised out of meta_json so a batch can be
    -- ordered and filtered by it without parsing 5,000 JSON blobs. Written by
    -- `advance` from the metadata stage; NULL until that stage has run.
    year            INTEGER
);
CREATE INDEX IF NOT EXISTS idx_papers_status ON papers(status);
CREATE INDEX IF NOT EXISTS idx_papers_arm    ON papers(arm);
CREATE INDEX IF NOT EXISTS idx_papers_stage  ON papers(stage, status);

-- Bulk classification bookkeeping. A batch is recorded BEFORE its spool file
-- is deleted, so the worst outcome of a crash mid-submit is a duplicate
-- submission, never a set of papers that silently never come back.
CREATE TABLE IF NOT EXISTS batches (
    batch_id    TEXT PRIMARY KEY,
    model       TEXT,
    n           INTEGER,
    status      TEXT NOT NULL,              -- open|closed
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batch_members (
    batch_id     TEXT NOT NULL,
    custom_id    TEXT NOT NULL,
    openalex_id  TEXT NOT NULL,
    meta_json    TEXT NOT NULL,
    PRIMARY KEY (batch_id, custom_id)
);
CREATE INDEX IF NOT EXISTS idx_bm_work ON batch_members(openalex_id);

CREATE TABLE IF NOT EXISTS evidence (
    openalex_id  TEXT NOT NULL,
    ev_id        INTEGER NOT NULL,
    quote        TEXT NOT NULL,
    section      TEXT,
    locator      TEXT,
    source       TEXT,
    PRIMARY KEY (openalex_id, ev_id)
);

CREATE TABLE IF NOT EXISTS verifications (
    openalex_id  TEXT NOT NULL,
    axis         TEXT NOT NULL,
    identifier   TEXT NOT NULL,
    status       TEXT NOT NULL,
    http_status  INTEGER,
    final_url    TEXT,
    note         TEXT,
    checked_at   TEXT NOT NULL,
    PRIMARY KEY (openalex_id, axis, identifier)
);

CREATE TABLE IF NOT EXISTS failures (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    openalex_id  TEXT NOT NULL,
    stage        TEXT NOT NULL,
    error_type   TEXT,
    message      TEXT,
    at           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS usage (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    openalex_id   TEXT,
    model         TEXT,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    cost_usd      REAL,
    at            TEXT NOT NULL
);
"""

#: A paper that failed this many times transiently is parked as `dead` so the
#: run stops paying for it. v1 retried forever and grew failures.csv unbounded.
MAX_ATTEMPTS = 3

#: DOI registrants that mint DOIs for preprints, self-deposits and data
#: collections rather than for versions of record. A row whose *own* DOI carries one of these prefixes
#: is a preprint record, not a published article -- which matters because the
#: study's unit is the version of record, and because the same paper is often
#: enrolled twice, once under its arXiv DOI and once under its publisher's.
#: Venues that are repositories rather than journals. A row whose venue is one
#: of these is a preprint or self-deposit record even when it carries no DOI of
#: its own -- and a row with no DOI is exactly what the prefix test cannot see.
PREPRINT_VENUE_PREFIXES: tuple[str, ...] = (
    "arXiv",
    "Zenodo",
    "Research Square",
    "SSRN",
    "Preprints.org",
    "bioRxiv",
    "medRxiv",
    "TechRxiv",
    "OSF",
    "Figshare",
    "ChemRxiv",
)

#: Repository and platform names that appear anywhere in a venue string rather
#: than at its start -- "DSpace@MIT", "Research Repository (Delft University of
#: Technology)", "Institutional Repository of Leibniz Universitaet Hannover".
#: A prefix test cannot reach these; the names are too varied to enumerate.
#: Validated before use: these substrings reject 0 of the 6,073 papers already
#: coded, and 140 rows in the classify queue. No journal in this corpus carries
#: "Repository", "Theses" or "Dissertation" in its title.
PREPRINT_VENUE_SUBSTRINGS: tuple[str, ...] = (
    "Repository",
    "repository",
    "ResearchData",
    "Dataverse",
    "DSpace",
    "EPrints",
    "Qeios",
    "eScholarship",
    "Publication Server",
    "Institutional",
    "Theses",
    "Dissertation",
    "Research Portal",
    "Open Research",
    "Strathprints",
    "Publikationsportal",
    "Open Portal",
)

PREPRINT_DOI_PREFIXES: tuple[str, ...] = (
    "10.48550/",  # arXiv
    "10.5281/",  # Zenodo
    "10.21203/",  # Research Square
    "10.2139/",  # SSRN
    "10.20944/",  # Preprints.org
    "10.6084/",  # figshare, incl. the `m9.figshare.c.` collection records
    "10.60893/",  # publisher-run figshare (AIP and others), supplementary only
    # Added 3 Oct (S100) after batch 108 exported `W4413166857`, an engrXiv
    # preprint with an empty venue string, so neither the prefix nor the venue
    # test fired. Measured on dap_frame at the time: 273 queued rows carry one
    # of these seven prefixes and the venue test would already have caught only
    # 83 of them. Five rows were coded before the fix and are not changed by it
    # -- `--publisher-doi-only` is a scope on the claim, not an exclusion.
    "10.36227/",  # TechRxiv            (125 queued)
    "10.26434/",  # ChemRxiv             (73)
    "10.31219/",  # OSF Preprints        (28)
    "10.22541/",  # Authorea             (21)
    "10.1101/",  # bioRxiv and medRxiv  (10)
    "10.31224/",  # engrXiv               (9)
    "10.31234/",  # PsyArXiv              (7)
)


def _scheme_family(version: Optional[str]) -> object:
    """The part of a scheme_version that must match for rows to pool: major and
    minor. Patch versions mark rulings on the same vocabulary (Diana, 28 Oct).
    Anything that is not dotted numbers, NULL included, is its own family.
    """
    parts = (version or "").split(".")
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        return tuple(int(p) for p in parts[:2])
    return version


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self._local = threading.local()
        with self._conn() as c:
            c.executescript(SCHEMA_SQL)
            # An existing database predates the `year` column. Adding it is
            # additive: no stored value is read, rewritten or lost.
            cols = {r["name"] for r in c.execute("PRAGMA table_info(papers)")}
            if "year" not in cols:
                c.execute("ALTER TABLE papers ADD COLUMN year INTEGER")
            c.execute("CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year)")
            self._guard_unique_dois(c)
        self.backfill_years()

    @staticmethod
    def _guard_unique_dois(c: sqlite3.Connection) -> None:
        """One live row per DOI, enforced by the database rather than by care.

        OpenAlex mints more than one work id for the same DOI, and
        `is_classified` keys on the id, so a paper enrolled twice is classified
        twice and the two records can disagree. 399 DOIs in dap_frame were held
        by two rows before this went in.

        The index is partial on purpose. A row with no DOI is outside it, since
        6.5% of the corpus has none and an empty string is not an identity; and
        an `excluded` row is outside it too, because `exclude` keeps the row and
        its codes as a tombstone, so the losing copy of a resolved duplicate
        must be allowed to keep sitting there with the same DOI.

        A database that already holds duplicates gets a warning and no index,
        not an exception: refusing to open it would leave the operator unable to
        run the `exclude` that fixes it.
        """
        try:
            c.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_papers_live_doi "
                "ON papers(doi) WHERE doi IS NOT NULL AND doi != '' "
                "AND status != 'excluded'"
            )
        except sqlite3.IntegrityError:
            dups = [
                r["doi"]
                for r in c.execute(
                    "SELECT doi FROM papers WHERE doi IS NOT NULL AND doi != '' "
                    "AND status != 'excluded' GROUP BY doi HAVING COUNT(*) > 1 LIMIT 5"
                )
            ]
            print(
                f"warning: {len(dups)}+ DOIs are held by more than one live row, "
                f"so the uniqueness index was not created. First: "
                f"{', '.join(dups)}. Resolve with `exclude` and reopen.",
                file=sys.stderr,
            )

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=60, isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        try:
            yield conn
        except Exception:
            # Only roll back if a transaction is actually open. The connection
            # runs in autocommit (isolation_level=None), so an unconditional
            # ROLLBACK raises OperationalError and *replaces* the real
            # exception — turning every store-layer bug into a misleading
            # "cannot rollback" message.
            if conn.in_transaction:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
            raise

    # -- enrolment ---------------------------------------------------------
    def enrol(self, rows: Iterable[dict]) -> int:
        """Add corpus rows as `pending`. Idempotent — re-running never duplicates.

        A row may carry the harvest's `year`. Storing it lets the batch window
        skip an old paper before the metadata stage spends requests on it; the
        metadata stage still overwrites it with OpenAlex's year.
        """
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        n = 0
        with self._conn() as c:
            c.execute("BEGIN")
            for r in rows:
                y = r.get("year")
                cur = c.execute(
                    "INSERT OR IGNORE INTO papers "
                    "(openalex_id, doi, status, arm, updated_at, year) "
                    "VALUES (?,?,'pending',?,?,?)",
                    (
                        r["openalex_id"],
                        r.get("doi", ""),
                        r.get("arm", ""),
                        now,
                        int(y) if y is not None else None,
                    ),
                )
                n += cur.rowcount
            c.execute("COMMIT")
        return n

    def pending(self, limit: Optional[int] = None, arm: Optional[str] = None) -> list[dict]:
        q = (
            "SELECT openalex_id, doi, arm, attempts FROM papers "
            "WHERE status IN ('pending','failed') AND attempts < ?"
        )
        args: list = [MAX_ATTEMPTS]
        if arm:
            q += " AND arm = ?"
            args.append(arm)
        q += " ORDER BY attempts ASC, openalex_id ASC"
        if limit:
            q += f" LIMIT {int(limit)}"
        with self._conn() as c:
            return [dict(r) for r in c.execute(q, args)]

    # -- results -----------------------------------------------------------
    def save(self, rec: PaperRecord, verifications: list = ()) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        flat = rec.flat()
        with self._conn() as c:
            c.execute("BEGIN")
            c.execute(
                "UPDATE papers SET status='done', scheme_version=?, arm=?, "
                "record_json=?, last_error=NULL, updated_at=? WHERE openalex_id=?",
                (rec.scheme_version, rec.arm.value, json.dumps(flat), now, rec.openalex_id),
            )
            c.execute("DELETE FROM evidence WHERE openalex_id=?", (rec.openalex_id,))
            c.executemany(
                "INSERT INTO evidence VALUES (?,?,?,?,?,?)",
                [
                    (rec.openalex_id, e.id, e.quote, e.section, e.locator, e.source)
                    for e in rec.evidence
                ],
            )
            for axis, vr in verifications:
                c.execute(
                    "INSERT OR REPLACE INTO verifications VALUES (?,?,?,?,?,?,?,?)",
                    (
                        rec.openalex_id,
                        axis,
                        vr.identifier,
                        vr.status.value,
                        vr.http_status,
                        vr.final_url,
                        vr.note,
                        now,
                    ),
                )
            c.execute("COMMIT")

    def fail(
        self,
        openalex_id: str,
        stage: str,
        error_type: str,
        message: str,
        *,
        permanent: bool = False,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._conn() as c:
            c.execute("BEGIN")
            c.execute(
                "INSERT INTO failures (openalex_id, stage, error_type, message, at) "
                "VALUES (?,?,?,?,?)",
                (openalex_id, stage, error_type, redact(message)[:2000], now),
            )
            c.execute(
                "UPDATE papers SET status=?, attempts=attempts+1, last_error=?, "
                "updated_at=? WHERE openalex_id=?",
                ("dead" if permanent else "failed", f"{stage}:{error_type}", now, openalex_id),
            )
            c.execute("COMMIT")

    def record_usage(self, openalex_id: str, usage, *, discount: float = 1.0) -> None:
        """`discount` is 0.5 for work billed through the Message Batches API, so
        the running cost meter reflects what is actually spent rather than the
        synchronous list price.
        """
        with self._conn() as c:
            c.execute(
                "INSERT INTO usage (openalex_id, model, input_tokens, output_tokens, "
                "cost_usd, at) VALUES (?,?,?,?,?,?)",
                (
                    openalex_id,
                    usage.model,
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.cost_usd * discount,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ),
            )

    def total_cost(self) -> float:
        with self._conn() as c:
            row = c.execute("SELECT COALESCE(SUM(cost_usd),0) AS t FROM usage").fetchone()
            return float(row["t"])

    # -- stage machine ------------------------------------------------------
    #: Papers move forward only. A stage runner claims work at its input stage
    #: and advances it; a crash anywhere leaves the paper at its last completed
    #: stage, so re-running the same command resumes rather than restarts.
    STAGES = [
        "enrolled",
        "metadata",
        "retrieved",
        "classify_queued",
        "classified",
        "verified",
        "done",
    ]

    def backfill_years(self) -> int:
        """Fill `year` from meta_json for rows written before the column existed.

        Read-only with respect to everything else: it touches one column, only
        where it is NULL, and only from a value already stored in that same row.
        """
        with self._conn() as c:
            cur = c.execute(
                "UPDATE papers SET year = CAST(json_extract(meta_json, '$.year') AS INTEGER) "
                "WHERE year IS NULL AND meta_json IS NOT NULL "
                "AND json_extract(meta_json, '$.year') IS NOT NULL"
            )
            return cur.rowcount or 0

    def years(
        self, *, stage: Optional[str] = None, coded_only: bool = False
    ) -> dict[Optional[int], int]:
        """Publication year histogram. `coded_only` counts only papers that
        carry a classification, which is what "reviewed papers" means.
        """
        q = "SELECT year, COUNT(*) AS n FROM papers WHERE 1=1"
        args: list = []
        if stage:
            q += " AND stage = ?"
            args.append(stage)
        if coded_only:
            q += " AND record_json IS NOT NULL"
        q += " GROUP BY year ORDER BY year IS NULL, year DESC"
        with self._conn() as c:
            return {r["year"]: r["n"] for r in c.execute(q, args)}

    def claim(
        self,
        stage: str,
        limit: int,
        arm: Optional[str] = None,
        *,
        year_min: Optional[int] = None,
        newest_first: bool = False,
        publisher_doi_only: bool = False,
        skip_coded_dois: bool = False,
    ) -> list[dict]:
        """Papers sitting at `stage` and still workable.

        `year_min` drops papers published before it. A paper whose year is not
        yet known is KEPT, not dropped: the year arrives with the metadata
        stage, so filtering an unknown year would silently discard every paper
        that has not reached it. `unknown_years()` counts what that admits.

        `publisher_doi_only` drops rows whose own DOI was minted by a preprint
        registrar (`PREPRINT_DOI_PREFIXES`) and rows whose venue is a repository
        rather than a journal (`PREPRINT_VENUE_PREFIXES` at the start of the
        string, `PREPRINT_VENUE_SUBSTRINGS` anywhere in it). Both tests are needed:
        an arXiv row often carries no DOI at all, which the prefix test cannot
        see. A row with neither a DOI nor a venue is KEPT, for the same reason an
        unknown year is: absence is not evidence of a preprint.
        Nothing is excluded or modified -- the rows stay pending and claimable
        the moment the filter is off.

        `skip_coded_dois` drops rows whose DOI is already carried by a paper at
        `verified/done`. OpenAlex sometimes mints two work ids for one DOI, and
        `is_classified` keys on the id, so without this the same paper is coded
        twice under two ids. Measured on dap_frame: 259 publisher DOIs are held
        by two rows, and 74 of those have one copy coded and one still queued.
        """
        q = (
            "SELECT openalex_id, doi, arm, meta_json, attempts FROM papers "
            "WHERE stage = ? AND status IN ('pending','failed') AND attempts < ?"
        )
        args: list = [stage, MAX_ATTEMPTS]
        if arm:
            q += " AND arm = ?"
            args.append(arm)
        if year_min is not None:
            q += " AND (year IS NULL OR year >= ?)"
            args.append(int(year_min))
        if publisher_doi_only:
            for pfx in PREPRINT_DOI_PREFIXES:
                q += " AND (doi IS NULL OR doi = '' OR doi NOT LIKE ?)"
                args.append(pfx + "%")
            for pfx in PREPRINT_VENUE_PREFIXES:
                q += (
                    " AND (json_extract(meta_json, '$.venue') IS NULL "
                    "OR json_extract(meta_json, '$.venue') NOT LIKE ?)"
                )
                args.append(pfx + "%")
            for sub in PREPRINT_VENUE_SUBSTRINGS:
                q += (
                    " AND (json_extract(meta_json, '$.venue') IS NULL "
                    "OR json_extract(meta_json, '$.venue') NOT GLOB ?)"
                )
                args.append("*" + sub + "*")
        if skip_coded_dois:
            q += (
                " AND (doi IS NULL OR doi = '' OR doi NOT IN "
                "(SELECT doi FROM papers WHERE stage='verified' AND status='done' "
                "AND doi IS NOT NULL AND doi != ''))"
            )
        order = "year IS NULL ASC, year DESC, " if newest_first else ""
        q += f" ORDER BY {order}attempts ASC, openalex_id ASC LIMIT {int(limit)}"
        with self._conn() as c:
            out = []
            for r in c.execute(q, args):
                d = dict(r)
                d["meta"] = json.loads(d.pop("meta_json") or "{}")
                out.append(d)
            return out

    def advance(self, openalex_id: str, stage: str, meta: dict | None = None) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._conn() as c:
            if meta is None:
                c.execute(
                    "UPDATE papers SET stage=?, status='pending', "
                    "last_error=NULL, updated_at=? WHERE openalex_id=?",
                    (stage, now, openalex_id),
                )
            else:
                y = meta.get("year")
                c.execute(
                    "UPDATE papers SET stage=?, status='pending', meta_json=?, "
                    "year=COALESCE(?, year), last_error=NULL, updated_at=? "
                    "WHERE openalex_id=?",
                    (
                        stage,
                        json.dumps(meta),
                        int(y) if isinstance(y, int) else None,
                        now,
                        openalex_id,
                    ),
                )

    def stage_counts(self) -> dict[str, int]:
        with self._conn() as c:
            return {
                f"{r['stage']}/{r['status']}": r["n"]
                for r in c.execute(
                    "SELECT stage, status, COUNT(*) AS n FROM papers "
                    "GROUP BY stage, status ORDER BY stage"
                )
            }

    # -- batch bookkeeping --------------------------------------------------
    def record_batch(
        self, *, batch_id: str, model: str, n: int, members: list[tuple[str, dict]]
    ) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._conn() as c:
            c.execute("BEGIN")
            c.execute(
                "INSERT OR REPLACE INTO batches VALUES (?,?,?,'open',?)", (batch_id, model, n, now)
            )
            c.executemany(
                "INSERT OR REPLACE INTO batch_members VALUES (?,?,?,?)",
                [(batch_id, cid, meta["openalex_id"], json.dumps(meta)) for cid, meta in members],
            )
            c.executemany(
                "UPDATE papers SET stage='classify_queued', batch_id=?, updated_at=? "
                "WHERE openalex_id=?",
                [(batch_id, now, meta["openalex_id"]) for _, meta in members],
            )
            c.execute("COMMIT")

    def open_batches(self) -> list[str]:
        with self._conn() as c:
            return [
                r["batch_id"]
                for r in c.execute(
                    "SELECT batch_id FROM batches WHERE status='open' ORDER BY created_at"
                )
            ]

    def batch_members(self, batch_id: str) -> dict[str, dict]:
        with self._conn() as c:
            return {
                r["custom_id"]: json.loads(r["meta_json"])
                for r in c.execute(
                    "SELECT custom_id, meta_json FROM batch_members WHERE batch_id=?", (batch_id,)
                )
            }

    def batch_model(self, batch_id: str) -> str:
        with self._conn() as c:
            r = c.execute("SELECT model FROM batches WHERE batch_id=?", (batch_id,)).fetchone()
            return r["model"] if r else ""

    def close_batch(self, batch_id: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE batches SET status='closed' WHERE batch_id=?", (batch_id,))

    def paper_meta(self, openalex_id: str) -> dict:
        """Full stored metadata for one paper, passages included. The batch spool
        keeps only a slim copy (no passages, no DAS text), so anything that must
        check a result against what the model was shown reads it from here.
        """
        with self._conn() as c:
            r = c.execute(
                "SELECT meta_json FROM papers WHERE openalex_id=?", (openalex_id,)
            ).fetchone()
        return json.loads(r["meta_json"]) if r and r["meta_json"] else {}

    def record_preprint_das(self, openalex_id: str, result: dict) -> str:
        """Store whether the paper's preprint carries an availability statement.

        A fact about a second copy of the paper, kept beside the codes rather
        than in them: the codes describe the version of record. It lives in
        `meta_json` under a key that `requeue_for_recode` does not strip, so it
        survives the recode that brings in the published text.
        """
        with self._conn() as c:
            c.execute("BEGIN")
            r = c.execute(
                "SELECT meta_json FROM papers WHERE openalex_id=?", (openalex_id,)
            ).fetchone()
            if r is None:
                c.execute("ROLLBACK")
                return "not enrolled"
            meta = json.loads(r["meta_json"] or "{}")
            meta["preprint_das"] = result
            c.execute(
                "UPDATE papers SET meta_json=? WHERE openalex_id=?", (json.dumps(meta), openalex_id)
            )
            c.execute("COMMIT")
        return ""

    def is_classified(self, openalex_id: str) -> bool:
        with self._conn() as c:
            r = c.execute("SELECT stage FROM papers WHERE openalex_id=?", (openalex_id,)).fetchone()
            return bool(r) and r["stage"] in ("classified", "verified", "done")

    def save_classified(self, rec, *, batch_id: str = "") -> None:
        """Store a classified record and advance it to the verify stage."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._conn() as c:
            c.execute("BEGIN")
            c.execute(
                "UPDATE papers SET stage='classified', status='pending', "
                "scheme_version=?, arm=?, record_json=?, batch_id=?, "
                "last_error=NULL, updated_at=? WHERE openalex_id=?",
                (
                    rec.scheme_version,
                    rec.arm.value,
                    json.dumps(rec.flat()),
                    batch_id,
                    now,
                    rec.openalex_id,
                ),
            )
            c.execute("DELETE FROM evidence WHERE openalex_id=?", (rec.openalex_id,))
            c.executemany(
                "INSERT INTO evidence VALUES (?,?,?,?,?,?)",
                [
                    (rec.openalex_id, e.id, e.quote, e.section, e.locator, e.source)
                    for e in rec.evidence
                ],
            )
            c.execute("COMMIT")

    def requeue_for_sync(self, openalex_id: str, *, reason: str) -> None:
        """A batch result that failed schema validation. There is no
        conversation to continue in a batch, so it goes back to the synchronous
        path where the validator's complaint can be handed to the model.
        """
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._conn() as c:
            c.execute("BEGIN")
            c.execute(
                "INSERT INTO failures (openalex_id, stage, error_type, message, at) "
                "VALUES (?,?,?,?,?)",
                (openalex_id, "classify_batch", "SchemaError", reason, now),
            )
            c.execute(
                "UPDATE papers SET stage='retrieved', status='pending', "
                "attempts=attempts+1, last_error='batch schema failure', "
                "updated_at=? WHERE openalex_id=?",
                (now, openalex_id),
            )
            c.execute("COMMIT")

    #: What `stage_retrieve_one` adds on top of the metadata facts. Requeueing
    #: strips exactly these, so the re-run starts from metadata rather than from
    #: a stale retrieval.
    RETRIEVAL_KEYS = (
        "passages",
        "retrieval",
        "text_source",
        "text_url",
        "text_complete",
        "pdf_parser",
        "retrieval_attempts",
        "passages_are_structural",
        "structural_context",
    )

    def requeue_for_retrieval(self, openalex_id: str, *, reason: str) -> str:
        """Send a retrieved, not-yet-classified paper back through retrieval.

        For a paper whose stored text is known to be the wrong copy, such as a
        preprint picked up while the version of record was throttled. Returns ""
        on success, otherwise why it was refused. A paper that already carries a
        classification is refused: its codes were judged against the text it
        has, and swapping that text underneath them would detach the codes from
        their evidence. The requeue is logged to `failures` as an audit trail.
        """
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._conn() as c:
            c.execute("BEGIN")
            r = c.execute(
                "SELECT stage, meta_json FROM papers WHERE openalex_id=?", (openalex_id,)
            ).fetchone()
            why = (
                "not enrolled"
                if r is None
                else ""
                if r["stage"] == "retrieved"
                else f"at stage {r['stage']}; only retrieved, unclassified papers can be requeued"
            )
            if why:
                c.execute("ROLLBACK")
                return why
            meta = json.loads(r["meta_json"] or "{}")
            for k in self.RETRIEVAL_KEYS:
                meta.pop(k, None)
            c.execute(
                "INSERT INTO failures (openalex_id, stage, error_type, message, at) "
                "VALUES (?,?,?,?,?)",
                (openalex_id, "retrieve", "requeued", reason[:2000], now),
            )
            c.execute(
                "UPDATE papers SET stage='metadata', status='pending', meta_json=?, "
                "last_error='retrieve:requeued', updated_at=? WHERE openalex_id=?",
                (json.dumps(meta), now, openalex_id),
            )
            c.execute("COMMIT")
        return ""

    def clear_attempts(self, openalex_id: str, *, reason: str) -> str:
        """Give a paper that ran out of attempts at its stage another try.

        `fail` bumps `attempts` whatever the cause, so a schema rejection in
        `--mode manual` -- a coder-side mistake, not a network one -- spends the
        same budget as a timeout. Three malformed lines therefore retire a paper
        for good: `claim` skips it at MAX_ATTEMPTS, and every `requeue_*` refuses
        it because there is no classification to replace. This is the way back,
        and it is deliberately narrow: it resets the counter and nothing else,
        so the stored text and metadata are untouched.

        Returns "" on success, otherwise why it was refused.
        """
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._conn() as c:
            c.execute("BEGIN")
            r = c.execute(
                "SELECT status, attempts, record_json FROM papers WHERE openalex_id=?",
                (openalex_id,),
            ).fetchone()
            why = (
                "not enrolled"
                if r is None
                else f"status is {r['status']}, not failed; nothing to retry"
                if r is not None and r["status"] != "failed"
                else "already carries a classification; use requeue or recode"
                if r is not None and r["record_json"]
                else ""
            )
            if why:
                c.execute("ROLLBACK")
                return why
            c.execute(
                "INSERT INTO failures (openalex_id, stage, error_type, message, at) "
                "VALUES ((SELECT openalex_id FROM papers WHERE openalex_id=?), "
                "(SELECT stage FROM papers WHERE openalex_id=?), ?, ?, ?)",
                (
                    openalex_id,
                    openalex_id,
                    "attempts_cleared",
                    f"attempts {r['attempts']} -> 0: {reason}"[:2000],
                    now,
                ),
            )
            c.execute(
                "UPDATE papers SET status='pending', attempts=0, "
                "last_error=NULL, updated_at=? WHERE openalex_id=?",
                (now, openalex_id),
            )
            c.execute("COMMIT")
        return ""

    def requeue_for_reclassification(self, openalex_id: str, *, reason: str) -> str:
        """Send a paper back to the classifier, keeping the text it already has.

        This is for one case only: a paper the gate stubbed out without ever
        reading it. Those rows carry `not_applicable` on both axes, an empty
        `model_id` and `needs_human_review`, and 199 of them have retrievable
        text sitting in `meta_json` that nobody has looked at.

        It is the narrow sibling of `requeue_for_recode`, which discards the
        text too. Here the text is not in question, so re-retrieving would spend
        requests to arrive at the same passages, and risk coming back with less
        than is already stored.

        It REFUSES any paper whose record names a coder. A real classification
        is a judgement about evidence, and quietly replacing one with another
        coder's is how a two-coder study loses the thing it is measuring — use
        `requeue_for_recode` and its audit trail for that. Returns "" on
        success, else the refusal.
        """
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._conn() as c:
            c.execute("BEGIN")
            r = c.execute(
                "SELECT meta_json, record_json FROM papers WHERE openalex_id=?", (openalex_id,)
            ).fetchone()
            if r is None:
                c.execute("ROLLBACK")
                return "not enrolled"
            rec = json.loads(r["record_json"] or "{}")
            if not rec:
                c.execute("ROLLBACK")
                return "no classification to replace"
            if rec.get("model_id"):
                c.execute("ROLLBACK")
                return f"carries a real classification by {rec['model_id']}; use recode"
            meta = json.loads(r["meta_json"] or "{}")
            # `retrieval` says text was fetched once, not that anything readable
            # survived into the row. 140 papers are marked `partial_text` with
            # zero passages: fetched before `structural_passages` existed, so
            # every keyword window was dropped as reference-list text and
            # nothing was stored. Those need re-retrieval, not reclassification,
            # and admitting them here only burns a claim and re-stubs them.
            from .classify import has_classifiable_evidence

            if not has_classifiable_evidence(
                crossref_summary={
                    k: meta.get(k) or [] for k in ("das_texts", "dataset_ids", "confirmed_datasets")
                },
                passages=meta.get("passages") or [],
            ):
                c.execute("ROLLBACK")
                return "no stored evidence to classify from; use recode to re-retrieve"
            c.execute(
                "INSERT INTO failures (openalex_id, stage, error_type, message, at) "
                "VALUES (?,?,?,?,?)",
                (
                    openalex_id,
                    "classify",
                    "reclassify",
                    (reason[:1000] + " || discarded stub: " + (r["record_json"] or ""))[:8000],
                    now,
                ),
            )
            c.execute("DELETE FROM evidence WHERE openalex_id=?", (openalex_id,))
            c.execute(
                "UPDATE papers SET stage='retrieved', status='pending', attempts=0, "
                "record_json=NULL, batch_id=NULL, last_error='classify:reclassify', "
                "updated_at=? WHERE openalex_id=?",
                (now, openalex_id),
            )
            c.execute("COMMIT")
        return ""

    def requeue_for_recode(self, openalex_id: str, *, reason: str) -> str:
        """Discard a finished paper's text *and* its codes, and start it over.

        `requeue_for_retrieval` refuses a classified paper because swapping the
        text underneath a classification detaches the codes from the evidence
        they were judged against. That objection is to keeping the codes. When
        the stored text is known to be the wrong document — a catalogue record,
        a preprint standing in for the version of record, an extraction that
        dropped the availability statement — the codes are not worth keeping
        either: they are a reading of a document that is not the paper.

        So this throws both away together, which leaves nothing detached, and
        logs what it did. The paper returns to `metadata` and is re-retrieved
        and re-classified from scratch. Returns "" on success, else the refusal.

        Deliberately not reversible in place: the previous record is written to
        `failures` before it is cleared, so the audit trail holds what was
        discarded and why.
        """
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._conn() as c:
            c.execute("BEGIN")
            r = c.execute(
                "SELECT stage, status, meta_json, record_json FROM papers WHERE openalex_id=?",
                (openalex_id,),
            ).fetchone()
            if r is None:
                c.execute("ROLLBACK")
                return "not enrolled"
            meta = json.loads(r["meta_json"] or "{}")
            for k in self.RETRIEVAL_KEYS:
                meta.pop(k, None)
            meta.pop("passages_are_structural", None)
            meta.pop("structural_context", None)
            prior = r["record_json"] or ""
            c.execute(
                "INSERT INTO failures (openalex_id, stage, error_type, message, at) "
                "VALUES (?,?,?,?,?)",
                (
                    openalex_id,
                    "classify",
                    "recoded",
                    (reason[:1000] + " || discarded record: " + prior)[:8000],
                    now,
                ),
            )
            c.execute("DELETE FROM evidence WHERE openalex_id=?", (openalex_id,))
            c.execute(
                "UPDATE papers SET stage='metadata', status='pending', "
                "attempts=0, record_json=NULL, meta_json=?, batch_id=NULL, "
                "last_error='classify:recoded', updated_at=? "
                "WHERE openalex_id=?",
                (json.dumps(meta), now, openalex_id),
            )
            c.execute("COMMIT")
        return ""

    #: Statuses that take a paper out of the corpus while keeping its row.
    #: `excluded` is a paper that should never be counted (a duplicate, a
    #: correction notice); `outdated` is one outside the study window, kept
    #: apart so the two reasons can be counted separately.
    REMOVED = ("excluded", "outdated")

    def exclude(self, openalex_id: str, *, reason: str, as_status: str = "excluded") -> str:
        """Take a paper out of the corpus without deleting it.

        For a paper that should not be counted at all — a duplicate enrolment,
        a correction notice — or, with `as_status="outdated"`, one that falls
        outside the study window. A DELETE would not hold: `enrol` is INSERT OR
        IGNORE, so the next enrol from the CSV would bring the paper straight
        back as `pending`. A removed row blocks that, is never claimed by a
        stage, and is not exported, since export reads only `done`.

        The record, evidence and verifications are kept as they are, so an
        exclusion is reversible and nothing about the paper is lost. The reason
        goes to `failures`. Returns "" on success, else the refusal.
        """
        if as_status not in self.REMOVED:
            raise ValueError(f"as_status must be one of {self.REMOVED}, not {as_status!r}")
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._conn() as c:
            c.execute("BEGIN")
            r = c.execute(
                "SELECT stage, status FROM papers WHERE openalex_id=?", (openalex_id,)
            ).fetchone()
            if r is None:
                c.execute("ROLLBACK")
                return "not enrolled"
            if r["status"] in self.REMOVED:
                c.execute("ROLLBACK")
                return f"already {r['status']}"
            c.execute(
                "INSERT INTO failures (openalex_id, stage, error_type, message, at) "
                "VALUES (?,?,?,?,?)",
                (
                    openalex_id,
                    r["stage"],
                    as_status,
                    f"{reason[:1000]} || was {r['stage']}/{r['status']}",
                    now,
                ),
            )
            c.execute(
                "UPDATE papers SET status=?, last_error=?, updated_at=? WHERE openalex_id=?",
                (as_status, f"{as_status}:{reason[:200]}", now, openalex_id),
            )
            c.execute("COMMIT")
        return ""

    def adjudicate(self, rec: PaperRecord, *, reason: str, reverify: bool) -> str:
        """Replace a finished paper's codes with a human coder's adjudication.

        For a record a human has settled by hand — two copies of one paper
        merged field by field, an identifier the extraction broke and the coder
        read correctly. The record is validated here, through the same
        `schema.py` gate as every classifier, so a human cannot write what a
        model could not.

        The previous record goes to `failures` first. Evidence rows are left as
        they are: an adjudication settles codes, it does not re-quote the paper.
        `reverify` sends the paper back to the verify stage — required whenever
        an identifier changed, because `Actual` is written only by verify.py.
        Returns "" on success, else the refusal.
        """
        rec.validate()
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._conn() as c:
            c.execute("BEGIN")
            r = c.execute(
                "SELECT status, stage, record_json FROM papers WHERE openalex_id=?",
                (rec.openalex_id,),
            ).fetchone()
            why = (
                "not enrolled"
                if r is None
                else ""
                if r["status"] == "done" and r["record_json"]
                else f"at {r['stage']}/{r['status']}; only finished papers can be adjudicated"
            )
            if why:
                c.execute("ROLLBACK")
                return why
            c.execute(
                "INSERT INTO failures (openalex_id, stage, error_type, message, at) "
                "VALUES (?,?,?,?,?)",
                (
                    rec.openalex_id,
                    r["stage"],
                    "adjudicated",
                    (reason[:1000] + " || replaced record: " + r["record_json"])[:8000],
                    now,
                ),
            )
            c.execute(
                "UPDATE papers SET record_json=?, scheme_version=?, stage=?, status=?, "
                "last_error=NULL, updated_at=? WHERE openalex_id=?",
                (
                    json.dumps(rec.flat()),
                    rec.scheme_version,
                    "classified" if reverify else r["stage"],
                    "pending" if reverify else r["status"],
                    now,
                    rec.openalex_id,
                ),
            )
            c.execute("COMMIT")
        return ""

    #: Rows that belong to one paper and travel with it when it is adopted.
    _PAPER_TABLES = ("evidence", "verifications", "failures", "usage")

    def adopt(self, src_path: str | Path, openalex_id: str, *, reason: str) -> str:
        """Copy one paper, as it stands, from another database into this one.

        For merging a separate run into the main store. The paper row comes
        across whole (stage, codes, text, coder id) with its evidence,
        verifications, failure history and usage, so nothing about how it was
        coded is lost or re-dated.

        Append only. A paper already here is refused, never overwritten: two
        codings of one paper are a comparison for a human to settle, not a
        conflict to resolve silently. A coded row from another scheme version
        is refused too, since pooling versions is what `export_csv` guards
        against. The adoption and its source go to `failures`. Returns "" on
        success, else the refusal.
        """
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        src = sqlite3.connect(str(src_path), timeout=60)
        src.row_factory = sqlite3.Row
        try:
            row = src.execute("SELECT * FROM papers WHERE openalex_id=?", (openalex_id,)).fetchone()
            if row is None:
                return f"not in {src_path}"
            if row["record_json"] and row["scheme_version"] not in (None, SCHEME_VERSION):
                return f"coded under scheme {row['scheme_version']}, not {SCHEME_VERSION}"
            children = {
                t: [
                    dict(r)
                    for r in src.execute(f"SELECT * FROM {t} WHERE openalex_id=?", (openalex_id,))
                ]
                for t in self._PAPER_TABLES
            }
        finally:
            src.close()

        with self._conn() as c:
            c.execute("BEGIN")
            if c.execute("SELECT 1 FROM papers WHERE openalex_id=?", (openalex_id,)).fetchone():
                c.execute("ROLLBACK")
                return "already enrolled here"
            cols = [
                k
                for k in row.keys()
                if k in {r["name"] for r in c.execute("PRAGMA table_info(papers)")}
            ]
            c.execute(
                f"INSERT INTO papers ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                [row[k] for k in cols],
            )
            for table, rows in children.items():
                for r in rows:
                    r.pop("id", None)  # autoincrement keys are local
                    c.execute(
                        f"INSERT OR IGNORE INTO {table} ({', '.join(r)}) "
                        f"VALUES ({', '.join('?' * len(r))})",
                        list(r.values()),
                    )
            c.execute(
                "INSERT INTO failures (openalex_id, stage, error_type, message, at) "
                "VALUES (?,?,?,?,?)",
                (
                    openalex_id,
                    row["stage"],
                    "adopted",
                    f"{reason[:1000]} || from {src_path} as {row['stage']}/{row['status']}",
                    now,
                ),
            )
            c.execute("COMMIT")
        return ""

    def counts(self) -> dict[str, int]:
        with self._conn() as c:
            return {
                r["status"]: r["n"]
                for r in c.execute("SELECT status, COUNT(*) AS n FROM papers GROUP BY status")
            }

    # -- export ------------------------------------------------------------
    def export_csv(
        self,
        analysis_path: str | Path,
        evidence_path: str | Path | None = None,
        review_path: str | Path | None = None,
    ) -> dict[str, int]:
        """Write the analysis view. Rows whose scheme_versions differ in major
        or minor refuse to export together — the vocabulary differs, and
        pooling them is exactly the silent corruption we are guarding against.
        A patch difference (2.0.0 beside 2.0.1) pools: a patch records a ruling
        on how the same tokens are applied, and the scheme_version column keeps
        every row traceable to the ruling it was coded or recoded under.
        """
        with self._conn() as c:
            versions = [
                r[0]
                for r in c.execute("SELECT DISTINCT scheme_version FROM papers WHERE status='done'")
            ]
            families = {_scheme_family(v) for v in versions}
            if len(families) > 1:
                raise RuntimeError(
                    f"refusing to export: rows span scheme versions "
                    f"{sorted(versions, key=str)}, which differ in more than the patch "
                    f"number. Migrate them to {SCHEME_VERSION} first."
                )
            rows = [
                json.loads(r["record_json"])
                for r in c.execute(
                    "SELECT record_json FROM papers WHERE status='done' AND record_json IS NOT NULL"
                )
            ]

        Path(analysis_path).parent.mkdir(parents=True, exist_ok=True)
        with open(analysis_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FLAT_COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        out = {"analysis": len(rows)}

        if evidence_path:
            with self._conn() as c, open(evidence_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["openalex_id", "ev_id", "quote", "section", "locator", "source"])
                n = 0
                for r in c.execute("SELECT * FROM evidence ORDER BY openalex_id, ev_id"):
                    w.writerow(
                        [
                            r["openalex_id"],
                            r["ev_id"],
                            r["quote"],
                            r["section"],
                            r["locator"],
                            r["source"],
                        ]
                    )
                    n += 1
                out["evidence"] = n

        if review_path:
            flagged = [r for r in rows if r.get("needs_human_review")]
            with open(review_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FLAT_COLUMNS, extrasaction="ignore")
                w.writeheader()
                w.writerows(flagged)
            out["review_queue"] = len(flagged)
        return out


__all__ = ["Store", "MAX_ATTEMPTS"]
