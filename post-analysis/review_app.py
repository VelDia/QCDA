"""Blind human review of classifier output, one paper at a time.

    ../data_availability_pipeline/.venv/bin/python review_app.py [human_review.csv] [--port 8765]

Serves a coding form at http://127.0.0.1:8765. Each save writes these columns
back into the CSV (atomically; a one-time copy of the original is kept as
<csv>.bak):

    human_review_a_case, human_review_a_case_num,
    human_review_b_case, human_review_b_case_num,
    human_review_confidence, human_review_evidence, human_review_notes,
    human_review_at

Case tokens, numbers and confidence tokens are imported from dap/schema.py, so
the form cannot offer or store a value the scheme does not have. The
classifier's own codes are never sent to the page: the review stays blind.
"""

import argparse
import csv
import json
import os
import shutil
import sqlite3
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "data_availability_pipeline" / "v2"))
from dap.schema import CASE_LABEL, CASE_NUMBER, Case, Confidence  # noqa: E402

NEW_COLS = [
    "human_review_a_case",
    "human_review_a_case_num",
    "human_review_b_case",
    "human_review_b_case_num",
    "human_review_confidence",
    "human_review_evidence",
    "human_review_notes",
    "human_review_at",
]

# One-line reminders shown under each option; the definitions are the
# classifier prompt's (dap/classify.py), shortened.
CASE_HINT = {
    Case.NOT_APPLICABLE: "The axis does not apply: no external data analysed (A) / nothing generated (B)",
    Case.OPENLY_ACCESSIBLE: "A concrete identifier — DOI, URL, accession, repository",
    Case.ACCESS_NOT_PROVIDED: 'Identifiable (e.g. "we use ImageNet") but no path given',
    Case.INSUFFICIENTLY_SPECIFIED: 'Cannot even identify the material ("an internal corpus")',
    Case.ACCESS_UNRESOLVABLE: "A specific path is given, and it fails",
    Case.ON_REQUEST: "Offered on contact, however hedged",
    Case.EXPLICITLY_RESTRICTED: "Stated that it cannot be shared",
}

CASES = [
    {"token": c.value, "num": CASE_NUMBER[c], "label": CASE_LABEL[c], "hint": CASE_HINT[c]}
    for c in Case
]
CONFIDENCES = [c.value for c in Confidence]


def load_copies(db: Path, ids: list[str]) -> dict[str, dict]:
    """Find where each paper can be read, and the passages the pipeline stored.

    Opened read-only: SQLite is the pipeline's state and this app never writes
    it. Only meta_json is read -- record_json holds the classifier's codes and
    rationale, and the page must not see them.

    Args:
        db: The pipeline database. A missing file gives an empty result.
        ids: The OpenAlex ids of the papers under review.

    Returns:
        For each id found, a dict with `links` (the copies of the paper:
        `name`, `url`, whether it is a PDF, and whether it is the copy the
        pipeline read), `text_source`, and `passages` (`keywords` and `text`).
    """
    if not db.exists():
        return {}
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    q = ",".join("?" * len(ids))
    out = {}
    for oid, meta in con.execute(
        f"SELECT openalex_id, meta_json FROM papers WHERE openalex_id IN ({q})", ids
    ):
        m = json.loads(meta or "{}")
        read_url = m.get("text_url") or ""
        links, seen = [], set()
        for loc in (m.get("oa_locations") or {}).get("locations") or []:
            url = loc.get("pdf_url") or loc.get("landing_page_url")
            if not url or url in seen:
                continue
            seen.add(url)
            links.append(
                {
                    "name": (loc.get("source") or {}).get("display_name") or "copy",
                    "url": url,
                    "pdf": bool(loc.get("pdf_url")),
                    "read": url == read_url,
                }
            )
        if read_url and read_url not in seen:
            links.append(
                {
                    "name": "pipeline's copy",
                    "url": read_url,
                    "pdf": read_url.endswith(".pdf"),
                    "read": True,
                }
            )
        out[oid] = {
            "links": links,
            "text_source": m.get("text_source") or "",
            "passages": [
                {"keywords": p.get("keywords") or [], "text": p.get("text") or ""}
                for p in m.get("passages") or []
            ],
        }
    con.close()
    return out


class Store:
    """The review CSV, held in memory and written back on every save.

    A one-time copy of the original file is kept beside it as `<csv>.bak`.

    Attributes:
        path: The CSV file under review.
        lock: Serialises saves from the server's request threads.
        fields: The CSV columns, with the human-review columns appended.
        rows: The CSV rows, one dict per paper.
        copies: Reading links and stored passages by OpenAlex id (see
            `load_copies`).
    """

    def __init__(self, path: Path, db: Path) -> None:
        """Load the CSV, back it up once and read the papers' copies.

        Args:
            path: The CSV file under review.
            db: The pipeline database to read copies and passages from. It is
                opened read-only and may be missing.

        Raises:
            SystemExit: If `openalex_id` is not unique in the CSV.
        """
        self.path = path
        self.lock = threading.Lock()
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(path, backup)
        self.fields, self.rows = self._read()
        ids = [r["openalex_id"] for r in self.rows]
        if len(ids) != len(set(ids)):
            raise SystemExit("openalex_id is not unique in the CSV; refusing to start")
        self.copies = load_copies(db, ids)

    def _read(self) -> tuple[list[str], list[dict]]:
        """Read the CSV and add the human-review columns it lacks.

        Returns:
            A tuple `(fields, rows)`: the column names, and one dict per row with
            every human-review column present (empty where not yet filled).
        """
        with open(self.path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = list(reader.fieldnames)
            rows = list(reader)
        for col in NEW_COLS:
            if col not in fields:
                fields.append(col)
        for r in rows:
            for col in NEW_COLS:
                r.setdefault(col, "")
        return fields, rows

    def _write(self) -> None:
        """Write every row back to the CSV atomically, through a temporary file."""
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.fields, lineterminator="\n")
            w.writeheader()
            w.writerows(self.rows)
        os.replace(tmp, self.path)

    def public_rows(self) -> list[dict]:
        """List the rows as the page may see them.

        Returns:
            One dict per paper with its identity, status, whether it has been
            classified, and the human-review columns. The classifier's codes are
            left out, so the review stays blind.
        """
        out = []
        for r in self.rows:
            out.append(
                {
                    "id": r["openalex_id"],
                    "doi": r["doi"],
                    "title": r["title"],
                    "year": r["year"],
                    "arm": r["arm"],
                    "status": r["status"],
                    "classified": bool(r["a_case"] or r["b_case"]),
                    **{c: r[c] for c in NEW_COLS},
                }
            )
        return out

    def save(self, payload: dict) -> dict:
        """Validate one review and write it to the CSV.

        Args:
            payload: The form's answer: `id` (the paper's OpenAlex id), `a_case`,
                `b_case`, `confidence`, and optionally `evidence` and `notes`.

        Returns:
            The human-review columns as stored for that paper.

        Raises:
            ValueError: If a case or the confidence is not a token of the scheme,
                or if `id` is not in the CSV.
        """
        tokens = {c.value: c for c in Case}
        a, b = payload.get("a_case"), payload.get("b_case")
        conf = payload.get("confidence")
        if a not in tokens or b not in tokens:
            raise ValueError("both axes need a case from the scheme")
        if conf not in CONFIDENCES:
            raise ValueError("confidence must be one of " + ", ".join(CONFIDENCES))
        with self.lock:
            row = next((r for r in self.rows if r["openalex_id"] == payload.get("id")), None)
            if row is None:
                raise ValueError("unknown openalex_id")
            row.update(
                {
                    "human_review_a_case": a,
                    "human_review_a_case_num": str(CASE_NUMBER[tokens[a]]),
                    "human_review_b_case": b,
                    "human_review_b_case_num": str(CASE_NUMBER[tokens[b]]),
                    "human_review_confidence": conf,
                    "human_review_evidence": (payload.get("evidence") or "").strip(),
                    "human_review_notes": (payload.get("notes") or "").strip(),
                    "human_review_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
            )
            self._write()
            return {c: row[c] for c in NEW_COLS}


def make_handler(store: Store) -> type[BaseHTTPRequestHandler]:
    """Build the HTTP request handler class for one review store.

    Args:
        store: The review the handler serves and saves to.

    Returns:
        A `BaseHTTPRequestHandler` subclass bound to `store`.
    """

    class Handler(BaseHTTPRequestHandler):
        """Serves the review page and its JSON API."""

        def _send(self, code: int, body: str | bytes, ctype: str = "application/json") -> None:
            """Send a complete response.

            Args:
                code: The HTTP status code.
                body: The response body; text is encoded as UTF-8.
                ctype: The media type, without the charset.
            """
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            """Serve the page, the row list, or one paper's copies and passages."""
            if self.path == "/":
                self._send(200, PAGE, "text/html")
            elif self.path.startswith("/api/paper?"):
                oid = parse_qs(urlsplit(self.path).query).get("id", [""])[0]
                self._send(200, json.dumps(store.copies.get(oid, {"links": [], "passages": []})))
            elif self.path == "/api/rows":
                self._send(
                    200,
                    json.dumps(
                        {
                            "rows": store.public_rows(),
                            "cases": CASES,
                            "confidences": CONFIDENCES,
                            "csv": str(store.path),
                        }
                    ),
                )
            else:
                self._send(404, json.dumps({"error": "not found"}))

        def do_POST(self) -> None:
            """Save one review; an invalid one is answered with 400 and the reason."""
            if self.path != "/api/save":
                return self._send(404, json.dumps({"error": "not found"}))
            try:
                n = int(self.headers.get("Content-Length", 0))
                saved = store.save(json.loads(self.rfile.read(n)))
            except ValueError as e:
                return self._send(400, json.dumps({"error": str(e)}))
            self._send(200, json.dumps({"saved": saved}))

        def log_message(self, fmt: str, *args: object) -> None:
            """Log saves only, to stderr; page and API reads are not logged.

            Args:
                fmt: The format string, as `BaseHTTPRequestHandler` passes it.
                *args: The values for `fmt`.
            """
            if self.command == "POST":
                sys.stderr.write("%s\n" % (fmt % args))

    return Handler


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Human Review</title>
<style>
:root{--bg:#f6f5f2;--panel:#fff;--ink:#1d1d1b;--muted:#6b6a65;--line:#e3e1db;
--accent:#2f5d8a;--accent-soft:#e6eef6;--ok:#2e7d4f;--warn:#a5631b;--sel:#fff7e0}
@media (prefers-color-scheme:dark){:root{--bg:#171716;--panel:#20201e;--ink:#ecebe7;
--muted:#9b9a94;--line:#353431;--accent:#8db6e0;--accent-soft:#25313d;--ok:#6fc392;--warn:#e0a35c;--sel:#3a3320}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.app{display:grid;grid-template-columns:300px 1fr;height:100vh}
aside{border-right:1px solid var(--line);display:flex;flex-direction:column;min-height:0;background:var(--panel)}
.side-head{padding:12px;border-bottom:1px solid var(--line)}
.progress{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin:8px 0 10px}
.progress>div{height:100%;background:var(--ok)}
select,textarea,button{font:inherit;color:inherit}
select{width:100%;padding:5px;border:1px solid var(--line);border-radius:6px;background:var(--bg)}
.list{overflow:auto;flex:1}
.item{padding:8px 12px;border-bottom:1px solid var(--line);cursor:pointer;display:flex;gap:8px}
.item:hover{background:var(--bg)} .item.cur{background:var(--accent-soft)}
.item .t{flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.item .c{font-variant-numeric:tabular-nums;color:var(--ok);font-weight:600;white-space:nowrap}
.item .c.none{color:var(--muted);font-weight:400}
main{overflow:auto;padding:20px 28px 90px}
h1{font-size:19px;margin:0 0 6px;line-height:1.3}
.meta{color:var(--muted);display:flex;flex-wrap:wrap;gap:6px 14px;align-items:center}
.badge{border:1px solid var(--warn);color:var(--warn);border-radius:10px;padding:0 7px;font-size:12px}
.open{display:inline-block;margin-top:12px;background:var(--accent);color:var(--panel);text-decoration:none;
padding:7px 14px;border-radius:7px;font-weight:600}
.axes{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:20px}
.axis{background:var(--panel);border:2px solid var(--line);border-radius:10px;padding:12px}
.axis.active{border-color:var(--accent)}
.axis h2{font-size:15px;margin:0}.axis .def{color:var(--muted);font-size:12.5px;margin:2px 0 10px}
.opt{display:flex;gap:10px;padding:7px 8px;border-radius:7px;cursor:pointer;border:1px solid transparent}
.opt:hover{background:var(--bg)} .opt.on{background:var(--sel);border-color:var(--warn)}
.opt .n{width:22px;height:22px;border-radius:50%;border:1px solid var(--line);display:grid;place-items:center;
font-weight:600;font-size:12px;flex:none}
.opt.on .n{background:var(--warn);color:var(--panel);border-color:var(--warn)}
.opt .l{font-weight:600}.opt .tok{color:var(--muted);font-family:ui-monospace,Menlo,monospace;font-size:11.5px}
.opt .h{color:var(--muted);font-size:12.5px}
.row{margin-top:18px}
.row label{font-weight:600;display:block;margin-bottom:5px}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:7px;overflow:hidden}
.seg button{border:0;background:var(--panel);padding:6px 16px;cursor:pointer;border-right:1px solid var(--line)}
.seg button:last-child{border-right:0}.seg button.on{background:var(--accent);color:var(--panel)}
textarea{width:100%;min-height:70px;padding:8px;border:1px solid var(--line);border-radius:7px;background:var(--panel);resize:vertical}
footer{position:fixed;left:300px;right:0;bottom:0;background:var(--panel);border-top:1px solid var(--line);
padding:10px 28px;display:flex;gap:10px;align-items:center}
footer button{padding:7px 14px;border-radius:7px;border:1px solid var(--line);background:var(--panel);cursor:pointer}
footer .primary{background:var(--accent);color:var(--panel);border-color:var(--accent);font-weight:600}
.status{margin-left:auto;color:var(--muted);font-size:12.5px}
.status.err{color:#c0392b}.status.ok{color:var(--ok)}
kbd{font-family:ui-monospace,Menlo,monospace;font-size:11px;border:1px solid var(--line);border-bottom-width:2px;
border-radius:4px;padding:0 4px;background:var(--bg)}
.keys{color:var(--muted);font-size:12px;margin-top:22px}
.copies{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px;align-items:center}
.copies a{font-size:12.5px;border:1px solid var(--line);border-radius:14px;padding:2px 10px;color:var(--accent);
text-decoration:none;background:var(--panel)}
.copies a.read{border-color:var(--ok)} .copies small{color:var(--muted)}
details.pass{margin-top:16px;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:8px 12px}
details.pass summary{cursor:pointer;font-weight:600}
.p{border-top:1px solid var(--line);padding:8px 0}.p:first-of-type{border-top:0}
.p .kw{font-size:11.5px;color:var(--muted);font-family:ui-monospace,Menlo,monospace}
.p pre{white-space:pre-wrap;font:12px/1.45 ui-monospace,Menlo,monospace;margin:4px 0 0;max-height:260px;overflow:auto}
mark{background:#ffe48a;color:#1d1d1b;border-radius:2px}
.overlay{position:fixed;inset:0;background:rgba(0,0,0,.35);display:none;align-items:flex-start;justify-content:center;
padding:40px 16px;z-index:10;overflow:auto}
.overlay.show{display:flex}
.card{background:var(--panel);border-radius:12px;max-width:980px;width:100%;padding:18px 20px;box-shadow:0 10px 40px rgba(0,0,0,.25)}
.card h2{margin:0 0 4px;font-size:17px}
table.lk{border-collapse:collapse;width:100%;font-size:13px;margin-top:10px}
table.lk th,table.lk td{border-bottom:1px solid var(--line);padding:7px 8px;text-align:left;vertical-align:top}
table.lk th{color:var(--muted);font-weight:600;font-size:12px}
table.lk td.num{font-weight:700;text-align:center}
.pairs{margin:14px 0 0;padding-left:18px;font-size:13px}.pairs li{margin:4px 0}
@media (max-width:1150px){.app{grid-template-columns:210px 1fr}footer{left:210px;padding:8px 14px;flex-wrap:wrap}
 main{padding:14px 16px 110px}.axes{gap:10px}.opt .h{display:none}.opt{padding:5px 6px}.item{padding:6px 10px;font-size:12.5px}}
@media (max-width:640px){.app{grid-template-columns:1fr}aside{display:none}.axes{grid-template-columns:1fr}footer{left:0}}
</style></head><body>
<div class="app">
<aside><div class="side-head">
<div><b id="count"></b></div><div class="progress"><div id="bar"></div></div>
<select id="filter"><option value="classified">Classifier-coded rows</option>
<option value="todo">Not yet reviewed</option><option value="all">All rows</option></select>
</div><div class="list" id="list"></div></aside>
<main id="main"></main>
</div>
<footer><button id="prev">← Prev</button><button id="save">Save</button>
<button id="next" class="primary">Save &amp; next →</button>
<label style="font-size:12.5px;color:var(--muted)"><input type="checkbox" id="auto" checked> open paper on next</label>
<button id="lookup">Case table <kbd>?</kbd></button>
<span class="status" id="status"></span></footer>
<div class="overlay" id="overlay"><div class="card">
<h2>Case lookup</h2><div style="color:var(--muted);font-size:12.5px">The same seven cases on both axes. # is the publication number stored in
<code>*_case_num</code>; it is not ordered by openness, so use the rank column for that. <kbd>Esc</kbd> or <kbd>?</kbd> closes this.</div>
<table class="lk"><thead><tr><th>#</th><th>Case</th><th>Axis A: external data</th><th>Axis B: self-generated data</th><th>Openness</th></tr></thead>
<tbody id="lkbody"></tbody></table>
<ul class="pairs">
<li><b>3 vs 4:</b> could a reader identify the material? “We evaluate on ImageNet” → 3. “An internal corpus”, or “parameters tuned empirically” → 4.</li>
<li><b>3 vs 5:</b> 5 only when a <i>specific</i> path is given and it fails. If no path is given, it’s 3.</li>
<li><b>6 vs 7:</b> any offer on contact, however hedged → 6. A statement that the data cannot be shared → 7.</li>
<li><b>2</b> needs an actual identifier (DOI, URL, accession, repository). “Data are available” with no path → 3.</li>
<li><b>B = 1</b> only if the paper produces nothing: a derivation, position piece or review. A simulation-only paper <i>has</i> Axis B data.</li>
<li>Running on shared hardware (a cloud QPU) generates data (B), not external data (A). Citing or comparing to another paper’s figures is not A.</li>
</ul></div></div>
<script>
let rows=[],cases=[],confs=[],view=[],cur=0,active="a",draft={},paper=null,passOpen=false;
const RANK={not_applicable:"—",openly_accessible:"5 · now, anyone",on_request:"4 · via a human",access_not_provided:"3 · reader must hunt",
 insufficiently_specified:"2 · probably not",access_unresolvable:"1 · promised, broken",explicitly_restricted:"0 · refused"};
const EX={
 not_applicable:["No external dataset analysed","Produces nothing: derivation, review, position piece"],
 openly_accessible:["Dataset cited with DOI / URL / accession","Own data or code deposited at an address"],
 access_not_provided:["Named, no path: “we use ImageNet”","Identifiable, but not provided and no path"],
 insufficiently_specified:["Can’t identify it: “an internal corpus”","Procedure too vague to reproduce"],
 access_unresolvable:["Given path fails (404, dead host)","Given path fails (404, dead host)"],
 on_request:["Third-party data from the authors on request","“From the corresponding author on reasonable request”"],
 explicitly_restricted:["Cannot be shared: licence, privacy, NDA","Cannot be shared: proprietary, export control"]};
const HL=/(data availability|code availability|availability statement|(data|code|datasets?|results?) (that support[^.]{0,40})?(are|is|will be) (publicly |openly |freely )?(available|accessible|provided|deposited)|on (reasonable )?request|upon request|corresponding author|github\.com|gitlab|zenodo|figshare|dryad|osf\.io|kaggle|doi\.org\/10\.|https?:\/\/\S+|supplementa\w+|repository|cannot be (made )?(shared|available)|not publicly available|proprietary)/gi;
const $=s=>document.querySelector(s);
const esc=s=>(s||"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const paperUrl=r=>r.doi?"https://doi.org/"+r.doi:r.id;
const done=r=>!!r.human_review_a_case;

async function load(){
  const d=await (await fetch("/api/rows")).json();
  rows=d.rows;cases=d.cases;confs=d.confidences;renderLookup();
  let f="classified";try{f=localStorage.getItem("hr_filter")||f}catch(e){}
  $("#filter").value=f;applyFilter();
  const first=view.findIndex(r=>!done(r));select(first<0?0:first);
}
function applyFilter(){
  const f=$("#filter").value;try{localStorage.setItem("hr_filter",f)}catch(e){}
  view=rows.filter(r=>f==="all"||(f==="classified"&&r.classified)||(f==="todo"&&r.classified&&!done(r)));
  renderList();
}
function renderList(){
  const base=rows.filter(r=>r.classified),n=base.filter(done).length;
  $("#count").textContent=`${n} / ${base.length} reviewed`;
  $("#bar").style.width=(base.length?100*n/base.length:0)+"%";
  $("#list").innerHTML=view.map((r,i)=>`<div class="item${i===cur?" cur":""}" data-i="${i}">
   <span class="t" title="${esc(r.title)}">${esc(r.title)||"<i>(no title) "+esc(r.doi)+"</i>"}</span>
   <span class="c${done(r)?"":" none"}">${done(r)?"A"+r.human_review_a_case_num+" B"+r.human_review_b_case_num:"—"}</span></div>`).join("");
  document.querySelectorAll(".item").forEach(e=>e.onclick=()=>select(+e.dataset.i));
}
async function select(i,openPaper){
  if(!view.length){$("#main").innerHTML="<p>Nothing in this view.</p>";return}
  cur=Math.max(0,Math.min(view.length-1,i));const r=view[cur];
  if(openPaper)window.open(paperUrl(r),"paper");  // before any await: needs the click/keypress
  draft={a_case:r.human_review_a_case,b_case:r.human_review_b_case,confidence:r.human_review_confidence,
    evidence:r.human_review_evidence,notes:r.human_review_notes};
  active=draft.a_case?"b":"a";paper=null;
  render();renderList();
  document.querySelector(".item.cur")?.scrollIntoView({block:"nearest"});
  const p=await (await fetch("/api/paper?id="+encodeURIComponent(r.id))).json();
  if(view[cur]===r){paper=p;renderPaper()}
}
function renderLookup(){
  $("#lkbody").innerHTML=cases.map(c=>`<tr><td class="num">${c.num}</td><td><b>${esc(c.label)}</b><br><span class="tok" style="color:var(--muted);font-family:ui-monospace,Menlo,monospace;font-size:11.5px">${c.token}</span></td>
   <td>${esc(EX[c.token][0])}</td><td>${esc(EX[c.token][1])}</td><td>${RANK[c.token]}</td></tr>`).join("");
}
function toggleLookup(show){$("#overlay").classList.toggle("show",show??!$("#overlay").classList.contains("show"))}
function highlight(t){return esc(t).replace(HL,m=>`<mark>${m}</mark>`)}
function renderPaper(){
  const r=view[cur],box=$("#copies"),pb=$("#passages");if(!box)return;
  const links=[`<a href="${esc(paperUrl(r))}" target="paper">${r.doi?"Publisher (doi.org)":"OpenAlex"}</a>`];
  for(const l of (paper?.links||[])){
    if(l.url===paperUrl(r))continue;
    links.push(`<a href="${esc(l.url)}" target="paper" class="${l.read?"read":""}" title="${esc(l.url)}">${esc(l.name)}${l.pdf?" · PDF":""}${l.read?" ✓ read by pipeline":""}</a>`);
  }
  box.innerHTML=`<small>Copies:</small>`+links.join("");
  // passages with availability wording first; otherwise the stored order
  const hits=t=>(t.match(HL)||[]).length;
  const ps=(paper?.passages||[]).map((p,i)=>({...p,i,h:hits(p.text)})).sort((x,y)=>(y.h>0)-(x.h>0)||x.i-y.i);
  pb.innerHTML=ps.length?`<details class="pass"${passOpen?" open":""}><summary>Stored passages (${ps.length})${paper.text_source?" · from "+esc(paper.text_source):""} <kbd>P</kbd></summary>
   ${ps.map(p=>`<div class="p"><div class="kw">#${p.i+1} · ${esc(p.keywords.join(", "))}${p.h?` · <b style="color:var(--warn)">${p.h} availability match${p.h>1?"es":""}</b>`:""}</div><pre>${highlight(p.text)}</pre></div>`).join("")}</details>`
   :`<p style="color:var(--muted);font-size:12.5px">No stored passages for this row${paper?"":" (loading…)"}.</p>`;
  pb.querySelector("details")?.addEventListener("toggle",e=>passOpen=e.target.open);
}
function axisHtml(ax,title,def){
  return `<section class="axis${active===ax?" active":""}" data-ax="${ax}"><h2>${title}</h2><div class="def">${def}</div>
  ${cases.map(c=>`<div class="opt${draft[ax+"_case"]===c.token?" on":""}" data-ax="${ax}" data-tok="${c.token}" title="${esc(c.hint)}">
   <span class="n">${c.num}</span><div><span class="l">${esc(c.label)}</span> <span class="tok">${c.token}</span>
   <div class="h">${esc(c.hint)}</div></div></div>`).join("")}</section>`;
}
function render(){
  const r=view[cur];
  $("#main").innerHTML=`<h1>${esc(r.title)||"<i>(no title in CSV)</i>"}</h1>
  <div class="meta"><span>${esc(r.year)}</span><span>${esc(r.arm)}</span>
   <span>${r.doi?esc(r.doi):"no DOI — opens OpenAlex"}</span><span>${esc(r.id.replace("https://openalex.org/",""))}</span>
   ${r.status!=="done"?`<span class="badge">${esc(r.status)}</span>`:""}
   ${r.human_review_at?`<span>reviewed ${esc(r.human_review_at.slice(0,16).replace("T"," "))}</span>`:""}</div>
  <a class="open" href="${esc(paperUrl(r))}" target="paper">Open paper ↗</a>
  <div class="copies" id="copies"></div><div id="passages"></div>
  <div class="axes">${axisHtml("a","Axis A — External data","Existed independently of this paper and is analysed by it: datasets, benchmarks, records, prior artifacts.")}
  ${axisHtml("b","Axis B — Self-generated data","Produced by this paper: measurements, simulation output, generated instances, training runs.")}</div>
  <div class="row"><label>Confidence</label><div class="seg">${confs.map(c=>`<button data-conf="${c}" class="${draft.confidence===c?"on":""}">${c}</button>`).join("")}</div></div>
  <div class="row"><label for="ev">Evidence <span style="font-weight:400;color:var(--muted)">— quote or section from the paper</span></label>
   <textarea id="ev">${esc(draft.evidence)}</textarea></div>
  <div class="row"><label for="no">Notes</label><textarea id="no">${esc(draft.notes)}</textarea></div>
  <div class="keys"><kbd>1</kbd>–<kbd>7</kbd> case on the highlighted axis (A, then B) · <kbd>A</kbd>/<kbd>B</kbd> switch axis ·
   <kbd>H</kbd> <kbd>M</kbd> <kbd>L</kbd> confidence · <kbd>O</kbd> open paper · <kbd>Enter</kbd> save &amp; next ·
   <kbd>⌘</kbd>+<kbd>Enter</kbd> from a text box · <kbd>←</kbd>/<kbd>→</kbd> move without saving ·
   <kbd>P</kbd> stored passages · <kbd>?</kbd> case table</div>`;
  document.querySelectorAll(".opt").forEach(e=>e.onclick=()=>{active=e.dataset.ax;pick(e.dataset.tok)});
  document.querySelectorAll(".axis").forEach(e=>e.addEventListener("click",()=>{if(active!==e.dataset.ax){active=e.dataset.ax;render()}}));
  document.querySelectorAll(".seg button").forEach(e=>e.onclick=()=>{draft.confidence=e.dataset.conf;render()});
  $("#ev").oninput=e=>draft.evidence=e.target.value;$("#no").oninput=e=>draft.notes=e.target.value;
  renderPaper();
}
function pick(tok){draft[active+"_case"]=tok;if(active==="a")active="b";render()}
function setStatus(t,cls){const s=$("#status");s.textContent=t;s.className="status "+(cls||"")}
async function save(advance){
  const r=view[cur];
  if(!draft.a_case||!draft.b_case||!draft.confidence){setStatus("Pick both axes and a confidence first","err");return false}
  const res=await fetch("/api/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({id:r.id,...draft})});
  const d=await res.json();
  if(!res.ok){setStatus(d.error,"err");return false}
  Object.assign(r,d.saved);
  setStatus(`Saved ${r.id.replace("https://openalex.org/","")}: A${d.saved.human_review_a_case_num} B${d.saved.human_review_b_case_num}`,"ok");
  if(advance){
    if($("#filter").value==="todo"){applyFilter();select(cur,$("#auto").checked)}
    else select(cur+1,$("#auto").checked);
  }else{render();renderList()}
  return true;
}
document.addEventListener("keydown",e=>{
  const typing=["TEXTAREA","INPUT","SELECT"].includes(document.activeElement.tagName);
  if(e.key==="Enter"&&(e.metaKey||e.ctrlKey)){e.preventDefault();save(true);return}
  if(typing){if(e.key==="Escape")document.activeElement.blur();return}
  const k=e.key.toLowerCase();
  if(k==="?"||k==="/"){e.preventDefault();toggleLookup();return}
  if(k==="escape"&&$("#overlay").classList.contains("show")){toggleLookup(false);return}
  if(k==="p"){passOpen=!passOpen;const d=document.querySelector("details.pass");if(d)d.open=passOpen;return}
  if(/^[1-7]$/.test(k)){pick(cases.find(c=>c.num===+k).token)}
  else if(k==="a"||k==="b"){active=k;render()}
  else if(k==="h"||k==="m"||k==="l"){draft.confidence={h:"high",m:"medium",l:"low"}[k];render()}
  else if(k==="o"){window.open(paperUrl(view[cur]),"paper")}
  else if(k==="enter"){e.preventDefault();save(true)}
  else if(k==="arrowright"){select(cur+1)}else if(k==="arrowleft"){select(cur-1)}
  else return;
});
$("#filter").onchange=()=>{applyFilter();select(0)};
$("#lookup").onclick=()=>toggleLookup();$("#overlay").onclick=e=>{if(e.target.id==="overlay")toggleLookup(false)};
$("#prev").onclick=()=>select(cur-1);$("#save").onclick=()=>save(false);$("#next").onclick=()=>save(true);
load();
</script></body></html>"""


def main() -> None:
    """Start the review server on 127.0.0.1 and serve until interrupted.

    Reads the command line: the CSV to review, `--port` and `--db`.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", nargs="?", default=str(HERE / "human_review.csv"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument(
        "--db",
        default=str(HERE.parent / "dap_final.sqlite"),
        help="pipeline database to read copies and passages from (read-only)",
    )
    args = ap.parse_args()
    store = Store(Path(args.csv).resolve(), Path(args.db).resolve())
    print(f"copies/passages for {len(store.copies)} rows from {args.db}", flush=True)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store))
    print(
        f"reviewing {store.path} ({len(store.rows)} rows) at http://127.0.0.1:{args.port}",
        flush=True,
    )
    srv.serve_forever()


if __name__ == "__main__":
    main()
