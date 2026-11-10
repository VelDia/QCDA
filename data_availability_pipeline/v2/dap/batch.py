"""
dap.batch — bulk classification through the Message Batches API.

WHY THIS EXISTS. The synchronous path makes one API call per paper and waits.
At corpus scale that is the wrong shape twice over: it pays full price, and its
throughput is bounded by round-trip latency times worker count. The Batches API
takes up to 100,000 requests in a single submission at **half** the per-token
price, and the work is done server-side with no open connection — so throughput
stops being a function of how many workers you can keep alive.

The trade is latency: most batches finish within an hour, the ceiling is 24.
For a bulk corpus run that is irrelevant. For a ten-paper shakedown it is
intolerable, which is why the synchronous path in `classify.py` stays.

SHAPE OF THE STAGE. Classification is split into two commands that do not have
to run in the same process, or the same day:

    submit   — drain spooled requests, create batches, record their ids
    collect  — poll open batches, pull results, validate, advance papers

Requests are spooled to JSONL on disk rather than held in the database, because
the passages are the bulk of the payload and there is no reason to store them
twice. A spool file is deleted only after its batch id is durably recorded, so
a crash between writing and submitting costs a re-spool, never a lost paper.

Results stay retrievable for 29 days, so `collect` can be re-run freely; it is
idempotent against papers that already advanced.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .classify import TIER_FAST, BATCH_DISCOUNT, _PRICES, _assemble, Usage
from .schema import Arm, Retrieval, SchemaError, TextSource
from .classify import grounding_text

#: Well under the 100,000-request / 256 MB ceiling. Smaller batches surface
#: failures sooner and keep any single loss small; there is no throughput
#: penalty for splitting, because batches process concurrently.
BATCH_SIZE = int(os.environ.get("DAP_BATCH_SIZE", "5000"))

#: A single request must stay comfortably inside the 256 MB batch cap; a paper
#: whose passages exceed this is spooled truncated rather than dropped.
MAX_REQUEST_BYTES = 400_000


@dataclass
class SpoolEntry:
    custom_id: str          # == openalex_id, sanitised
    params: dict
    meta: dict              # carried through so `collect` can rebuild the record


def safe_custom_id(openalex_id: str) -> str:
    """custom_id must match ^[a-zA-Z0-9_-]{1,64}$ — OpenAlex ids are URLs."""
    tail = openalex_id.rsplit("/", 1)[-1]
    cleaned = "".join(c if (c.isalnum() or c in "_-") else "_" for c in tail)
    return cleaned[:64] or "unknown"


class Spool:
    """Append-only JSONL staging area for requests awaiting submission."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _open_file(self) -> Path:
        existing = sorted(self.root.glob("spool_*.jsonl"))
        if existing:
            last = existing[-1]
            if sum(1 for _ in last.open()) < BATCH_SIZE:
                return last
        return self.root / f"spool_{len(existing):05d}.jsonl"

    def add(self, entry: SpoolEntry) -> None:
        line = json.dumps({"custom_id": entry.custom_id,
                           "params": entry.params, "meta": entry.meta})
        if len(line.encode()) > MAX_REQUEST_BYTES:
            # trim the passages rather than drop the paper; note it so the
            # record carries the fact that it was classified on less evidence
            msgs = entry.params["messages"]
            msgs[0]["content"] = msgs[0]["content"][:60_000] + \
                "\n\n[passages truncated to fit the batch request size limit]"
            entry.meta["truncated_input"] = True
            line = json.dumps({"custom_id": entry.custom_id,
                               "params": entry.params, "meta": entry.meta})
        with self._open_file().open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def ready_files(self, *, include_partial: bool = False) -> list[Path]:
        files = sorted(self.root.glob("spool_*.jsonl"))
        if include_partial:
            return files
        return [p for p in files if sum(1 for _ in p.open()) >= BATCH_SIZE]

    @staticmethod
    def read(path: Path) -> list[dict]:
        return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


# ---------------------------------------------------------------------------
def submit(client, store, spool: Spool, *, flush: bool = False,
           dry_run: bool = False, log=print) -> int:
    """Create batches from spooled requests. Returns the number submitted."""
    files = spool.ready_files(include_partial=flush)
    if not files:
        log("nothing ready to submit "
            f"(spool files fill at {BATCH_SIZE}; use --flush to send a partial one)")
        return 0

    submitted = 0
    for path in files:
        rows = Spool.read(path)
        if not rows:
            path.unlink(missing_ok=True)
            continue
        if dry_run:
            log(f"[dry-run] would submit {len(rows)} requests from {path.name}")
            submitted += 1
            continue

        requests = [{"custom_id": r["custom_id"], "params": r["params"]} for r in rows]
        batch = client.messages.batches.create(requests=requests)

        # Record the batch BEFORE deleting the spool: a crash here costs a
        # duplicate submission at worst, never a silently lost set of papers.
        store.record_batch(
            batch_id=batch.id,
            model=rows[0]["params"]["model"],
            n=len(rows),
            members=[(r["custom_id"], r["meta"]) for r in rows],
        )
        path.unlink(missing_ok=True)
        submitted += 1
        log(f"submitted {len(rows):>6} requests as {batch.id}")
    return submitted


def poll(client, store, *, log=print) -> dict[str, str]:
    """Refresh the processing status of every open batch."""
    out = {}
    for bid in store.open_batches():
        try:
            b = client.messages.batches.retrieve(bid)
        except Exception as e:
            log(f"  {bid}  could not retrieve: {type(e).__name__}")
            continue
        out[bid] = b.processing_status
        counts = getattr(b, "request_counts", None)
        detail = ""
        if counts is not None:
            detail = (f"  succeeded={getattr(counts,'succeeded',0)} "
                      f"errored={getattr(counts,'errored',0)} "
                      f"processing={getattr(counts,'processing',0)}")
        log(f"  {bid}  {b.processing_status}{detail}")
    return out


def collect(client, store, *, log=print) -> dict[str, int]:
    """Pull results from ended batches, validate, and advance the papers.

    Idempotent: a paper already advanced is skipped, so this can be re-run as
    often as you like while other batches are still processing.
    """
    tally = {"succeeded": 0, "errored": 0, "expired": 0, "invalid": 0, "skipped": 0}

    for bid in store.open_batches():
        try:
            b = client.messages.batches.retrieve(bid)
        except Exception as e:
            log(f"  {bid}: retrieve failed ({type(e).__name__}) — leaving open")
            continue
        if b.processing_status != "ended":
            continue

        members = store.batch_members(bid)          # custom_id -> meta
        model = store.batch_model(bid)

        for result in client.messages.batches.results(bid):
            cid = result.custom_id
            meta = members.get(cid)
            if meta is None:
                tally["skipped"] += 1
                continue
            if store.is_classified(meta["openalex_id"]):
                tally["skipped"] += 1
                continue

            kind = result.result.type
            if kind != "succeeded":
                tally[kind if kind in tally else "errored"] += 1
                store.fail(meta["openalex_id"], "classify_batch", kind,
                           f"batch {bid} returned {kind}",
                           permanent=(kind == "errored"))
                continue

            msg = result.result.message
            if getattr(msg, "stop_reason", None) == "max_tokens":
                tally["invalid"] += 1
                store.fail(meta["openalex_id"], "classify_batch", "max_tokens",
                           "record truncated; re-spool with fewer passages")
                continue

            block = next((c for c in msg.content if c.type == "tool_use"), None)
            if block is None:
                tally["invalid"] += 1
                store.fail(meta["openalex_id"], "classify_batch", "no_tool_use",
                           "model returned no tool_use block")
                continue

            usage = Usage(model=model,
                          input_tokens=getattr(msg.usage, "input_tokens", 0),
                          output_tokens=getattr(msg.usage, "output_tokens", 0))
            try:
                # The spool keeps only a slim copy of each paper's metadata, which
                # meant collect never saw the passages: grounding was not checked on
                # the census path, and das_source and locators could not be derived.
                # Read the real inputs back from the store.
                full = store.paper_meta(meta["openalex_id"])
                summary = {k: full.get(k) or [] for k in
                           ("das_texts", "dataset_ids", "confirmed_datasets")}
                rec = _assemble(
                    block.input,
                    openalex_id=meta["openalex_id"], doi=meta.get("doi", ""),
                    title=meta.get("title", ""), venue=meta.get("venue", ""),
                    year=meta.get("year"), arm=Arm.coerce(meta.get("arm", "unspecified")),
                    model=model,
                    retrieval=Retrieval.coerce(meta.get("retrieval", "metadata_only")),
                    text_source=TextSource.coerce(meta.get("text_source", "none")),
                    text_url=meta.get("text_url", ""), usage=usage,
                    grounding=grounding_text(crossref_summary=summary,
                                             passages=full.get("passages") or []),
                    passages=full.get("passages") or [],
                    crossref_das=bool(summary["das_texts"]))
            except SchemaError as e:
                # No self-correction loop here — there is no conversation to
                # continue. The paper goes back for a synchronous retry, where
                # the validator's complaint CAN be handed to the model.
                tally["invalid"] += 1
                store.requeue_for_sync(meta["openalex_id"], reason=str(e)[:500])
                continue

            store.save_classified(rec, batch_id=bid)
            store.record_usage(meta["openalex_id"], usage, discount=BATCH_DISCOUNT)
            tally["succeeded"] += 1

        store.close_batch(bid)
        log(f"  {bid}: collected")
    return tally


def estimate_cost(n_papers: int, *, model: str = TIER_FAST,
                  in_tokens: int = 6000, out_tokens: int = 1200) -> tuple[float, float]:
    """(synchronous, batched) USD for n papers at the given per-paper token use."""
    pin, pout = _PRICES.get(model, (0.0, 0.0))
    sync = n_papers * (in_tokens * pin + out_tokens * pout) / 1_000_000
    return sync, sync * BATCH_DISCOUNT


__all__ = ["Spool", "SpoolEntry", "submit", "poll", "collect",
           "safe_custom_id", "estimate_cost", "BATCH_SIZE"]
