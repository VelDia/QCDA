"""The release build: any subject, no arms, the text cache. All offline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dap.classify as C
import dap.pipeline as P
from dap.collect import Http, Query, collect
from dap.schema import Arm


# --------------------------------------------------------------------------- subject
def test_subject_goes_into_a_title_and_abstract_filter_by_default():
    q = Query(subject="protein folding", year_min=2021, year_max=2026)
    p = q.params()
    assert "search" not in p
    assert p["filter"] == ("title_and_abstract.search:protein folding,publication_year:2021-2026")


def test_title_and_fulltext_modes_and_extra_filter():
    assert (
        Query(subject="graphene", search_in="title").params()["filter"] == "title.search:graphene"
    )
    p = Query(subject="graphene", search_in="fulltext", extra_filter="type:article").params()
    assert p["search"] == "graphene" and p["filter"] == "type:article"
    assert Query(subject="x", year_min=2020).params()["filter"].endswith("publication_year:>2019")


def test_subject_is_required_and_a_comma_cannot_split_the_filter():
    with pytest.raises(ValueError):
        Query(subject="  ")
    with pytest.raises(ValueError):
        Query(subject="ice, snow")
    Query(subject="ice, snow", search_in="fulltext")  # plain search allows it


def _work(i):
    return {
        "id": f"https://openalex.org/W{i}",
        "doi": f"https://doi.org/10.1/{i}",
        "title": f"paper {i}",
        "publication_year": 2024,
        "type": "article",
        "primary_location": {"source": {"display_name": "J"}},
        "open_access": {"is_oa": True, "oa_status": "gold"},
        "cited_by_count": 3,
    }


class _Pages:
    """Fake OpenAlex: serves pages by cursor; optionally runs out of quota once."""

    def __init__(self, pages, quota_at=None):
        self.pages, self.quota_at, self.calls = pages, quota_at, []

    def __call__(self, url, params, timeout):
        self.calls.append(dict(params))
        cur = params["cursor"]
        if self.quota_at == cur:
            self.quota_at = None
            return SimpleNamespace(
                status_code=429,
                headers={"Retry-After": "86400"},
                raise_for_status=lambda: None,
                json=lambda: {},
            )
        idx = 0 if cur == "*" else int(cur)
        results = self.pages[idx] if idx < len(self.pages) else []
        nxt = str(idx + 1) if idx + 1 < len(self.pages) else None
        body = {"results": results, "meta": {"count": 5, "next_cursor": nxt}}
        return SimpleNamespace(
            status_code=200, headers={}, raise_for_status=lambda: None, json=lambda: body
        )


def test_collect_pages_dedups_and_records_the_query(tmp_path):
    get = _Pages([[_work(1), _work(2)], [_work(2), _work(3)]])
    out = tmp_path / "papers.csv"
    n = collect(
        Query(subject="graphene"), out, Http(get=get, sleep=lambda s: None), log=lambda m: None
    )
    assert n == 3
    rows = list(csv.DictReader(open(out)))
    assert [r["openalex_id"][-2:] for r in rows] == ["W1", "W2", "W3"]
    assert rows[0]["doi"] == "10.1/1"
    q = json.loads((tmp_path / "papers.csv.query.json").read_text())
    assert q["subject"] == "graphene"
    assert not (tmp_path / "papers.csv.cursor.json").exists(), "finished: no cursor left"


def test_collect_resumes_exactly_after_quota_exhaustion(tmp_path):
    get = _Pages([[_work(1)], [_work(2)], [_work(3)]], quota_at="1")
    out, http = tmp_path / "p.csv", Http(get=get, sleep=lambda s: None)
    assert collect(Query(subject="x"), out, http, log=lambda m: None) == 1
    assert (tmp_path / "p.csv.cursor.json").exists()
    assert collect(Query(subject="x"), out, http, log=lambda m: None) == 3
    assert [c["cursor"] for c in get.calls] == ["*", "1", "1", "2"]


def test_collect_refuses_to_mix_queries_in_one_file(tmp_path):
    get = _Pages([[_work(1)], [_work(2)]], quota_at="1")
    out, http = tmp_path / "p.csv", Http(get=get, sleep=lambda s: None)
    collect(Query(subject="x"), out, http, log=lambda m: None)
    with pytest.raises(SystemExit):
        collect(Query(subject="y"), out, http, log=lambda m: None)


def test_collect_output_enrols_without_any_arm_map(tmp_path):
    get = _Pages([[_work(1), _work(2)]])
    out = tmp_path / "papers.csv"
    collect(Query(subject="x"), out, Http(get=get, sleep=lambda s: None), log=lambda m: None)
    db = tmp_path / "d.sqlite"
    P.cmd_enrol(
        argparse.Namespace(db=str(db), input=str(out), year_min=None, year_max=None, limit=None)
    )
    import sqlite3

    arms = {r[0] for r in sqlite3.connect(db).execute("SELECT arm FROM papers")}
    assert arms == {Arm.UNSPECIFIED.value}


def test_cli_has_no_arm_flags_and_offers_collect():
    import subprocess

    root = Path(__file__).resolve().parents[1]
    h = subprocess.run(
        [sys.executable, "-m", "dap.pipeline", "enrol", "-h"],
        cwd=root,
        capture_output=True,
        text=True,
    ).stdout
    assert "--arm-map" not in h
    h = subprocess.run(
        [sys.executable, "-m", "dap.pipeline", "stage", "-h"],
        cwd=root,
        capture_output=True,
        text=True,
    ).stdout
    assert "--arm " not in h and "--arm\n" not in h
    h = subprocess.run(
        [sys.executable, "-m", "dap.pipeline", "collect", "-h"],
        cwd=root,
        capture_output=True,
        text=True,
    ).stdout
    assert "--subject" in h


# --------------------------------------------------------------------------- prompt
def test_prompt_is_subject_neutral_and_carries_no_arm():
    assert "quantum computing, machine learning" not in C.SYSTEM_PROMPT
    msg = C._build_user_message(
        title="t", venue="v", year=2024, arm=Arm.QUANTUM_COMPUTING, crossref_summary={}, passages=[]
    )
    assert "arm" not in msg.lower().replace("warm", "")


# --------------------------------------------------------------------------- text cache
def test_text_cache_is_off_unless_asked(tmp_path, monkeypatch):
    ft = SimpleNamespace(
        text="full text", source=SimpleNamespace(value="pdf"), url="http://x", complete=True
    )
    monkeypatch.delenv("DAP_TEXT_CACHE", raising=False)
    P._cache_text("https://openalex.org/W9", ft)
    assert not list(tmp_path.iterdir())
    monkeypatch.setenv("DAP_TEXT_CACHE", str(tmp_path / "texts"))
    P._cache_text("https://openalex.org/W9", ft)
    assert (tmp_path / "texts" / "W9.txt").read_text() == "full text"
    assert json.loads((tmp_path / "texts" / "W9.json").read_text())["text_url"] == "http://x"


# --------------------------------------------------------------------------- current models
def test_forced_tool_choice_only_where_the_model_accepts_it():
    assert C.tool_choice_for("claude-haiku-4-5-20251001") == {
        "type": "tool",
        "name": "record_classification",
    }
    for m in ("claude-opus-5-5", "claude-sonnet-5-5"):
        assert C.tool_choice_for(m) == {"type": "auto"}
        assert C.max_tokens_for(m) > 8000
        req = C.build_request(
            model=m,
            title="t",
            venue="v",
            year=2024,
            arm=Arm.UNSPECIFIED,
            openalex_id="W1",
            crossref_summary={},
            passages=[],
        )
        assert req["tool_choice"] == {"type": "auto"} and "temperature" not in req


def test_current_models_are_priced_so_the_budget_can_stop_them():
    assert C.Usage(input_tokens=1_000_000, output_tokens=0, model="claude-opus-5-5").cost_usd == 4.0


# --------------------------------------------------------------------------- privacy
# The contact address goes to Crossref (`mailto=`) and Unpaywall (`email=`) and
# nowhere else: not in a User-Agent, not to OpenAlex, not into the database.
ADDR = "someone.private@example.org"


def _recorder(calls, reply=None):
    import dap.sources as S

    def fake(cfg, url, **kw):
        calls.append((url, kw.get("params") or {}))
        if reply is None:
            raise S.PermanentError("HTTP 404")
        return reply

    return fake


def test_redact_masks_addresses_and_keys_but_keeps_the_url_readable():
    from dap.redact import redact

    msg = (
        "ConnectionError: HTTPSConnectionPool(host='api.unpaywall.org'): url: "
        "/v2/10.1/x?email=someone.private%40example.org&api_key=sk-123 "
        "(UA mailto:someone.private@example.org)"
    )
    out = redact(msg)
    assert "someone.private" not in out and "sk-123" not in out
    assert "api.unpaywall.org" in out and "/v2/10.1/x?email=<redacted>" in out


def test_no_user_agent_carries_the_address():
    import dap.sources as S
    import dap.verify as V

    assert "@" not in S.Config(contact_email=ADDR).user_agent
    assert "@" not in V._UA and "{" not in V._UA


def test_a_failed_request_error_does_not_carry_the_address(monkeypatch):
    import requests

    import dap.sources as S

    def boom(url, params=None, **kw):
        raise requests.ConnectionError(
            f"Max retries exceeded with url: {url}?email={ADDR.replace('@', '%40')}"
        )

    monkeypatch.setattr(S.requests, "get", boom)
    monkeypatch.setattr(S.time, "sleep", lambda s: None)
    monkeypatch.setattr(S.LIMITER, "wait", lambda url: None)
    with pytest.raises(S.TransientError) as e:
        S._get(
            S.Config(contact_email=ADDR, max_retries=1),
            "https://api.unpaywall.org/v2/10.1/x",
            params={"email": ADDR},
        )
    assert "someone.private" not in str(e.value)


def test_only_crossref_gets_the_address_at_metadata(monkeypatch):
    import dap.sources as S

    calls = []
    monkeypatch.setattr(S, "_get", _recorder(calls))
    monkeypatch.setattr(P, "_get", _recorder(calls, SimpleNamespace(json=lambda: {})))
    S.crossref_facts(S.Config(contact_email=ADDR), "10.1/x")
    P._openalex(S.Config(contact_email=ADDR), "https://openalex.org/W1")
    (cr_url, cr_params), (oa_url, oa_params) = calls
    assert "crossref" in cr_url and cr_params == {"mailto": ADDR}
    assert "openalex" in oa_url and ADDR not in json.dumps(oa_params)
    calls.clear()
    S.crossref_facts(S.Config(), "10.1/x")  # no address: public pool
    assert calls[0][1] == {}


def test_without_an_address_unpaywall_is_skipped_and_noted(monkeypatch):
    import dap.sources as S

    calls = []
    monkeypatch.setattr(S, "_get", _recorder(calls))
    ft = S.retrieve_full_text(
        S.Config(), work={}, doi="10.1/x", title="A title", facts=S.CrossrefFacts(found=True)
    )
    assert not any("unpaywall" in u for u, _ in calls)
    assert all(ADDR not in json.dumps(p) for _, p in calls)
    assert any(
        a["source"] == "unpaywall" and a["outcome"].startswith("skipped") for a in ft.attempts
    )


def test_the_store_never_writes_an_address_into_failures(tmp_path):
    from dap.store import Store

    s = Store(tmp_path / "s.sqlite")
    s.fail("https://openalex.org/W1", "retrieve", "transient", f"url: /v2/10.1/x?email={ADDR}")
    import sqlite3

    (msg,) = (
        sqlite3.connect(tmp_path / "s.sqlite").execute("SELECT message FROM failures").fetchone()
    )
    assert ADDR not in msg and "email=<redacted>" in msg


def test_collect_sends_no_address():
    seen = {}

    def get(url, params=None, **kw):
        seen.update(params)
        return SimpleNamespace(
            status_code=200,
            headers={},
            raise_for_status=lambda: None,
            json=lambda: {"results": [], "meta": {}},
        )

    Http(get=get, sleep=lambda s: None).works({"search": "x"})
    assert "mailto" not in seen and "email" not in seen
