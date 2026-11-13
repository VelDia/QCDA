"""Baselines: an LLM given the paper's full text and the class definitions only.

    # Claude (Anthropic SDK; ANTHROPIC_API_KEY or an `ant auth login` profile)
    python baseline.py --provider anthropic --model claude-opus-5-5
    # any OpenAI-compatible chat endpoint: OpenAI, Gemini, OpenRouter, a local server
    python baseline.py --provider openai --model gpt-5 --api-key-env OPENAI_API_KEY
    python baseline.py --provider openai --model gemini-2.5-pro \
        --base-url https://generativelanguage.googleapis.com/v1beta/openai \
        --api-key-env GEMINI_API_KEY
    python baseline.py --provider openai --model llama3.3 \
        --base-url http://localhost:11434/v1 --api-key-env ""

What the baseline is given, and why. Each paper's full text is read from
`--texts`, the directory the pipeline run filled with DAP_TEXT_CACHE
(run_pipeline.sh). So a baseline reads the SAME copy of each paper the
pipeline retrieved: any accuracy difference is the pipeline's design, not a
better or worse PDF. This is generous to the baselines, because in practice
someone would have to find and download every paper for them; that effort is
counted separately by evaluate.py. A paper the pipeline found no text for is
given to the baseline with its title only, as it was to the pipeline.

Variants (`--variant`):
  definitions       (default) the seven case definitions and two axis
                    definitions, as the human reviewers saw them, and the
                    answer format. Nothing else.
  pipeline_prompt   ablation: the pipeline's full system prompt (decision
                    rules, exceptions, evidence rules) on the full text, with
                    none of the machinery (no passage selection, metadata,
                    validator, grounding check, or verification).

One JSON line per paper is appended to `--out`; a rerun skips papers already
answered, so an interrupted run resumes. Nothing is retried silently: an
unparseable answer or a non-token case is recorded as invalid.

Text longer than `--max-chars` is cut to fit the model's context, and the row
says so (`truncated`); the default fits the full text of almost every paper.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from common import HERE, baseline_system, grounded, parse_answer, read_csv, wid

_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Providers. Each returns (reply_text, input_tokens, output_tokens, stop_reason).
# ---------------------------------------------------------------------------
class AnthropicProvider:
    """Claude through the official SDK.

    No sampling parameters: current models reject `temperature`, and thinking
    is adaptive by default. Streaming, because full papers make long requests.

    Attributes:
        client: The `anthropic.Anthropic` client the requests are sent with.
        model: The model id.
        effort: The `output_config.effort` level, or None to leave it unset.
        max_tokens: The output-token ceiling of each request.
    """

    def __init__(self, model: str, effort: str | None, max_tokens: int) -> None:
        """Create the client from the environment's credentials.

        Args:
            model: The model id, e.g. `claude-opus-5-5`.
            effort: The effort level to request, or None for the model default.
            max_tokens: The output-token ceiling of each request.
        """
        import anthropic

        self.client = anthropic.Anthropic()
        self.model, self.effort, self.max_tokens = model, effort, max_tokens

    def __call__(self, system: str, user: str) -> tuple[str, int, int, str]:
        """Send one request and wait for the complete reply.

        Args:
            system: The system prompt.
            user: The user message: the paper header and its full text.

        Returns:
            A tuple `(reply_text, input_tokens, output_tokens, stop_reason)`.
        """
        kw = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if self.effort:
            kw["output_config"] = {"effort": self.effort}
        with self.client.messages.stream(**kw) as stream:
            msg = stream.get_final_message()
        text = "".join(b.text for b in msg.content if b.type == "text")
        return text, msg.usage.input_tokens, msg.usage.output_tokens, msg.stop_reason


class OpenAICompatibleProvider:
    """Any endpoint that serves `POST {base_url}/chat/completions`.

    Covers OpenAI, Gemini's and OpenRouter's compatible endpoints, and local
    servers (Ollama, vLLM).

    Attributes:
        requests: The `requests` module, imported on construction.
        model: The model id.
        base: The base URL, without a trailing slash.
        key: The API key, or an empty string when none is sent.
        temperature: The sampling temperature, or None to omit it.
        max_tokens: The completion-token ceiling; 0 omits it.
        timeout: The request timeout in seconds.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key_env: str,
        temperature: float | None,
        max_tokens: int,
        timeout: float = 600,
    ) -> None:
        """Read the API key and store the request settings.

        Exits with a message if `api_key_env` names a variable that is not set.

        Args:
            model: The model id the endpoint expects.
            base_url: The endpoint's base URL, e.g. `https://api.openai.com/v1`.
            api_key_env: The environment variable that holds the API key. An
                empty string sends no `Authorization` header (local servers).
            temperature: The sampling temperature, or None to omit it, because
                several current models reject it.
            max_tokens: The completion-token ceiling; 0 omits it.
            timeout: The request timeout in seconds.
        """
        import requests

        self.requests = requests
        self.model, self.base = model, base_url.rstrip("/")
        self.key = os.environ.get(api_key_env, "") if api_key_env else ""
        if api_key_env and not self.key:
            sys.exit(f"{api_key_env} is not set")
        self.temperature, self.max_tokens, self.timeout = temperature, max_tokens, timeout

    def __call__(self, system: str, user: str) -> tuple[str, int, int, str]:
        """Send one chat completion request.

        A 429 or 5xx answer is retried up to five times, waiting for
        `Retry-After` or with exponential backoff.

        Args:
            system: The system prompt.
            user: The user message: the paper header and its full text.

        Returns:
            A tuple `(reply_text, input_tokens, output_tokens, stop_reason)`.
            `requests.HTTPError` propagates from any other error status, and
            from the last answer once the retries are used up.
        """
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.max_tokens:
            body["max_completion_tokens"] = self.max_tokens
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        for attempt in range(5):
            r = self.requests.post(
                f"{self.base}/chat/completions", json=body, headers=headers, timeout=self.timeout
            )
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(float(r.headers.get("Retry-After") or 2**attempt))
                continue
            r.raise_for_status()
            d = r.json()
            ch = (d.get("choices") or [{}])[0]
            u = d.get("usage") or {}
            return (
                (ch.get("message") or {}).get("content") or "",
                u.get("prompt_tokens", 0),
                u.get("completion_tokens", 0),
                ch.get("finish_reason", ""),
            )
        r.raise_for_status()


# ---------------------------------------------------------------------------
def user_message(paper: dict, text: str) -> str:
    """Build the user message for one paper.

    Args:
        paper: The gold row of the paper: `title`, `doi`, `publication_year`.
        text: The paper's full text, already cut to the context limit. Empty
            when no text was retrieved; the message then says so.

    Returns:
        The header (title, DOI, year) followed by the full text.
    """
    head = (
        f"Title: {paper.get('title', '')}\n"
        f"DOI: {paper.get('doi', '')}   Year: {paper.get('publication_year', '')}\n\n"
    )
    if not text:
        return head + "(No full text is available for this paper.)"
    return head + "FULL TEXT OF THE PAPER:\n\n" + text


def run_one(
    paper: dict,
    *,
    provider: Callable[[str, str], tuple[str, int, int, str]],
    system: str,
    texts: Path,
    max_chars: int,
    name: str,
    model: str,
    variant: str,
) -> dict:
    """Ask one baseline about one paper and build its result row.

    Nothing is raised for a failed request: the error is recorded in the row
    (`ok` False), so the paper is retried by the next run.

    Args:
        paper: The gold row of the paper; `openalex_id` is required.
        provider: The model to ask: called with `(system, user)`, returns
            `(reply_text, input_tokens, output_tokens, stop_reason)`.
        system: The system prompt (see `common.baseline_system`).
        texts: The directory of cached full texts, one `<work id>.txt` each.
        max_chars: The longest text given to the model; anything longer is
            cut and the row is marked `truncated`.
        name: The system name recorded in the row.
        model: The model id recorded in the row.
        variant: The prompt variant recorded in the row.

    Returns:
        The result row: provenance, timing, token counts, the raw reply, and
        for each axis the case, the confidence, the quoted evidence and
        whether that quote is in the text the model was given.
    """
    w = wid(paper["openalex_id"])
    p = texts / f"{w}.txt"
    full = p.read_text(encoding="utf-8") if p.exists() else ""
    given = full[:max_chars]
    row = {
        "openalex_id": paper["openalex_id"],
        "system": name,
        "model": model,
        "variant": variant,
        "has_text": bool(full),
        "text_chars": len(full),
        "truncated": len(full) > max_chars,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    t0 = time.monotonic()
    try:
        reply, tin, tout, stop = provider(system, user_message(paper, given))
    except Exception as e:  # recorded, never swallowed
        row.update(
            ok=False,
            error=f"{type(e).__name__}: {str(e)[:300]}",
            seconds=round(time.monotonic() - t0, 2),
        )
        return row
    ans, err = parse_answer(reply)
    row.update(
        ok=True,
        seconds=round(time.monotonic() - t0, 2),
        input_tokens=tin,
        output_tokens=tout,
        stop_reason=stop,
        parse_error=err,
        reply=reply,
    )
    for ax, k in (("axis_a", "a"), ("axis_b", "b")):
        row[f"{k}_case"] = ans[ax]["case"]
        row[f"{k}_conf"] = ans[ax]["confidence"]
        row[f"{k}_evidence"] = ans[ax]["evidence"]
        row[f"{k}_evidence_grounded"] = grounded(ans[ax]["evidence"], given)
    return row


def main() -> None:
    """Run one baseline over the gold papers, appending one JSON line each.

    Reads the command line (see the module docstring), skips the papers the
    output file already answers and runs the rest on a thread pool. Exits
    with a message if the texts directory is empty.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gold", default=str(HERE / "gold" / "frame_review.csv"))
    ap.add_argument("--texts", default=str(HERE / "texts"))
    ap.add_argument("--provider", choices=["anthropic", "openai"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", help="system name in the results (default: model-variant)")
    ap.add_argument("--variant", choices=["definitions", "pipeline_prompt"], default="definitions")
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="openai provider only; omitted by default because several current models reject it",
    )
    ap.add_argument(
        "--effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="anthropic provider only",
    )
    ap.add_argument("--max-tokens", type=int, default=32000)
    ap.add_argument("--max-chars", type=int, default=600_000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out")
    a = ap.parse_args()

    name = a.name or f"{a.model}-{a.variant}"
    out = Path(a.out or HERE / "runs" / f"baseline_{name}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    texts = Path(a.texts)
    if not texts.is_dir() or not any(texts.glob("*.txt")):
        sys.exit(
            f"no texts in {texts}: run run_pipeline.sh first, so the baseline "
            f"reads the same copies the pipeline read"
        )

    papers = read_csv(a.gold)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            if r.get("ok"):
                done.add(r["openalex_id"])
    todo = [p for p in papers if p["openalex_id"] not in done][: a.limit or None]
    print(f"{name}: {len(todo)} to run, {len(done)} already done -> {out}", file=sys.stderr)

    if a.provider == "anthropic":
        provider = AnthropicProvider(a.model, a.effort, a.max_tokens)
    else:
        provider = OpenAICompatibleProvider(
            a.model, a.base_url, a.api_key_env, a.temperature, a.max_tokens
        )
    system = baseline_system(a.variant)

    def work(p: dict) -> None:
        row = run_one(
            p,
            provider=provider,
            system=system,
            texts=texts,
            max_chars=a.max_chars,
            name=name,
            model=a.model,
            variant=a.variant,
        )
        with _lock, open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"  {wid(p['openalex_id'])}  A={row.get('a_case', '-'):<25} "
            f"B={row.get('b_case', '-'):<25} {row.get('error', '')}",
            file=sys.stderr,
        )

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(work, todo))


if __name__ == "__main__":
    main()
