"""Fixtures and fakes — everything the pipeline touches, faked at the boundary.

The point is that no layer should require a network, an API key, or a live
publisher to be exercised. Anything that can only be tested against the real
internet is a design problem, not an acceptable gap: it means the failure will
first be observed in production, at scale, spending money.

The Crossref payloads below are shaped from the documented REST API format
(assertion / relation / link / reference). They are deliberately awkward —
inconsistent assertion naming, a relation type in a different case, a reference
list mixing article and dataset DOIs — because that is what real payloads look
like and tidy fixtures prove nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dap.sources import Config

# ---------------------------------------------------------------------------
# Crossref
# ---------------------------------------------------------------------------
CROSSREF_WITH_DAS = {
    "message": {
        "type": "journal-article",
        "container-title": ["npj Quantum Information"],
        "published": {"date-parts": [[2024, 3, 11]]},
        "assertion": [
            # noise that must NOT be picked up as a DAS
            {"name": "received", "value": "2023-08-01", "label": "Received"},
            {"name": "accepted", "value": "2024-01-15", "label": "Accepted"},
            # the real thing, under a name no lookup would guess
            {
                "name": "Data_Availability",
                "label": "Data availability statement",
                "value": "The datasets generated during the current study are "
                "available in the Zenodo repository, "
                "https://doi.org/10.5281/zenodo.10499999",
            },
            # a second, separate statement about code
            {
                "name": "CodeAvail",
                "label": "Code availability",
                "value": "Analysis code is available at https://github.com/g/proj",
            },
        ],
        "relation": {
            "is-supplemented-by": [
                {"id-type": "doi", "id": "10.5281/zenodo.10499999", "asserted-by": "subject"}
            ],
            # must be ignored — not a dataset relation
            "has-preprint": [
                {"id-type": "doi", "id": "10.48550/arXiv.2308.00001", "asserted-by": "object"}
            ],
        },
        "reference": [
            {"key": "r1", "DOI": "10.1103/PhysRevLett.100.100501"},
            {"key": "r2", "DOI": "10.5061/dryad.abc123"},  # a dataset
            {"key": "r3", "unstructured": "Smith et al, no DOI"},
        ],
        "link": [
            {
                "URL": "https://api.springer.com/fulltext/10.1038/x",
                "content-type": "application/xml",
                "content-version": "vor",
                "intended-application": "text-mining",
            },
            {
                "URL": "https://example.org/similarity",
                "content-type": "application/pdf",
                "content-version": "vor",
                "intended-application": "similarity-checking",
            },  # must be ignored
        ],
        "license": [{"URL": "http://creativecommons.org/licenses/by/4.0/"}],
    }
}

CROSSREF_BARE = {
    "message": {
        "type": "proceedings-article",
        "container-title": ["Proceedings of ISCA"],
        "published": {"date-parts": [[2022]]},
        # no assertion, no relation, no link — the common conference case
        "author": [{"given": "K.", "family": "Ueno"}, {"given": "M.", "family": "Tanaka"}],
        "reference": [{"key": "r1", "DOI": "10.1145/3000000.3000001"}],
    }
}

OPENALEX_ARXIV = {
    "title": "Transmon readout fidelity improvements",
    "publication_year": 2024,
    "primary_location": {"source": {"display_name": "Physical Review Applied"}},
    "locations": [
        {
            "source": {"display_name": "Physical Review Applied"},
            "landing_page_url": "https://link.aps.org/doi/10.1103/PhysRevApplied.21.1",
        },
        {
            "source": {"display_name": "arXiv (Cornell University)"},
            "landing_page_url": "https://arxiv.org/abs/2401.12345v2",
            "pdf_url": "https://arxiv.org/pdf/2401.12345v2",
        },
    ],
    "ids": {"openalex": "https://openalex.org/W123"},
}

OPENALEX_NO_ARXIV = {
    "title": "A systems paper",
    "locations": [
        {"source": {"display_name": "ACM"}, "landing_page_url": "https://dl.acm.org/doi/10.1145/1"}
    ],
    "ids": {"openalex": "https://openalex.org/W999"},
}

DATACITE_DATASET_ATTRS = {"types": {"resourceTypeGeneral": "Dataset"}, "publisher": "Zenodo"}
DATACITE_ARTICLE_ATTRS = {"types": {"resourceTypeGeneral": "JournalArticle"}}

DATACITE_DATASET = {"data": {"attributes": DATACITE_DATASET_ATTRS}}
DATACITE_ARTICLE = {"data": {"attributes": DATACITE_ARTICLE_ATTRS}}


def datacite_bulk(*pairs: tuple[str, dict]) -> dict:
    """Build a response of DataCite's *bulk* endpoint.

    In that shape `data` is a list and each item carries its own `id`. The
    single-DOI endpoint returns a bare object instead, which is why a
    single-shaped fixture cannot stand in for a bulk one.

    Args:
        *pairs: `(doi, attributes)` pairs, one per record in the response.

    Returns:
        The response body.
    """
    return {"data": [{"id": doi, "type": "dois", "attributes": attrs} for doi, attrs in pairs]}


#: A paper whose DOI is registered with DataCite, not Crossref — an institutional
#: repository deposit. Shaped on the real record for 10.17169/refubium-4067,
#: which Crossref 404s. Note `resourceTypeGeneral` is absent, as it is there:
#: a DataCite record is not required to carry one.
DATACITE_PAPER = {
    "data": {
        "attributes": {
            "titles": [{"title": "A game of surface codes"}],
            "publisher": "Freie Universitat Berlin",
            "publicationYear": 2019,
            "types": {},
            "descriptions": [
                {
                    "descriptionType": "Abstract",
                    "description": "We present a lattice surgery scheme.",
                },
                {
                    "descriptionType": "Data Availability",
                    "description": "Source data are available at Zenodo 10.5281/zenodo.1.",
                },
            ],
            "relatedIdentifiers": [
                {
                    "relationType": "IsSupplementedBy",
                    "relatedIdentifierType": "DOI",
                    "relatedIdentifier": "10.5281/zenodo.1",
                },
                {
                    "relationType": "Cites",
                    "relatedIdentifierType": "DOI",
                    "relatedIdentifier": "10.1103/PhysRevA.1",
                },
                {
                    "relationType": "IsPartOf",
                    "relatedIdentifierType": "URL",
                    "relatedIdentifier": "https://refubium.fu-berlin.de/",
                },
            ],
            "rightsList": [{"rightsUri": "http://creativecommons.org/licenses/by/4.0"}],
        }
    }
}


class FakeResponse:
    """A stand-in for `requests.Response`.

    Attributes:
        status_code: The HTTP status.
        headers: The response headers.
        text: The body as text.
        content: The body as bytes.
        url: The final URL, after redirects.
    """

    def __init__(
        self,
        payload: dict | None = None,
        *,
        status: int = 200,
        headers: dict | None = None,
        text: str = "",
        content: bytes = b"",
        url: str = "https://final.example/x",
    ) -> None:
        """Script one response.

        Args:
            payload: What `json()` returns; None makes `json()` raise.
            status: The HTTP status.
            headers: The response headers; JSON content type by default.
            text: The body as text.
            content: The body as bytes.
            url: The final URL, after redirects.
        """
        self._payload = payload
        self.status_code = status
        self.headers = headers or {"Content-Type": "application/json"}
        self.text = text
        self.content = content
        self.url = url

    def json(self) -> dict:
        """Return the scripted JSON payload.

        Returns:
            The payload given on construction.

        Raises:
            ValueError: If no payload was given.
        """
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def fake_get_factory(routes: dict) -> Callable[..., FakeResponse]:
    """Build a stand-in for `sources._get`.

    Args:
        routes: Maps a substring of the URL to the `FakeResponse` to return,
            or to an exception to raise.

    Returns:
        A function with the signature of `sources._get`. It raises
        AssertionError for a URL that no route matches.
    """

    def _fake_get(cfg: Config, url: str, **kw: object) -> FakeResponse:
        for needle, resp in routes.items():
            if needle in url:
                if isinstance(resp, Exception):
                    raise resp
                return resp
        raise AssertionError(f"unrouted URL in test: {url}")

    return _fake_get


# ---------------------------------------------------------------------------
# Anthropic client
# ---------------------------------------------------------------------------
class FakeBlock:
    """A stand-in for a `tool_use` content block.

    Attributes:
        type: Always `"tool_use"`.
        id: The id of the tool call.
        input: The tool input, i.e. the classification payload.
        name: Always `"record_classification"`.
    """

    def __init__(self, payload: dict, block_id: str = "tu_1") -> None:
        """Script one tool call.

        Args:
            payload: The tool input the model "returned".
            block_id: The id of the tool call.
        """
        self.type = "tool_use"
        self.id = block_id
        self.input = payload
        self.name = "record_classification"


class FakeTextBlock:
    """A stand-in for a text content block: a reply with no tool call.

    Attributes:
        type: Always `"text"`.
        text: The reply text.
    """

    type = "text"
    text = "I think the answer is..."


class FakeMessage:
    """A stand-in for an Anthropic `Message`.

    Attributes:
        content: The content blocks.
        stop_reason: Why the model stopped.
        usage: An object with `input_tokens` and `output_tokens`.
    """

    def __init__(
        self, blocks: list, *, stop_reason: str = "tool_use", in_tok: int = 1000, out_tok: int = 400
    ) -> None:
        """Script one model reply.

        Args:
            blocks: The content blocks of the reply.
            stop_reason: Why the model stopped.
            in_tok: The input tokens reported.
            out_tok: The output tokens reported.
        """
        self.content = blocks
        self.stop_reason = stop_reason
        self.usage = SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok)


class FakeAnthropic:
    """A stand-in for the Anthropic client that replays scripted responses.

    It records every request so tests can assert on what was actually sent
    (e.g. that the schema rejection was handed back to the model).

    Attributes:
        calls: The keyword arguments of every `create` call, in order.
        messages: The client itself, so `client.messages.create` works.
    """

    def __init__(self, responses: list[FakeMessage]) -> None:
        """Script the client.

        Args:
            responses: The replies to return, one per `create` call, in order.
        """
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.messages = self

    def create(self, **kw: object) -> FakeMessage:
        """Record a request and return the next scripted reply.

        Args:
            **kw: The request, as `messages.create` would receive it.

        Returns:
            The next scripted reply.

        Raises:
            AssertionError: If every scripted reply has been used.
        """
        self.calls.append(kw)
        if not self._responses:
            raise AssertionError("FakeAnthropic ran out of scripted responses")
        return self._responses.pop(0)


def good_payload(**overrides: object) -> dict:
    """Build a schema-valid classification payload.

    Args:
        **overrides: Top-level keys to replace in the payload.

    Returns:
        The payload, as the `record_classification` tool would receive it.
    """
    base = {
        "evidence": [
            {
                "id": 1,
                "quote": "available in the Zenodo repository",
                "section": "Data availability",
                "locator": "",
            },
            {
                "id": 2,
                "quote": "The datasets generated during the current study",
                "section": "Data availability",
                "locator": "",
            },
        ],
        "paper_type": {
            "value": "simulation",
            "confidence": "high",
            "rationale": "methods describe simulated instances only",
            "evidence_ids": [2],
        },
        "axis_a": {
            "case": {
                "value": "not_applicable",
                "confidence": "high",
                "rationale": "no pre-existing dataset is analysed",
                "evidence_ids": [2],
            },
            "data_type": {
                "value": "unclear",
                "confidence": "medium",
                "rationale": "no external data present",
                "evidence_ids": [2],
            },
            "declared": {
                "value": "none_stated",
                "confidence": "high",
                "rationale": "no statement about external data",
                "evidence_ids": [2],
            },
            "granularity": {
                "value": "not_applicable",
                "confidence": "high",
                "rationale": "axis absent",
                "evidence_ids": [2],
            },
            "pinned": "none",
            "restriction": "none_given",
            "repo_types": [],
            "identifiers": [],
            "names": [],
        },
        "axis_b": {
            "case": {
                "value": "openly_accessible",
                "confidence": "high",
                "rationale": "simulation output deposited with a versioned DOI",
                "evidence_ids": [1],
            },
            "data_type": {
                "value": "synthetic",
                "confidence": "high",
                "rationale": "generated by simulation",
                "evidence_ids": [2],
            },
            "declared": {
                "value": "public",
                "confidence": "high",
                "rationale": "statement names a Zenodo record",
                "evidence_ids": [1],
            },
            "granularity": {
                "value": "processed",
                "confidence": "medium",
                "rationale": "processed outputs described",
                "evidence_ids": [1],
            },
            "pinned": "doi_versioned",
            "restriction": "none_given",
            "repo_types": ["zenodo"],
            "identifiers": ["10.5281/zenodo.10499999"],
            "names": ["circuit instance set"],
            "generation_code": True,
            "generation_params": True,
            "seed_specified": False,
            "environment_specified": False,
        },
        "das_present": True,
        "needs_human_review": False,
        "review_reason": "",
    }
    base.update(overrides)
    return base
