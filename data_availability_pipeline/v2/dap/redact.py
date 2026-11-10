"""Keep contact addresses and keys out of anything the pipeline stores or prints.

A failed request's error text carries its full URL, query string included, and
the Unpaywall lookup puts the contact address in that query string. Before this
module, those messages went into the `failures` table verbatim, so a shared
database or log carried the operator's email address.

`redact` rewrites the value of any credential-like query parameter
(`email=`, `mailto=`, `api_key=`, ...) and any `mailto:` address, wherever it
appears in a string. It is applied where errors are built (`sources._get`) and
again where they are stored (`Store.fail`), so a new call site cannot reintroduce
the leak by forgetting one of them.
"""
from __future__ import annotations

import re

MASK = "<redacted>"

_PARAM = re.compile(
    r"(?i)\b(email|mailto|api_key|apikey|api-key|access_token|token)=([^&\s'\"<>()]+)")
_MAILTO = re.compile(r"(?i)\bmailto:[^\s'\"<>();]+")


def redact(text: object) -> str:
    """`text` as a string, with credential values replaced by MASK."""
    s = _PARAM.sub(lambda m: f"{m.group(1)}={MASK}", str(text))
    return _MAILTO.sub(f"mailto:{MASK}", s)
