#!/usr/bin/env python3
"""Assign the study arm from title keywords: `keyword_arms.csv`.

Replaces two arm lists that disagreed. `frame_arms.csv` called every paper in
a dedicated quantum venue quantum computing, physics included. `arms.csv`
(`build_arms.py`) used OpenAlex's single primary topic, which missed papers
titled "Quantum computing for nuclear reactions", and left 323,373 papers
`unresolved` because OpenAlex's daily budget ran out partway through.

This runs offline over the harvest's own titles, so every paper gets an arm,
and each row records the tier and the terms that produced it, so any single
assignment can be checked by reading one line.

Measured on 2017-2026 (371,367 papers), with the rule Diana chose on 25 Sep 2026
(tier 1 + tier 2 = quantum_computing):

  * recall against OpenAlex topic T10682 (Quantum Computing Algorithms and
    Architecture): 77%. Titles only; the misses are papers whose title names
    the application, not the method ("Quantum planning for swarm robotics").
  * 2,833 already-coded venue-frame papers match no term. They are physics
    published in quantum venues. Their stored arm is NOT rewritten; this file
    is joined at analysis time.

    python keyword_arms.py --input ../../openalex_papers.csv --out keyword_arms.csv \\
        --year-min 2017 --year-max 2026

Reads one CSV, writes one CSV. No network.
"""
from __future__ import annotations

import argparse
import collections
import re
import sys

import pandas as pd

#: Bump when a pattern changes; every row records the version that assigned it.
RULE_VERSION = "kw-1"

#: Quantum computing proper.
TIER1 = {
    "qubit": r"\bqu(?:bit|dit|trit)s?\b",
    "quantum computing": r"quantum[- ]comput",
    "quantum circuit": r"quantum circuits?\b",
    "quantum algorithm": r"quantum algorithm",
    "quantum gate": r"quantum (?:logic )?gates?\b",
    "error correction / fault tolerance":
        r"quantum error|error[- ]correct\w* (?:code|quantum)|fault[- ]toleran\w* quantum"
        r"|quantum fault|surface codes?\b|stabili[sz]er codes?|logical qubits?"
        r"|quantum (?:ldpc|codes?)\b|error mitigation",
    "NISQ": r"\bnisq\b",
    "variational / VQE / QAOA": r"variational quantum|\bvqe\b|\bqaoa\b|quantum approximate optimi",
    "annealing": r"quantum anneal|\bd-?wave\b",
    "quantum machine learning":
        r"quantum machine learning|quantum neural|quantum (?:kernel|classifier|perceptron"
        r"|reinforcement|generative|convolutional)|\bqml\b|quantum-enhanced (?:machine )?learning"
        r"|quantum-enhanced feature|quantum feature (?:maps?|spaces?)",
    "advantage / supremacy": r"quantum (?:advantage|supremacy|speed-?ups?)",
    "software / compilers":
        r"quantum (?:software|programs?|programming|compil|computer science)|qiskit|\bcirq\b|transpil",
    "Grover / Shor": r"\bgrover\b|\bgrover'?s\b|\bshor'?s\b",
    "processors / hardware":
        r"quantum (?:processors?|hardware|computers?|chips?|devices? for comput)"
        r"|superconducting (?:qubit|quantum|processor)|transmons?\b|spin qubits?",
    "quantum walk": r"quantum walks?",
    "measurement-based / blind": r"measurement-based quantum|blind quantum|cluster states?",
    "circuit-level terms":
        r"\bcnot\b|\bclifford\b|\bt-count\b|\btrotter|random circuit sampling|boson sampling"
        r"|quantum volume|randomi[sz]ed benchmarking",
    "algorithmic primitives":
        r"hybrid quantum[- ]classical|quantum[- ]classical hybrid|quantum (?:amplitude|phase) estimation"
        r"|quantum fourier|quantum signal processing|quantum singular value|block[- ]encoding"
        r"|quantum (?:krylov|eigensolver|optimi[sz]ation|search|oracle|query|sampling|image"
        r"|state preparation|linear (?:system|solver))",
    "named platforms": r"ibm quantum|ibmq|\bsycamore\b|rigetti|ionq|quantinuum",
    "quantum codes": r"quantum [\w-]+ codes?\b|quantum (?:tensor|convolutional|polar|reed|bch|stabilizer)[\w -]*codes?",
}

#: Quantum information and communication: what OpenAlex topic T10020 covers,
#: which the earlier topic map also counted as quantum computing.
TIER2 = {
    "QKD / quantum cryptography":
        r"quantum key distribution|\bqkd\b|quantum cryptograph|quantum secure|quantum secret sharing"
        r"|quantum (?:digital )?signature|quantum private|quantum authenticat|quantum random number",
    "quantum networks / communication":
        r"quantum (?:network|internet|repeater|communication|teleportation|channels?\b|state transfer)"
        r"|entanglement (?:distribution|purification|distillation|swapping|routing)",
    "quantum information":
        r"quantum information|quantum (?:channel )?capacit|quantum (?:state )?(?:tomography"
        r"|discrimination|certification|verification)|quantum entropy|quantum resource theor",
    "quantum memory": r"quantum memor",
    # Not a bare "quantum simulation": in device physics that means a classical
    # transport calculation, and it pulled in transistor papers.
    "quantum simulators":
        r"quantum simulators?|digital quantum simulat|analog(?:ue)? quantum simulat"
        r"|quantum simulation of (?:lattice|gauge|spin|hubbard|many|fermion|boson|open quantum)",
    "trapped ions": r"trapped[- ]ions?",
}

#: Post-quantum cryptography is classical cryptography: quantum computers are
#: the threat it defends against, not its subject.
POST_QUANTUM = r"post-quantum|quantum[- ](?:resistant|safe)|lattice-based cryptograph"

#: Terms that a post-quantum or chemistry title uses in passing. Only a term
#: outside this set outranks those exclusions.
GENERIC = frozenset({"quantum computing", "processors / hardware", "Grover / Shor",
                     "advantage / supremacy"})

#: "Quantum computational study" in chemistry is a DFT calculation run on a
#: classical machine. Measured: 24 titles, all DFT, docking or spectroscopy.
CHEMISTRY_STRONG = r"docking|hirshfeld|homo-lumo|ft-ir|fukui|\bdft\b|density functional|corrosion|toxicity"
CHEMISTRY = CHEMISTRY_STRONG + r"|spectroscop|synthesis|drug"

TIERS = ("tier1", "tier2", "post_quantum", "chemistry", "no_match")


def _matches(title: str, table: dict) -> list[str]:
    return [k for k, p in table.items() if re.search(p, title)]


def assign(title) -> tuple[str, str, list[str]]:
    """(arm, tier, matched terms) for one title."""
    t = title.lower() if isinstance(title, str) else ""
    t1, t2 = _matches(t, TIER1), _matches(t, TIER2)
    if re.search(POST_QUANTUM, t) and set(t1) <= GENERIC:
        return "out_of_scope", "post_quantum", ["post-quantum"] + t1
    if (set(t1) == {"quantum computing"}
            and not re.search(r"quantum computing|quantum computers?\b|processor|qubit", t)
            and ((re.search(r"quantum computational\b", t) and re.search(CHEMISTRY, t))
                 or re.search(CHEMISTRY_STRONG, t))):
        return "out_of_scope", "chemistry", t1
    if t1:
        return "quantum_computing", "tier1", t1 + t2
    if t2:
        return "quantum_computing", "tier2", t2
    return "out_of_scope", "no_match", []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="keyword_arms.csv")
    ap.add_argument("--year-min", type=int)
    ap.add_argument("--year-max", type=int)
    a = ap.parse_args()

    df = pd.read_csv(a.input, usecols=["openalex_id", "title", "publication_year"],
                     low_memory=False).drop_duplicates("openalex_id")
    yr = pd.to_numeric(df.publication_year, errors="coerce")
    keep = pd.Series(True, index=df.index)
    if a.year_min: keep &= yr >= a.year_min
    if a.year_max: keep &= yr <= a.year_max
    df = df[keep]

    out = df.title.map(assign)
    res = pd.DataFrame({"openalex_id": df.openalex_id,
                        "arm": out.str[0], "tier": out.str[1],
                        "matched_terms": out.str[2].map("; ".join),
                        "rule_version": RULE_VERSION})
    res.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(res):,} papers")
    print("arms:", dict(collections.Counter(res.arm)))
    print("tiers:", dict(collections.Counter(res.tier)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
