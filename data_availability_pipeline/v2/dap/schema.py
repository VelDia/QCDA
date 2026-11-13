"""dap.schema — the single source of truth for the coding scheme.

Nothing in this pipeline writes a value that has not passed through here. Every
controlled vocabulary is a closed enum; every record is validated before it is
persisted; an out-of-vocabulary value raises rather than being written.

This is the direct answer to how the v1 output drifted: there, the model's JSON
was flattened straight into CSV, so `request`, `upon_request` and
`available_on_request` all became real values of a five-value field, and a field
documented as a 1-12 integer code ended up holding English sentences.

SCHEME_VERSION is stamped on every row. Its major or minor number must be
incremented whenever a token is added, removed, or has its meaning changed;
rows differing there must never be pooled without an explicit, documented
migration. The patch number records a ruling on how the same tokens are applied
(Diana, 28 Oct 2026): rows differing only in patch pool on export, and the
stamp says which ruling each row was coded under.

Framework (v2):
    Axis A — external data: data that existed independently of this paper and
             that the paper performs analysis on.
    Axis B — self-generated data: data this paper produced by running an
             experiment, simulation, or generative procedure.
    A paper is scored independently on each axis. Purely theoretical work is
    the case where BOTH axes independently return `not_applicable`.

NOTE ON v1 DATA: the v1 scheme used the OPPOSITE axis assignment (its Axis A was
original/self-generated data, its Axis B was reused/external). v1 rows must be
axis-swapped by the migration in `migrate.py`; they cannot be read as v2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

# 2.0.1 (28 Oct 2026, S219): an explicit statement that no data was used,
# created, generated or analysed codes Axis B `not_applicable`. Axis A is still
# coded from the paper (Diana's clarification). Vocabulary unchanged.
SCHEME_VERSION = "2.0.1"


class _Tok(str, Enum):
    """Base class of every controlled vocabulary: members are their own token.

    Stored data and analysis code reference `.value` (the token), never the
    human-readable label. Relabelling for publication is then a documentation
    change, not a data migration.
    """

    def __str__(self) -> str:  # so f-strings and csv writers emit the token
        """Return the token, so f-strings and CSV writers emit it.

        Returns:
            The member's value.
        """
        return self.value

    @classmethod
    def tokens(cls) -> set[str]:
        """Return every token of this vocabulary.

        Returns:
            The set of member values.
        """
        return {m.value for m in cls}

    @classmethod
    def coerce(cls, v: object) -> "_Tok":
        """Accept only an exact token of this vocabulary.

        Deliberately strict: no case-folding, no whitespace stripping, no
        synonym table. A near-miss means the prompt or the model is wrong and
        we want to see it, not paper over it.

        Args:
            v: A member of this vocabulary, or the exact string of one of its
                tokens.

        Returns:
            The member of this vocabulary that `v` names.

        Raises:
            SchemaError: If `v` is neither a member nor an exact token.
        """
        if isinstance(v, cls):
            return v
        if isinstance(v, str) and v in cls.tokens():
            return cls(v)
        raise SchemaError(
            f"{cls.__name__}: {v!r} is not a valid token. Allowed: {sorted(cls.tokens())}"
        )


class SchemaError(ValueError):
    """Raised on any value that is not in its controlled vocabulary."""


# ---------------------------------------------------------------------------
# Cases — applied identically to both axes
# ---------------------------------------------------------------------------
class Case(_Tok):
    """The access state of one axis's data; applied identically to both axes.

    Attributes:
        NOT_APPLICABLE: The axis does not apply: no external data analysed
            (Axis A), or nothing generated (Axis B).
        OPENLY_ACCESSIBLE: A concrete identifier is given: a DOI, URL,
            accession number or repository.
        ACCESS_NOT_PROVIDED: The material is identifiable but no path to it is
            given.
        INSUFFICIENTLY_SPECIFIED: The material cannot even be identified.
        ACCESS_UNRESOLVABLE: A specific path is given, and it fails. Derived
            from a failed verification, never asserted from text.
        ON_REQUEST: Offered on contact, however hedged.
        EXPLICITLY_RESTRICTED: Stated that the data cannot be shared.
    """

    NOT_APPLICABLE = "not_applicable"
    OPENLY_ACCESSIBLE = "openly_accessible"
    ACCESS_NOT_PROVIDED = "access_not_provided"
    INSUFFICIENTLY_SPECIFIED = "insufficiently_specified"
    ACCESS_UNRESOLVABLE = "access_unresolvable"
    ON_REQUEST = "on_request"
    EXPLICITLY_RESTRICTED = "explicitly_restricted"


#: Publication numbering (matches the LaTeX table) and display labels.
#: Numbers are NOT monotone in openness — see OPENNESS_RANK for ordinal work.
CASE_NUMBER: dict[Case, int] = {
    Case.NOT_APPLICABLE: 1,
    Case.OPENLY_ACCESSIBLE: 2,
    Case.ACCESS_NOT_PROVIDED: 3,
    Case.INSUFFICIENTLY_SPECIFIED: 4,
    Case.ACCESS_UNRESOLVABLE: 5,
    Case.ON_REQUEST: 6,
    Case.EXPLICITLY_RESTRICTED: 7,
}

CASE_LABEL: dict[Case, str] = {
    Case.NOT_APPLICABLE: "Not applicable",
    Case.OPENLY_ACCESSIBLE: "Openly accessible",
    Case.ACCESS_NOT_PROVIDED: "Access not provided",
    Case.INSUFFICIENTLY_SPECIFIED: "Insufficiently specified",
    Case.ACCESS_UNRESOLVABLE: "Access unresolvable",
    Case.ON_REQUEST: "Available on request",
    Case.EXPLICITLY_RESTRICTED: "Explicitly restricted",
}

#: Derived ordinal for "how obtainable is this, in practice". Use this for any
#: correlation or trend analysis; never sort on CASE_NUMBER.
#: None = not on the scale (the axis does not apply).
OPENNESS_RANK: dict[Case, Optional[int]] = {
    Case.NOT_APPLICABLE: None,
    Case.OPENLY_ACCESSIBLE: 5,  # obtainable now, by anyone
    Case.ON_REQUEST: 4,  # obtainable, gated on a human
    Case.ACCESS_NOT_PROVIDED: 3,  # maybe obtainable, reader must hunt
    Case.INSUFFICIENTLY_SPECIFIED: 2,  # probably not obtainable
    Case.ACCESS_UNRESOLVABLE: 1,  # promised and broken
    Case.EXPLICITLY_RESTRICTED: 0,  # refused on the record
}


# ---------------------------------------------------------------------------
# Supporting vocabularies
# ---------------------------------------------------------------------------
class DataType(_Tok):
    """Whether an axis's data is empirical or computationally generated.

    Attributes:
        REAL: Empirical or measured data, including processed measurements.
        SYNTHETIC: Simulated or computationally generated data.
        MIXED: Both real and synthetic data.
        UNCLEAR: The paper does not make the kind of data clear.
    """

    REAL = "real"  # empirical/measured, incl. processed measurements
    SYNTHETIC = "synthetic"  # simulated / computationally generated
    MIXED = "mixed"
    UNCLEAR = "unclear"


class Declared(_Tok):
    """What the paper SAYS about access. Extracted from text/metadata only.

    Attributes:
        PUBLIC: The paper states that the data is publicly available.
        ON_REQUEST: The paper offers the data on request.
        RESTRICTED: The paper states that access is restricted.
        NONE_STATED: The paper says nothing about access.
        UNCLEAR: What the paper says about access cannot be determined.
    """

    PUBLIC = "public"
    ON_REQUEST = "on_request"
    RESTRICTED = "restricted"
    NONE_STATED = "none_stated"
    UNCLEAR = "unclear"


class Actual(_Tok):
    """What VERIFICATION found for a declared access path.

    Only `verify.py` may set anything but NOT_ATTEMPTED.

    Attributes:
        ACCESSIBLE: The path resolves and delivers the data.
        PAYWALLED: The path leads to a paywall.
        AUTH_REQUIRED: The path requires a login.
        NOT_FOUND: The path does not resolve.
        RESOLVES_NO_DATA: The landing page loads but the dataset is absent.
        ERROR: The check did not complete (a timeout, a DNS failure, a 503).
            This is not a failed check and does not change the case.
        NOT_ATTEMPTED: No verification has been run.
        NOT_APPLICABLE: There is no access path to verify.
    """

    ACCESSIBLE = "accessible"
    PAYWALLED = "paywalled"
    AUTH_REQUIRED = "auth_required"
    NOT_FOUND = "not_found"
    RESOLVES_NO_DATA = "resolves_no_data"  # landing page loads, dataset absent
    ERROR = "error"
    NOT_ATTEMPTED = "not_attempted"
    NOT_APPLICABLE = "not_applicable"


class Pinned(_Tok):
    """Granularity of the access path, separate from whether it works.

    Attributes:
        DOI_VERSIONED: A versioned dataset DOI.
        COMMIT_PINNED: A git commit, tag or release.
        UNPINNED_URL: A bare URL or a repository root.
        NONE: No access path is given.
    """

    DOI_VERSIONED = "doi_versioned"  # versioned dataset DOI
    COMMIT_PINNED = "commit_pinned"  # git commit / tag / release
    UNPINNED_URL = "unpinned_url"  # bare URL or repo root
    NONE = "none"


class Granularity(_Tok):
    """The processing level at which an axis's data is offered.

    Attributes:
        RAW: Raw data only.
        PROCESSED: Processed data only.
        BOTH: Both raw and processed data.
        UNCLEAR: The paper does not make the level clear.
        NOT_APPLICABLE: The axis does not apply.
    """

    RAW = "raw"
    PROCESSED = "processed"
    BOTH = "both"
    UNCLEAR = "unclear"
    NOT_APPLICABLE = "not_applicable"


class Restriction(_Tok):
    """Reason accompanying `Case.EXPLICITLY_RESTRICTED`. Never inferred.

    Attributes:
        PROPRIETARY: The data is proprietary.
        CONFIDENTIAL: The data is confidential.
        INTELLECTUAL_PROP: Intellectual-property grounds.
        COMMERCIAL: Commercial grounds.
        INSTITUTIONAL: An institutional policy.
        THIRD_PARTY: The data belongs to a third party.
        VENDOR_HARDWARE: A hardware vendor's terms, e.g. a QPU vendor NDA.
        EXPORT_CONTROL: Export-control rules.
        PRIVACY: Privacy of the data subjects.
        SIZE_TECHNICAL: The data is too large, or other technical grounds.
        OTHER_STATED: Another reason, stated in the paper.
        NONE_GIVEN: No reason is given.
    """

    PROPRIETARY = "proprietary"
    CONFIDENTIAL = "confidential"
    INTELLECTUAL_PROP = "intellectual_property"
    COMMERCIAL = "commercial"
    INSTITUTIONAL = "institutional"
    THIRD_PARTY = "third_party"
    VENDOR_HARDWARE = "vendor_hardware"  # e.g. QPU vendor NDA
    EXPORT_CONTROL = "export_control"
    PRIVACY = "privacy"
    SIZE_TECHNICAL = "size_technical"
    OTHER_STATED = "other_stated"
    NONE_GIVEN = "none_given"


class RepoType(_Tok):
    """The kind of repository an access path points to.

    Deliberately spans physics, ML and systems: the v1 list was physics-only,
    which would have coded most of the ML arm as `other`.

    Attributes:
        ZENODO: Zenodo.
        FIGSHARE: Figshare.
        DRYAD: Dryad.
        OSF: The Open Science Framework.
        GITHUB: GitHub.
        GITLAB: GitLab.
        HUGGINGFACE: Hugging Face.
        KAGGLE: Kaggle.
        OPENML: OpenML.
        PAPERSWITHCODE: Papers with Code.
        IEEE_DATAPORT: IEEE DataPort.
        MATERIALS_CLOUD: Materials Cloud.
        INSTITUTIONAL: An institutional repository.
        PUBLISHER_SUPP: The publisher's supplementary material.
        DOMAIN_ARCHIVE: A domain archive, e.g. UCI, PDB or the ImageNet host.
        CLOUD_BUCKET: A cloud storage link (S3, GCS, Azure).
        PERSONAL_SITE: A personal or group website.
        OTHER: A repository of a kind not listed here.
        UNKNOWN: The kind of repository cannot be determined.
    """

    ZENODO = "zenodo"
    FIGSHARE = "figshare"
    DRYAD = "dryad"
    OSF = "osf"
    GITHUB = "github"
    GITLAB = "gitlab"
    HUGGINGFACE = "huggingface"
    KAGGLE = "kaggle"
    OPENML = "openml"
    PAPERSWITHCODE = "paperswithcode"
    IEEE_DATAPORT = "ieee_dataport"
    MATERIALS_CLOUD = "materials_cloud"
    INSTITUTIONAL = "institutional_repository"
    PUBLISHER_SUPP = "publisher_supplementary"
    DOMAIN_ARCHIVE = "domain_archive"  # e.g. UCI, PDB, ImageNet host
    CLOUD_BUCKET = "cloud_bucket"  # S3/GCS/Azure link
    PERSONAL_SITE = "personal_site"
    OTHER = "other"
    UNKNOWN = "unknown"


class PaperType(_Tok):
    """The kind of work a paper reports.

    Needed for matching across arms: quantum has far more pure theory, and an
    unmatched comparison would measure that instead of sharing practice.

    Attributes:
        THEORY: Theoretical work.
        SIMULATION: A simulation study.
        EXPERIMENTAL: Experimental work.
        SYSTEMS_BUILD: A system that was built.
        BENCHMARK_STUDY: A benchmark study.
        DATASET_PAPER: A paper that presents a dataset.
        TOOL_PAPER: A paper that presents a tool.
        SURVEY_REVIEW: A survey or review.
        POSITION: A position paper.
        UNCLEAR: The kind of work cannot be determined.
    """

    THEORY = "theory"
    SIMULATION = "simulation"
    EXPERIMENTAL = "experimental"
    SYSTEMS_BUILD = "systems_build"
    BENCHMARK_STUDY = "benchmark_study"
    DATASET_PAPER = "dataset_paper"
    TOOL_PAPER = "tool_paper"
    SURVEY_REVIEW = "survey_review"
    POSITION = "position"
    UNCLEAR = "unclear"


class Arm(_Tok):
    """The study arm of a paper. Metadata, not part of the coding scheme.

    Attributes:
        QUANTUM_COMPUTING: Quantum computing.
        MACHINE_LEARNING: Machine learning.
        HPC_SYSTEMS: High-performance computing and systems.
        OTHER_CS: Other computer science.
        OUT_OF_SCOPE: Outside the study's scope.
        UNSPECIFIED: No arm. The general tool has no arms, so every paper of
            a new corpus carries this value; the others stay so case-study
            databases still load.
    """

    QUANTUM_COMPUTING = "quantum_computing"
    MACHINE_LEARNING = "machine_learning"
    HPC_SYSTEMS = "hpc_systems"
    OTHER_CS = "other_cs"
    OUT_OF_SCOPE = "out_of_scope"
    # The general tool has no arms: every paper of a new corpus is
    # `unspecified`. The other values stay so case-study databases still load.
    # Metadata, not a case token: SCHEME_VERSION is unaffected (rule 3).
    UNSPECIFIED = "unspecified"


class Confidence(_Tok):
    """How certain the classifier is of a judgement.

    Attributes:
        HIGH: High confidence; requires at least one piece of evidence.
        MEDIUM: Medium confidence.
        LOW: Low confidence; a low-confidence case routes the paper to human
            review.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Retrieval(_Tok):
    """How much of a paper the classifier was given.

    Attributes:
        METADATA_ONLY: Classified from Crossref/DataCite metadata alone.
        FULL_TEXT: The full text was retrieved.
        PARTIAL_TEXT: Only the abstract or front matter was retrieved.
        FAILED: Nothing could be retrieved.
    """

    METADATA_ONLY = "metadata_only"  # classified from Crossref/DataCite alone
    FULL_TEXT = "full_text"
    PARTIAL_TEXT = "partial_text"  # abstract / front matter only
    FAILED = "failed"


class TextSource(_Tok):
    """Where the text of a paper was retrieved from.

    Attributes:
        CROSSREF_TDM: A text-mining link deposited in Crossref.
        OPENALEX_OA: An open-access location listed by OpenAlex.
        UNPAYWALL: A location found through Unpaywall.
        ARXIV: arXiv.
        PMC: PubMed Central.
        CORE: CORE.
        PUBLISHER_TDM_API: A publisher's text-mining API.
        LOCAL_PDF: A PDF fetched by hand for a paper the OA cascade may not
            reach.
        NONE: No text was retrieved.
    """

    CROSSREF_TDM = "crossref_tdm"
    OPENALEX_OA = "openalex_oa"
    UNPAYWALL = "unpaywall"
    ARXIV = "arxiv"
    PMC = "pmc"
    CORE = "core"
    PUBLISHER_TDM_API = "publisher_tdm_api"
    #: A PDF fetched by hand for a paper the OA cascade may not reach.
    LOCAL_PDF = "local_pdf"
    NONE = "none"


# ---------------------------------------------------------------------------
# Judgement: every classified value carries its own warrant
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Judgement:
    """A single classified value plus why it was chosen.

    This is the unit the classifier must return for every substantive field.
    `rationale` is the model's own reasoning in one sentence; `evidence_ids`
    point into the record's evidence list. A judgement with no evidence and
    HIGH confidence is rejected by validate() — the model cannot assert
    certainty it has no warrant for.

    Attributes:
        value: The classified value: a token of the field's vocabulary.
        confidence: How certain the classifier is of the value.
        rationale: One sentence on why this value was chosen.
        evidence_ids: The ids of the evidence items that support the value.
    """

    value: Any
    confidence: Confidence
    rationale: str = ""
    evidence_ids: list[int] = field(default_factory=list)

    def validate(
        self, vocab: type[_Tok] | None, *, field_name: str, allow_unwarranted: bool = False
    ) -> "Judgement":
        """Check this judgement against its vocabulary and the warrant rules.

        Args:
            vocab: The vocabulary `value` must belong to, or None to leave the
                value unchecked.
            field_name: The name of the field, used in error messages.
            allow_unwarranted: Whether HIGH confidence is accepted without
                evidence. Set for machine-set fields, which carry no quote.

        Returns:
            This judgement, with `value` and `confidence` coerced to members of
            their vocabularies.

        Raises:
            SchemaError: If the value or confidence is not a valid token, if the
                rationale is empty or longer than 600 characters, or if
                confidence is HIGH with no evidence and `allow_unwarranted` is
                not set.
        """
        if vocab is not None:
            self.value = vocab.coerce(self.value)
        self.confidence = Confidence.coerce(self.confidence)
        if not self.rationale.strip():
            raise SchemaError(f"{field_name}: rationale is required")
        if len(self.rationale) > 600:
            raise SchemaError(f"{field_name}: rationale too long ({len(self.rationale)} chars)")
        if self.confidence is Confidence.HIGH and not self.evidence_ids and not allow_unwarranted:
            raise SchemaError(f"{field_name}: confidence=high requires at least one evidence_id")
        return self


@dataclass(slots=True)
class Evidence:
    """A verbatim quote that supports one or more judgements.

    Attributes:
        id: The number judgements refer to; unique within a record.
        quote: The quoted text: verbatim, at most 40 words, never paraphrased.
        section: The section of the paper the quote is from.
        locator: The page number or DOM anchor; empty if unknown.
        source: The retrieval source the quote came from.
    """

    id: int
    quote: str  # verbatim, <= 40 words, never paraphrased
    section: str = ""
    locator: str = ""  # page number or DOM anchor; "" if unknown
    source: str = ""  # which retrieval source this came from

    def validate(self) -> "Evidence":
        """Check that the quote is present and short enough.

        Returns:
            This evidence item.

        Raises:
            SchemaError: If the quote is empty or longer than 40 words.
        """
        q = self.quote.strip()
        if not q:
            raise SchemaError(f"evidence[{self.id}]: empty quote")
        if len(q.split()) > 40:
            raise SchemaError(f"evidence[{self.id}]: quote is {len(q.split())} words, limit is 40")
        return self


@dataclass(slots=True)
class AxisRecord:
    """One axis of one paper.

    Field names are axis-neutral so Axis A and Axis B are structurally
    identical, which is what makes them comparable.

    Attributes:
        case: The access state of this axis's data (`Case`).
        data_type: Empirical or computationally generated (`DataType`).
        declared: What the paper says about access (`Declared`).
        actual: What verification found (`Actual`). Set by `verify.py`, not
            by the model.
        granularity: The processing level of the data (`Granularity`).
        pinned: How precisely the access path is pinned.
        restriction: The stated reason for a restriction.
        repo_types: The kinds of repository the access paths point to.
        identifiers: The DOIs and URLs given, verbatim.
        names: The dataset or source names given.
        generation_code: Axis B: whether the generating code is given; None
            if not assessed.
        generation_params: Axis B: whether the generation parameters are
            given; None if not assessed.
        seed_specified: Axis B: whether the random seed is given; None if not
            assessed.
        environment_specified: Axis B: whether the software environment is
            given; None if not assessed.
    """

    case: Judgement
    data_type: Judgement
    declared: Judgement
    actual: Judgement  # set by verify.py, not by the model
    granularity: Judgement
    pinned: Pinned = Pinned.NONE
    restriction: Restriction = Restriction.NONE_GIVEN
    repo_types: list[RepoType] = field(default_factory=list)
    identifiers: list[str] = field(default_factory=list)  # DOIs / URLs, verbatim
    names: list[str] = field(default_factory=list)  # dataset or source names
    # Axis B reconstructability — what separates openly_accessible from
    # insufficiently_specified when the "data" is something the paper generated.
    generation_code: Optional[bool] = None
    generation_params: Optional[bool] = None
    seed_specified: Optional[bool] = None
    environment_specified: Optional[bool] = None

    def validate(self, axis: str) -> "AxisRecord":
        """Check every field of this axis and the rules that tie them together.

        Args:
            axis: The name of the axis (`"axis_a"` or `"axis_b"`), used in error
                messages.

        Returns:
            This record, with every value coerced to a member of its vocabulary.

        Raises:
            SchemaError: If a value is not a valid token, or if a cross-field
                rule is broken: `not_applicable` with identifiers or
                repositories, `openly_accessible` with no identifier,
                `explicitly_restricted` with no reason, `access_unresolvable`
                with no failed verification, or `on_request` when the paper does
                not declare it.
        """
        self.case.validate(Case, field_name=f"{axis}.case")
        self.data_type.validate(DataType, field_name=f"{axis}.data_type")
        self.declared.validate(Declared, field_name=f"{axis}.declared")
        self.actual.validate(
            Actual, field_name=f"{axis}.actual", allow_unwarranted=True
        )  # machine-set, no quote
        self.granularity.validate(Granularity, field_name=f"{axis}.granularity")
        self.pinned = Pinned.coerce(self.pinned)
        self.restriction = Restriction.coerce(self.restriction)
        self.repo_types = [RepoType.coerce(r) for r in self.repo_types]

        case = self.case.value
        # --- cross-field consistency: the rules the v1 data violated silently ---
        if case is Case.NOT_APPLICABLE:
            if self.identifiers or self.repo_types:
                raise SchemaError(
                    f"{axis}: case=not_applicable but identifiers/repositories were "
                    f"extracted — the axis cannot be absent and present at once"
                )
        if case is Case.OPENLY_ACCESSIBLE and not self.identifiers:
            raise SchemaError(
                f"{axis}: case=openly_accessible requires at least one identifier. "
                f"A stated claim with no path is access_not_provided or "
                f"insufficiently_specified."
            )
        if case is Case.EXPLICITLY_RESTRICTED and self.restriction is Restriction.NONE_GIVEN:
            raise SchemaError(f"{axis}: case=explicitly_restricted requires a restriction reason")
        if case is Case.ACCESS_UNRESOLVABLE:
            if self.actual.value in (Actual.NOT_ATTEMPTED, Actual.ACCESSIBLE):
                raise SchemaError(
                    f"{axis}: case=access_unresolvable but actual={self.actual.value}. "
                    f"This case is derived from a failed verification, never asserted."
                )
        if case is Case.ON_REQUEST and self.declared.value is not Declared.ON_REQUEST:
            raise SchemaError(f"{axis}: case=on_request requires declared=on_request")
        return self


@dataclass(slots=True)
class PaperRecord:
    """The complete coded record of one paper.

    Attributes:
        openalex_id: The paper's OpenAlex id.
        doi: The paper's DOI; empty if it has none.
        title: The paper's title.
        year: The publication year, or None if unknown.
        venue: The journal or conference.
        arm: The study arm.
        paper_type: The kind of work the paper reports (`PaperType`).
        axis_a: The record of Axis A, external data.
        axis_b: The record of Axis B, self-generated data.
        evidence: The quotes the judgements refer to.
        das_present: Whether the paper carries a data availability statement.
        das_source: Where that statement was found: `crossref_assertion`,
            `full_text` or `none`.
        needs_human_review: Whether the paper is routed to the review queue.
        review_reason: Why the paper needs review; required when it does.
        scheme_version: The version of the coding scheme the record was coded
            under.
        prompt_version: The version of the classifier prompt.
        model_id: The model, or the manual coder, that produced the record.
        retrieval: How much of the paper the classifier was given.
        text_source: Where the text was retrieved from.
        text_url: The URL of the copy that was read.
        processed_at: When the record was produced, as an ISO timestamp.
        input_tokens: The input tokens the classification used.
        output_tokens: The output tokens the classification used.
    """

    openalex_id: str
    doi: str
    title: str
    year: Optional[int]
    venue: str
    arm: Arm
    paper_type: Judgement
    axis_a: AxisRecord  # external data
    axis_b: AxisRecord  # self-generated data
    evidence: list[Evidence]
    das_present: bool
    das_source: str  # crossref_assertion | full_text | none
    needs_human_review: bool
    review_reason: str
    # provenance — machine-set, never typed
    scheme_version: str = SCHEME_VERSION
    prompt_version: str = ""
    model_id: str = ""
    retrieval: Retrieval = Retrieval.FAILED
    text_source: TextSource = TextSource.NONE
    text_url: str = ""
    processed_at: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    def validate(self) -> "PaperRecord":
        """Check the whole record: both axes, the evidence and the review flag.

        Returns:
            This record, with every value coerced to a member of its vocabulary.

        Raises:
            SchemaError: If `openalex_id` is empty, a value is not a valid token,
                an axis breaks a cross-field rule, an evidence id is duplicated
                or referenced but missing, a review is flagged without a reason,
                or a low-confidence case is not flagged for review.
        """
        if not self.openalex_id:
            raise SchemaError("openalex_id is required")
        self.arm = Arm.coerce(self.arm)
        self.retrieval = Retrieval.coerce(self.retrieval)
        self.text_source = TextSource.coerce(self.text_source)
        self.paper_type.validate(PaperType, field_name="paper_type")

        ids = set()
        for ev in self.evidence:
            ev.validate()
            if ev.id in ids:
                raise SchemaError(f"duplicate evidence id {ev.id}")
            ids.add(ev.id)

        self.axis_a.validate("axis_a")
        self.axis_b.validate("axis_b")

        # every evidence_id referenced must exist
        for axis_name, axis in (("axis_a", self.axis_a), ("axis_b", self.axis_b)):
            for fname in ("case", "data_type", "declared", "granularity"):
                j: Judgement = getattr(axis, fname)
                missing = [i for i in j.evidence_ids if i not in ids]
                if missing:
                    raise SchemaError(f"{axis_name}.{fname}: evidence_ids {missing} do not exist")

        if self.needs_human_review and not self.review_reason.strip():
            raise SchemaError("needs_human_review=True requires review_reason")

        # A low-confidence case on either axis must be flagged, not quietly kept.
        for axis_name, axis in (("axis_a", self.axis_a), ("axis_b", self.axis_b)):
            if axis.case.confidence is Confidence.LOW and not self.needs_human_review:
                raise SchemaError(
                    f"{axis_name}.case has low confidence but needs_human_review is "
                    f"False — low confidence must route to the review queue"
                )
        return self

    # -- derived, for analysis --------------------------------------------
    @property
    def is_purely_theoretical(self) -> bool:
        """Whether both axes independently return `not_applicable`."""
        return (
            self.axis_a.case.value is Case.NOT_APPLICABLE
            and self.axis_b.case.value is Case.NOT_APPLICABLE
        )

    def flat(self) -> dict[str, Any]:
        """Flatten the record into one row for CSV/SQL export.

        Returns:
            A dict of plain values keyed by the names in `FLAT_COLUMNS`. Tokens
            only, never labels.
        """
        out: dict[str, Any] = dict(
            openalex_id=self.openalex_id,
            doi=self.doi,
            title=self.title,
            year=self.year,
            venue=self.venue,
            arm=self.arm.value,
            paper_type=self.paper_type.value.value,
            paper_type_conf=self.paper_type.confidence.value,
            das_present=self.das_present,
            das_source=self.das_source,
            purely_theoretical=self.is_purely_theoretical,
            needs_human_review=self.needs_human_review,
            review_reason=self.review_reason,
            scheme_version=self.scheme_version,
            prompt_version=self.prompt_version,
            model_id=self.model_id,
            retrieval=self.retrieval.value,
            text_source=self.text_source.value,
            text_url=self.text_url,
            processed_at=self.processed_at,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )
        for prefix, axis in (("a", self.axis_a), ("b", self.axis_b)):
            case: Case = axis.case.value
            out[f"{prefix}_case"] = case.value
            out[f"{prefix}_case_num"] = CASE_NUMBER[case]
            out[f"{prefix}_openness_rank"] = OPENNESS_RANK[case]
            out[f"{prefix}_case_conf"] = axis.case.confidence.value
            out[f"{prefix}_case_rationale"] = axis.case.rationale
            out[f"{prefix}_data_type"] = axis.data_type.value.value
            out[f"{prefix}_declared"] = axis.declared.value.value
            out[f"{prefix}_actual"] = axis.actual.value.value
            out[f"{prefix}_granularity"] = axis.granularity.value.value
            out[f"{prefix}_pinned"] = axis.pinned.value
            out[f"{prefix}_restriction"] = axis.restriction.value
            out[f"{prefix}_repo_types"] = ";".join(r.value for r in axis.repo_types)
            out[f"{prefix}_identifiers"] = ";".join(axis.identifiers)
            out[f"{prefix}_names"] = ";".join(axis.names)
            out[f"{prefix}_gen_code"] = axis.generation_code
            out[f"{prefix}_gen_params"] = axis.generation_params
            out[f"{prefix}_seed"] = axis.seed_specified
            out[f"{prefix}_environment"] = axis.environment_specified
        return out


FLAT_COLUMNS: list[str] = list(
    PaperRecord(
        openalex_id="x",
        doi="",
        title="",
        year=None,
        venue="",
        arm=Arm.OTHER_CS,
        paper_type=Judgement(PaperType.UNCLEAR, Confidence.LOW, "init"),
        axis_a=AxisRecord(
            case=Judgement(Case.NOT_APPLICABLE, Confidence.LOW, "init"),
            data_type=Judgement(DataType.UNCLEAR, Confidence.LOW, "init"),
            declared=Judgement(Declared.NONE_STATED, Confidence.LOW, "init"),
            actual=Judgement(Actual.NOT_ATTEMPTED, Confidence.LOW, "init"),
            granularity=Judgement(Granularity.NOT_APPLICABLE, Confidence.LOW, "init"),
        ),
        axis_b=AxisRecord(
            case=Judgement(Case.NOT_APPLICABLE, Confidence.LOW, "init"),
            data_type=Judgement(DataType.UNCLEAR, Confidence.LOW, "init"),
            declared=Judgement(Declared.NONE_STATED, Confidence.LOW, "init"),
            actual=Judgement(Actual.NOT_ATTEMPTED, Confidence.LOW, "init"),
            granularity=Judgement(Granularity.NOT_APPLICABLE, Confidence.LOW, "init"),
        ),
        evidence=[],
        das_present=False,
        das_source="none",
        needs_human_review=True,
        review_reason="init",
    )
    .flat()
    .keys()
)


# ---------------------------------------------------------------------------
# JSON Schema for the model's constrained output (Anthropic tool-use input_schema)
# ---------------------------------------------------------------------------
def _judgement_schema(vocab: type[_Tok], desc: str) -> dict:
    """Build the JSON Schema of one judgement for the classification tool.

    Args:
        vocab: The vocabulary the judgement's value must come from.
        desc: The description of the field shown to the model.

    Returns:
        The JSON Schema object for a `{value, confidence, rationale,
        evidence_ids}` judgement.
    """
    return {
        "type": "object",
        "description": desc,
        "properties": {
            "value": {"type": "string", "enum": sorted(vocab.tokens())},
            "confidence": {"type": "string", "enum": sorted(Confidence.tokens())},
            "rationale": {
                "type": "string",
                "maxLength": 600,
                "description": "One sentence: why this value and not the nearest "
                "alternative. Reference the evidence, do not restate "
                "the definition.",
            },
            "evidence_ids": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["value", "confidence", "rationale", "evidence_ids"],
    }


def _axis_schema(axis_label: str) -> dict:
    """Build the JSON Schema of one axis for the classification tool.

    Args:
        axis_label: The description of the axis shown to the model.

    Returns:
        The JSON Schema object for the axis.
    """
    return {
        "type": "object",
        "description": axis_label,
        "properties": {
            "case": _judgement_schema(Case, "The access state of this axis's data."),
            "data_type": _judgement_schema(DataType, "Empirical vs computationally generated."),
            "declared": _judgement_schema(Declared, "What the paper SAYS about availability."),
            "granularity": _judgement_schema(Granularity, "Raw vs processed."),
            "pinned": {"type": "string", "enum": sorted(Pinned.tokens())},
            "restriction": {"type": "string", "enum": sorted(Restriction.tokens())},
            "repo_types": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(RepoType.tokens())},
            },
            "identifiers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "DOIs/URLs copied verbatim from the text. "
                "Never construct or guess one.",
            },
            "names": {"type": "array", "items": {"type": "string"}},
            "generation_code": {"type": ["boolean", "null"]},
            "generation_params": {"type": ["boolean", "null"]},
            "seed_specified": {"type": ["boolean", "null"]},
            "environment_specified": {"type": ["boolean", "null"]},
        },
        "required": [
            "case",
            "data_type",
            "declared",
            "granularity",
            "pinned",
            "restriction",
            "repo_types",
            "identifiers",
            "names",
        ],
    }


CLASSIFICATION_TOOL_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "evidence": {
            "type": "array",
            "description": "Verbatim supporting quotes. Assign each a unique integer "
            "id and reference those ids from the judgements.",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "quote": {
                        "type": "string",
                        "description": "Verbatim, 40 words maximum. Copy exactly; "
                        "never paraphrase into this field.",
                    },
                    "section": {"type": "string"},
                    "locator": {
                        "type": "string",
                        "description": "Page or anchor. Leave empty if unknown — never invent one.",
                    },
                },
                "required": ["id", "quote"],
            },
        },
        "paper_type": _judgement_schema(PaperType, "What kind of paper this is."),
        "axis_a": _axis_schema(
            "Axis A — EXTERNAL data: data that existed independently of this paper "
            "(pre-collected datasets, benchmarks, real-world records, prior artifacts) "
            "and that this paper analyses."
        ),
        "axis_b": _axis_schema(
            "Axis B — SELF-GENERATED data: data this paper produced itself by running "
            "an experiment, simulation, or generative procedure."
        ),
        "das_present": {"type": "boolean"},
        "needs_human_review": {"type": "boolean"},
        "review_reason": {"type": "string"},
    },
    "required": [
        "evidence",
        "paper_type",
        "axis_a",
        "axis_b",
        "das_present",
        "needs_human_review",
        "review_reason",
    ],
}


def describe_scheme() -> str:
    """Render the case scheme as text for the classifier prompt.

    Injected into the prompt so the prompt and the validator can never
    disagree about the vocabulary.

    Returns:
        The case vocabulary, one numbered line per case token.
    """
    lines = ["CASE VOCABULARY (choose exactly one token per axis):"]
    for case in Case:
        lines.append(f"  {CASE_NUMBER[case]}. {case.value}  — {CASE_LABEL[case]}")
    return "\n".join(lines)


__all__ = [
    "SCHEME_VERSION",
    "SchemaError",
    "Case",
    "CASE_NUMBER",
    "CASE_LABEL",
    "OPENNESS_RANK",
    "DataType",
    "Declared",
    "Actual",
    "Pinned",
    "Granularity",
    "Restriction",
    "RepoType",
    "PaperType",
    "Arm",
    "Confidence",
    "Retrieval",
    "TextSource",
    "Judgement",
    "Evidence",
    "AxisRecord",
    "PaperRecord",
    "FLAT_COLUMNS",
    "CLASSIFICATION_TOOL_SCHEMA",
    "describe_scheme",
]
