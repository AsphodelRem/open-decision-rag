"""Public arguments contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from odr.schema.records import FrozenModel

EXPANSION_LIMIT = 100000
DEPTH_LIMIT = 64
CONTEXT_CHAR_LIMIT = 32000
MAX_OPTIONS = 52
SUMMARY_TITLE_LIMIT = 20
DEFAULT_ITERATIONS = 300
CANVAS_MINIMUM = 16
CANVAS_MAXIMUM = 256
DEFAULT_SEED = 42


class BuilderArguments(FrozenModel):
    """Semantic indexing parameters.

    :ivar hierarchy_type: Configured hierarchy construction algorithm.
    :ivar n_clusters: Requested number of document clusters.
    :ivar random_seed: Seed used by deterministic KMeans initialization.
    :ivar n_init: Number of KMeans initializations.
    :ivar max_iter: Maximum KMeans iterations.
    :ivar make_group_summary: Whether required group summaries are generated; disabling is unsupported.
    :ivar small_corpus_policy: Whether to shrink cluster count or reject insufficient unique vectors.
    :ivar max_documents_per_group: Target document limit for recursive groups.
    :ivar branching_factor: Maximum requested branching width inside recursive groups.
    :ivar max_depth: Maximum permitted hierarchy or navigation depth.
    """

    hierarchy_type: Literal["flat", "metadata", "kmeans", "recursive"] = "flat"
    n_clusters: int = Field(default=8, ge=1)
    random_seed: int = 42
    n_init: int = Field(default=10, ge=1)
    max_iter: int = Field(default=DEFAULT_ITERATIONS, ge=1)
    make_group_summary: Literal[True] = True
    small_corpus_policy: Literal["shrink", "strict"] = "shrink"
    max_documents_per_group: int = Field(default=SUMMARY_TITLE_LIMIT, ge=1)
    branching_factor: int = Field(default=4, ge=2)
    max_depth: int = Field(default=8, ge=1)


class StorageArguments(FrozenModel):
    """Persistent index location.

    :ivar storage_folder: Index bundle directory, or no persistent publication.
    :ivar tree_backend: Requested tree storage backend.
    """

    storage_folder: str | None = None
    tree_backend: Literal["memory", "sqlite"] = "sqlite"


class VLLMDecisionArguments(FrozenModel):
    """Versioned configuration for direct DiffusionGemma decision reads.

    :ivar tokenizer_name: Matching Hugging Face tokenizer or local directory.
    :ivar tokenizer_revision: Pinned tokenizer commit or local artifact identity.
    :ivar deployment_revision: Identity of server version, weights and configuration.
    :ivar canvas_length: Maximum seeded canvas width configured on vLLM.
    :ivar seed: Deterministic base seed for answer-slot noise.
    """

    tokenizer_name: str = Field(min_length=1)
    tokenizer_revision: str = Field(min_length=1)
    deployment_revision: str = Field(min_length=1)
    canvas_length: int = Field(
        default=DEPTH_LIMIT, ge=CANVAS_MINIMUM, le=CANVAS_MAXIMUM
    )
    seed: int = Field(default=DEFAULT_SEED, ge=0)


class SearchArguments(FrozenModel):
    """Generic navigation and shortlist budgets.

    :ivar profile: Configured search profile; only generic is currently supported.
    :ivar return_top_k: Maximum number of returned passages.
    :ivar branch_top_k: Maximum branches retained at each internal stage.
    :ivar choice_top_m: Number of top binary candidates compared by choice.
    :ivar candidate_pool_size: Maximum passage pool retained before choice reranking.
    :ivar choice_mode: Shortlist reorder or winner-only promotion policy.
    :ivar scoring: Binary relevance or document-parent weighting policy.
    :ivar max_context_chars: Maximum provider-rendered context characters; excludes server chat templates and is not a token limit.
    :ivar max_context_tokens: Optional exact prompt-plus-output token budget; requires tokenizer capability.
    :ivar max_depth: Maximum permitted hierarchy or navigation depth.
    :ivar max_expanded_nodes: Maximum internal nodes expanded by a query.
    """

    profile: Literal[
        "generic",
        "multidoc2dial",
        "conditionalqa",
        "mtrag_ibm_cloud",
        "natural_questions",
    ] = "generic"
    return_top_k: int = Field(default=5, ge=1)
    branch_top_k: int = Field(default=10, ge=1)
    choice_top_m: int = Field(default=0, ge=0, le=MAX_OPTIONS)
    candidate_pool_size: int | None = Field(default=None, ge=1)
    choice_mode: Literal["reorder_shortlist", "promote_top1"] = "reorder_shortlist"
    scoring: Literal["binary", "parent_product"] = "binary"
    max_context_chars: int = Field(default=CONTEXT_CHAR_LIMIT, ge=1)
    max_context_tokens: int | None = Field(default=None, ge=1)
    max_depth: int = Field(default=DEPTH_LIMIT, ge=1)
    max_expanded_nodes: int = Field(default=EXPANSION_LIMIT, ge=1)

    @model_validator(mode="after")
    def validate_pool(self) -> SearchArguments:
        """Shortlist must fit in the retained pool.

        :returns: The validated search arguments.
        :raises ValueError: If the input, stored data, or operation result is invalid.
        """
        required = max(self.return_top_k, self.choice_top_m)
        if self.candidate_pool_size is not None and self.candidate_pool_size < required:
            raise ValueError(
                "candidate_pool_size must cover return_top_k and choice_top_m"
            )
        return self
