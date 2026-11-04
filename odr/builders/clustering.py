"""Numerical document clustering with the required NumPy and KMeans backends."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, cast

import numpy as np

from odr.builders import numeric_types as numeric
from odr.models import types as models
from odr.schema import arguments as config

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


def normalized_matrix(
    members: Sequence[int], vectors: models.Vectors
) -> NDArray[np.float64]:
    """Validate finite, nonzero document vectors.

    :param members: Document row indices assigned to a group.
    :param vectors: Ordered document embedding vectors.
    :returns: Finite document vectors normalized to unit length.
    :raises ValueError: If the input, stored data, or operation result is invalid.
    """
    matrix = np.asarray([vectors[index] for index in members], dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Invalid embedding matrix")
    norms = np.linalg.norm(matrix, axis=1)
    if (norms == 0).any():
        raise ValueError("Cannot normalize zero vectors")
    return matrix / norms.reshape(-1, 1)


def cluster_labels(
    matrix: NDArray[np.float64], settings: config.BuilderArguments
) -> NDArray[np.int64]:
    """Run ordinary KMeans with explicit reproducibility parameters.

    :param matrix: Normalized document feature matrix.
    :param settings: Provider options forwarded to the selected backend.
    :returns: Cluster label for every feature-matrix row.
    :raises ValueError: If the input, stored data, or operation result is invalid.
    """
    factory = cast(numeric.ClusterFactory, import_module("sklearn.cluster").KMeans)

    count = min(settings.n_clusters, len(np.unique(matrix, axis=0)))
    if count < settings.n_clusters and settings.small_corpus_policy == "strict":
        raise ValueError("Insufficient unique vectors for requested clusters")
    estimator = factory(
        n_clusters=count,
        random_state=settings.random_seed,
        n_init=settings.n_init,
        max_iter=settings.max_iter,
        algorithm="lloyd",
    )
    return estimator.fit_predict(matrix)


def typical_members(
    matrix: NDArray[np.float64], positions: NDArray[np.int64], members: Sequence[int]
) -> list[int]:
    """Sort title inputs by centroid distance with deterministic ties.

    :param matrix: Normalized document feature matrix.
    :param positions: Feature-matrix row indices belonging to a cluster.
    :param members: Document row indices assigned to a group.
    :returns: Document indices ordered by centroid distance.
    """
    center = matrix[positions].mean(axis=0)
    distances = []
    for index in positions:
        distance = np.linalg.norm(matrix[index] - center)
        distances.append((distance, index))
    return [members[position] for (distance, position) in sorted(distances)]


def partition(
    members: Sequence[int], vectors: models.Vectors, settings: config.BuilderArguments
) -> list[list[int]]:
    """Maintain stable row to document mapping.

    :param members: Document row indices assigned to a group.
    :param vectors: Ordered document embedding vectors.
    :param settings: Provider options forwarded to the selected backend.
    :returns: Cluster memberships preserving row-to-document mapping.
    """
    matrix = normalized_matrix(members, vectors)
    labels = cluster_labels(matrix, settings)
    groups = []
    for label in sorted(set(labels)):
        positions = np.flatnonzero(labels == label)
        groups.append(typical_members(matrix, positions, members))
    return groups
