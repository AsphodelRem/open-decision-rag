"""Typed boundaries for the clustering backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class ClusterEstimator(Protocol):
    """Clustering estimator accepting a document feature matrix."""

    def fit_predict(self, matrix: NDArray[np.float64]) -> NDArray[np.int64]:
        """Assign document rows to clusters.

        :param matrix: Normalized document vectors.
        :returns: Cluster label for each row.
        """
        ...


class ClusterFactory(Protocol):
    """Constructor interface of the KMeans implementation."""

    def __call__(self, **settings: Any) -> ClusterEstimator:
        """Create an estimator with explicit build settings.

        :param settings: Provider estimator configuration.
        :returns: Configured KMeans estimator.
        """
        ...
