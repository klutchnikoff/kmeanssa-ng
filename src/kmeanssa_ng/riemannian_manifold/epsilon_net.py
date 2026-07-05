"""Strategies that place a quasi-uniform epsilon-net on a Riemannian manifold.

An epsilon-net is a finite set of points whose covering radius is small and
whose spacing is roughly uniform; it is what turns a continuous geodesic space
into a clusterable quantum graph. Every strategy returns an ``(n, d_ambient)``
array of points lying on the manifold.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    from .space import RiemannianManifold


class EpsilonNetStrategy(ABC):
    """Abstract base class for epsilon-net placement strategies."""

    def __init__(self, random_state: int | np.random.Generator | None = None):
        self.random_state = random_state

    @abstractmethod
    def build(self, manifold: RiemannianManifold, n: int) -> np.ndarray:
        """Return ``n`` points on ``manifold`` forming a quasi-uniform net."""
        raise NotImplementedError


class UniformNet(EpsilonNetStrategy):
    """Plain uniform sampling. A baseline: fast, but irregular for finite n."""

    def build(self, manifold: RiemannianManifold, n: int) -> np.ndarray:
        return manifold.random_uniform(n, self.random_state)


class RepulsionNet(EpsilonNetStrategy):
    """Uniform init relaxed by truncated Riesz repulsion (Riemannian gradient flow).

    Fully *intrinsic*: each point is pushed away from its ``k`` nearest neighbours
    -- found by geodesic distance -- along geodesics, with a step that is a
    shrinking fraction of the local spacing, until the configuration freezes into
    a near-regular net. Being driven only by the manifold's intrinsic ``log``,
    ``exp`` and ``norm``, it is correct on any compact Riemannian manifold,
    including quotient spaces (e.g. the Bolza surface) where an ambient embedding
    would misrepresent neighbours across identified boundaries.

    Neighbour search is O(n^2) in the point count. When the manifold is
    isometrically embedded and ambient (chordal) proximity matches geodesic
    proximity -- as for S^2 in R^3 -- :class:`RepulsionNetExtrinsicSpeedup`
    replaces it by an O(n log n) KD-tree search on the embedding.

    Args:
        k: Number of nearest neighbours each point repels.
        n_iter: Number of relaxation steps.
        riesz_s: Exponent of the repulsion kernel (force ~ 1 / d^(s+1)).
        cooling: Larger values cool more slowly (step ~ 1 / (1 + t / cooling)).
        random_state: Seed or Generator for the uniform initialisation.
    """

    # Cap on the transient (chunk * n, dim) buffer of the neighbour search.
    _NEIGHBOUR_BUFFER = 2_000_000

    def __init__(
        self,
        k: int = 12,
        n_iter: int = 600,
        riesz_s: float = 1.0,
        cooling: float = 300.0,
        random_state: int | np.random.Generator | None = None,
    ):
        super().__init__(random_state)
        self.k = k
        self.n_iter = n_iter
        self.riesz_s = riesz_s
        self.cooling = cooling

    def _neighbours(self, manifold: RiemannianManifold, X: np.ndarray) -> np.ndarray:
        """Indices of each point's ``k`` nearest neighbours by *geodesic* distance.

        Distances come from the manifold's own ``log``/``norm`` (so the metric --
        quotient-aware or not -- is whatever the space defines), evaluated in row
        chunks to bound memory. O(n^2); self is excluded. Returns ``(n, k)``.
        """
        n = X.shape[0]
        idx = np.empty((n, self.k), dtype=int)
        chunk = max(1, self._NEIGHBOUR_BUFFER // n)
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            m = hi - lo
            base = np.repeat(X[lo:hi], n, axis=0)
            tiled = np.tile(X, (m,) + (1,) * (X.ndim - 1))
            d = manifold.norm(base, manifold.log(base, tiled)).reshape(m, n)
            d[np.arange(m), np.arange(lo, hi)] = np.inf  # exclude self
            idx[lo:hi] = np.argpartition(d, self.k - 1, axis=1)[:, : self.k]
        return idx

    def build(self, manifold: RiemannianManifold, n: int) -> np.ndarray:
        # NOTE: on non-compact domains (e.g. raw hyperbolic space) unbounded
        # repulsion pushes points to the boundary and leaves interior voids.
        # Supporting those needs a bounded region with a soft confining potential;
        # this is orthogonal to the intrinsic/extrinsic neighbour choice. Compact
        # spaces (spheres, the Bolza surface) need no confinement.
        rng = np.random.default_rng(self.random_state)
        X = manifold.random_uniform(n, rng)
        dim = X.shape[1]
        s = self.riesz_s
        for t in range(self.n_iter):
            frac = 0.25 / (1 + t / self.cooling)
            idx = self._neighbours(manifold, X)
            base = np.repeat(X, self.k, axis=0)
            neighbours = X[idx].reshape(-1, dim)
            tangent = manifold.log(base, neighbours)  # base -> neighbour
            dist = np.maximum(manifold.norm(base, tangent), 1e-6)
            unit = tangent / dist[:, None]
            weight = 1.0 / dist ** (s + 1)  # Riesz repulsion
            force = -(weight[:, None] * unit).reshape(n, self.k, dim).sum(axis=1)
            local = np.median(dist.reshape(n, self.k), axis=1, keepdims=True)
            f_norm = np.maximum(manifold.norm(X, force), 1e-12)[:, None]
            # Move at most a shrinking fraction of the local spacing (stable).
            step = np.minimum(
                frac * local, frac * local * f_norm / (f_norm.mean() + 1e-12)
            )
            X = manifold.exp(X, step * (force / f_norm))
        return X


class RepulsionNetExtrinsicSpeedup(RepulsionNet):
    """:class:`RepulsionNet` with an *extrinsic* neighbour search (optimisation).

    Overrides only the neighbour step, replacing the O(n^2) geodesic search by an
    O(n log n) Euclidean k-NN (KD-tree) on ``manifold.embed(X)``. The repulsion
    dynamics are otherwise identical.

    Valid ONLY when ``embed`` is an isometric embedding whose ambient (chordal)
    neighbour ordering matches the geodesic one -- true for a compact manifold
    embedded in ambient space (e.g. S^2 in R^3), FALSE for a quotient space: there,
    points glued across an identified boundary are geodesic neighbours yet lie far
    apart in the embedding, so the KD-tree returns the wrong neighbours. Use the
    intrinsic :class:`RepulsionNet` in that case.
    """

    def _neighbours(self, manifold: RiemannianManifold, X: np.ndarray) -> np.ndarray:
        embedded = manifold.embed(X)
        _, idx = (
            NearestNeighbors(n_neighbors=self.k + 1).fit(embedded).kneighbors(embedded)
        )
        return idx[:, 1:]  # drop self


class FibonacciNet(EpsilonNetStrategy):
    """Deterministic Fibonacci lattice on the 2-sphere.

    A near-optimal, low-discrepancy net available only for S^2; ``random_state``
    is ignored since the construction is deterministic.
    """

    def build(self, manifold: RiemannianManifold, n: int) -> np.ndarray:
        if not (manifold.is_sphere and manifold.dim == 2):
            raise ValueError("FibonacciNet is only defined for the 2-sphere S^2.")
        i = np.arange(n)
        golden = (1 + 5**0.5) / 2
        z = 1 - 2 * (i + 0.5) / n
        r = np.sqrt(1 - z * z)
        theta = 2 * np.pi * i / golden
        return np.c_[r * np.cos(theta), r * np.sin(theta), z]
