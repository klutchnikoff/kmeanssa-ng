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

    Each point is pushed away from its ``k`` nearest neighbours along geodesics,
    with a step that is a shrinking fraction of the local spacing, until the
    configuration freezes into a near-regular net. Being driven only by the
    manifold's ``log``/``exp``, it works on any (compact) Riemannian manifold.

    Args:
        k: Number of nearest neighbours each point repels.
        n_iter: Number of relaxation steps.
        riesz_s: Exponent of the repulsion kernel (force ~ 1 / d^(s+1)).
        cooling: Larger values cool more slowly (step ~ 1 / (1 + t / cooling)).
        random_state: Seed or Generator for the uniform initialisation.
    """

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

    def build(self, manifold: RiemannianManifold, n: int) -> np.ndarray:
        # TODO: on non-compact domains (e.g. hyperbolic space) unbounded repulsion
        # pushes points to the boundary and leaves interior voids. Support a
        # bounded region with a soft confining potential instead of relying on the
        # manifold being compact.
        rng = np.random.default_rng(self.random_state)
        X = manifold.random_uniform(n, rng)
        dim = X.shape[1]
        s = self.riesz_s
        for t in range(self.n_iter):
            frac = 0.25 / (1 + t / self.cooling)
            _, idx = (
                NearestNeighbors(n_neighbors=self.k + 1)
                .fit(manifold.embed(X))
                .kneighbors(manifold.embed(X))
            )
            idx = idx[:, 1:]  # drop self
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
