"""The Bolza surface: a compact genus-2 hyperbolic space, closed form.

The Bolza surface is the quotient of the hyperbolic plane by the Fuchsian group
that glues opposite sides of a regular hyperbolic octagon (interior angle pi/4).
It is the most symmetric closed genus-2 surface, of constant curvature -1, and a
natural negatively curved counterpart to the sphere.

This module has two layers:

* closed-form Poincaré-disk geometry and the Fuchsian group (module-level
  functions on complex arrays) -- distance, exp/log, the eight side-pairings,
  a geometric group ball, the quotient distance and uniform sampling of the
  fundamental octagon;
* :class:`BolzaSurface`, a first-class :class:`RiemannianManifold` whose geodesic
  operations are *quotient-aware* (every geodesic step is folded back into the
  fundamental domain, every neighbour is taken to its nearest copy). It carries
  no geomstats backend -- only a lightweight :class:`_PoincareDisk` chart -- so it
  is a concrete demonstration that a space can be defined without one, and the
  space on which the intrinsic :class:`RepulsionNet` is exercised.

Points are complex numbers in the open unit disk at the geometry layer, and
``(..., 2)`` real ``(Re, Im)`` arrays at the :class:`BolzaSurface` layer (matching
the ambient-array convention of the other manifolds). Tangent vectors are ambient
vectors with the conformal norm.
"""

from __future__ import annotations

import numpy as np

from .space import RiemannianManifold

SQRT2 = np.sqrt(2.0)
N_SIDES = 8

# Regular octagon, interior angle pi/4: the eight vertices sit at Euclidean radius
# R0 = 2^{-1/4} (from cosh(R_hyp) = cot^2(pi/8) = 3 + 2 sqrt2, then r = tanh(R/2)).
R0 = 2.0**-0.25
VERTEX_ANGLES = np.arange(N_SIDES) * (np.pi / 4.0)
VERTICES = R0 * np.exp(1j * VERTEX_ANGLES)

# Side s_k joins v_k and v_{k+1}; its midpoint lies in direction (k + 1/2) pi/4.
SIDE_DIRS = (np.arange(N_SIDES) + 0.5) * (np.pi / 4.0)

# Side-pairing translation length: cosh(ell/2) = 1 + sqrt2 (= cot(pi/8)).
COSH_HALF = 1.0 + SQRT2
SINH_HALF = np.sqrt(COSH_HALF * COSH_HALF - 1.0)  # = sqrt(2 + 2 sqrt2)


# --------------------------------------------------------------------------- #
# Poincaré disk H^2 (closed form)
# --------------------------------------------------------------------------- #
def dist(z, w):
    """Hyperbolic distance in the Poincaré disk. Accepts scalars or arrays."""
    z = np.asarray(z, dtype=complex)
    w = np.asarray(w, dtype=complex)
    r = np.abs(z - w) / np.abs(1.0 - np.conj(w) * z)
    return 2.0 * np.arctanh(np.clip(r, 0.0, 1.0 - 1e-15))


def _mobius_to_origin(z, base):
    """Isometry sending ``base`` -> 0 (a Möbius map of the disk), applied to z."""
    return (z - base) / (1.0 - np.conj(base) * z)


def _mobius_from_origin(z, base):
    """Inverse of :func:`_mobius_to_origin`: sends 0 -> base."""
    return (z + base) / (1.0 + np.conj(base) * z)


def log_map(base, point):
    """Riemannian log at ``base`` toward ``point``, as an ambient complex tangent.

    Consistent with :func:`exp_map` and :func:`tangent_norm`:
    ``tangent_norm(base, log_map(base, point)) == dist(base, point)``.
    """
    base = np.asarray(base, dtype=complex)
    point = np.asarray(point, dtype=complex)
    p0 = _mobius_to_origin(point, base)
    mag = np.abs(p0)
    safe = np.where(mag > 0, mag, 1.0)
    v0 = np.arctanh(np.clip(mag, 0.0, 1.0 - 1e-15)) * (p0 / safe)
    v0 = np.where(mag > 0, v0, 0.0)
    return (1.0 - np.abs(base) ** 2) * v0  # push forward to the frame at ``base``


def exp_map(base, tangent):
    """Riemannian exp at ``base`` of ambient complex ``tangent`` (inverse of log)."""
    base = np.asarray(base, dtype=complex)
    tangent = np.asarray(tangent, dtype=complex)
    v0 = tangent / (1.0 - np.abs(base) ** 2)  # pull back to the frame at 0
    mag = np.abs(v0)
    safe = np.where(mag > 0, mag, 1.0)
    p0 = np.tanh(mag) * (v0 / safe)
    p0 = np.where(mag > 0, p0, 0.0)
    return _mobius_from_origin(p0, base)


def tangent_norm(base, tangent):
    """Conformal norm of an ambient complex tangent at ``base``."""
    base = np.asarray(base, dtype=complex)
    tangent = np.asarray(tangent, dtype=complex)
    return 2.0 * np.abs(tangent) / (1.0 - np.abs(base) ** 2)


# --------------------------------------------------------------------------- #
# Fuchsian group of the Bolza surface (SU(1,1) Möbius transformations)
# --------------------------------------------------------------------------- #
def _translation(theta):
    """Hyperbolic translation of length ell (cosh(ell/2)=1+sqrt2) along ``theta``.

    Returned as the SU(1,1) matrix ``[[alpha, beta], [conj beta, conj alpha]]`` with
    ``alpha = cosh(ell/2)``, ``beta = e^{i theta} sinh(ell/2)``; det = 1.
    """
    beta = np.exp(1j * theta) * SINH_HALF
    return np.array([[COSH_HALF, beta], [np.conj(beta), COSH_HALF]], dtype=complex)


def apply_mobius(matrix, z):
    """Apply an SU(1,1) Möbius transformation to points z (scalar or array)."""
    a, b = matrix[0, 0], matrix[0, 1]
    c, d = matrix[1, 0], matrix[1, 1]
    z = np.asarray(z, dtype=complex)
    return (a * z + b) / (c * z + d)


def inverse_mobius(matrix):
    """Inverse of an SU(1,1) matrix (det = 1): ``[[d, -b], [-c, a]]``."""
    return np.array(
        [[matrix[1, 1], -matrix[0, 1]], [-matrix[1, 0], matrix[0, 0]]], dtype=complex
    )


# The eight side-pairings: crossing side s_k is a translation along its midpoint
# direction. Opposite sides give inverse translations, so the s_{k+4} pairing is
# the inverse of the s_k pairing; the four generators are k = 0, 1, 2, 3.
SIDE_PAIRINGS = [_translation(SIDE_DIRS[k]) for k in range(N_SIDES)]
GENERATORS = SIDE_PAIRINGS[:4]

# Rigorous cutoff for an exact quotient distance between any two points of the
# closed fundamental domain: d(0, g.0) <= 2 * circumradius + diameter ~ 6.98.
# (Empirically exact already at 5.5; 7.0 keeps a safe margin. 265 elements.)
DEFAULT_CUTOFF = 7.0
_DEFAULT_BALL = None


def _key(matrix, decimals=6):
    """Hashable rounded key identifying a Möbius action by (alpha, beta)."""
    a = complex(round(matrix[0, 0].real, decimals), round(matrix[0, 0].imag, decimals))
    b = complex(round(matrix[0, 1].real, decimals), round(matrix[0, 1].imag, decimals))
    return (a, b)


def group_ball(radius):
    """Group elements as words of length <= ``radius`` in the eight side-pairings,
    deduplicated by their action (identity first). Word-length balls grow
    exponentially; :func:`group_near` is the tool for quotient-distance work. Kept
    for reference and verification."""
    identity = np.eye(2, dtype=complex)
    elements = [identity]
    seen = {_key(identity)}
    frontier = [identity]
    for _ in range(radius):
        nxt = []
        for g in frontier:
            for s in SIDE_PAIRINGS:
                h = g @ s
                k = _key(h)
                if k not in seen:
                    seen.add(k)
                    elements.append(h)
                    nxt.append(h)
        frontier = nxt
    return elements


def group_near(cutoff):
    """Group elements whose copy of the fundamental domain lies near the origin:
    all g with ``d_H(0, g.0) <= cutoff``. The geometrically correct truncation for
    the quotient distance -- far tighter than a word-length ball. BFS over the
    side-pairings, expanding only elements still inside the cutoff (copies adjacent
    along a geodesic have monotonically growing centre distance, so nothing within
    the cutoff is missed). Returns a list of SU(1,1) matrices, identity first."""
    identity = np.eye(2, dtype=complex)
    elements = [identity]
    seen = {_key(identity)}
    frontier = [identity]
    while frontier:
        nxt = []
        for g in frontier:
            for s in SIDE_PAIRINGS:
                h = g @ s
                k = _key(h)
                if k in seen:
                    continue
                seen.add(k)
                if dist(0.0 + 0j, apply_mobius(h, 0.0 + 0j)) <= cutoff:
                    elements.append(h)
                    nxt.append(h)
        frontier = nxt
    return elements


def default_ball():
    """Lazily built and cached geometric group ball used by the quotient metric."""
    global _DEFAULT_BALL
    if _DEFAULT_BALL is None:
        _DEFAULT_BALL = group_near(DEFAULT_CUTOFF)
    return _DEFAULT_BALL


def _copy_centres(ball):
    """Centres g.0 of the copies for a group ball (identity first -> origin first)."""
    return np.array([apply_mobius(g, 0.0 + 0j) for g in ball])


def quotient_distance(z, w, ball=None):
    """Quotient distance ``d([z],[w])`` on the surface: ``min_g d_H(z, g.w)``.

    ``z``, ``w`` broadcastable complex arrays; ``ball`` a list of matrices (defaults
    to the cached geometric ball, exact over the whole fundamental domain)."""
    if ball is None:
        ball = default_ball()
    z = np.asarray(z, dtype=complex)
    w = np.asarray(w, dtype=complex)
    best = dist(z, w)
    for g in ball[1:]:
        best = np.minimum(best, dist(z, apply_mobius(g, w)))
    return best


def in_fundamental_domain(z, tol=1e-9):
    """True where z lies in the Dirichlet fundamental octagon: at least as close to
    the origin as to any neighbouring copy centre. Vectorised over ``z``."""
    z = np.asarray(z, dtype=complex)
    centres = _copy_centres(default_ball())[1:]
    d0 = dist(0.0 + 0j, z)
    dmin = dist(centres[:, None], z.reshape(1, -1)).min(axis=0).reshape(z.shape)
    return d0 <= dmin + tol


def sample_fundamental_domain(n, rng):
    """``n`` points uniform w.r.t. hyperbolic area in the fundamental octagon.

    Rejection sampling: draw uniformly by hyperbolic area in the bounding disk of
    hyperbolic radius = the octagon circumradius (the radial density is
    proportional to sinh r, inverted in closed form), then keep the points that
    fall in the Dirichlet domain. Acceptance ~ area(octagon)/area(disk) ~ 0.41.
    Returns a complex array of length ``n``."""
    centres = _copy_centres(default_ball())[1:]
    rho_max = float(dist(0.0 + 0j, VERTICES[0]))  # octagon circumradius (~2.448)
    kept = []
    while sum(len(k) for k in kept) < n:
        m = 2 * n + 32
        u = rng.random(m)
        r_hyp = np.arccosh(1.0 + u * (np.cosh(rho_max) - 1.0))  # inverse CDF of sinh
        z = np.tanh(r_hyp / 2.0) * np.exp(1j * 2 * np.pi * rng.random(m))
        d0 = dist(0.0 + 0j, z)
        inside = d0 <= dist(centres[:, None], z[None, :]).min(axis=0) + 1e-12
        kept.append(z[inside])
    return np.concatenate(kept)[:n]


def fold_to_domain(z, max_iter=1000):
    """Representative of ``z`` in the fundamental octagon (Dirichlet reduction).

    Greedily applies the side-pairing that most decreases the distance to the
    origin, repeating until none does -- at which point ``z`` is at least as close
    to 0 as to any copy centre, i.e. in the Dirichlet domain. Each step strictly
    decreases ``d_H(0, z)`` over a discrete orbit, so it terminates; unlike a fixed
    group ball this reduces points arbitrarily far away. Vectorised over ``z``."""
    z = np.asarray(z, dtype=complex).copy()
    for _ in range(max_iter):
        d0 = dist(0.0 + 0j, z)
        best, best_d = z.copy(), d0.copy()
        for g in SIDE_PAIRINGS:
            gz = apply_mobius(g, z)
            d = dist(0.0 + 0j, gz)
            take = d < best_d - 1e-12
            best = np.where(take, gz, best)
            best_d = np.where(take, d, best_d)
        if not np.any(best_d < d0 - 1e-12):
            return best
        z = best
    return z


def nearest_copy(base, point, ball=None):
    """The copy ``g.point`` nearest to ``base`` (the representative realising the
    quotient distance). Vectorised over broadcast ``base``/``point``."""
    if ball is None:
        ball = default_ball()
    base = np.asarray(base, dtype=complex)
    point = np.asarray(point, dtype=complex)
    best = np.broadcast_to(point, np.broadcast(base, point).shape).copy()
    best_d = dist(base, point)
    for g in ball[1:]:
        gp = apply_mobius(g, point)
        d = dist(base, gp)
        take = d < best_d
        best = np.where(take, gp, best)
        best_d = np.where(take, d, best_d)
    return best


# --------------------------------------------------------------------------- #
# BolzaSurface: a first-class RiemannianManifold with no geomstats backend
# --------------------------------------------------------------------------- #
class _PoincareDisk:
    """Minimal chart descriptor standing in for a geomstats manifold.

    :class:`RiemannianManifold` only reads ``dim``/``shape`` from its backend, and
    :class:`RiemannianPoint` validates by shape (falling back to ``belongs`` only on
    a shape mismatch). Providing this tiny plain-Python object -- no geomstats --
    is what lets :class:`BolzaSurface` be a manifold without one."""

    dim = 2
    shape = (2,)
    intrinsic = True

    def belongs(self, coordinates, atol=1e-6):
        coordinates = np.asarray(coordinates, dtype=float)
        return bool(np.linalg.norm(coordinates) < 1.0 + atol)


def _to_complex(arr):
    """(..., 2) real -> (...) complex."""
    arr = np.asarray(arr, dtype=float)
    return arr[..., 0] + 1j * arr[..., 1]


def _to_real(z):
    """(...) complex -> (..., 2) real."""
    z = np.asarray(z, dtype=complex)
    return np.stack([z.real, z.imag], axis=-1)


class BolzaSurface(RiemannianManifold):
    """The Bolza surface as a quotient-aware Riemannian manifold.

    Points are ``(..., 2)`` real ``(Re, Im)`` coordinates in the fundamental
    octagon of the Poincaré disk. The geodesic operations are intrinsic to the
    *surface*: :meth:`log` takes each target to its nearest copy under the Fuchsian
    group, and :meth:`exp` folds every step back into the fundamental domain. With
    :meth:`random_uniform`, :meth:`embed`, :meth:`log`, :meth:`exp` and :meth:`norm`
    all quotient-aware, the intrinsic :class:`RepulsionNet` builds a correct net,
    whereas an embedding-based (extrinsic) search would fail across the identified
    boundary.

    Args:
        ball: Group ball used by the quotient metric (defaults to the cached
            geometric ball, exact over the fundamental domain).
    """

    def __init__(self, ball=None) -> None:
        super().__init__(_PoincareDisk())
        self.ball = default_ball() if ball is None else ball

    # -- quotient-aware geodesic operations on (..., 2) real arrays -------- #
    def exp(self, base_point: np.ndarray, tangent_vec: np.ndarray) -> np.ndarray:
        base = _to_complex(base_point)
        tangent = _to_complex(tangent_vec)
        moved = exp_map(base, tangent)
        return _to_real(fold_to_domain(moved))

    def log(self, base_point: np.ndarray, point: np.ndarray) -> np.ndarray:
        base = _to_complex(base_point)
        target = nearest_copy(base, _to_complex(point), self.ball)
        return _to_real(log_map(base, target))

    def norm(self, base_point: np.ndarray, tangent_vec: np.ndarray) -> np.ndarray:
        return tangent_norm(_to_complex(base_point), _to_complex(tangent_vec))

    def to_tangent(self, base_point: np.ndarray, ambient_vec: np.ndarray) -> np.ndarray:
        # The disk chart is 2-dimensional: every ambient vector is already tangent.
        return np.asarray(ambient_vec, dtype=float)

    def random_tangent(
        self, base_point: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Metric-isotropic tangent draw on the conformal disk chart.

        The hyperbolic metric is conformal with factor 2/(1 - |z|^2), so
        scaling an ambient Gaussian by (1 - |z|^2)/2 yields a vector whose
        components are standard normal in an orthonormal frame of the metric
        -- the same Brownian law at every point of the surface.
        """
        gaussian = rng.standard_normal(self.shape)
        z = _to_complex(np.asarray(base_point, dtype=float))
        return gaussian * (1.0 - np.abs(z) ** 2) / 2.0

    def embed(self, points: np.ndarray) -> np.ndarray:
        """Not available: the quotient has no faithful Euclidean embedding.

        ``embed`` exists so that an *extrinsic* neighbour search (a KD-tree on
        ambient coordinates) can accelerate strategies on isometrically embedded
        manifolds. The Bolza surface is a quotient: points glued across the octagon
        boundary are close on the surface yet far apart in any disk-coordinate
        image, so no such embedding is faithful. Returning the disk coordinates
        anyway would make extrinsic neighbour search silently wrong, so this raises
        instead. Use the intrinsic strategies (:class:`RepulsionNet`, which searches
        by geodesic distance), not the extrinsic ones.
        """
        raise NotImplementedError(
            "BolzaSurface has no faithful Euclidean embedding for neighbour search; "
            "use the intrinsic epsilon-net strategies (e.g. RepulsionNet), not the "
            "extrinsic ones (RepulsionNetExtrinsicSpeedup, build_epsilon_net_graph)."
        )

    def random_uniform(
        self, n: int, random_state: int | np.random.Generator | None = None
    ) -> np.ndarray:
        """``(n, 2)`` points uniform by hyperbolic area in the fundamental octagon."""
        rng = np.random.default_rng(random_state)
        return _to_real(sample_fundamental_domain(n, rng))

    # -- point-level metric (Space interface), quotient distance ---------- #
    def distance(self, point1, point2) -> float:
        d = quotient_distance(
            _to_complex(point1.coordinates), _to_complex(point2.coordinates), self.ball
        )
        return float(np.asarray(d).item())
