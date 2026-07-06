"""Tests for the Bolza surface: closed-form geometry, Fuchsian group, and the
quotient-aware BolzaSurface manifold (including its intrinsic epsilon-net path)."""

import numpy as np
import pytest

from kmeanssa_ng import create_bolza_surface
from kmeanssa_ng.riemannian_manifold import bolza as bg
from kmeanssa_ng.riemannian_manifold import RepulsionNet, RepulsionNetExtrinsicSpeedup


def _rand_disk(rng, n, rmax=0.84):
    return rmax * np.sqrt(rng.random(n)) * np.exp(1j * 2 * np.pi * rng.random(n))


class TestGeometry:
    def test_side_pairings_are_su11(self):
        err = max(
            abs(abs(g[0, 0]) ** 2 - abs(g[0, 1]) ** 2 - 1.0) for g in bg.SIDE_PAIRINGS
        )
        assert err < 1e-9

    def test_exp_log_norm_consistency(self):
        rng = np.random.default_rng(0)
        a, b = _rand_disk(rng, 2000), _rand_disk(rng, 2000)
        v = bg.log_map(a, b)
        assert np.max(np.abs(bg.tangent_norm(a, v) - bg.dist(a, b))) < 1e-9
        assert np.max(np.abs(bg.exp_map(a, v) - b)) < 1e-9

    def test_side_pairing_maps_opposite_side(self):
        V = bg.VERTICES
        for k in range(4):
            img = bg.apply_mobius(
                bg.SIDE_PAIRINGS[k], np.array([V[(k + 4) % 8], V[(k + 5) % 8]])
            )
            target = {complex(np.round(V[k], 6)), complex(np.round(V[(k + 1) % 8], 6))}
            got = {complex(np.round(img[0], 6)), complex(np.round(img[1], 6))}
            assert got == target

    def test_opposite_pairing_is_inverse(self):
        err = max(
            np.max(
                np.abs(bg.SIDE_PAIRINGS[k + 4] - bg.inverse_mobius(bg.SIDE_PAIRINGS[k]))
            )
            for k in range(4)
        )
        assert err < 1e-9

    def test_octagon_tiles_without_overlap(self):
        ball = bg.default_ball()
        centres = np.array([bg.apply_mobius(g, 0j) for g in ball])
        rng = np.random.default_rng(1)
        z = _rand_disk(rng, 1000, rmax=0.6)

        def in_octagon(w):
            d0 = bg.dist(0j, w)
            dj = bg.dist(centres[1:, None], w[None, :]).min(axis=0)
            return d0 <= dj + 1e-9

        counts = np.zeros(len(z), int)
        for g in ball:
            counts += in_octagon(bg.apply_mobius(bg.inverse_mobius(g), z)).astype(int)
        assert np.all(counts == 1)  # exactly one copy: no gaps, no overlaps

    def test_all_eight_vertices_are_one_surface_point(self):
        # The genus-2 signature: the octagon's eight corners are a single point.
        V = bg.VERTICES
        mx = max(bg.quotient_distance(V[0], V[j]) for j in range(1, 8))
        assert mx < 1e-6

    def test_glued_points_are_identified(self):
        rng = np.random.default_rng(2)
        z = _rand_disk(rng, 500, rmax=0.7)
        w = bg.apply_mobius(bg.GENERATORS[1], z)  # z and g.z are the same surface point
        assert np.max(bg.quotient_distance(z, w)) < 1e-6

    def test_quotient_distance_is_a_metric(self):
        rng = np.random.default_rng(3)
        z, w = _rand_disk(rng, 500, rmax=0.7), _rand_disk(rng, 500, rmax=0.7)
        assert (
            np.max(np.abs(bg.quotient_distance(z, w) - bg.quotient_distance(w, z)))
            < 1e-9
        )
        assert np.all(bg.quotient_distance(z, w) <= bg.dist(z, w) + 1e-9)

    def test_sampling_is_in_domain_and_uniform(self):
        rng = np.random.default_rng(7)
        pts = bg.sample_fundamental_domain(5000, rng)
        assert np.all(bg.in_fundamental_domain(pts))
        half = float(bg.dist(0j, bg.VERTICES[0])) / 2  # inner disk fully inside octagon
        measured = np.mean(bg.dist(0j, pts) <= half)
        expected = (
            2 * np.pi * (np.cosh(half) - 1) / (4 * np.pi)
        )  # area(inner)/area(octagon)
        assert abs(measured - expected) < 0.03


class TestBolzaSurface:
    def test_random_uniform_shape_and_domain(self):
        surface = create_bolza_surface()
        pts = surface.random_uniform(400, random_state=0)
        assert pts.shape == (400, 2)
        assert np.all(bg.in_fundamental_domain(pts[:, 0] + 1j * pts[:, 1]))

    def test_reproducible(self):
        s = create_bolza_surface()
        np.testing.assert_array_equal(
            s.random_uniform(100, random_state=5), s.random_uniform(100, random_state=5)
        )

    def test_norm_of_log_is_quotient_distance(self):
        surface = create_bolza_surface()
        rng = np.random.default_rng(4)
        a = surface.random_uniform(300, rng)
        b = surface.random_uniform(300, rng)
        d = surface.norm(a, surface.log(a, b))
        za, zb = a[:, 0] + 1j * a[:, 1], b[:, 0] + 1j * b[:, 1]
        assert np.max(np.abs(d - bg.quotient_distance(za, zb))) < 1e-9

    def test_exp_stays_in_domain(self):
        surface = create_bolza_surface()
        rng = np.random.default_rng(6)
        base = surface.random_uniform(300, rng)
        tangent = 0.5 * rng.standard_normal(
            (300, 2)
        )  # sizeable steps cross the boundary
        moved = surface.exp(base, tangent)
        assert np.all(bg.in_fundamental_domain(moved[:, 0] + 1j * moved[:, 1]))

    def test_exp_log_roundtrip_on_surface(self):
        surface = create_bolza_surface()
        rng = np.random.default_rng(8)
        a = surface.random_uniform(300, rng)
        b = surface.random_uniform(300, rng)
        recovered = surface.exp(a, surface.log(a, b))
        zr = recovered[:, 0] + 1j * recovered[:, 1]
        zb = b[:, 0] + 1j * b[:, 1]
        assert np.max(bg.quotient_distance(zr, zb)) < 1e-8

    def test_distance_matches_quotient(self):
        from kmeanssa_ng.riemannian_manifold import RiemannianPoint

        surface = create_bolza_surface()
        p = RiemannianPoint(surface, np.array([0.1, 0.2]))
        q = RiemannianPoint(surface, np.array([-0.3, 0.05]))
        expected = float(bg.quotient_distance(0.1 + 0.2j, -0.3 + 0.05j))
        assert abs(surface.distance(p, q) - expected) < 1e-12


class TestIntrinsicNetOnBolza:
    def _min_spacing(self, surface, pts):
        za = pts[:, 0] + 1j * pts[:, 1]
        d = bg.quotient_distance(za[:, None], za[None, :])
        np.fill_diagonal(d, np.inf)
        return d.min()

    def test_intrinsic_repulsion_builds_valid_net(self):
        # The point of making RepulsionNet intrinsic: it produces a spread-out net
        # on a quotient space, where the extrinsic (embedding) search cannot.
        surface = create_bolza_surface()
        n = 120
        uniform = surface.random_uniform(n, random_state=0)
        repelled = RepulsionNet(n_iter=40, random_state=0).build(surface, n)
        assert repelled.shape == (n, 2)
        assert np.all(bg.in_fundamental_domain(repelled[:, 0] + 1j * repelled[:, 1]))
        # repulsion increases the smallest pairwise (surface) spacing
        assert self._min_spacing(surface, repelled) > self._min_spacing(
            surface, uniform
        )

    def test_extrinsic_speedup_rejected_on_quotient(self):
        # The extrinsic search relies on embed(), which a quotient cannot provide
        # faithfully. Rather than return silently-wrong neighbours, BolzaSurface
        # refuses it -- so the extrinsic strategy raises instead of building a bad
        # net. The intrinsic RepulsionNet is the correct path.
        surface = create_bolza_surface()
        with pytest.raises(NotImplementedError):
            RepulsionNetExtrinsicSpeedup(n_iter=1, random_state=1).build(surface, 20)

    def test_intrinsic_graph_is_connected_quotient(self):
        # The whole pipeline: intrinsic net -> intrinsic epsilon-net graph. The
        # extrinsic builder is unavailable (embed raises); the intrinsic one builds
        # a connected quantum graph whose edges follow the quotient metric.
        import networkx as nx
        from kmeanssa_ng.riemannian_manifold import build_epsilon_net_graph

        surface = create_bolza_surface()
        n = 120
        pts = RepulsionNet(n_iter=30, random_state=0).build(surface, n)
        qg = build_epsilon_net_graph(surface, pts, ell=0.6, intrinsic=True)
        assert qg.number_of_nodes() == n
        assert qg.number_of_edges() > 0
        assert nx.is_connected(qg)


class TestManifoldGuarantees:
    """Regression tests: the manifold API guarantees hold on the quotient.

    They pin down three former defects: the Brownian direction was an ambient
    Gaussian (metric norm ~3x larger near the octagon boundary than at the
    center), the Frechet mean crashed (geomstats estimator on the bare chart)
    and would have ignored the quotient anyway, and off-manifold points were
    accepted then folded into plausible-looking distances.
    """

    @staticmethod
    def _boundary_point(margin=0.02):
        """A domain point close to the middle of an octagon side."""
        direction = np.exp(1j * np.pi / 8)
        lo, hi = 0.0, 0.999
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if bg.in_fundamental_domain(np.array([mid * direction]))[0]:
                lo = mid
            else:
                hi = mid
        return (lo - margin) * direction

    def test_random_tangent_is_metric_isotropic(self):
        """The Brownian direction has the same law everywhere on the surface."""
        surface = create_bolza_surface()
        rng = np.random.default_rng(0)
        origin = np.zeros(2)
        near_boundary = bg._to_real(np.array(0.8 * np.exp(1j * np.pi / 4)))

        def mean_norm(base):
            norms = [
                float(surface.norm(base, surface.random_tangent(base, rng)))
                for _ in range(2000)
            ]
            return np.mean(norms)

        expected = np.sqrt(np.pi / 2)  # E||g|| for a 2D standard normal
        assert mean_norm(origin) == pytest.approx(expected, rel=0.05)
        assert mean_norm(near_boundary) == pytest.approx(expected, rel=0.05)

    def test_frechet_mean_is_quotient_aware(self):
        """The Karcher mean of two points glued across a side stays with them."""
        from kmeanssa_ng import KarcherFrechetMean, RiemannianPoint

        surface = create_bolza_surface()
        p = bg._to_real(np.array(self._boundary_point()))
        # Step across the identified boundary: exp folds back into the domain
        outward = bg._to_real(
            np.array(0.3 * np.exp(1j * np.pi / 8) / 2 * (1 - 0.77**2))
        )
        q = surface.exp(p, outward)

        gap = surface.distance(RiemannianPoint(surface, p), RiemannianPoint(surface, q))
        assert gap < 0.5  # close on the surface...
        assert np.linalg.norm(p - q) > 1.0  # ...but far apart in the chart

        mean = KarcherFrechetMean().update(
            [RiemannianPoint(surface, p), RiemannianPoint(surface, q)], surface
        )
        d_p = surface.distance(mean, RiemannianPoint(surface, p))
        d_q = surface.distance(mean, RiemannianPoint(surface, q))
        # Geodesic midpoint on the surface; a chart-level mean would sit near
        # the disk center, ~2 hyperbolic units away from both points.
        assert d_p == pytest.approx(gap / 2, abs=1e-6)
        assert d_q == pytest.approx(gap / 2, abs=1e-6)

    def test_off_manifold_point_rejected(self):
        """A point outside the disk raises instead of yielding fake distances."""
        from kmeanssa_ng import RiemannianPoint

        surface = create_bolza_surface()
        with pytest.raises(ValueError, match="belong"):
            RiemannianPoint(surface, np.array([5.0, 7.0]))

    def test_sa_frechet_mean_runs_intrinsically(self):
        """The generic SA-based Fréchet mean works unchanged on the quotient.

        With ``n_samples`` oversampling (with replacement) a 2-point cluster,
        the inner annealing gets a full schedule and must beat the trivial
        strategy of parking on one of the observations.
        """
        from kmeanssa_ng import RiemannianPoint, SimulatedAnnealingFrechetMean

        surface = create_bolza_surface()
        p = bg._to_real(np.array(self._boundary_point()))
        outward = bg._to_real(
            np.array(0.3 * np.exp(1j * np.pi / 8) / 2 * (1 - 0.77**2))
        )
        q = surface.exp(p, outward)
        P, Q = RiemannianPoint(surface, p), RiemannianPoint(surface, q)
        gap = surface.distance(P, Q)

        mean = SimulatedAnnealingFrechetMean(
            n_samples=200, random_state=0, lambda0=0.5, beta0=3.0, step_size=0.02
        ).update([P, Q], surface)

        energy = (surface.distance(mean, P) ** 2 + surface.distance(mean, Q) ** 2) / 2
        at_observation = gap**2 / 2  # energy of parking on p or q
        assert energy < 0.95 * at_observation
