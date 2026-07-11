"""Test riemannian manifold functionality."""

import numpy as np
import pytest
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.hyperboloid import Hyperboloid

from kmeanssa_ng.riemannian_manifold.sampling import UniformManifoldSampling
from kmeanssa_ng.riemannian_manifold import (
    RiemannianCenter,
    RiemannianManifold,
    RiemannianPoint,
    Sphere,
    create_hyperbolic_space,
    create_sphere,
)
from kmeanssa_ng import SimulatedAnnealing
from kmeanssa_ng.core.strategies.initialization import KMeansPlusPlus, RandomInit


class TestRiemannianManifold:
    """Tests for RiemannianManifold class."""

    def test_create_manifold(self):
        """Test creating a Riemannian manifold from geomstats object."""
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)

        assert space.manifold == sphere

    def test_distance(self):
        """Test distance computation between two points."""
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)

        # Create two points on the sphere (extrinsic coordinates)
        coords1 = np.array([1.0, 0.0, 0.0])
        coords2 = np.array([0.0, 1.0, 0.0])

        point1 = RiemannianPoint(space, coords1)
        point2 = RiemannianPoint(space, coords2)

        dist = space.distance(point1, point2)

        # Distance should be π/2 for orthogonal unit vectors
        assert isinstance(dist, float)
        assert dist > 0
        np.testing.assert_allclose(dist, np.pi / 2, rtol=1e-5)

    def test_sample_points(self):
        """Test sampling random points from the manifold."""
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)

        n = 10
        points = space.sample_points(n, strategy=UniformManifoldSampling())

        assert len(points) == n
        assert all(isinstance(p, RiemannianPoint) for p in points)

        # Check that points belong to the manifold (on unit sphere)
        for point in points:
            assert space.manifold.belongs(point.coordinates)
            norm = np.linalg.norm(point.coordinates)
            np.testing.assert_allclose(norm, 1.0, rtol=1e-5)

    def test_sample_centers(self):
        """Test sampling random centers from the manifold."""
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)

        k = 3

        points = space.sample_points(k, strategy=UniformManifoldSampling())
        sa = SimulatedAnnealing(points, k=k)
        centers = RandomInit().initialize_centers(sa)

        assert len(centers) == k
        assert all(isinstance(c, RiemannianCenter) for c in centers)

        # Check that centers are on the sphere (unit norm for extrinsic coords)
        for center in centers:
            norm = np.linalg.norm(center.coordinates)
            np.testing.assert_allclose(norm, 1.0, rtol=1e-5)

    def test_calculate_energy(self):
        """Test energy calculation."""
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)

        # Sample observations
        points = space.sample_points(20, strategy=UniformManifoldSampling())

        # Sample centers
        sa = SimulatedAnnealing(points, k=3)
        centers = RandomInit().initialize_centers(sa)

        # Calculate energy on an explicit observation list
        energy = space.calculate_energy(centers, observations=points)

        assert isinstance(energy, float)
        assert energy >= 0  # Energy is sum of squared distances, must be non-negative

    def test_calculate_energy_no_observations(self):
        """Test energy calculation fails without observations."""
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)

        # Create dummy observations to initialize centers
        dummy_points = space.sample_points(10, strategy=UniformManifoldSampling())
        sa = SimulatedAnnealing(dummy_points, k=3)
        centers = RandomInit().initialize_centers(sa)

        # Observations belong to the caller: omitting them must fail loudly
        with pytest.raises(ValueError, match="observations"):
            space.calculate_energy(centers)

    def test_calculate_energy_no_centers(self):
        """Test energy calculation fails without centers."""
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)
        points = space.sample_points(10, strategy=UniformManifoldSampling())

        with pytest.raises(ValueError, match="Centers list cannot be empty"):
            space.calculate_energy([], observations=points)

    def test_calculate_energy_with_how_parameter(self):
        """Only 'empirical' is a valid mode on a manifold; others fail loudly."""
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)
        points = space.sample_points(20, strategy=UniformManifoldSampling())

        sa = SimulatedAnnealing(points, k=3, energy_mode="empirical")
        centers = RandomInit().initialize_centers(sa)
        energy_default = space.calculate_energy(centers, observations=points)
        energy_empirical = space.calculate_energy(
            centers, how="empirical", observations=points
        )

        assert energy_default == energy_empirical  # "empirical" is the default
        assert isinstance(energy_default, float)
        assert energy_default >= 0

        # No silent reinterpretation of the other modes on a manifold
        with pytest.raises(ValueError, match="empirical"):
            space.calculate_energy(centers, how="obs", observations=points)
        with pytest.raises(ValueError, match="empirical"):
            space.calculate_energy(centers, how="uniform", observations=points)
        with pytest.raises(ValueError, match="empirical"):
            space.calculate_energy(centers, how="node_measure", observations=points)

    def test_two_annealings_share_a_space_independently(self):
        """Two SAs on one manifold evaluate energies on their own observations.

        Regression: the space used to hold the observation list, so creating a
        second annealer silently redefined the energy of the first one.
        """
        space = RiemannianManifold(Hypersphere(dim=2))
        north = [RiemannianPoint(space, np.array([0.0, 0.0, 1.0])) for _ in range(5)]
        south = [RiemannianPoint(space, np.array([0.0, 0.0, -1.0])) for _ in range(5)]

        sa_north = SimulatedAnnealing(
            north, k=1, random_state=0, energy_mode="empirical"
        )
        sa_south = SimulatedAnnealing(
            south, k=1, random_state=0, energy_mode="empirical"
        )

        center = [space.center_from_point(north[0])]
        assert sa_north.calculate_energy(center) == pytest.approx(0.0)
        assert sa_south.calculate_energy(center) == pytest.approx(np.pi**2)

    def test_distances_from_centers(self):
        """Test computing distances from multiple centers to a target point."""
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)

        # Sample centers and a target point

        # Sample centers and a target point
        points = space.sample_points(20, strategy=UniformManifoldSampling())
        sa = SimulatedAnnealing(points, k=5)
        centers = RandomInit().initialize_centers(sa)
        target = space.sample_points(1, strategy=UniformManifoldSampling())[0]

        # Compute distances
        distances = space.distances_from_centers(centers, target)

        # Verify result
        assert isinstance(distances, np.ndarray)
        assert distances.shape == (5,)
        assert np.all(distances >= 0)  # All distances should be non-negative

        # Verify distances match individual distance calculations
        for i, center in enumerate(centers):
            individual_dist = space.distance(center, target)
            assert np.isclose(distances[i], individual_dist)


class TestRiemannianPoint:
    """Tests for RiemannianPoint class."""

    def test_create_point(self):
        """Test creating a point on a manifold."""
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)
        coords = np.array([1.0, 0.0, 0.0])

        point = RiemannianPoint(space, coords)

        assert point.space == space
        np.testing.assert_array_equal(point.coordinates, coords)

    def test_create_point_invalid_coords(self):
        """Test creating a point with invalid coordinates."""
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)

        # Wrong dimension (should be 3 for sphere embedded in R^3)
        coords = np.array([2.0, 0.0])

        with pytest.raises(ValueError):
            RiemannianPoint(space, coords)

    def test_create_point_none_space(self):
        """Test creating a point with None space."""
        coords = np.array([1.0, 0.0, 0.0])

        with pytest.raises(ValueError, match="space cannot be None"):
            RiemannianPoint(None, coords)

    def test_create_point_non_array(self):
        """Test creating a point with non-array coordinates."""
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)

        with pytest.raises(ValueError, match="coordinates must be a numpy array"):
            RiemannianPoint(space, [1.0, 0.0, 0.0])

    def test_point_str(self):
        """Test string representation of point."""
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)
        coords = np.array([1.0, 0.0, 0.0])
        point = RiemannianPoint(space, coords)

        s = str(point)
        assert "RiemannianPoint" in s
        assert "Hypersphere" in s

    def test_point_repr(self):
        """Test detailed representation of point."""
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)
        coords = np.array([1.0, 0.0, 0.0])
        point = RiemannianPoint(space, coords)

        r = repr(point)
        assert "RiemannianPoint" in r
        assert "coordinates" in r


class TestRiemannianCenter:
    """Tests for RiemannianCenter class."""

    def test_create_center(self):
        """Test creating a center from a point."""
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)
        coords = np.array([1.0, 0.0, 0.0])
        point = RiemannianPoint(space, coords)

        center = RiemannianCenter(point)

        assert center.space == space
        np.testing.assert_array_equal(center.coordinates, coords)

    def test_brownian_motion(self):
        """Test Brownian motion on manifold."""
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)
        coords = np.array([1.0, 0.0, 0.0])
        point = RiemannianPoint(space, coords)
        center = RiemannianCenter(point)

        initial_coords = center.coordinates.copy()
        center.brownian_motion(time_to_travel=0.1)

        # Coordinates should have changed
        assert not np.allclose(center.coordinates, initial_coords)

        # Should still be on the sphere
        norm = np.linalg.norm(center.coordinates)
        np.testing.assert_allclose(norm, 1.0, rtol=1e-5)

    def test_brownian_step_radial_law_is_chi(self):
        """The step is sqrt(t) * V with V ~ N(0, I) tangent (DYN-3 regression).

        On S^2 the geodesic displacement of one step exp(x, sqrt(t) V) has
        length sqrt(t) |V| with |V| ~ chi_2, so E[dist]/sqrt(t) = E[chi_2] =
        sqrt(pi/2) ~ 1.2533. The previous code multiplied V by an extra scalar
        N(0,1); that product law has E[|step|]/sqrt(t) = E[|N|] E[chi_2] = 1.0
        (the same second moment, hence the covariance looked right, but a
        heavier-tailed, non-Gaussian step). The mean displacement separates
        the two laws cleanly.
        """
        space = RiemannianManifold(Hypersphere(dim=2))
        base = np.array([1.0, 0.0, 0.0])
        t = 0.01
        rng = np.random.default_rng(0)

        n = 4000
        displacements = np.empty(n)
        for i in range(n):
            center = RiemannianCenter(RiemannianPoint(space, base), rng=rng)
            center.brownian_motion(time_to_travel=t)
            displacements[i] = space.distance(center, RiemannianPoint(space, base))

        mean_normalized = displacements.mean() / np.sqrt(t)
        assert mean_normalized == pytest.approx(np.sqrt(np.pi / 2), abs=0.04)
        assert mean_normalized > 1.1  # excludes the product-law value 1.0

    def test_brownian_motion_reproducible_across_global_rng(self):
        """Brownian motion depends only on the center's generator, not global RNG.

        The random tangent direction must be drawn from ``center._rng`` so that a
        given ``random_state`` reproduces the same walk regardless of the global
        numpy RNG state (which geomstats' ``random_tangent_vec`` would otherwise use).
        """
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)
        point = RiemannianPoint(space, np.array([1.0, 0.0, 0.0]))

        def walk(seed):
            center = RiemannianCenter(point, rng=np.random.default_rng(seed))
            for _ in range(5):
                center.brownian_motion(time_to_travel=0.1)
            return center.coordinates.copy()

        np.random.seed(1)
        first = walk(42)
        np.random.seed(999_999)  # perturb the global RNG between the two walks
        second = walk(42)

        np.testing.assert_array_equal(first, second)

    def test_brownian_motion_zero_time(self):
        """Test Brownian motion with zero time."""
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)
        coords = np.array([1.0, 0.0, 0.0])
        point = RiemannianPoint(space, coords)
        center = RiemannianCenter(point)

        initial_coords = center.coordinates.copy()
        center.brownian_motion(time_to_travel=0.0)

        # Coordinates should not change
        np.testing.assert_array_equal(center.coordinates, initial_coords)

    def test_brownian_motion_invalid_time(self):
        """Test Brownian motion with invalid time."""
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)
        coords = np.array([1.0, 0.0, 0.0])
        point = RiemannianPoint(space, coords)
        center = RiemannianCenter(point)

        with pytest.raises(ValueError, match="time_to_travel must be non-negative"):
            center.brownian_motion(time_to_travel=-0.1)

        with pytest.raises(ValueError, match="time_to_travel must be a number"):
            center.brownian_motion(time_to_travel="invalid")

    def test_drift(self):
        """Test drift toward target point."""
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)

        # Start at a point
        coords1 = np.array([1.0, 0.0, 0.0])
        point1 = RiemannianPoint(space, coords1)
        center = RiemannianCenter(point1)

        # Target at another point (orthogonal, not antipodal)
        coords2 = np.array([0.0, 1.0, 0.0])
        point2 = RiemannianPoint(space, coords2)

        initial_coords = center.coordinates.copy()
        center.drift(point2, prop_to_travel=0.5)

        # Should have moved toward target
        assert not np.allclose(center.coordinates, initial_coords)

        # Should still be on the sphere
        norm = np.linalg.norm(center.coordinates)
        np.testing.assert_allclose(norm, 1.0, rtol=1e-5)

        # Should be closer to target than before
        initial_dist = space.distance(point1, point2)
        new_center_point = RiemannianPoint(space, center.coordinates)
        new_dist = space.distance(new_center_point, point2)
        assert new_dist < initial_dist

    def test_drift_zero_proportion(self):
        """Test drift with zero proportion."""
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)
        coords1 = np.array([1.0, 0.0, 0.0])
        coords2 = np.array([0.0, 1.0, 0.0])
        point1 = RiemannianPoint(space, coords1)
        point2 = RiemannianPoint(space, coords2)
        center = RiemannianCenter(point1)

        initial_coords = center.coordinates.copy()
        center.drift(point2, prop_to_travel=0.0)

        # Should not move
        np.testing.assert_array_equal(center.coordinates, initial_coords)

    def test_drift_full_proportion(self):
        """Test drift with full proportion."""
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)
        coords1 = np.array([1.0, 0.0, 0.0])
        coords2 = np.array([0.0, 1.0, 0.0])
        point1 = RiemannianPoint(space, coords1)
        point2 = RiemannianPoint(space, coords2)
        center = RiemannianCenter(point1)

        center.drift(point2, prop_to_travel=1.0)

        # Should reach target (or very close)
        np.testing.assert_allclose(center.coordinates, coords2, atol=1e-5)

    def test_drift_invalid_proportion(self):
        """Test drift with invalid proportion."""
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)
        coords1 = np.array([1.0, 0.0, 0.0])
        coords2 = np.array([0.0, 1.0, 0.0])
        point1 = RiemannianPoint(space, coords1)
        point2 = RiemannianPoint(space, coords2)
        center = RiemannianCenter(point1)

        with pytest.raises(ValueError, match="prop_to_travel must be in \\[0, 1\\]"):
            center.drift(point2, prop_to_travel=-0.1)

        with pytest.raises(ValueError, match="prop_to_travel must be in \\[0, 1\\]"):
            center.drift(point2, prop_to_travel=1.5)

        with pytest.raises(ValueError, match="prop_to_travel must be a number"):
            center.drift(point2, prop_to_travel="invalid")

    def test_drift_none_target(self):
        """Test drift with None target."""
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)
        coords = np.array([1.0, 0.0, 0.0])
        point = RiemannianPoint(space, coords)
        center = RiemannianCenter(point)

        with pytest.raises(ValueError, match="target_point cannot be None"):
            center.drift(None, prop_to_travel=0.5)

    def test_clone(self):
        """Test cloning a center."""
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)
        coords = np.array([1.0, 0.0, 0.0])
        point = RiemannianPoint(space, coords)
        center = RiemannianCenter(point)

        cloned = center.clone()

        # Should be a different object
        assert cloned is not center
        assert cloned.coordinates is not center.coordinates

        # But with same values
        assert cloned.space == center.space
        np.testing.assert_array_equal(cloned.coordinates, center.coordinates)

        # Modifying clone should not affect original
        cloned.brownian_motion(0.1)
        assert not np.allclose(cloned.coordinates, center.coordinates)

    def test_center_str(self):
        """Test string representation of center."""
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)
        coords = np.array([1.0, 0.0, 0.0])
        point = RiemannianPoint(space, coords)
        center = RiemannianCenter(point)

        s = str(center)
        assert "Center" in s
        assert "Hypersphere" in s

    def test_center_repr(self):
        """Test detailed representation of center."""
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)
        coords = np.array([1.0, 0.0, 0.0])
        point = RiemannianPoint(space, coords)
        center = RiemannianCenter(point)

        r = repr(center)
        assert "RiemannianCenter" in r
        assert "coordinates" in r


class TestGenerators:
    """Tests for generator functions."""

    def test_create_sphere(self):
        """Test creating a sphere."""
        sphere = create_sphere(dim=2)

        assert isinstance(sphere, RiemannianManifold)
        assert isinstance(sphere.manifold, Hypersphere)
        assert sphere.manifold.dim == 2

    def test_create_sphere_different_dims(self):
        """Test creating spheres of different dimensions."""
        for dim in [1, 2, 3, 5]:
            sphere = create_sphere(dim=dim)
            assert sphere.manifold.dim == dim

    def test_create_sphere_invalid_dim(self):
        """Test creating sphere with invalid dimension."""
        with pytest.raises(ValueError):
            create_sphere(dim=-1)

        with pytest.raises(ValueError):
            create_sphere(dim=0)

    def test_create_hyperbolic_space(self):
        """Test creating hyperbolic space."""
        hyperbolic = create_hyperbolic_space(dim=2)

        assert isinstance(hyperbolic, RiemannianManifold)
        assert isinstance(hyperbolic.manifold, Hyperboloid)
        assert hyperbolic.manifold.dim == 2

    def test_create_hyperbolic_different_dims(self):
        """Test creating hyperbolic spaces of different dimensions."""
        for dim in [1, 2, 3, 5]:
            hyperbolic = create_hyperbolic_space(dim=dim)
            assert hyperbolic.manifold.dim == dim

    def test_create_hyperbolic_invalid_dim(self):
        """Test creating hyperbolic space with invalid dimension."""
        with pytest.raises(ValueError):
            create_hyperbolic_space(dim=-1)

        with pytest.raises(ValueError):
            create_hyperbolic_space(dim=0)


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_clustering_workflow(self):
        """Test a complete clustering workflow on a sphere."""
        # Create sphere
        sphere = create_sphere(dim=2)

        # Sample observations
        points = sphere.sample_points(50, strategy=UniformManifoldSampling())
        assert len(points) == 50

        # Initialize centers with k-means++
        k = 3

        sa = SimulatedAnnealing(points, k=k)
        centers = KMeansPlusPlus().initialize_centers(sa)
        assert len(centers) == k

        # Calculate initial energy
        initial_energy = sphere.calculate_energy(centers, observations=points)
        assert initial_energy > 0

        # Perform some drift operations
        for center in centers:
            # Drift toward random observation
            target = np.random.choice(points)
            center.drift(target, prop_to_travel=0.1)

        # Calculate new energy
        new_energy = sphere.calculate_energy(centers, observations=points)
        assert new_energy > 0

        # Energy should have changed
        assert new_energy != initial_energy

    def test_different_manifolds(self):
        """Test that the same operations work on different sphere dimensions.

        Hyperbolic space refuses uniform sampling: uniform-in-volume does not
        exist on a non-compact manifold, and the previous geomstats fallback
        drew from the global RNG (non-reproducible).
        """
        hyperbolic = create_hyperbolic_space(dim=2)
        with pytest.raises(NotImplementedError):
            hyperbolic.sample_points(5, strategy=UniformManifoldSampling())

        manifolds = [
            create_sphere(dim=2),
            create_sphere(dim=3),
        ]

        for space in manifolds:
            # Sample and compute
            points = space.sample_points(20, strategy=UniformManifoldSampling())

            sa = SimulatedAnnealing(points, k=3)
            centers = KMeansPlusPlus().initialize_centers(sa)
            energy = space.calculate_energy(centers, observations=points)

            assert len(points) == 20
            assert len(centers) == 3
            assert energy >= 0

    @pytest.mark.slow
    def test_brownian_motion_stays_on_manifold(self):
        """Test that repeated Brownian motion keeps centers on manifold."""
        sphere = create_sphere(dim=2)
        coords = np.array([1.0, 0.0, 0.0])
        point = RiemannianPoint(sphere, coords)
        center = RiemannianCenter(point)

        # Perform many Brownian steps
        for _ in range(100):
            center.brownian_motion(0.01)

            # Should still be on sphere
            norm = np.linalg.norm(center.coordinates)
            np.testing.assert_allclose(norm, 1.0, rtol=1e-5)


class TestGeodesicOperations:
    """Tests for the exp/log/norm/random_uniform geodesic wrappers."""

    def test_dim_and_is_sphere(self):
        assert create_sphere(dim=2).dim == 2
        assert create_sphere(dim=2).is_sphere is True
        assert create_hyperbolic_space(dim=2).is_sphere is False

    def test_random_uniform_reproducible_and_on_manifold(self):
        sphere = create_sphere(dim=2)
        a = sphere.random_uniform(50, random_state=42)
        b = sphere.random_uniform(50, random_state=42)
        np.testing.assert_array_equal(a, b)
        assert a.shape == (50, 3)
        np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-12)

    def test_random_uniform_accepts_generator(self):
        sphere = create_sphere(dim=2)
        rng = np.random.default_rng(0)
        pts = sphere.random_uniform(10, random_state=rng)
        assert pts.shape == (10, 3)

    def test_exp_log_round_trip(self):
        sphere = create_sphere(dim=2)
        base = sphere.random_uniform(20, random_state=1)
        target = sphere.random_uniform(20, random_state=2)
        recovered = sphere.exp(base, sphere.log(base, target))
        np.testing.assert_allclose(recovered, target, atol=1e-6)

    def test_norm_equals_geodesic_distance(self):
        sphere = create_sphere(dim=2)
        base = sphere.random_uniform(20, random_state=3)
        target = sphere.random_uniform(20, random_state=4)
        tangent = sphere.log(base, target)
        expected = np.array(
            [
                sphere.distance(
                    RiemannianPoint(sphere, base[i]),
                    RiemannianPoint(sphere, target[i]),
                )
                for i in range(len(base))
            ]
        )
        np.testing.assert_allclose(sphere.norm(base, tangent), expected, atol=1e-6)

    def test_geodesic_wrappers_generalise_to_hyperbolic(self):
        hyp = create_hyperbolic_space(dim=2)
        base = np.array([[1.0, 0.0, 0.0]])
        tangent = np.array([[0.0, 0.3, 0.4]])  # Riemannian norm 0.5
        point = hyp.exp(base, tangent)
        assert bool(hyp.manifold.belongs(point)[0])
        np.testing.assert_allclose(hyp.norm(base, tangent), 0.5, atol=1e-6)
        np.testing.assert_allclose(hyp.log(base, point), tangent, atol=1e-6)

    def test_random_uniform_unsupported_manifold_raises(self):
        with pytest.raises(NotImplementedError, match="hyperspheres only"):
            create_hyperbolic_space(dim=2).random_uniform(5, random_state=0)


class TestSphere:
    """The closed-form Sphere must match the generic geomstats implementation."""

    def test_create_sphere_is_sphere_subclass(self):
        assert isinstance(create_sphere(2), Sphere)

    def test_closed_forms_match_geomstats(self):
        """exp/log/distance/norm/to_tangent equal geomstats, single and batched."""
        geo = RiemannianManifold(Hypersphere(2))  # generic geomstats reference
        sph = create_sphere(2)  # closed-form overrides
        rng = np.random.default_rng(0)

        def unit(shape):
            x = rng.standard_normal(shape)
            return x / np.linalg.norm(x, axis=-1, keepdims=True)

        for batch in ((), (16,)):  # single point and a batch
            p, q = unit(batch + (3,)), unit(batch + (3,))
            v = geo.to_tangent(p, rng.standard_normal(batch + (3,)))
            np.testing.assert_allclose(sph.exp(p, v), geo.exp(p, v), atol=1e-10)
            np.testing.assert_allclose(sph.log(p, q), geo.log(p, q), atol=1e-9)
            np.testing.assert_allclose(sph.norm(p, v), geo.norm(p, v), atol=1e-10)
            w = rng.standard_normal(batch + (3,))
            np.testing.assert_allclose(
                sph.to_tangent(p, w), geo.to_tangent(p, w), atol=1e-12
            )
        p, q = unit((3,)), unit((3,))
        np.testing.assert_allclose(
            sph.distance(RiemannianPoint(sph, p), RiemannianPoint(sph, q)),
            geo.distance(RiemannianPoint(geo, p), RiemannianPoint(geo, q)),
            atol=1e-12,
        )

    def test_exp_projects_radial_component(self):
        """exp must project a non-tangent input onto the tangent space (as geomstats
        does), so a radial component does not push the result off the sphere."""
        sph = create_sphere(2)
        p = np.array([1.0, 0.0, 0.0])
        tangent = np.array([0.0, 0.4, 0.3])
        moved = sph.exp(p, tangent + 0.5 * p)  # add a radial component
        np.testing.assert_allclose(np.linalg.norm(moved), 1.0, atol=1e-12)
        np.testing.assert_allclose(moved, sph.exp(p, tangent), atol=1e-12)


class TestManifoldGuarantees:
    """The advertised manifold guarantees hold (or refuse loudly)."""

    def test_brownian_motion_refuses_on_hyperboloid(self):
        """No metric-isotropic frame is known there: refuse, don't bias."""
        space = create_hyperbolic_space(dim=2)
        origin = np.array([1.0, 0.0, 0.0])  # hyperboloid base point
        center = space.center_from_point(RiemannianPoint(space, origin))

        with pytest.raises(NotImplementedError, match="random_tangent"):
            center.brownian_motion(0.1)

    def test_frechet_mean_update_exact_on_sphere(self):
        """The Karcher mean of two symmetric points is their geodesic midpoint."""
        from kmeanssa_ng import KarcherFrechetMean

        space = RiemannianManifold(Hypersphere(dim=2))

        def unit(v):
            v = np.asarray(v, dtype=float)
            return v / np.linalg.norm(v)

        points = [
            RiemannianPoint(space, unit([0.3, 0.0, 1.0])),
            RiemannianPoint(space, unit([-0.3, 0.0, 1.0])),
        ]
        mean = KarcherFrechetMean().update(points, space)

        north = RiemannianPoint(space, np.array([0.0, 0.0, 1.0]))
        assert space.distance(mean, north) < 1e-8

    def test_point_membership_is_validated(self):
        """Off-manifold, non-finite or mis-shaped coordinates are rejected."""
        space = RiemannianManifold(Hypersphere(dim=2))

        with pytest.raises(ValueError, match="belong"):
            RiemannianPoint(space, np.array([2.0, 0.0, 0.0]))  # off the sphere

        with pytest.raises(ValueError, match="finite"):
            RiemannianPoint(space, np.array([np.nan, 0.0, 1.0]))

        with pytest.raises(ValueError, match="shape"):
            RiemannianPoint(space, np.array([1.0, 0.0]))
