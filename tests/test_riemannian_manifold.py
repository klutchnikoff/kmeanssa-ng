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
        assert isinstance(space.observations, list)

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
        assert len(space.observations) == n

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

        # Calculate energy
        energy = space.calculate_energy(centers)

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

        # Now, for the actual test, ensure the space has no observations
        space.observations = []

        with pytest.raises(ValueError, match="No observations available"):
            space.calculate_energy(centers)

    def test_calculate_energy_no_centers(self):
        """Test energy calculation fails without centers."""
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)
        space.sample_points(10, strategy=UniformManifoldSampling())

        with pytest.raises(ValueError, match="Centers list cannot be empty"):
            space.calculate_energy([])

    def test_calculate_energy_with_how_parameter(self):
        """Test energy calculation accepts 'how' parameter for API compatibility."""
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)
        points = space.sample_points(20, strategy=UniformManifoldSampling())

        sa = SimulatedAnnealing(points, k=3)
        centers = RandomInit().initialize_centers(sa)
        # Both calls should give the same result since 'how' is ignored
        energy_default = space.calculate_energy(centers)
        energy_obs = space.calculate_energy(centers, how="obs")
        energy_uniform = space.calculate_energy(centers, how="uniform")

        assert energy_default == energy_obs
        assert energy_default == energy_uniform  # 'uniform' is ignored
        assert isinstance(energy_default, float)
        assert energy_default >= 0

    def test_compute_clusters(self):
        """Test compute_clusters (stub implementation)."""
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)
        points = space.sample_points(20, strategy=UniformManifoldSampling())

        sa = SimulatedAnnealing(points, k=3)
        centers = RandomInit().initialize_centers(sa)

        # Should not raise an error
        space.compute_clusters(centers)

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
        initial_energy = sphere.calculate_energy(centers)
        assert initial_energy > 0

        # Perform some drift operations
        for center in centers:
            # Drift toward random observation
            target = np.random.choice(points)
            center.drift(target, prop_to_travel=0.1)

        # Calculate new energy
        new_energy = sphere.calculate_energy(centers)
        assert new_energy > 0

        # Energy should have changed
        assert new_energy != initial_energy

    def test_different_manifolds(self):
        """Test that the same operations work on different manifolds."""
        manifolds = [
            create_sphere(dim=2),
            create_sphere(dim=3),
            create_hyperbolic_space(dim=2),
        ]

        for space in manifolds:
            # Sample and compute
            points = space.sample_points(20, strategy=UniformManifoldSampling())

            sa = SimulatedAnnealing(points, k=3)
            centers = KMeansPlusPlus().initialize_centers(sa)
            energy = space.calculate_energy(centers)

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
