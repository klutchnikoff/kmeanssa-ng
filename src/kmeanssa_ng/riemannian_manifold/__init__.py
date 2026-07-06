"""Riemannian Manifold module for kmeanssa-ng."""

from .bolza import BolzaSurface
from .center import RiemannianCenter
from .epsilon_net import (
    EpsilonNetStrategy,
    FibonacciNet,
    RepulsionNet,
    RepulsionNetExtrinsicSpeedup,
    UniformNet,
)
from .generators import (
    create_bolza_surface,
    create_hyperbolic_space,
    create_sphere,
)
from .graph import (
    approximate_geodesic_space,
    build_epsilon_net_graph,
    estimate_covering_radius,
)
from .point import RiemannianPoint
from .space import RiemannianManifold, Sphere
from .update import KarcherFrechetMean

__all__ = [
    "RiemannianManifold",
    "Sphere",
    "BolzaSurface",
    "RiemannianPoint",
    "RiemannianCenter",
    "create_sphere",
    "create_hyperbolic_space",
    "create_bolza_surface",
    "KarcherFrechetMean",
    "EpsilonNetStrategy",
    "UniformNet",
    "RepulsionNet",
    "RepulsionNetExtrinsicSpeedup",
    "FibonacciNet",
    "approximate_geodesic_space",
    "build_epsilon_net_graph",
    "estimate_covering_radius",
]
