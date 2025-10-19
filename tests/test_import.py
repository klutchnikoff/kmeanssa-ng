"""Test basic imports and package structure."""

import kmeanssa_ng


def test_version():
    """Test that version is defined."""
    assert kmeanssa_ng.__version__ == "0.1.0"


def test_core_imports():
    """Test that core classes can be imported."""
    from kmeanssa_ng import Center, Point, SimulatedAnnealing, Space

    assert Point is not None
    assert Center is not None
    assert Space is not None
    assert SimulatedAnnealing is not None


def test_quantum_graph_imports():
    """Test that quantum graph classes can be imported."""
    from kmeanssa_ng import QGCenter, QGPoint, QuantumGraph

    assert QGPoint is not None
    assert QGCenter is not None
    assert QuantumGraph is not None


def test_generator_imports():
    """Test that generators can be imported."""
    from kmeanssa_ng import (
        as_quantum_graph,
        complete_quantum_graph,
        generate_random_sbm,
        generate_sbm,
        generate_simple_graph,
        generate_simple_random_graph,
    )

    assert generate_simple_graph is not None
    assert generate_simple_random_graph is not None
    assert generate_sbm is not None
    assert generate_random_sbm is not None
    assert as_quantum_graph is not None
    assert complete_quantum_graph is not None
