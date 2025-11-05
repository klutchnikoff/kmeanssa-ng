---
title: Advanced Usage
---


The `kmeanssa-ng` library is designed with extensibility in mind,
following a classic Strategy design pattern. This allows you to create
your own custom components, such as initialization or robustification
strategies, without altering the core package code.

## Implementing a Custom Initialization Strategy

The main algorithm, `SimulatedAnnealing`, accepts an
`initialization_strategy` object. This object must be an instance of a
class that inherits from the `InitializationStrategy` abstract base
class and implements its required methods.

Let’s demonstrate this by creating a custom strategy that initializes
centers at fixed, predefined locations.

### 1. Define the Custom Strategy

First, create a new class that inherits from
`kmeanssa_ng.core.strategies.initialization.InitializationStrategy`. You
must implement the `initialize_centers` method.

``` python
from kmeanssa_ng.core.strategies.initialization import InitializationStrategy
from kmeanssa_ng.core.simulated_annealing import SimulatedAnnealing

class MyFixedInit(InitializationStrategy):
    """Initializes centers at predefined positions."""
    def __init__(self, fixed_centers):
        super().__init__()
        self.fixed_centers = fixed_centers

    def initialize_centers(self, sa: SimulatedAnnealing):
        """Returns the list of predefined centers."""
        print("Using a custom fixed initialization strategy!")
        
        # It's good practice to ensure the number of centers is correct.
        if len(self.fixed_centers) != sa.k:
            raise ValueError(
                f"The number of predefined centers ({len(self.fixed_centers)}) "
                f"does not match the required number of clusters k={sa.k}."
            )
        
        return self.fixed_centers
```

### 2. Use the Custom Strategy

Now, you can instantiate your custom strategy and pass it to the
`run_interleaved` or `run_sequential` method of the `SimulatedAnnealing`
object.

``` python
# Example setup
from kmeanssa_ng import generate_sbm
from kmeanssa_ng.quantum_graph.sampling import UniformNodeSampling

# Generate space and data
my_space = generate_sbm(sizes=[50, 50], p=[[0.8, 0.1], [0.1, 0.8]])

observations = my_space.sample_points(200, strategy=UniformNodeSampling())
k = 4

# Create the predefined centers for our strategy
points_for_centers = my_space.sample_points(k, strategy=UniformNodeSampling())
predefined_centers = [my_space.center_from_point(p) for p in points_for_centers]

# Instantiate your custom strategy
my_strategy = MyFixedInit(predefined_centers)

# Run the algorithm with your strategy
sa = SimulatedAnnealing(observations, k=k)
final_centers = sa.run_interleaved(initialization_strategy=my_strategy)

print("\nClustering completed successfully using a custom strategy.")
```

    Using a custom fixed initialization strategy!

    Clustering completed successfully using a custom strategy.

This same principle applies to other strategic components in the
library, such as `RobustificationStrategy`, allowing for a high degree
of customization to fit your specific use case.
