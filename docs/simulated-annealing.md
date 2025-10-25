# Simulated Annealing


## General Principle

Simulated annealing is a probabilistic optimization technique inspired
by the metallurgical process of annealing, where metals are heated and
slowly cooled to reduce defects and reach a low-energy state.

The key idea is to allow the algorithm to:

- **Explore** the solution space through random movements (like thermal
  fluctuations)
- **Exploit** promising regions by moving toward better solutions
- **Gradually reduce randomness** over time (cooling), converging to a
  good solution

This balance between exploration and exploitation helps escape local
minima—a critical advantage over greedy algorithms like Lloyd’s method.

## Mechanisms in K-means Context

In `kmeanssa-ng`, cluster centers are dynamic entities that move through
the metric space using two complementary mechanisms:

#### Brownian Motion (Exploration)

Centers perform **random walks** in the metric space, characterized by:

- Random direction selection
- Distance traveled proportional to $\sqrt{\Delta t}$ (diffusion
  scaling)
- Allows escape from local minima through stochastic exploration

For a center $c$ and time parameter $\Delta t$:

$$ c \leftarrow c + \text{Brownian}(\Delta t) $$

#### Drift (Exploitation)

Centers are **pulled toward** the observations assigned to their
cluster:

- Each center drifts toward a randomly selected point in its cluster
- Distance traveled: proportion $\alpha$ of the geodesic distance
- Reduces cluster energy by moving centers closer to their observations

For a center $c$, target observation $x$, and drift proportion
$\alpha  [0,1]$:

$$ c \leftarrow c + \alpha  (x - c) $$

(where the notation is geometric; on graphs this means moving along the
geodesic path)

#### Temperature Schedule

The “temperature” controls the balance between exploration and
exploitation over time. `kmeanssa-ng` uses an **inhomogeneous Poisson
process** to generate a decreasing temperature schedule:

$$ T(n) = \sqrt{\sum_{i=1}^{n} E_i + 1} - 1 $$

where $E_i  \text{Exp}(\lambda)$ are exponential random variables.

Key properties:

- Temperature decreases over time (cooling)
- Controlled by parameter $\lambda$ (intensity)
- Stochastic schedule adds robustness

## Two Algorithm Variants

`kmeanssa-ng` implements two strategies for combining Brownian motion
and drift:

#### V1: Interleaved (Default)

At each iteration:

1.  Randomly select one observation $x_i$
2.  Perform Brownian motion on all centers:
    $c_j \leftarrow c_j + \text{Brownian}(\Delta t)$
3.  Find nearest center $c^*$ to $x_i$
4.  Apply drift: $c^* \leftarrow c^* + \alpha  (x_i - c^*)$

This approach **alternates** exploration and exploitation at a
fine-grained level.

#### V2: Sequential

Separates exploration and exploitation into distinct phases:

1.  **Brownian phase**: Iterate through all observations, performing
    only Brownian motion
2.  **Drift phase**: Iterate through all observations, performing only
    drift

This approach separates the two mechanisms temporally.

**Choice**: V1 (interleaved) is generally recommended as the default,
but V2 can be useful for specific problem structures.

## Robustification

To improve stability, `kmeanssa-ng` uses **robustification**: instead of
returning the final centers, it averages results from the last $p\%$ of
iterations (default: 10%). This reduces sensitivity to late-iteration
fluctuations.
