---
title: Clustering Methods
---


This page introduces the $k$-means problem on a general metric space and
the two algorithms the library provides to solve it.

## The $k$-means problem

Classical $k$-means places $k$ centres to minimise the average squared
Euclidean distance to the nearest centre. On a metric space
$(\mathcal{M}, d)$ the same objective is written with the space’s own
distance,

$$
U(c_1, \ldots, c_k) = \int_{\mathcal{M}} \min_{j=1,\ldots,k} d^2(x, c_j)\, P(\mathrm{d}x),
$$

where $P$ is the probability measure being quantised (see
[Concepts](concepts.md) for why this is the right generalisation). The
centres $c_j$ live in $\mathcal{M}$, so minimising $U$ means *moving
centres on the space itself* — which is exactly what breaks the
Euclidean recipe of “average the assigned points”: on a graph or a
curved manifold, an average is not defined.

## Algorithms

The library’s **main algorithm** is **[Simulated
Annealing](simulated-annealing.md)** — the online scheme of the
companion paper. Centres explore the space by Brownian motion and drift
toward incoming observations, with a cooling schedule that lets them
escape poor local minima. It only asks the space for a Brownian motion
and a drift, so it runs on *any* space (graphs, manifolds, your own),
and it is the method used throughout this guide.

A classical **[Lloyd iteration](lloyd.md)** is also provided, as a
familiar **reference method**. It alternates assignment and
centre-update steps until it converges — fast, but it descends to the
nearest critical point of $U$, without the annealing’s ability to escape
local minima. And unlike Simulated Annealing, each update step must
*recompute* a centre (a Fréchet mean), so Lloyd needs an update rule
specific to the space.

For a first run, start with [Simulated
Annealing](simulated-annealing.md); reach for [Lloyd](lloyd.md) when you
want a classical baseline to compare against.
