# kmeanssa-ng

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Pipeline Status](https://plmlab.math.cnrs.fr/nicolas.klutchnikoff/kmeanssa-ng/badges/main/pipeline.svg)](https://plmlab.math.cnrs.fr/nicolas.klutchnikoff/kmeanssa-ng/-/pipelines)
[![Coverage Report](https://plmlab.math.cnrs.fr/nicolas.klutchnikoff/kmeanssa-ng/badges/main/coverage.svg)](https://plmlab.math.cnrs.fr/nicolas.klutchnikoff/kmeanssa-ng/-/commits/main)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Documentation](https://img.shields.io/badge/docs-GitLab%20Pages-blue)](https://kmeanssa-ng-b901a8.pages.math.cnrs.fr/)
<!-- Future badges (uncomment when ready):
[![PyPI version](https://badge.fury.io/py/kmeanssa-ng.svg)](https://badge.fury.io/py/kmeanssa-ng)
-->

> **K-means clustering on quantum graphs and metric spaces using simulated annealing**

`kmeanssa-ng` enables clustering on **quantum graphs** — networks where data points can exist anywhere along edges, not just at vertices. Perfect for road networks, neural pathways, river systems, and any scenario where your data lives on a connected structure.

## ✨ Key Features

- 🌐 **Quantum Graph Support** - Points anywhere on edges, geodesic distances
- 🔥 **Simulated Annealing** - Robust k-means with Brownian motion + drift  
- 🎯 **Flexible Architecture** - Extend to any metric space
- ⚡ **Smart Initialization** - k-means++ for faster convergence
- 🛡️ **Robustification** - Stable results via iteration averaging
- 📝 **Fully Typed** - Modern Python with complete type hints

## 🚀 Quick Start

```python
from kmeanssa_ng import generate_sbm, SimulatedAnnealing

# Create quantum graph with 2 clusters
graph = generate_sbm(sizes=[50, 50], p=[[0.8, 0.1], [0.1, 0.8]])
graph.precomputing()  # Cache shortest paths

# Cluster 100 points with k-means++
points = graph.sample_points(100)
sa = SimulatedAnnealing(points, k=2)
centers = sa.run(initialization="kpp")
```

## 📖 Documentation

**[📚 Full Documentation →](https://kmeanssa-ng-b901a8.pages.math.cnrs.fr/)

- **[🎯 Quick Start Guide](https://kmeanssa-ng-b901a8.pages.math.cnrs.fr/getting-started/quickstart/)** - Get running in 5 minutes
- **[📐 Core Concepts](https://kmeanssa-ng-b901a8.pages.math.cnrs.fr/getting-started/concepts/)** - Quantum graphs & simulated annealing
- **[🔧 API Reference](https://kmeanssa-ng-b901a8.pages.math.cnrs.fr/api/core/)** - Complete API documentation
- **[💡 Examples](https://kmeanssa-ng-b901a8.pages.math.cnrs.fr/examples/basic-clustering/)** - Practical use cases
- **[🏗️ User Guides](https://kmeanssa-ng-b901a8.pages.math.cnrs.fr/user-guide/quantum-graphs/)** - Deep dives into features

## 📦 Installation

### From source (current)
```bash
git clone https://plmlab.math.cnrs.fr/nicolas.klutchnikoff/kmeanssa-ng.git
cd kmeanssa-ng && pdm install
```

### Future PyPI release
```bash
pip install kmeanssa-ng  # Coming soon
```

## 🏷️ Citation

```bibtex
@software{kmeanssa_ng,
  author = {Klutchnikoff, Nicolas},
  title = {kmeanssa-ng: K-means clustering on quantum graphs},
  year = {2025},
  url = {https://plmlab.math.cnrs.fr/nicolas.klutchnikoff/kmeanssa-ng}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Author:** Nicolas Klutchnikoff ([nicolas.klutchnikoff@univ-rennes2.fr](mailto:nicolas.klutchnikoff@univ-rennes2.fr))
