# TODO: 3 Tâches Finales pour Publication Recherche/Enseignement

**Objectif** : Rendre le package complètement prêt pour la recherche et l'enseignement

**Estimation totale** : 3-4 heures

---

## 1. Propagation Complète du Generator (2-3 heures)

**Objectif** : Reproductibilité 100% end-to-end avec `random_state`

### Actuellement
- ✅ `SimulatedAnnealing` : shuffle et Poisson utilisent l'état global
- ❌ Strategies : utilisent encore `random` module directement
- ❌ Centers : Brownian motion utilise `random` module
- ❌ Spaces : certaines opérations utilisent `random` module

### À Modifier

#### **a) Initialization Strategies** (`core/strategies/initialization.py`)

**RandomInit** :
```python
class RandomInit(InitializationStrategy):
    def initialize(self, sa: SimulatedAnnealing) -> None:
        # AVANT :
        points = rd.choices(sa.observations, k=sa.k)

        # APRÈS :
        points = sa._rng.choice(sa.observations, size=sa.k, replace=False)
```

**KMeansPlusPlus** :
```python
class KMeansPlusPlus(InitializationStrategy):
    def initialize(self, sa: SimulatedAnnealing) -> None:
        # AVANT :
        first_point = rd.choice(sa.observations)
        # ...
        next_point = rd.choices(sa.observations, weights=probabilities, k=1)[0]

        # APRÈS :
        first_point = sa._rng.choice(sa.observations)
        # ...
        next_point = sa._rng.choice(sa.observations, p=probabilities)
```

**Fichiers à modifier** :
- `src/kmeanssa_ng/core/strategies/initialization.py`
- Remplacer `import random as rd` par accès à `sa._rng`

---

#### **b) Centers - Brownian Motion**

**QuantumGraph Center** (`quantum_graph/center.py`) :

Actuellement ligne 77, 243 :
```python
return rd.choice(best_neighbors)
next_node = rd.choice(list(self.space.neighbors(cur_node)))
```

**Solution 1 (Simple)** : Utiliser `np.random` global au lieu de `random`
```python
# En haut du fichier, changer :
import random as rd
# Par :
import numpy as np

# Puis :
return np.random.choice(best_neighbors)
next_node = np.random.choice(list(self.space.neighbors(cur_node)))
```

**Solution 2 (Idéale mais plus complexe)** : Passer le rng via les méthodes
```python
def brownian_motion(self, h: float, rng: np.random.Generator | None = None) -> None:
    if rng is None:
        rng = np.random.default_rng()
    # ... utiliser rng au lieu de rd
```

**Recommandation** : Solution 1 pour l'instant (rapide), Solution 2 pour perfection future

**Fichiers à modifier** :
- `src/kmeanssa_ng/quantum_graph/center.py` (lignes 5, 77, 243)
- `src/kmeanssa_ng/riemannian_manifold/center.py` (vérifier s'il y a du random)

---

#### **c) Spaces - Opérations Aléatoires**

**QuantumGraph Space** (`quantum_graph/space.py`) :

Lignes concernées : 5, 474, 587, 590, 606, 619, 669
```python
import random as rd
# ...
idx = rd.choice(all_idx)
nodes = rd.choices(list(self.nodes()), k=n)
neighbor = rd.choice(list(self.neighbors(node)))
```

**Solution** : Remplacer par `np.random` global
```python
import numpy as np
# ...
idx = np.random.choice(all_idx)
nodes = np.random.choice(list(self.nodes()), size=n, replace=True)
neighbor = np.random.choice(list(self.neighbors(node)))
```

**Fichiers à modifier** :
- `src/kmeanssa_ng/quantum_graph/space.py`
- `src/kmeanssa_ng/quantum_graph/sampling.py` (lignes 9, 64, 67, 137, 139, 214, 217)

---

#### **d) Generators** (peut rester inchangé si utilisé avant SA)

`quantum_graph/generators.py` utilise `random` mais c'est OK car appelé **avant** `SimulatedAnnealing.__init__()`.

Si besoin de reproductibilité complète :
```python
# L'utilisateur fait :
random.seed(42)
np.random.seed(42)
graph = generate_simple_graph(...)
points = graph.sample_points(...)
sa = SimulatedAnnealing(points, k=3, random_state=42)
```

---

### Tests à Ajouter/Modifier

**Nouveau test de reproductibilité end-to-end** :
```python
def test_full_reproducibility_with_random_state():
    """Test that random_state gives fully reproducible results."""
    import random, numpy as np

    # Setup
    random.seed(1000)
    np.random.seed(1000)
    graph = generate_simple_graph(n_a=3)
    points = graph.sample_points(30, strategy=UniformNodeSampling())

    # Run 1
    sa1 = SimulatedAnnealing(points, k=3, random_state=42)
    centers1 = sa1.run_interleaved(
        initialization_strategy=KMeansPlusPlus(),
        robustification_strategy=MinimizeEnergy(),
        robust_prop=0.1,
    )

    # Run 2 - même seed
    sa2 = SimulatedAnnealing(points, k=3, random_state=42)
    centers2 = sa2.run_interleaved(
        initialization_strategy=KMeansPlusPlus(),
        robustification_strategy=MinimizeEnergy(),
        robust_prop=0.1,
    )

    # Assert identical results
    assert len(centers1) == len(centers2)
    for c1, c2 in zip(centers1, centers2):
        assert c1.edge == c2.edge
        assert abs(c1.position - c2.position) < 1e-10
```

Ajouter dans `tests/test_simulated_annealing.py`

---

### Checklist

- [ ] Modifier `core/strategies/initialization.py` (RandomInit, KMeansPlusPlus)
- [ ] Modifier `quantum_graph/center.py` (remplacer `rd` par `np.random`)
- [ ] Modifier `quantum_graph/space.py` (remplacer `rd` par `np.random`)
- [ ] Modifier `quantum_graph/sampling.py` (remplacer `rd` par `np.random`)
- [ ] Vérifier `riemannian_manifold/center.py` (si du random)
- [ ] Ajouter test end-to-end dans `tests/test_simulated_annealing.py`
- [ ] Modifier `IDEAS/test_random_state.py` pour vérifier que Test 1 et 3 passent
- [ ] Lancer tous les tests : `pdm run pytest tests/ -v`
- [ ] Vérifier ruff : `pdm run ruff check src/`

---

## 2. Documentation energy_mode (30 minutes)

**Objectif** : Expliquer la différence entre "uniform" et "obs"

### Dans le Code

**Fichier** : `src/kmeanssa_ng/core/simulated_annealing.py`

**Ligne 110**, remplacer :
```python
energy_mode: Energy calculation mode, either "uniform" or "obs".
    TODO: Document the difference between these modes.
```

**Par** :
```python
energy_mode: Energy calculation mode, either "uniform" or "obs".
    Controls how the k-means energy is computed.

    - **"uniform"** (default): Classical k-means energy where each
      observation contributes equally. The energy is the average of
      squared distances from observations to their nearest centers.

      Formula: E = (1/n) Σᵢ min_j d²(xᵢ, cⱼ)

      Use this for standard k-means clustering on discrete datasets.

    - **"obs"**: Observation-weighted energy suitable for continuous
      probability distributions on quantum graphs. Each observation
      represents a probability density, and the energy accounts for
      the underlying measure on the space.

      [TODO: Ajouter la formule exacte depuis votre article]

      Use this when clustering probability distributions on metric
      graphs with non-uniform measures.

    For theoretical justification and detailed mathematical derivation,
    see Section X.X in [YourReference2025].
```

### Dans la Documentation

**Fichier** : `docs-src/simulated-annealing.qmd`

Ajouter une nouvelle section après "## Two Algorithm Variants" (ligne ~90) :

```markdown
## Energy Calculation Modes

The algorithm supports two energy calculation modes controlled by the
`energy_mode` parameter:

### Uniform Mode (Default)

Classical k-means energy where each observation contributes equally:

$$ E_{\text{uniform}} = \frac{1}{n} \sum_{i=1}^{n} \min_{j=1,\ldots,k} d^2(x_i, c_j) $$

**When to use**:
- Standard clustering of finite datasets
- Discrete observations with equal importance
- Classical k-means applications

### Obs Mode

Observation-weighted energy for continuous distributions:

[TODO: Ajouter la formule depuis votre article]

$$ E_{\text{obs}} = ... $$

**When to use**:
- Clustering probability distributions on quantum graphs
- Non-uniform measures on metric spaces
- [TODO: autres cas d'usage depuis l'article]

**Example**:
```python
# Standard k-means (default)
sa = SimulatedAnnealing(points, k=3, energy_mode="uniform")

# For probability distributions on graphs
sa = SimulatedAnnealing(points, k=3, energy_mode="obs")
```

For mathematical details, see Section X.X in [@YourReference2025].
```

### Checklist

- [ ] Modifier docstring dans `src/kmeanssa_ng/core/simulated_annealing.py`
- [ ] Ajouter section dans `docs-src/simulated-annealing.qmd`
- [ ] Compléter avec les formules de votre article
- [ ] Vérifier que ça compile : `quarto preview docs-src/`

---

## 3. Ajouter Référence HAL/ArXiv (15 minutes)

**Objectif** : Lier la documentation à votre article

### Emplacements des TODO

#### **a) Code - simulated_annealing.py**

**Ligne 76** (lambda0) :
```python
# AVANT :
TODO: Add reference to article on HAL/ArXiv when available.

# APRÈS :
See [1] for theoretical justification and convergence analysis.
```

**Ligne 93** (beta0) :
```python
# AVANT :
TODO: Add reference to article on HAL/ArXiv when available.

# APRÈS :
See [1] for theoretical justification and convergence analysis.
```

**Lignes 145-149** (References section) :
```python
# AVANT :
References:
    TODO: Add reference to your article:
    [1] Your Name. "Title of your paper". HAL/ArXiv, 2025.
        URL: https://...

# APRÈS :
References:
    [1] N. Klutchnikoff et al. "Titre de votre article".
        ArXiv preprint arXiv:XXXX.XXXXX, 2025.
        URL: https://arxiv.org/abs/XXXX.XXXXX
        HAL: https://hal.archives-ouvertes.fr/hal-XXXXXXX
```

#### **b) Documentation - simulated-annealing.qmd**

**Lignes 211-213** (Theoretical Foundation) :
```markdown
# AVANT :
<!-- TODO: Add your HAL/ArXiv reference here -->
<!-- [1] Your Name. "Title of your paper". HAL/ArXiv, 2025. -->
<!--     URL: https://... -->

# APRÈS :
**References**:

- Klutchnikoff, N., [autres auteurs] (2025). *Titre de votre article*.
  ArXiv preprint. [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)
- Version HAL : [hal-XXXXXXX](https://hal.archives-ouvertes.fr/hal-XXXXXXX)
```

#### **c) Documentation - TODO_documentation_parameters.md**

Mettre à jour le fichier `IDEAS/TODO_documentation_parameters.md` pour indiquer que cette tâche est faite.

### Format de Citation Standard

Pour cohérence, utilisez ce format partout :

```
Klutchnikoff, N., [Co-Author1], [Co-Author2] (2025).
"K-means Clustering on Quantum Graphs via Simulated Annealing".
ArXiv preprint arXiv:XXXX.XXXXX.
```

### Checklist

- [ ] Trouver/remplacer "TODO: Add reference" dans `src/kmeanssa_ng/core/simulated_annealing.py`
- [ ] Compléter références dans `docs-src/simulated-annealing.qmd`
- [ ] Compléter références dans `docs-src/concepts.qmd` (si mentionné)
- [ ] Mettre à jour `IDEAS/TODO_documentation_parameters.md`
- [ ] Vérifier que la doc compile : `quarto preview docs-src/`

---

## Validation Finale

Après les 3 tâches, vérifier :

### Tests
```bash
pdm run pytest tests/ -v
# Tous les tests doivent passer, y compris le nouveau test de reproductibilité
```

### Linting
```bash
pdm run ruff check src/
pdm run ruff format src/
```

### Documentation
```bash
cd docs-src
quarto preview
# Vérifier que tout s'affiche correctement
```

### Test Manuel de Reproductibilité
```bash
pdm run python IDEAS/test_random_state.py
# Test 1 et Test 3 doivent maintenant montrer "Identical results: True"
```

---

## Commit et Publication

Une fois les 3 tâches terminées :

```bash
git add -A
git commit -m "feat: complete reproducibility + energy_mode docs + article references

- Propagate random Generator to all components for 100% reproducibility
- Document energy_mode (uniform vs obs) with mathematical formulas
- Add HAL/ArXiv references throughout code and documentation

Package now ready for research and teaching use."

git push origin feature/improvements-critical
```

Puis créer la MR finale sur GitLab.

---

## Notes

- Ces 3 tâches rendent le package **suffisant** pour recherche et enseignement
- Les perfectionnements additionnels (exceptions custom, tests edge cases, etc.)
  peuvent être faits progressivement selon les besoins
- **Estimation totale : 3-4 heures**
- Ne pas hésiter à itérer après feedback des premiers utilisateurs
