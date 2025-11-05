# TODO: Finaliser la documentation des paramètres

## Fichiers modifiés

1. **`src/kmeanssa_ng/core/simulated_annealing.py`** - Docstring de `__init__()`
2. **`docs-src/simulated-annealing.qmd`** - Nouvelle section "Algorithm Parameters"

## TODO à compléter

### Dans le code (`simulated_annealing.py`)

#### Ligne 76 et 93 : Ajouter la référence à votre article
```python
TODO: Add reference to article on HAL/ArXiv when available.
```

Remplacer par :
```python
See [1] for theoretical justification and detailed analysis.
```

#### Ligne 110 : Documenter energy_mode
```python
energy_mode: Energy calculation mode, either "uniform" or "obs".
    TODO: Document the difference between these modes.
```

À compléter avec :
- Explication de "uniform" vs "obs"
- Quand utiliser chacun
- Impact sur les performances

#### Lignes 117-119 : Ajouter la référence complète
```python
References:
    TODO: Add reference to your article:
    [1] Your Name. "Title of your paper". HAL/ArXiv, 2025.
        URL: https://...
```

### Dans la documentation (`docs-src/simulated-annealing.qmd`)

#### Ligne 117 : Guidance spécifique par problème
```html
<!-- TODO: Add specific guidance based on problem characteristics (graph size, number of clusters, etc.) -->
```

Suggestions :
- Pour graphes < 100 nœuds : ...
- Pour graphes > 1000 nœuds : ...
- Pour k petit (< 5) : ...
- Pour k grand (> 20) : ...

#### Ligne 143 : Interaction lambda0/beta0
```html
<!-- TODO: Discuss the interaction between lambda0 and beta0 -->
```

Points à aborder :
- Le ratio lambda0/beta0 contrôle l'équilibre exploration/exploitation
- lambda0 élevé + beta0 élevé : exploration rapide puis convergence forte
- lambda0 faible + beta0 faible : recherche lente mais douce

#### Ligne 167 : Vérification du step_size
```html
<!-- TODO: Add guidance on how to check if step_size is appropriate -->
```

Suggestions :
- Observer la stabilité de l'énergie
- Comparer les résultats avec step_size/2
- Si pas de différence significative, step_size est OK

#### Ligne 203-205 : Résultats empiriques et tuning
```html
<!-- TODO: Add empirical results showing performance with different parameter settings -->
<!-- TODO: Add section on parameter tuning strategies -->
<!-- TODO: Discuss computational cost vs. quality tradeoffs -->
```

Idées :
- Tableau de résultats sur des benchmarks
- Stratégie de recherche en grille
- Graphiques temps de calcul vs qualité

#### Lignes 211-213 : Référence HAL/ArXiv
```html
<!-- TODO: Add your HAL/ArXiv reference here -->
<!-- [1] Your Name. "Title of your paper". HAL/ArXiv, 2025. -->
<!--     URL: https://... -->
```

À remplacer par la référence complète quand l'article sera publié.

## Vérifications suggérées

### Valeurs à confirmer/ajuster

Les plages de valeurs suggérées sont basées sur l'intuition :
- **lambda0**: [0.3, 0.8] (faible), 1.0 (défaut), [1.5, 3.0] (élevé)
- **beta0**: [0.3, 0.8] (faible), [1.0, 2.0] (défaut), [2.0, 5.0] (élevé)
- **step_size**: [0.001, 0.01] (petit), 0.01 (défaut), [0.05, 0.1] (grand)

→ **À vérifier avec vos expériences et votre article**

### Tests empiriques recommandés

1. Tester les trois configurations suggérées (Quick/Thorough/Balanced) sur vos benchmarks
2. Documenter les performances observées
3. Ajuster les recommandations si nécessaire

## Prochaines étapes

1. ✅ Structure mise en place avec TODO
2. ⬜ Compléter les TODO quand l'article sera prêt
3. ⬜ Valider les plages de valeurs empiriquement
4. ⬜ Ajouter des exemples concrets avec résultats
5. ⬜ Régénérer la documentation : `quarto render docs-src/`
