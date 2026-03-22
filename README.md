# Factor Portfolio Pipeline

Ce projet fournit un squelette propre en Python pour une chaîne complète de gestion de portefeuille quantitative :

1. **estimation d’un alpha** avec un modèle linéaire simple,
2. **estimation d’un modèle de risque factoriel** avec OLS,
3. **construction d’une matrice de covariance factorielle**,
4. **optimisation convexe du portefeuille** avec `cvxpy`.

Le fichier principal est : `factor_portfolio_pipeline.py`

---

## 1. Architecture générale

Le code est organisé autour de trois classes :

```python
class AlphaModel:
    def fit(self, X, y): ...
    def predict(self, X): ...

class FactorRiskModel:
    def fit(self, returns, factors): ...
    def expected_covariance(self): ...
    def exposures(self): ...

class PortfolioOptimizer:
    def solve(self, mu, Sigma, exposures, constraints): ...
```

### Rôle de chaque bloc

- **`AlphaModel`** : apprend un signal attendu par actif, noté en général \( \mu_i \) ou \( \alpha_i \).
- **`FactorRiskModel`** : estime les expositions factorielles \( \beta \), le risque spécifique, puis la covariance de portefeuille.
- **`PortfolioOptimizer`** : transforme rendement attendu + risque + contraintes en **poids optimaux**.

---

## 2. Formules utilisées

### 2.1 Modèle d’alpha

Le modèle d’alpha utilisé dans ce squelette est une régression linéaire simple :

\[
y = X\theta + \varepsilon
\]

où :

- \( X \) = matrice de features,
- \( y \) = cible de rendement,
- \( \theta \) = coefficients appris,
- \( \varepsilon \) = bruit.

Avec constante, cela devient :

\[
y_i = c + \sum_{j=1}^{p} \theta_j x_{i,j} + \varepsilon_i
\]

Puis la prédiction d’alpha est :

\[
\hat{\alpha}_i = \hat{c} + \sum_{j=1}^{p} \hat{\theta}_j x_{i,j}
\]

---

### 2.2 Modèle factoriel de rendement

Pour chaque actif \( i \), on suppose :

\[
r_{i,t} = c_i + \beta_i^{\top} f_t + \varepsilon_{i,t}
\]

avec :

- \( r_{i,t} \) = rendement de l’actif \( i \) au temps \( t \),
- \( f_t \) = vecteur des facteurs au temps \( t \),
- \( \beta_i \) = exposition factorielle de l’actif \( i \),
- \( c_i \) = intercept,
- \( \varepsilon_{i,t} \) = risque spécifique.

Dans le cas Fama-French 3 facteurs, on a souvent :

\[
r_{i,t} = c_i + \beta_{i,MKT} \cdot MKT_t + \beta_{i,SMB} \cdot SMB_t + \beta_{i,HML} \cdot HML_t + \varepsilon_{i,t}
\]

---

### 2.3 Estimation OLS

L’estimation OLS cherche à minimiser la somme des résidus au carré :

\[
\min_{\theta} \sum_{t=1}^{T} (y_t - x_t^{\top}\theta)^2
\]

La solution fermée théorique est :

\[
\hat{\theta} = (X^{\top}X)^{-1}X^{\top}y
\]

En pratique, `statsmodels` se charge de la résolution numérique.

---

### 2.4 Covariance factorielle

Une fois les expositions estimées, on construit la covariance des actifs avec :

\[
\Sigma = B\Sigma_f B^{\top} + D
\]

avec :

- \( B \in \mathbb{R}^{N \times K} \) = matrice des expositions factorielles,
- \( \Sigma_f \in \mathbb{R}^{K \times K} \) = covariance des facteurs,
- \( D \in \mathbb{R}^{N \times N} \) = matrice diagonale du risque spécifique.

Cette décomposition dit :

- une partie du risque vient des **facteurs communs**,
- une autre partie vient du **bruit propre à chaque actif**.

---

### 2.5 Rendement attendu du portefeuille

Pour un vecteur de poids \( w \), le rendement espéré est :

\[
\mathbb{E}[R_p] = \mu^{\top} w
\]

avec :

- \( \mu \) = rendement attendu par actif,
- \( w \) = poids du portefeuille.

---

### 2.6 Risque du portefeuille

Le risque quadratique du portefeuille est :

\[
\mathrm{Var}(R_p) = w^{\top} \Sigma w
\]

et sa volatilité est :

\[
\sigma_p = \sqrt{w^{\top} \Sigma w}
\]

---

### 2.7 Problème d’optimisation

L’objectif implémenté est de type **mean-variance** :

\[
\max_w \; \mu^{\top} w - \lambda \, w^{\top} \Sigma w
\]

avec :

- \( \mu^{\top} w \) = rendement espéré,
- \( w^{\top}\Sigma w \) = risque,
- \( \lambda > 0 \) = aversion au risque.

---

### 2.8 Contraintes possibles

Le code permet de gérer les contraintes suivantes.

#### Budget / exposition nette

\[
\sum_{i=1}^{N} w_i = c
\]

- `c = 1` pour un portefeuille net long classique,
- `c = 0` pour un portefeuille dollar-neutral.

#### Bornes par actif

Long only :

\[
0 \leq w_i \leq w_{max}
\]

Long/short borné :

\[
|w_i| \leq w_{max}
\]

#### Gross exposure

\[
\sum_{i=1}^{N} |w_i| \leq G
\]

#### Neutralité factorielle

Pour un facteur donné \( b \) :

\[
b^{\top} w = \tau
\]

- \( \tau = 0 \) : neutralité stricte,
- \( \tau \neq 0 \) : exposition cible.

#### Pénalité de turnover

Si \( w^{prev} \) est le portefeuille précédent, on peut pénaliser les changements :

\[
\text{penalty} = \eta \lVert w - w^{prev} \rVert_1
\]

L’objectif devient alors :

\[
\max_w \; \mu^{\top} w - \lambda w^{\top}\Sigma w - \eta \lVert w - w^{prev} \rVert_1
\]

---

## 3. Explication détaillée du code

---

### 3.1 `AlphaModel`

Cette classe sert à apprendre un signal d’alpha à partir de features.

#### Attributs

- `add_intercept` : ajoute une constante à la régression.
- `results_` : objet `statsmodels` contenant les résultats OLS.
- `feature_names_` : noms des features utilisées à l’entraînement.

#### `fit(X, y)`

Cette méthode :

1. vérifie les types (`DataFrame` pour `X`, `Series` pour `y`),
2. aligne `X` et `y` sur les mêmes index,
3. nettoie les `NaN` et `Inf`,
4. ajoute une constante si demandé,
5. entraîne un modèle OLS via `statsmodels.OLS(...).fit()`.

Le but est d’apprendre une relation entre les variables explicatives et une cible de rendement.

#### `predict(X)`

Cette méthode :

1. vérifie que le modèle a bien été entraîné,
2. ne garde que les features vues pendant `fit`,
3. ajoute la constante si nécessaire,
4. renvoie un `pd.Series` appelé `alpha`.

Le résultat peut être utilisé comme **rendement attendu par actif** dans l’optimisation.

---

### 3.2 `FactorRiskModel`

Cette classe estime un modèle de risque factoriel actif par actif.

#### Principe

Pour chaque actif, le code ajuste la relation :

\[
r_{i,t} = c_i + \beta_i^{\top}f_t + \varepsilon_{i,t}
\]

Cela donne :

- un intercept par actif,
- une exposition par facteur,
- une variance spécifique.

#### Attributs

- `intercepts_` : intercept estimé pour chaque actif,
- `beta_df_` : table des expositions factorielles,
- `residual_var_` : variance des résidus par actif,
- `factor_cov_` : covariance empirique des facteurs,
- `asset_names_` et `factor_names_` : noms utiles.

#### `fit(returns, factors)`

Cette méthode :

1. aligne les dates entre les rendements et les facteurs,
2. supprime les lignes contenant des valeurs manquantes,
3. construit la matrice explicative `X` à partir des facteurs,
4. boucle sur chaque actif,
5. entraîne une régression OLS,
6. extrait les coefficients et la variance spécifique.

Concrètement, si `returns` a `N` colonnes, le code effectue `N` régressions séparées.

#### `expected_covariance()`

Cette méthode construit :

\[
\Sigma = B\Sigma_f B^{\top} + D
\]

Étapes internes :

1. lit les expositions `B`,
2. lit la covariance des facteurs `Sigma_f`,
3. construit la diagonale `D` à partir des variances spécifiques,
4. calcule la covariance totale des actifs.

Le résultat est un `DataFrame` carré de taille `(N, N)`.

#### `exposures()`

Cette méthode renvoie simplement la matrice des expositions factorielles, utile pour :

- analyser le portefeuille,
- imposer des contraintes de neutralité,
- calculer les expositions agrégées du portefeuille.

---

### 3.3 `PortfolioOptimizer`

Cette classe résout l’optimisation avec `cvxpy`.

#### Pourquoi `cvxpy` ?

Parce que l’objectif est quadratique et les contraintes sont affines ou convexes :

- terme linéaire en rendement,
- terme quadratique en risque,
- contraintes de somme,
- contraintes de bornes,
- contrainte L1 sur le turnover ou la gross exposure.

C’est exactement le cadre naturel d’un problème convexe de portefeuille.

#### Attribut

- `solver` : solveur utilisé, par défaut `OSQP`.

#### `solve(mu, Sigma, exposures, constraints)`

##### Entrées

- `mu` : `Series` des rendements attendus,
- `Sigma` : matrice de covariance des actifs,
- `exposures` : matrice des expositions factorielles,
- `constraints` : dictionnaire d’options.

##### Étapes internes

1. aligne `Sigma` et `exposures` sur les actifs de `mu`,
2. lit les hyperparamètres (`risk_aversion`, `max_weight`, etc.),
3. crée une variable d’optimisation `w`,
4. construit l’objectif
   \(
   \mu^\top w - \lambda w^\top\Sigma w
   \),
5. ajoute éventuellement la pénalité de turnover,
6. ajoute les contraintes,
7. résout le problème avec `cvxpy`,
8. renvoie les poids sous forme de `pd.Series`.

##### Contraintes prises en charge

- `net_exposure`
- `max_weight`
- `long_only`
- `gross_limit`
- `w_prev`
- `turnover_penalty`
- `factor_targets`

##### Exemple d’interprétation

Si on met :

```python
constraints={
    "risk_aversion": 5.0,
    "net_exposure": 1.0,
    "max_weight": 0.25,
    "long_only": False,
    "gross_limit": 2.0,
    "factor_targets": {"MKT": 0.0},
}
```

cela signifie :

- portefeuille net investi à 100%,
- autorisation de positions longues et courtes,
- poids absolu maximum de 25% par actif,
- exposition brute totale limitée à 200%,
- neutralité au facteur marché `MKT`.

---

## 4. Bloc `__main__` : exemple complet

Le bas du fichier fournit un exemple minimal exécutable.

### Étape 1 — Simulation de facteurs

```python
factors = pd.DataFrame(...)
```

On génère des facteurs artificiels `MKT`, `SMB`, `HML`.

### Étape 2 — Simulation de rendements d’actifs

```python
returns = pd.DataFrame(
    factors.values @ true_betas.T.values + specific_noise,
    ...
)
```

Ici, les rendements sont construits comme :

\[
R = F B^{\top} + \text{bruit spécifique}
\]

### Étape 3 — Simulation des features d’alpha

```python
X_alpha = pd.DataFrame(...)
y_alpha = pd.Series(...)
```

On fabrique un petit problème de prédiction supervisée.

### Étape 4 — Entraînement du modèle d’alpha

```python
alpha_model = AlphaModel().fit(X_alpha, y_alpha)
alpha = alpha_model.predict(X_alpha)
```

### Étape 5 — Estimation du modèle de risque

```python
risk_model = FactorRiskModel().fit(returns, factors)
Sigma = risk_model.expected_covariance()
expo = risk_model.exposures()
```

### Étape 6 — Optimisation

```python
optimizer = PortfolioOptimizer(solver="OSQP")
weights = optimizer.solve(...)
```

Puis le code affiche :

- l’alpha prédit,
- les expositions factorielles,
- les poids optimisés,
- la somme des poids,
- la gross exposure.

---

## 5. Comment utiliser ce squelette avec de vraies données

### 5.1 Alpha

Remplacer `X_alpha` et `y_alpha` par de vraies variables, par exemple :

- momentum,
- mean reversion,
- volumes,
- microstructure features,
- signaux ML.

### 5.2 Rendements actifs

`returns` doit être une matrice `(T, N)` avec :

- index = dates,
- colonnes = actifs,
- valeurs = rendements.

### 5.3 Facteurs

`factors` doit être une matrice `(T, K)` avec :

- index = mêmes dates que `returns`,
- colonnes = facteurs comme `MKT`, `SMB`, `HML`, `MOM`, etc.

### 5.4 Optimisation

Tu peux ensuite ajouter :

- coûts de transaction,
- neutralité sectorielle,
- limites de liquidité,
- turnover maximum,
- contraintes par univers,
- budget long/short séparé.

---

## 6. Limites actuelles du squelette

Ce code est volontairement pédagogique. Il manque encore plusieurs éléments d’une version hedge fund production-grade :

- estimateurs robustes de covariance,
- shrinkage de covariance,
- robustesse aux outliers,
- rolling window / walk-forward,
- backtest complet,
- coûts de transaction réalistes,
- contraintes sectorielles et de liquidité,
- prévision factorielle plus élaborée.

---

## 7. Références

### Théorie des facteurs

1. **Fama, E. F., & French, K. R. (1993)**, *Common risk factors in the returns on stocks and bonds*.
2. **Kenneth R. French Data Library** — description et construction des facteurs Fama-French.

### OLS / régression

3. **statsmodels documentation** — `statsmodels.regression.linear_model.OLS`
4. **statsmodels documentation** — régression linéaire et résultats OLS.

### Optimisation convexe

5. **CVXPY documentation** — quadratic program example.
6. **CVXPY documentation** — examples et solveurs.

---

## 8. Pistes d’amélioration

Quelques extensions naturelles :

- remplacer `AlphaModel` par `Ridge`, `Lasso`, `XGBoost`, `LightGBM`,
- utiliser des excess returns \( R_i - R_f \),
- ajouter `Newey-West` ou estimateurs robustes,
- faire des betas roulants,
- faire une optimisation multi-période,
- intégrer un backtest complet.

---

## 9. Résumé en une phrase

Le pipeline implémente la logique suivante :

\[
\text{features} \rightarrow \text{alpha} \rightarrow \text{betas / covariance} \rightarrow \text{optimisation} \rightarrow \text{poids du portefeuille}
\]

C’est une base claire pour construire un moteur quant plus avancé.
