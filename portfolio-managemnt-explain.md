# Explication vulgarisee du projet `portfolio-management`

Ce projet est un mini-pipeline de gestion de portefeuille quantitatif.

L'idee generale est la suivante : on donne au programme des donnees sur plusieurs actifs, il essaie d'estimer lesquels ont le meilleur potentiel, il mesure les risques, puis il calcule une repartition optimale du capital entre ces actifs.

Ce n'est pas un outil de trading complet pret pour la production. C'est plutot un squelette propre qui montre les grandes briques classiques d'un systeme de portfolio management quantitatif.

## Ce que le projet contient

Le code principal est dans `main.py`.

Il contient trois grandes classes :

1. `AlphaModel`
2. `FactorRiskModel`
3. `PortfolioOptimizer`

A la fin du fichier, il y a aussi un exemple executable avec des donnees simulees.

## La logique globale

Le pipeline suit cette logique :

```text
Donnees d'entree
    -> estimation du rendement attendu
    -> estimation du risque
    -> optimisation du portefeuille
    -> poids finaux par actif
```

En langage plus concret :

- on cree ou on recoit une liste d'actifs, par exemple `Asset_0`, `Asset_1`, etc. ;
- on estime combien chaque actif pourrait rapporter ;
- on estime quels risques chaque actif apporte au portefeuille ;
- on demande a un optimiseur de choisir les poids ;
- le resultat final dit combien mettre dans chaque actif.

## 1. `AlphaModel` : essayer de predire le rendement

En finance quantitative, un `alpha` est une estimation du rendement supplementaire qu'on pense pouvoir obtenir.

Par exemple, si le modele dit :

```text
Asset_3 alpha = 0.025
Asset_7 alpha = -0.010
```

cela signifie que le modele pense que `Asset_3` est plus attractif que `Asset_7`.

Dans ce projet, l'alpha est estime avec une regression lineaire OLS.

Une regression lineaire, c'est simplement une formule du type :

```text
rendement attendu = constante
                  + coefficient_1 * feature_1
                  + coefficient_2 * feature_2
                  + ...
```

Les `features` sont des variables explicatives. Dans un vrai projet, ca pourrait etre :

- le momentum d'une action ;
- sa volatilite recente ;
- son ratio value ;
- un signal macro ;
- un signal alternatif.

Ici, dans l'exemple, les features sont simulees aleatoirement. Le but est de montrer le mecanisme, pas de produire un vrai signal de marche.

### Ce que fait `AlphaModel.fit(X, y)`

`fit` apprend la relation entre :

- `X` : les features des actifs ;
- `y` : le rendement cible qu'on aimerait predire.

Le modele nettoie aussi les donnees :

- il garde seulement les indices communs entre `X` et `y` ;
- il retire les valeurs infinies ;
- il retire les lignes avec des valeurs manquantes.

### Ce que fait `AlphaModel.predict(X)`

`predict` applique le modele appris sur de nouvelles features et renvoie une serie `alpha`.

Cette serie devient ensuite l'entree `mu` de l'optimiseur, c'est-a-dire le rendement attendu par actif.

## 2. `FactorRiskModel` : comprendre d'ou vient le risque

Un portefeuille n'est pas risque uniquement parce que chaque actif bouge.

Il est aussi risque parce que plusieurs actifs peuvent bouger ensemble pour les memes raisons.

Exemple :

- si tout le marche baisse, beaucoup d'actions baissent ensemble ;
- si les petites capitalisations souffrent, les actifs exposes a ce facteur souffrent ensemble ;
- si les actions value montent, les actifs exposes au facteur value peuvent monter ensemble.

Ces raisons communes sont appelees des `facteurs`.

Dans le code, les facteurs simules sont :

```text
MKT, SMB, HML
```

Ce sont des noms typiques des modeles Fama-French :

- `MKT` : exposition au marche global ;
- `SMB` : exposition aux petites capitalisations ;
- `HML` : exposition au style value contre growth.

Tu n'as pas besoin de connaitre Fama-French en detail pour comprendre le projet. L'idee importante est seulement : chaque actif a une sensibilite differente a chaque facteur.

### Exemple intuitif d'exposition

Si un actif a :

```text
MKT = 1.2
```

alors il reagit fortement au marche.

Si un autre actif a :

```text
MKT = 0.2
```

alors il est beaucoup moins sensible au marche.

Ces sensibilites sont appelees des `betas` ou expositions factorielles.

### Ce que fait `FactorRiskModel.fit(returns, factors)`

Pour chaque actif, le modele fait une regression :

```text
rendement de l'actif = constante
                     + beta_MKT * MKT
                     + beta_SMB * SMB
                     + beta_HML * HML
                     + bruit specifique
```

Le modele estime donc :

- l'intercept de chaque actif ;
- ses expositions aux facteurs ;
- sa variance specifique, c'est-a-dire le risque qui ne vient pas des facteurs ;
- la covariance des facteurs entre eux.

### Ce que fait `expected_covariance()`

Cette methode construit une matrice de covariance des actifs.

Dit simplement, cette matrice repond a la question :

> Quand tel actif bouge, est-ce que les autres ont tendance a bouger avec lui ?

C'est essentiel pour optimiser un portefeuille, parce que deux actifs risques individuellement peuvent parfois reduire le risque global s'ils ne bougent pas toujours dans le meme sens.

## 3. `PortfolioOptimizer` : choisir les poids

L'optimiseur recoit :

- `mu` : les rendements attendus, donc les alphas ;
- `Sigma` : la matrice de risque ;
- `exposures` : les expositions factorielles ;
- `constraints` : les contraintes du portefeuille.

Il cherche ensuite les poids `w`.

Un poids, c'est la part du portefeuille allouee a un actif.

Exemple :

```text
Asset_0 = 0.20
Asset_1 = 0.10
Asset_2 = -0.05
```

Cela veut dire :

- 20 % du portefeuille sur `Asset_0` ;
- 10 % sur `Asset_1` ;
- -5 % sur `Asset_2`, donc une position short.

## L'objectif de l'optimiseur

Le programme maximise une idee simple :

```text
rendement attendu - penalite de risque
```

Donc il cherche un compromis :

- prendre les actifs avec un bon alpha ;
- eviter de prendre trop de risque ;
- respecter les contraintes imposees.

Le parametre important est :

```python
"risk_aversion": 5.0
```

Plus ce nombre est eleve, plus le portefeuille devient prudent.

Plus ce nombre est faible, plus l'optimiseur accepte du risque pour aller chercher du rendement.

## Les contraintes implementees

Dans l'exemple de `main.py`, l'optimisation utilise :

```python
{
    "risk_aversion": 5.0,
    "net_exposure": 1.0,
    "max_weight": 0.25,
    "long_only": False,
    "gross_limit": 2.0,
    "factor_targets": {"MKT": 0.0},
}
```

### `net_exposure`

```python
"net_exposure": 1.0
```

La somme des poids doit valoir 1.

Exemple :

```text
0.30 + 0.40 + 0.30 = 1.00
```

Cela correspond a un portefeuille investi a 100 % en net.

### `max_weight`

```python
"max_weight": 0.25
```

Chaque actif est limite a 25 % du portefeuille en valeur absolue.

Comme `long_only` vaut `False`, les poids peuvent aller de `-0.25` a `+0.25`.

### `long_only`

```python
"long_only": False
```

Le portefeuille peut etre long/short.

Une position `long` signifie qu'on achete l'actif.

Une position `short` signifie qu'on parie sur sa baisse ou qu'on vend l'actif a decouvert.

Si `long_only` etait `True`, tous les poids devraient etre positifs ou nuls.

### `gross_limit`

```python
"gross_limit": 2.0
```

La gross exposure est la somme des valeurs absolues des positions.

Exemple :

```text
poids = 0.80, 0.40, -0.30
gross exposure = |0.80| + |0.40| + |-0.30| = 1.50
```

Dans le projet, la gross exposure maximale est fixee a 2.0.

Cela autorise donc un peu de levier, mais pas sans limite.

### `factor_targets`

```python
"factor_targets": {"MKT": 0.0}
```

Cette contrainte demande que l'exposition totale du portefeuille au facteur marche `MKT` soit egale a 0.

Intuitivement, le portefeuille essaie d'etre neutre au marche.

Cela veut dire que le portefeuille ne doit pas gagner ou perdre principalement parce que le marche global monte ou baisse. Il doit plutot dependre des choix relatifs entre actifs.

## Ce que fait l'exemple executable

Quand tu lances :

```bash
python main.py
```

le script :

1. cree 252 jours de donnees simulees ;
2. cree 10 actifs fictifs ;
3. cree 3 facteurs fictifs : `MKT`, `SMB`, `HML` ;
4. simule les rendements des actifs ;
5. simule des features pour le modele d'alpha ;
6. entraine le modele d'alpha ;
7. estime le modele de risque factoriel ;
8. calcule la matrice de covariance ;
9. optimise le portefeuille ;
10. affiche les alphas, les expositions factorielles et les poids optimises.

## Ce que le projet affiche

Le script imprime trois blocs principaux.

### `Alpha predit`

Ce sont les rendements attendus par actif selon le modele.

Plus l'alpha est eleve, plus l'actif est considere attractif par le modele.

### `Expositions factorielles estimees`

Ce tableau montre la sensibilite de chaque actif aux facteurs `MKT`, `SMB` et `HML`.

Cela sert a mesurer et controler les risques communs.

### `Poids optimises`

Ce sont les allocations finales choisies par l'optimiseur.

Un poids positif signifie une position long.

Un poids negatif signifie une position short.

Le script affiche aussi :

```text
Somme des poids
Gross exposure
```

Ces deux lignes permettent de verifier que les contraintes principales sont respectees.

## Lien avec les options, calls et puts

Ce projet ne valorise pas directement des options.

Il ne calcule pas :

- le prix d'un call ;
- le prix d'un put ;
- les greeks ;
- une volatilite implicite ;
- une strategie optionnelle.

Mais il est quand meme lie a la finance de marche.

La difference est que ce projet travaille au niveau portefeuille :

- quels actifs acheter ou shorter ;
- combien mettre sur chaque actif ;
- comment controler le risque global ;
- comment eviter une exposition trop forte a un facteur.

Si on voulait etendre ce projet aux options, on pourrait remplacer les actifs simples par des options ou ajouter des contraintes sur les greeks :

- delta ;
- gamma ;
- vega ;
- theta.

Par exemple, on pourrait demander un portefeuille delta-neutral, comme ici on demande une neutralite au facteur `MKT`.

## Personal Project

J'ai implemente un mini-systeme de gestion de portefeuille quantitatif en Python.

Le but du projet est de montrer comment un portefeuille peut etre construit de maniere plus methodique qu'un simple choix manuel d'actifs. Le programme estime d'abord quels actifs semblent les plus interessants, puis il mesure les risques communs entre eux, par exemple leur sensibilite au marche. Ensuite, il utilise un optimiseur mathematique pour proposer une repartition du capital qui cherche a maximiser le rendement attendu tout en limitant le risque.

En pratique, le projet reproduit les grandes etapes d'un processus utilise en asset management quantitatif :

- construction d'un signal de rendement attendu ;
- estimation du risque via des facteurs de marche ;
- calcul d'une matrice de covariance ;
- optimisation des poids du portefeuille sous contraintes ;
- controle de l'exposition au marche et du levier.

Les donnees utilisees dans l'exemple sont simulees, donc le projet n'est pas fait pour etre branche directement sur les marches. Son interet est pedagogique : il montre comment relier la prediction, la gestion du risque et la construction finale d'un portefeuille dans un meme pipeline.

## En resume

Ce projet implemente une chaine quantitative simple :

```text
Alpha = ce que je pense pouvoir gagner
Risque = ce que je peux perdre ou subir comme variation
Contraintes = les regles que je m'impose
Optimiseur = la machine qui choisit les poids
```

La sortie finale du projet est donc une allocation de portefeuille : une liste de poids indiquant quelle part mettre dans chaque actif.

Le projet est surtout pedagogique. Il montre les briques fondamentales d'un portfolio manager quantitatif, mais avec des donnees simulees et un modele volontairement simple.
