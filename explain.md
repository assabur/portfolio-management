# Explication vulgarisee du projet

Ce projet montre comment construire un portefeuille quantitatif de facon methodique.

L'idee n'est pas de choisir des actifs "au feeling". Le programme suit plutot une chaine logique :

```text
1. Estimer ce qui peut rapporter
2. Mesurer les risques
3. Imposer des regles de gestion
4. Calculer les poids du portefeuille
```

Le resultat final est une liste de poids. Chaque poids dit quelle part du portefeuille mettre sur un actif.

Exemple :

```text
Asset_0 =  0.20  -> 20 % du portefeuille
Asset_1 =  0.10  -> 10 % du portefeuille
Asset_2 = -0.05  -> position short de 5 %
```

Le code principal est dans `main.py`.

## Vue simple

Le projet repond a une question centrale :

> Comment repartir mon capital entre plusieurs actifs en cherchant un bon rendement, sans prendre n'importe quel risque ?

Pour y repondre, il separe le probleme en trois blocs :

- `AlphaModel` : estime quels actifs semblent interessants.
- `FactorRiskModel` : mesure les risques et les dependances entre actifs.
- `PortfolioOptimizer` : choisit les poids finaux en respectant des contraintes.

## 1. AlphaModel : trouver les actifs attractifs

En finance quantitative, un `alpha` est une estimation du rendement attendu d'un actif.

Dit simplement :

```text
alpha eleve  -> actif potentiellement interessant
alpha faible -> actif moins interessant
alpha negatif -> actif potentiellement a eviter ou a shorter
```

Dans ce projet, l'alpha est estime avec une regression lineaire.

Une regression lineaire essaie d'apprendre une relation du type :

```text
rendement attendu = constante
                  + coefficient_1 * feature_1
                  + coefficient_2 * feature_2
                  + ...
```

Les `features` sont des informations disponibles sur les actifs.

Dans un vrai projet, les features pourraient etre :

- le momentum d'une action ;
- sa volatilite recente ;
- un ratio de valorisation ;
- un indicateur macroeconomique ;
- un signal alternatif.

Dans ce repo, les donnees sont simulees. Le but est pedagogique : montrer la mecanique du pipeline.

## 2. FactorRiskModel : comprendre le risque

Un portefeuille ne depend pas seulement du risque de chaque actif pris separement.

Deux actifs peuvent bouger ensemble parce qu'ils sont sensibles aux memes forces de marche. Ces forces communes sont appelees des `facteurs`.

Dans l'exemple, les facteurs sont :

```text
MKT, SMB, HML
```

Interpretation simplifiee :

- `MKT` : sensibilite au marche global.
- `SMB` : sensibilite aux petites capitalisations.
- `HML` : sensibilite au style value/growth.

Chaque actif a une exposition differente a ces facteurs. Cette exposition est souvent appelee `beta`.

Exemple :

```text
Asset_A a un beta MKT de 1.2 -> il reagit fortement au marche
Asset_B a un beta MKT de 0.2 -> il reagit peu au marche
```

Le modele de risque estime donc :

- les expositions des actifs aux facteurs ;
- le risque specifique de chaque actif ;
- la covariance entre actifs.

La covariance est importante parce qu'elle indique si les actifs ont tendance a bouger ensemble.

## 3. Matrice de covariance : la carte du risque

La matrice de covariance est une grande table qui resume les relations de risque entre actifs.

Elle aide a repondre a des questions comme :

- Est-ce que deux actifs montent et baissent souvent ensemble ?
- Est-ce qu'un actif peut compenser partiellement le risque d'un autre ?
- Est-ce que le portefeuille est trop concentre sur le meme type de risque ?

Le projet la construit avec une formule classique de modele factoriel :

```text
risque total = risque venant des facteurs + risque specifique des actifs
```

Cette separation est utile parce qu'elle permet de mieux comprendre d'ou vient le risque.

## 4. PortfolioOptimizer : choisir les poids

L'optimiseur recoit trois informations principales :

- `mu` : les rendements attendus, produits par `AlphaModel`.
- `Sigma` : la matrice de covariance, produite par `FactorRiskModel`.
- `constraints` : les regles que le portefeuille doit respecter.

Il cherche ensuite les poids `w`.

Son objectif est :

```text
maximiser le rendement attendu - penaliser le risque
```

En langage simple :

```text
Je veux gagner le plus possible,
mais je ne veux pas prendre trop de risque pour le faire.
```

Le parametre `risk_aversion` controle ce compromis.

- Plus `risk_aversion` est eleve, plus le portefeuille devient prudent.
- Plus `risk_aversion` est faible, plus le portefeuille accepte du risque.

## Les contraintes du portefeuille

Dans `main.py`, l'exemple utilise notamment :

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

### net_exposure

`net_exposure` fixe la somme des poids.

Avec :

```python
"net_exposure": 1.0
```

la somme des poids doit valoir 1. Le portefeuille est donc investi a 100 % en net.

### max_weight

`max_weight` limite la taille maximale d'une position.

Avec :

```python
"max_weight": 0.25
```

aucun actif ne peut depasser 25 % du portefeuille en valeur absolue.

### long_only

Si `long_only` vaut `True`, tous les poids doivent etre positifs ou nuls.

Si `long_only` vaut `False`, le portefeuille peut contenir des positions short.

Une position short signifie que le portefeuille profite si l'actif baisse.

### gross_limit

La `gross exposure` est la somme des valeurs absolues des poids.

Exemple :

```text
poids = 0.80, 0.40, -0.30
gross exposure = |0.80| + |0.40| + |-0.30| = 1.50
```

Cette contrainte limite le levier total du portefeuille.

### factor_targets

`factor_targets` permet de controler l'exposition du portefeuille a certains facteurs.

Dans l'exemple :

```python
"factor_targets": {"MKT": 0.0}
```

Le portefeuille cherche a etre neutre au facteur marche.

Intuition :

```text
Le portefeuille ne doit pas principalement gagner ou perdre parce que tout le marche monte ou baisse.
Il doit plutot dependre des choix relatifs entre actifs.
```

## Ce que fait le script quand on le lance

Quand on execute :

```bash
python main.py
```

le script :

1. cree des dates de marche simulees ;
2. cree 10 actifs fictifs ;
3. simule 3 facteurs de risque ;
4. simule les rendements des actifs ;
5. cree des features pour l'alpha ;
6. entraine le modele d'alpha ;
7. estime le modele de risque factoriel ;
8. construit la matrice de covariance ;
9. optimise les poids ;
10. affiche les alphas, les expositions et les poids finaux.

## Ce que le projet n'est pas

Ce projet n'est pas encore un systeme de trading complet.

Il ne fait pas :

- de recuperation de donnees de marche reelles ;
- de backtest historique complet ;
- de gestion des frais de transaction reels ;
- de passage d'ordres ;
- de suivi de performance en production.

C'est un squelette pedagogique. Il montre les briques fondamentales d'un processus de portfolio management quantitatif.

## Resume mental

On peut retenir le projet comme ceci :

```text
AlphaModel
    -> "Quels actifs semblent prometteurs ?"

FactorRiskModel
    -> "Quels risques prennent ces actifs ?"

PortfolioOptimizer
    -> "Combien mettre sur chaque actif ?"
```

La sortie finale est une allocation de portefeuille : une liste de poids qui cherche a combiner rendement attendu, controle du risque et contraintes de gestion.
