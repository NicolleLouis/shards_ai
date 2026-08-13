# Forecast d'horizon par classes — Architecture

## Objective

Remplacer la régression du nombre exact de tours actifs restants par une classification adaptée à
la décision d'achat : `T0`, `T1`, `T2`, `T3`, `T4`, `T5` et `T6+`.

L'objectif est de savoir si la partie est dans les cinq derniers tours, et plus particulièrement
si elle est dans la zone `T0` à `T2`, plutôt que de prédire une valeur numérique moyenne peu utile.

## Current State

L'architecture 087 a produit un baseline `turn_number` et une V1 enrichie. Sur le test, la V1
régressait vers une moyenne d'environ deux tours dans la zone courte et distinguait mal `T0` et
`T1`. La régression exacte n'est donc plus adaptée au calcul de la pénalité d'achat.

## Target Behavior

Le générateur encode la cible :

```python
horizon_class = min(remaining_active_player_turns, 6)
```

avec le mapping stable :

```text
0 -> T0
1 -> T1
2 -> T2
3 -> T3
4 -> T4
5 -> T5
6 -> T6+
```

Le baseline utilise uniquement `turn_number`. V1 conserve les features d'état et les huit comptes
factionnels globaux de l'architecture 087.

Chaque modèle renvoie sept logits, transformables en probabilités par softmax. Les indicateurs
destinés à la future pénalité sont :

```python
p_le_2 = P(T0) + P(T1) + P(T2)
p_le_5 = sum(P(T0), ..., P(T5))
```

## Key Decisions

1. La régression est supprimée du module, de l'entraînement et du rapport ; aucune sortie numérique
   exacte n'est conservée comme objectif.
2. La classification comporte exactement sept classes ordonnées.
3. L'entraînement utilise une cross-entropy pondérée par classe, avec des poids inversement
   proportionnels à la racine carrée de la fréquence d'entraînement, afin que `T0` et `T1` ne soient
   pas ignorées au profit de `T6+`.
4. La comparaison reste appariée : mêmes parties, mêmes splits par `game_id`, mêmes seeds et même
   pondération pour baseline et V1.
5. Les métriques prioritaires sont la matrice de confusion, le rappel/précision par classe,
   l'exactitude à une classe près, `T0–T2` et `T6+`. L'accuracy globale seule est insuffisante.
6. Les artefacts de la régression 087 sont historiques et ne sont pas réutilisés comme modèles de
   classification. Une nouvelle génération du dataset est obligatoire.

## Non-Goals

- modifier le joueur V005 ou appliquer immédiatement la pénalité d'achat ;
- prédire un nombre exact de tours ;
- ajouter une nouvelle information cachée dans l'observation ;
- transformer les classes en seuil de décision avant l'évaluation complète.

## Proposed Architecture

`shards_ai/ai/horizon_forecast.py` contient un `HorizonClassifier` avec une sortie de dimension 7,
la conversion target, l'entraînement et les métriques. Le générateur écrit
`target_horizon_class` dans le dataset canonique et la projection baseline conserve cette cible.

Le rapport de training contient pour chaque modèle :

- accuracy et balanced accuracy ;
- accuracy à une classe près ;
- précision, rappel et support par classe ;
- rappel de `T0–T2` et de `T6+` ;
- matrice de confusion ;
- Brier score pour `P(T<=2)` et `P(T<=5)`.

## Rollout And Migration

La campagne se relance avec `scripts/run_horizon_forecast.sh`. Les checkpoints et rapports dans
`artifacts/horizon_forecast/models/` sont remplacés par les nouveaux artefacts de classification.
La campagne 087 reste disponible dans l'historique Git ou dans une sauvegarde externe, mais ses
checkpoints ne sont pas compatibles avec le nouveau chargeur.

## Testing Strategy

- vérifier le mapping `0..5,6+` ;
- vérifier les dimensions logits/probabilités ;
- vérifier la projection baseline ;
- vérifier la matrice de confusion et les métriques de groupes courts ;
- vérifier les poids de classes et des logits finis ;
- exécuter la campagne smoke puis les tests ciblés.
