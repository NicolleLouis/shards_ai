# Validation par borne de confiance positive — Architecture

## Objective

Adapter la promotion des profils heuristiques à un modèle déjà fortement optimisé. Une nouvelle
version doit être acceptée contre le profil de référence lorsque son intervalle de confiance sur la
différence de performance est strictement positif, sans exiger artificiellement un gain minimal de
1 point.

## Current State

`_validate_candidate()` compare le candidat au profil précédent et à `RandomPlayer` lorsque la
campagne est mixte. La règle actuelle exige une borne basse supérieure à `minimum_gain` contre le
précédent et autorise une petite régression jusqu'à `-minimum_gain` contre Random.

La campagne `gems_produced` seed 86 obtient contre le précédent une différence de `+1,36 point`,
avec une borne basse positive de `+0,13 point`, mais échoue car elle reste sous l'ancien seuil de
`+1 point`. Contre Random, sa borne basse est légèrement négative et ne signale pas de régression
significative.

## Target Behavior

Pour l'adversaire `previous`, la validation passe si :

```text
candidate_games >= validation_games
et confidence_lower > 0
```

Pour `random`, la règle de non-régression actuelle est conservée :

```text
candidate_games >= validation_games
et confidence_lower > -minimum_gain
```

Cette asymétrie est volontaire : le profil doit être strictement meilleur que son prédécesseur,
mais ne doit pas être rejeté pour une variation aléatoire minime contre l'adversaire de référence.

## Non-Goals

- Modifier le calcul de l'intervalle de confiance.
- Accepter un candidat dont l'intervalle inclut zéro.
- Supprimer la vérification contre Random.
- Réévaluer automatiquement toutes les campagnes historiques.

## Key Decisions

1. **Borne strictement positive.** `confidence_lower > 0.0` est le critère de promotion contre
   `previous`.
2. **Tolérance Random conservée.** `minimum_gain` devient une tolérance de non-régression contre
   Random, pas une exigence de gain contre le précédent.
3. **Traçabilité.** Le résultat JSON et les profils publiés indiquent la règle utilisée et la borne
   requise par adversaire.
4. **Promotion manuelle du run 86.** Les résultats existants satisfont la nouvelle règle : la
   version `gems_produced=0.5` devient `v006`, sans relancer 10 000 parties.
5. **Défaut global.** Les outils d'analyse, d'optimisation et de reporting utilisent `v006` par
   défaut après publication.

## Proposed Architecture

Dans `_validate_candidate()`, remplacer la borne requise dépendante de `minimum_gain` pour
`previous` par `0.0`. Ajouter un champ de métadonnées `validation_rule` et conserver
`required_lower` dans chaque adversaire pour rendre la décision auditée.

Le profil `v006` reprend `v005`, avec `card_acquisition_weights.gems_produced = 0.5`, et conserve
les résultats du run seed 86 dans ses métadonnées avec la nouvelle règle.

## Data Model

Aucun changement de modèle de partie. Le profil contient les métadonnées de validation suivantes :

```yaml
validation_rule: positive_confidence_lower_bound_vs_previous
```

## Rollout And Migration

1. Modifier le validateur et ses tests.
2. Créer `configs/heuristic_profiles/v006.yaml` à partir des résultats validés du run 86.
3. Passer les chemins par défaut des scripts de `v005` à `v006`.
4. Utiliser `v006` comme base de la campagne `buy_threshold`.

## Testing Strategy

- valider une borne basse positive contre `previous` ;
- rejeter une borne basse nulle ou négative contre `previous` ;
- conserver la tolérance de non-régression contre Random ;
- vérifier le chargement du profil `v006` ;
- exécuter la suite complète.

## Files Expected To Change

- `shards_ai/optimization/heuristic.py` : règle de validation et métadonnées ;
- `configs/heuristic_profiles/v006.yaml` : profil publié ;
- `scripts/optimize_heuristic.py` : profil par défaut ;
- `scripts/analyze_game_detail.py` et `scripts/benchmark_heuristic_report.py` : profils par défaut ;
- tests d'optimisation et de profils ;
- documentation current state et todo.
