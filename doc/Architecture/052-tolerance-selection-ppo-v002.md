# Tolérance minimale de sélection PPO v002

## Objective

Éviter que la sélection périodique rejette systématiquement un candidat qui progresse contre v008
pour une variation d'une seule partie contre Random ou v007, tout en conservant la validation finale
comme barrière stricte avant promotion.

## Current State

Le trainer compare les évaluations gloutonnes périodiques au meilleur état retenu. Toute baisse,
même d'une victoire sur 64, contre Random, v007 ou v008 entraîne une restauration. Les évaluations
intermédiaires peuvent donc montrer un meilleur score pondéré sans jamais être conservées.

## Target Behavior

Pendant la sélection périodique uniquement :

- Random et v007 acceptent une baisse maximale d'une victoire sur le panel d'évaluation ;
- v008 ne bénéficie d'aucune tolérance et ne doit pas régresser ;
- le score pondéré doit strictement progresser ;
- la validation indépendante de `validate_neural_profile.py` reste inchangée et sans tolérance avant
  promotion.

La tolérance est exprimée comme `1 / evaluation_games`, afin de rester cohérente si la taille du
panel change.

## Non-Goals

- modifier la loss PPO, le reward ou le mix d'adversaires ;
- accepter une régression contre v008 ;
- modifier la règle stricte de promotion ;
- créer un checkpoint parallèle.

## Key Decisions

- La tolérance est limitée aux adversaires de référence plus faciles (`random`, `v007`).
- Une amélioration pondérée est toujours obligatoire.
- La sélection est une heuristique d'exploration ; seule la validation finale décide de la promotion.

## Testing Strategy

- accepter une baisse d'une victoire contre v007 si le score pondéré progresse ;
- rejeter toute baisse contre v008 ;
- rejeter une baisse de plus d'une victoire contre Random ou v007 ;
- conserver les tests de validation finale stricte.

## Files Expected To Change

- `shards_ai/ai/rl_training.py` ;
- `scripts/train_neural_rl.py` ;
- `tests/ai/test_rl_training.py` ;
- `README.md` ;
- `doc/Current state/Neural player.md`.
