# Reward shaping de deckbuilding basé sur v008

## Objective

Ajouter au training PPO un signal dense sur les décisions de deckbuilding afin de mieux créditer
les achats et les bannissements qui améliorent la qualité du deck avant la récompense terminale.
La première version utilise une table fixe de valeurs dérivée de l'évaluation d'acquisition des
cartes par l'heuristique v008.

Le signal doit rester auxiliaire : la victoire ou la défaite terminale demeure l'objectif principal.

## Current State

`shards_ai/ai/rl_training.py` ne renseigne actuellement que la dernière transition d'une partie
avec `+1`, `-1` ou `0`. Le PPO utilise déjà `gamma=1.0` et peut propager ce signal avec GAE.

Le profil `configs/heuristic_profiles/v008.yaml` contient les coefficients d'acquisition utilisés
par `shards_ai/ai/heuristic_features.py`, mais aucune table fixe par identifiant de carte n'est
conservée pour le RL.

`GameRunner` permet d'observer l'état avant et après chaque action. L'observation de shaping actuelle
est volontairement minimale et ne contient pas encore les zones du deck du joueur.

## Target Behavior

Pour chaque transition du joueur neural correspondant à une décision de deckbuilding, calculer :

```text
reward = terminal_reward + beta * (potential(after) - potential(before))
```

La première fonction de potentiel est la moyenne des valeurs fixes des cartes possédées par le
joueur neural. Elle couvre les cartes dans la pioche, la main, la défausse, la zone de jeu et les
champions. Les actions de jeu ordinaires conservent un shaping nul car elles ne modifient pas la
qualité du deck.

Les actions de deckbuilding ciblées sont `BuyCard`, `RecruitFreeCard`, `RecruitMercenary`,
`GainMastery`, `BanishCard`, `SkipBanish` et `StopBuying`. La valeur moyenne est bornée et le
coefficient `beta` est faible afin d'éviter qu'un gain de moyenne ne remplace l'objectif de victoire.

## Non-Goals

- Apprendre immédiatement la valeur des cartes.
- Remplacer l'évaluation contextuelle complète de v008 dans l'heuristique.
- Modifier les observations accessibles au réseau neural.
- Ajouter un shaping sur les attaques ou les cartes jouées.
- Promouvoir automatiquement le checkpoint issu de cette expérience.

## Key Decisions

- La table initiale est dérivée des coefficients `card_acquisition_weights` de v008 et générée
  une fois dans `configs/neural_training_profiles/card_values_v008.yaml`.
- Le calcul de la table utilise une situation neutre reproductible ; il ne dépend donc pas d'une
  partie particulière et ne fuit pas d'information cachée dans l'observation du réseau.
- Le potentiel est une moyenne, plutôt qu'une somme, afin qu'un bannissement d'une carte sous la
  moyenne puisse produire un signal positif de thinning.
- Les transitions restent conservées pour le critic. Le shaping est ajouté à la reward de la
  transition choisie, avant GAE.
- Le premier essai démarre depuis le checkpoint stable v001, avec `gamma=1.0`, `gae_lambda=1.0`
  et un `beta` configurable.

## Open Questions

- La table fixe reflète-t-elle suffisamment les synergies et les dépendances de maîtrise ?
- Le signal favorise-t-il une réduction excessive du deck ou un banish prématuré ?
- Quelle valeur de `beta` améliore v008 sur une validation indépendante ?
- Une future table apprise doit-elle être une tête auxiliaire du critic ou un modèle séparé ?

## Proposed Architecture

`shards_ai/ai/card_value_shaping.py` chargera la table v008, calculera la valeur moyenne d'un
`GameState` pour un joueur et décidera si une action est éligible au shaping. Le collecteur PPO
utilisera `GameRunner.transition_observer` avec `Game.shaping_observation_for` enrichi des zones
connues du joueur neural.

Le shaping sera paramétré dans le profil PPO et sérialisé dans les métadonnées du checkpoint afin
que les runs soient reproductibles. Les paramètres seront aussi surchargeables en ligne de commande
pour les campagnes comparatives.

## Edge Cases

- Un deck vide a une moyenne nulle.
- Une carte inconnue de la table provoque une erreur de configuration, pas une valeur silencieuse.
- Les actions qui ne modifient pas les zones de cartes ont un delta nul.
- Le delta est borné avant multiplication par `beta`.
- Les cartes mercenaires recrutées immédiatement ne doivent pas être interprétées comme des cartes
  durables si elles ne restent pas dans les zones du deck.

## Testing Strategy

- Vérifier le chargement complet et déterministe de la table v008.
- Vérifier qu'un achat d'une carte au-dessus de la moyenne produit un delta positif.
- Vérifier que bannir une carte sous la moyenne produit un delta positif.
- Vérifier que jouer une carte ou attaquer produit un delta nul.
- Vérifier la reproductibilité séquentielle/parallèle du rollout avec shaping activé.
- Valider le candidat sur un panel indépendant d'au moins 200 parties par adversaire avant toute
  promotion.

## Rollout And Migration

Le shaping est désactivé par défaut pour préserver le comportement des profils existants. Une
campagne dédiée part de `configs/neural_profiles/v001.pt`, écrit uniquement dans
`artifacts/neural_training/checkpoint.pt`, puis est comparée à v001 et aux meilleurs candidats
précédents. Le fichier stable v001 n'est jamais modifié.

## Files Expected To Change

- `shards_ai/ai/card_value_shaping.py`
- `shards_ai/ai/rl_training.py`
- `shards_ai/game/game.py`
- `configs/neural_training_profiles/candidates/v002.yaml`
- `configs/neural_training_profiles/card_values_v008.yaml`
- `scripts/generate_card_value_table.py`
- `tests/ai/test_card_value_shaping.py`
- `tests/ai/test_rl_training.py`
