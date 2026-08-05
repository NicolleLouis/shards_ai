# Fine-tuning PPO conservateur du NeuralPlayer v002

## Objective

Éviter que le fine-tuning victoire/défaite éloigne progressivement la politique v001 de ses
comportements utiles, notamment l'achat de cartes, tout en augmentant la performance contre v008.
Le training reste un apprentissage par récompense terminale, mais devient conservateur et observable.

## Current State

`scripts/train_neural_rl.py` collecte des rollouts PPO contre Random, v007 et v008, puis met à jour
le checkpoint mutable `artifacts/neural_training/checkpoint.pt`. La politique v002 peut dériver de
v001 malgré un learning rate réduit. Le training logge la performance de la politique stochastique
du rollout, mais ne valide pas régulièrement la politique gloutonne ni ne retient explicitement son
meilleur état.

## Target Behavior

Chaque update PPO v002 doit :

1. collecter uniquement des récompenses terminales `+1`, `-1` ou `0` ;
2. limiter la dérive de la politique par rapport à une copie gelée de v001 ;
3. utiliser un coefficient d'entropie réduit et un mélange orienté vers v008 ;
4. évaluer périodiquement la politique gloutonne contre Random, v007 et v008 ;
5. conserver dans le checkpoint mutable le meilleur état selon la moyenne des taux de victoire par
   adversaire, sans créer de second checkpoint mutable.

## Non-Goals

- introduire un reward shaping de santé, maîtrise, achats ou dégâts ;
- réentraîner ou modifier un checkpoint stable sous `configs/neural_profiles/` ;
- changer l'architecture `NeuralActionScorer` ;
- promouvoir automatiquement v002 ;
- conserver un fichier `best.pt` parallèle au checkpoint canonique.

## Key Decisions

- **Référence gelée :** la copie actor-critic initialisée depuis `initial_checkpoint` v001 reste en
  mode évaluation pendant toute la campagne.
- **Régularisation :** ajouter `reference_kl_coefficient * KL(pi_reference || pi_current)` à la
  loss PPO, calculée sur toutes les actions légales de chaque transition.
- **Exploration :** passer `entropy_coefficient` de `0.01` à `0.001` pour limiter la dérive vers une
  politique trop diffuse.
- **Mix :** utiliser `20 %` Random, `30 %` v007 et `50 %` v008 afin de donner davantage de signal à
  l'adversaire cible tout en conservant les deux références plus faciles.
- **Validation périodique :** tous les `evaluation_interval_games` jeux, évaluer la politique
  gloutonne sur `evaluation_games` seeds fixes par adversaire. Le score est la moyenne des trois
  taux de victoire, afin que Random ne masque pas une régression v008.
- **Meilleur état :** conserver en mémoire les états modèle/optimiseur ayant le meilleur score ; en
  cas de régression, restaurer cet état et continuer depuis lui. Le checkpoint écrit reste toujours
  `artifacts/neural_training/checkpoint.pt`.

## Open Questions

- **Non-blocking :** calibrer après validation le coefficient KL et la taille du panel d'évaluation.
- **Non-blocking :** décider ultérieurement si une régularisation comportementale ciblée sur les
  achats est nécessaire malgré la KL vers v001.

## Proposed Architecture

`NeuralActorCritic` reçoit une référence optionnelle dans `ppo_update`. Les log-probabilités de la
référence sont calculées sans gradient avant les passes PPO, puis la KL de chaque transition est
ajoutée à la loss courante. La référence n'est jamais modifiée par l'optimiseur.

`evaluate_greedy_model` exporte temporairement les poids actor compatibles avec
`NeuralActionScorer`, construit un `NeuralPlayer` glouton et joue des parties bornées contre les
trois adversaires. Les seeds d'évaluation sont fixes et indépendants des seeds de collecte.

Le trainer conserve les états modèle et optimiseur du meilleur score en mémoire. Après chaque
évaluation, il écrit le meilleur état courant au chemin canonique avec les métriques cumulées. Une
reprise restaure cet état unique et poursuit le budget `total_games`.

## Testing Strategy

- tester que le reward terminal reste positif pour une victoire et négatif pour une défaite ;
- tester que la loss KL est nulle lorsque politique et référence sont identiques ;
- tester que le coefficient KL contribue positivement à la loss en cas de divergence ;
- tester que l'évaluation utilise les mêmes seeds et retourne un score par adversaire ;
- smoke test du trainer avec et sans régularisation ;
- exécuter la suite complète et vérifier la compatibilité du checkpoint avec `NeuralPlayer`.

## Files Expected To Change

- `configs/neural_training_profiles/candidates/v002.yaml` ;
- `shards_ai/ai/rl_training.py` ;
- `scripts/train_neural_rl.py` ;
- `tests/ai/test_rl_training.py` ;
- `doc/Current state/Neural player.md`.
