# Benchmark contrefactuel des états visités par le NeuralPlayer

## Objective

Mesurer l’écart réel entre un NeuralPlayer et l’heuristique v008 sur les états que le NeuralPlayer
rencontre effectivement en partie. À chaque décision du NeuralPlayer contre v008, calculer sans
modifier la partie l’action que v008 aurait choisie avec les mêmes actions légales.

Le rapport doit fournir :

- l’accord top-1 Neural/v008 sur ces états visités ;
- le regret heuristique `H(s,a_v008) - H(s,a_NN)` ;
- le rang de l’action v008 dans le classement des scores neural ;
- les métriques par phase et par `action_type` ;
- le numéro de la première divergence dans chaque partie.

## Current State

`GameRunner.run` expose un `decision_observer` appelé avant `Game.apply`. Le callback reçoit
l’observation du joueur, les actions légales et l’action choisie ; la closure peut lire l’état de
jeu avant transition. `HeuristicPlayer.choose_action` est déterministe pour v008 et peut donc être
appelé contrefactuellement sur cet état sans appliquer son action.

Le benchmark neural mix existant mesure principalement les victoires et les statistiques finales.
Il ne permet pas de distinguer une divergence de politique au premier tour d’une divergence tardive.

## Target Behavior

Pour chaque seed, le script joue Neural contre v008. À chaque décision du NeuralPlayer uniquement :

1. conserver l’état et les actions légales avant application ;
2. demander à v008 son action sur ces mêmes objets ;
3. scorer les mêmes actions avec le scoreur neural et classer les actions ;
4. enregistrer l’accord, le rang v008, le regret et le type de décision ;
5. mémoriser la première divergence de la partie.

Le calcul contrefactuel ne joue jamais l’action v008 : la trajectoire reste exactement celle du
NeuralPlayer contre l’adversaire v008.

## Non-Goals

- Modifier la trajectoire de jeu ou faire jouer deux branches concurrentes.
- Mesurer une victoire contrefactuelle de v008.
- Remplacer le benchmark de parties existant.
- Utiliser les états du dataset d’imitation comme approximation des états visités.

## Key Decisions

- Les états analysés sont exclusivement ceux visités par le NeuralPlayer dans les parties simulées.
- L’accord top-1 compare les objets `Action` exacts, pas seulement leur `action_type`.
- Le rang v008 est le rang de son action selon les logits neural, avec départage stable par l’ordre
  des actions légales. Le top-1 de la partie reste fondé sur l’action effectivement retournée par
  le NeuralPlayer, y compris son départage aléatoire des égalités.
- Le regret utilise `HeuristicPlayer.score_action` sur l’action v008 et l’action neural ; il mesure
  le coût selon la fonction heuristique, pas directement la différence de victoire.
- Les compteurs sont agrégés par phase et `action_type`; les premières divergences sont conservées
  par partie avec seed et numéro de décision neural.
- Le scoreur neural est chargé une fois et réutilisé entre les parties.

## Data Model

Le JSON contient la configuration, les agrégats `overall`, `by_phase`, `by_action_type`, et une
entrée `first_divergence_by_game` par partie. Chaque groupe contient `records`, `top1_agreement`,
`top3_agreement`, `mean_heuristic_regret`, `mean_heuristic_score`, `mean_heuristic_rank` et
`divergence_rate`.

## Proposed Architecture

Créer `shards_ai/analysis/visited_state_benchmark.py` pour les agrégateurs et le rendu HTML, et
`scripts/benchmark_neural_visited_states.py` pour le CLI et l’orchestration des parties. Le moteur
reste la source de vérité ; le benchmark utilise seulement `decision_observer` et une lecture de
l’état avant transition.

## Performance And Risks

Chaque décision neural nécessite un second forward neural pour obtenir les scores et un calcul
heuristique v008 pour toutes les actions. Le coût est proportionnel aux décisions réellement
jouées et reste séparé du benchmark de production. Une partie ne doit pas être ralentie par une
branche de simulation complète.

Le rang et le regret sont conditionnels à la trajectoire du NeuralPlayer : ils ne mesurent pas ce
qui se serait passé si v008 avait choisi ses actions. C’est précisément la propriété recherchée
pour localiser la première divergence causale apparente, mais pas une preuve de causalité sur le
résultat final.

## Testing Strategy

- tester l’accord, le top-3, le rang et le regret sur une fixture déterministe ;
- tester qu’une seule première divergence est conservée par partie ;
- tester que le callback n’applique jamais l’action contrefactuelle ;
- smoke test d’une partie Neural contre v008 avec sortie JSON/HTML.

## Files Expected To Change

- `shards_ai/analysis/visited_state_benchmark.py`
- `scripts/benchmark_neural_visited_states.py`
- `tests/analysis/test_visited_state_benchmark.py`
- `Makefile`
- `doc/Current state/Neural player.md`
