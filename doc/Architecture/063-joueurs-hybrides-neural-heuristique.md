# Joueurs hybrides Neural / Heuristic V8

## Objective

Mesurer, en partie réelle, quelles familles de décisions dégradent le plus la
performance du joueur neuronal actuel. L'expérience oppose le NeuralPlayer à
cinq adversaires répartis exactement à 20 % : NeuralPlayer, Heuristic V8 et
trois joueurs hybrides.

## Current State

`NeuralPlayer` reçoit une observation masquée et la liste des actions légales,
calcule un score par action puis sélectionne le maximum. `HeuristicPlayer`
reçoit l'état complet détaché et applique les poids du profil V8. Le
`GameRunner` choisit l'observation selon `observation_kind`; le moteur valide
ensuite l'action dans tous les cas.

## Target Behavior

Les trois joueurs hybrides jouent normalement avec le réseau, sauf pour la
famille de décisions ciblée :

1. `purchase_recruitment` : achats, recrutements et sortie de la phase d'achat
   (`BuyCard`, `RecruitMercenary`, `RecruitFreeCard`, `StopBuying`) sont choisis
   par Heuristic V8.
2. `play_phase` : toute décision prise pendant `Phase.PLAY` est choisie par
   Heuristic V8, afin d'inclure les cartes, champions, maîtrise, banishment et
   passage de phase lorsqu'ils apparaissent dans cette phase.
3. `banish` : toute décision dont les alternatives contiennent un
   `BanishCard` est choisie par Heuristic V8.

Le benchmark produit un JSON détaillé et un rapport HTML synthétique. Les
parties sont affectées aux cinq adversaires par rotation déterministe, ce qui
impose un nombre de parties multiple de cinq.

## Non-Goals

- Modifier les règles du moteur ou les actions légales.
- Réentraîner le réseau.
- Prétendre à une causalité parfaite : chaque intervention peut modifier la
  trajectoire des états ultérieurs. Il s'agit d'une ablation comportementale.

## Key Decisions

- Les hybrides sont des adaptateurs de joueur, pas des variantes du moteur.
- Leur observation publique est un `GameState` pour permettre l'heuristique;
  le délégué neuronal reconstruit ensuite l'observation masquée via le même
  objet `Game`.
- Le score et les poids utilisés sont ceux passés explicitement au benchmark
  (checkpoint de travail et profil Heuristic V8).
- Le NeuralPlayer testé est toujours le joueur évalué; le type d'opposant
  varie uniquement selon le calendrier 20 %.
- Les graines et les rôles des joueurs restent déterministes et sont conservés
  dans la sortie JSON.

## Open Questions

- Non bloquant : une seconde campagne avec permutation des rôles et plusieurs
  graines pourra réduire l'effet de la position de départ.
- Non bloquant : une analyse ultérieure pourra comparer les premières
  divergences, en plus du taux de victoire.

## Proposed Architecture

`HybridPlayer` possède un `NeuralPlayer` et un `HeuristicPlayer`. À chaque
décision il inspecte uniquement l'état et les actions légales fournis par le
runner. Si la politique ciblée s'applique, il délègue à l'heuristique; sinon
il produit l'observation neuronale masquée et délègue au réseau.

Le benchmark réutilise un scorer neuronal chargé en lecture seule, crée un
nouveau moteur et des RNG dérivés pour chaque partie, puis agrège les résultats
par type d'adversaire.

## Observability And Operations

Chaque partie conserve sa graine, son adversaire, son résultat, son nombre de
tours/actions et les compteurs de décisions neurales. Le HTML expose au moins
les parties, victoires, défaites, nulles et taux de victoire par adversaire.

## Testing Strategy

- Tests unitaires vérifiant le routage des trois politiques et le rejet d'une
  politique inconnue.
- Smoke test du benchmark sur cinq parties.
- Suite de tests Python complète si le temps le permet.

## Rollout And Migration

Aucun changement de données ni migration. Le nouveau benchmark est opt-in via
une cible Makefile et écrit uniquement dans `artifacts/neural_benchmark/`.

## Files Expected To Change

- `shards_ai/ai/hybrid_player.py`
- `shards_ai/ai/__init__.py`
- `benchmarks/benchmark_neural_hybrids.py`
- `tests/ai/test_hybrid_player.py`
- `Makefile`
- `doc/Current state/Neural player.md`
