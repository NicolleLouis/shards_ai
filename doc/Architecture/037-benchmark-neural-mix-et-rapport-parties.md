# Benchmark NeuralPlayer contre un mix d'adversaires

## Objective

Évaluer le checkpoint neural actuel sur une campagne de parties représentative avec 20 % de
`RandomPlayer`, 50 % de `HeuristicPlayer` v007 et 30 % de `HeuristicPlayer` v008. Le rapport doit
permettre de comparer les taux de victoire et de comprendre les parties via leur durée, la maîtrise,
les PV et les decks finaux.

## Current State

`benchmarks/benchmark_neural_players.py` ne joue actuellement que contre un adversaire choisi pour
toute la campagne et produit un JSON agrégé limité aux victoires, actions et latence. Les profils
v007 et v008 sont disponibles dans `configs/heuristic_profiles/`.

## Target Behavior

Une campagne de 1 000 parties utilise exactement :

- 200 parties contre Random ;
- 500 parties contre v007 ;
- 300 parties contre v008.

Chaque partie alterne le joueur qui commence selon son index. Le benchmark produit un JSON complet
et un rapport HTML autonome avec un résumé global, une section par adversaire, les distributions
de durée et la composition finale moyenne des decks de chaque joueur.

## Key Decisions

- La répartition est déterministe par blocs de dix parties (`RRVVVVVVVVV` avec `R=Random` et
  `V=v007/v008` selon le bloc), afin de conserver les proportions même avec une seed fixe.
- Les mêmes seeds de campagne sont utilisés pour toutes les comparaisons ultérieures.
- Le modèle neural est chargé une seule fois par campagne et partagé en lecture seule entre les
  parties ; les joueurs recréés par partie gardent leur RNG et leurs compteurs indépendants.
- Le deck final comprend les cartes de la main, pioche, défausse, zone de jeu et champions ; il est
  agrégé par `card_definition_id`, sans instance id ni ordre.
- Le rapport affiche séparément les statistiques du NeuralPlayer et de son adversaire.
- Les rapports et détails de parties restent hors de `doc/`, sous `artifacts/neural_benchmark/`.

## Open Questions

- Pour un nombre de parties qui n'est pas multiple de dix, les quotas arrondis au plus proche seront
  non bloquants ; les 1 000 parties demandées ont une répartition exacte.

## Proposed Architecture

Ajouter un benchmark de campagne qui :

1. charge le scorer du checkpoint une fois ;
2. choisit l'adversaire depuis le planning déterministe ;
3. construit les deux joueurs avec les flux RNG dérivés de la seed ;
4. exécute `GameRunner` sans modifier les règles ;
5. capture l'état final et les compteurs du NeuralPlayer ;
6. agrège les résultats globalement et par adversaire ;
7. écrit JSON détaillé et HTML lisible localement.

Le HTML contient des cartes de synthèse, une barre de taux de victoire par adversaire, des tableaux
de durée/actions/maîtrise/PV et une composition de deck finale agrégée par joueur. Les détails de
chaque partie restent consultables dans le JSON, sans surcharger le tableau principal.

## Metrics

Par adversaire : parties, victoires neural, victoires adverses, matchs nuls, taux de victoire,
actions, tours, durée murale, décisions neural, temps d'inférence, maîtrise finale et PV finaux.
Pour les decks : nombre moyen de copies et nombre de parties contenant chaque carte, séparément pour
le neural et l'adversaire.

## Edge Cases

- profil demandé absent ou checkpoint incompatible : échec explicite avant de commencer la campagne ;
- erreur pendant une partie : seed et adversaire inclus dans l'erreur ;
- partie terminée en nul par limite d'actions/tours ;
- deck final vide ou carte absente : représenté par une valeur nulle, sans suppression silencieuse ;
- campagnes non multiples de dix : répartition documentée dans le JSON.

## Testing Strategy

- test du planning 20/50/30 ;
- test de l'agrégation des résultats et des decks ;
- test de génération HTML sans ressource externe ;
- smoke-test de quelques parties avec le checkpoint baseline ;
- validation complète de la suite existante.

## Files Expected To Change

- `shards_ai/ai/neural_player.py` — partage optionnel du scorer chargé ;
- `benchmarks/benchmark_neural_mix.py` — campagne et agrégation ;
- `tests/ai/test_neural_benchmark.py` — planning et statistiques ;
- `doc/Current state/Neural player.md` — benchmark disponible ;
- `Makefile` — commande pratique de campagne 1 000 parties.
