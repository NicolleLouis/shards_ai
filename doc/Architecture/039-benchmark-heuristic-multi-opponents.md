# Benchmark Heuristic multi-adversaires Architecture

## Objective

Faire évoluer `scripts/benchmark_heuristic_report.py` pour mesurer le profil principal v008 sur
un échantillon équilibré composé de 50 % de parties contre Random et 50 % contre Heuristic v007.
Le rapport doit comparer v008 séparément à chacun de ses adversaires, notamment pour les résultats,
les decks finaux, les choix de cartes et les comportements spécifiques de v008.

## Current State

Le script lance uniquement v008 contre `RandomPlayer`. Il agrège les choix et les decks sous les
rôles `heuristic` et `random`, puis affiche un delta unique `Heuristic − Random`. Le profil principal
est configurable par `--profile`, avec v008 par défaut. Les comportements de v008 (mercenaires,
passage de phase et conversion gemme → maîtrise) sont déjà enregistrés.

## Target Behavior

- v008 reste toujours le joueur principal observé ; son siège alterne entre les deux joueurs.
- L’adversaire est choisi de façon déterministe et équilibrée : index de partie pair = Random,
  index impair = v007. Pour un nombre pair de parties, la répartition est exactement 50/50.
- Les résultats et métriques sont regroupés par adversaire (`random`, `v007`).
- Chaque groupe expose un delta v008 moins son adversaire, sans mélanger les deux populations.
- Le HTML met en avant le résultat global, les résultats par adversaire, les decks finaux et les
  comportements de v008. Les tableaux détaillés redondants de choix par rôle et par résultat sont
  retirés de la vue HTML ; les CSV/JSON restent disponibles pour l’analyse détaillée.

## Non-Goals

- Modifier les règles du jeu, les joueurs ou la politique de choix.
- Changer la définition d’un deck final ou des métriques de comportement existantes.
- Comparer v007 directement à Random dans cette campagne ; v007 sert uniquement d’adversaire.

## Key Decisions

1. La répartition 50/50 est déterministe par index plutôt que probabiliste afin de garantir une
   comparaison équilibrée et reproductible.
2. Les seeds de parties restent dérivées de la seed racine et de l’index ; changer l’adversaire
   ne change donc pas la reproductibilité de la campagne.
3. Les deltas sont calculés séparément pour chaque adversaire : `v008 − Random` et `v008 − v007`.
4. Le résultat JSON devient la source complète ; le HTML est une vue synthétique et les CSV
   détaillent les cartes et comportements par adversaire.
5. Les anciennes clés mono-adversaire ne sont pas nécessaires au nouveau rapport ; les tests
   valident le nouveau contrat explicite par adversaire.

## Open Questions

Aucune question bloquante pour cette évolution. Une future évolution pourra ajouter une troisième
catégorie d’adversaire, mais elle devra conserver la partition des statistiques.

## Proposed Architecture

`run_benchmark` construit deux accumulateurs de matchup, chacun contenant les statistiques de v008,
de l’adversaire, des résultats et des comportements. La sélection de l’adversaire intervient avant
la construction des joueurs. Après les parties, chaque accumulateur réutilise les fonctions
d’agrégation existantes (`build_statistics`, `build_delta_statistics`). `write_reports` écrit un
JSON global, un CSV de cartes par matchup et un HTML avec une section de synthèse par adversaire.

## Testing Strategy

- Vérifier avec une petite campagne que les deux adversaires sont présents et reçoivent chacun la
  moitié des parties.
- Vérifier les deltas indépendants, les decks finaux et les comportements de v008 dans chaque groupe.
- Vérifier que le HTML contient les deux deltas et ne contient plus les sections supprimées.
- Exécuter la suite de tests d’analyse et la suite complète disponible.

## Files Expected To Change

- `scripts/benchmark_heuristic_report.py`
- `tests/analysis/test_heuristic_benchmark_report.py`
- `doc/Current state/Analysis.md`
