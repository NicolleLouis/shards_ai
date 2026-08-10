# Benchmark panel neural complet

## Objective

Fournir une commande reproductible qui fait jouer un checkpoint neural testé contre Random, les
profils heuristiques `v007` et `v008`, puis les checkpoints neural promus `v001`, `v002` et `v003`.
Le résultat doit être disponible en JSON pour analyse et en HTML autonome pour lecture humaine.

## Current State

`make neural-benchmark-mix` appelle `benchmarks/benchmark_neural_mix.py`, qui répartit les parties
entre Random, v007 et v008 selon `20/30/50`. Il produit déjà les métriques de partie, de deck,
décisions, d'inférence, de maîtrise et de mercenaires, avec un HTML autonome.

## Target Behavior

- Ajouter `benchmarks/benchmark_neural_panel.py`.
- Le checkpoint testé est configurable et vaut par défaut `configs/neural_profiles/v003.pt`.
- Le panel est fixe et ordonné : `random`, `v007`, `v008`, `neural:v001`, `neural:v002`, `neural:v003`.
- `--games` désigne le nombre de parties par adversaire ; les seeds sont uniques et reproductibles.
- Les résultats restent séparés par adversaire ; aucune moyenne de win-rate entre profils n'est présentée comme une preuve de qualité.
- Le JSON conserve les parties détaillées et les résumés ; le HTML expose une table globale, des cartes de synthèse et une section détaillée par adversaire.

## Non-Goals

- Ne pas modifier la gate de promotion ni les checkpoints.
- Ne pas remplacer `neural-benchmark-mix`, qui reste utile pour son workload pondéré historique.
- Ne pas sélectionner ou promouvoir automatiquement un profil.

## Key Decisions

- Utiliser exactement le même workload et les mêmes métriques observables que `benchmark_neural_mix.py`.
- Jouer un nombre égal de parties par adversaire pour rendre les comparaisons lisibles.
- Charger tous les scorers une seule fois avant la campagne.
- Refuser explicitement un checkpoint ou un profil neural manquant avant de lancer des parties.
- Déclarer dans le HTML le checkpoint testé, les checkpoints adverses, le nombre de parties, la seed et le nombre de threads Torch.

## Proposed Architecture

Le nouveau script possède un runner générique pour un adversaire heuristique ou neural, réutilise la
structure de résumé des métriques du benchmark mix et rend une page HTML dédiée au panel. Chaque
partie conserve son adversaire, sa seed, son issue, son temps, ses actions, ses décisions neural,
son état final et ses événements de deckbuilding.

## Observability And Operations

Les sorties par défaut sont `artifacts/neural_benchmark/neural_panel.json` et
`artifacts/neural_benchmark/neural_panel.html`. Une progression est affichée régulièrement. Une
campagne interrompue laisse éventuellement un JSON absent ou partiel ; aucune promotion ne dépend
de ce benchmark.

## Testing Strategy

- Tester l'ordre et la cardinalité du panel.
- Tester le nombre total de parties calculé comme `games_per_opponent × 6`.
- Tester le rendu HTML avec un payload synthétique sans lancer de parties.
- Compiler le script et exécuter les tests du benchmark existant et du nouveau benchmark.

## Files Expected To Change

- `benchmarks/benchmark_neural_panel.py`
- `tests/ai/test_neural_panel_benchmark.py`
- `Makefile`
- `README.md`
- `doc/Current state/Neural player.md`
