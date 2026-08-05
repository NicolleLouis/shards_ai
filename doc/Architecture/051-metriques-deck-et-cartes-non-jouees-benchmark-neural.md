# Métriques de deck et de cartes non jouées du benchmark neural

## Objective

Rendre visible dans le benchmark neural mix si le NeuralPlayer achète et conserve réellement des
cartes, et s'il passe la phase de jeu alors qu'il dispose encore d'une carte jouable en main.

## Current State

`benchmarks/benchmark_neural_mix.py` agrège déjà la composition des decks finaux et les événements
d'achat de mercenaires, mais ne compare pas la taille totale des decks et ne signale pas les passages
avec des cartes jouables restantes.

## Target Behavior

Chaque partie enregistre la taille finale du deck neural et celle de l'adversaire, ainsi que le
nombre de décisions `PassPlayPhase` du neural réalisées alors qu'au moins une `PlayCard` était
légale. Elle enregistre aussi un booléen indiquant si cette situation est apparue au moins une fois
dans la partie.

Le résumé par adversaire expose les moyennes, minimums et maximums des tailles de deck, leur delta
Neural moins adversaire, le nombre de parties concernées et leur taux. Le rapport HTML affiche ces
informations dans le résumé de chaque matchup.

## Non-Goals

- juger automatiquement qu'une carte devait être jouée stratégiquement ;
- compter les cartes achetées uniquement à partir de la taille finale ;
- modifier le moteur ou le comportement du NeuralPlayer ;
- ajouter des logs bruts dans `doc/`.

## Key Decisions

- Une carte est considérée jouable uniquement si le moteur expose une action `PlayCard` dans les
  actions légales au moment du `PassPlayPhase`.
- Le signal est compté au niveau des décisions et au niveau des parties pour distinguer fréquence et
  existence.
- La taille du deck inclut les mêmes zones que la composition déjà agrégée : main, pioche, défausse,
  zone de jeu et champions.

## Testing Strategy

- tester l'agrégation de tailles et de deltas sur des parties synthétiques ;
- tester l'affichage HTML des nouvelles métriques ;
- préserver les tests du planning 20/50/30 et des graphiques existants ;
- exécuter la suite complète et une campagne courte si le checkpoint est disponible.

## Files Expected To Change

- `benchmarks/benchmark_neural_mix.py` ;
- `tests/ai/test_neural_benchmark.py` ;
- `doc/Current state/Neural player.md`.
