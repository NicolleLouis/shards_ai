# Test du biais de recrutement des mercenaires

## Objective

Mesurer si la préférence du NeuralPlayer pour l'achat long terme de mercenaires augmente la taille
du deck et réduit son efficacité. Le test doit modifier uniquement le choix des actions pendant un
benchmark, sans réentraîner ni modifier un checkpoint stable.

## Current State

`NeuralPlayer` choisit l'action au score maximal parmi les actions légales. Le benchmark
`benchmarks/benchmark_neural_mix.py` observe déjà les mercenaires recrutés immédiatement
(`RecruitMercenary`) et achetés long terme (`BuyCard`), ainsi que la taille finale du deck.

## Target Behavior

Un paramètre de benchmark `mercenary_mode_bias` permet de tester une politique dérivée :

- score augmenté pour `RecruitMercenary` sur une carte mercenaire ;
- score diminué du même montant pour `BuyCard` sur cette carte ;
- score inchangé pour les autres actions et les autres cartes.

La valeur zéro reproduit exactement le comportement actuel. Les variantes sont évaluées avec les
mêmes seeds, adversaires et checkpoint.

## Non-Goals

- modifier les règles ou les actions légales ;
- modifier le checkpoint ou le profil actif ;
- imposer un ratio de mercenaires comme vérité stratégique ;
- conclure à partir de la seule taille finale du deck.

## Key Decisions

- Le premier test est une intervention d'inférence, car elle isole directement le choix immédiat
  contre long terme sans changer les labels d'imitation.
- Les valeurs de biais doivent être explicites dans le manifest du benchmark.
- La taille du deck doit être analysée par adversaire et, lors d'une évolution ultérieure, par tour.
- Une amélioration avec biais ne prouvera pas encore que le réseau doit être réentraîné avec cette
  préférence ; elle justifiera un dataset ou une loss ciblée.

## Testing Strategy

- tester que le biais zéro ne change pas le choix ;
- tester qu'un biais positif favorise le recrutement immédiat sur une carte mercenaire ;
- exécuter baseline, biais modéré et biais fort sur les mêmes seeds ;
- comparer victoires, tailles de decks et ratios immédiat/long terme.

## Files Expected To Change

- `shards_ai/ai/neural_player.py` ;
- `benchmarks/benchmark_neural_mix.py` ;
- `tests/ai/test_neural_player.py` ;
- `tests/ai/test_neural_benchmark.py`.
