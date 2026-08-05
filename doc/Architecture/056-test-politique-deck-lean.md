# Test de politique deck lean

## Objective

Vérifier si la taille excessive du deck Neural vient d'une fréquence trop élevée des achats
`BuyCard`, en pénalisant temporairement ces actions pendant le benchmark.

## Target Behavior

Un paramètre `deck_lean_bias` soustrait une valeur explicite au score de chaque `BuyCard`. Les
actions `RecruitMercenary`, `GainMastery`, `BanishCard` et les autres actions restent inchangées.
La valeur zéro reproduit exactement la politique de référence.

## Key Decisions

- Le biais est appliqué uniquement à l'inférence de benchmark ; aucun checkpoint n'est modifié.
- Les mêmes seeds, adversaires et checkpoint sont utilisés pour les variantes.
- Les métriques prioritaires sont la victoire, la taille du deck par adversaire et le delta de deck.
- Une réduction du deck sans gain de victoire ne validera pas l'hypothèse causale.

## Non-Goals

- modifier les règles d'achat ;
- imposer une taille cible au deck ;
- remplacer l'apprentissage ciblé des mercenaires ;
- promouvoir une variante biaisée comme checkpoint stable.

## Testing Strategy

- tester le biais zéro et un biais positif sur les choix légaux ;
- benchmarker des valeurs faibles et modérées sur 1 000 parties ;
- comparer les victoires, les achats, le deck final et les mercenaires immédiats/long terme.

## Files Expected To Change

- `shards_ai/ai/neural_player.py` ;
- `benchmarks/benchmark_neural_mix.py` ;
- `tests/ai/test_neural_player.py`.
