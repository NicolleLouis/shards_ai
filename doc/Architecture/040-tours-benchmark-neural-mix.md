# Tours de jeu dans le benchmark Neural mix

## Objective

Afficher dans le benchmark `neural-benchmark-mix` le nombre moyen de tours joués par partie,
ainsi que le nombre moyen de tours par joueur, afin de compléter les métriques de durée et
d’actions.

## Current State

`benchmarks/benchmark_neural_mix.py` capture déjà `GameState.turn_number` dans chaque résultat et
agrège cette valeur sous la clé `turns`. Le rapport HTML l’affiche dans le tableau comparatif,
mais pas dans le résumé principal de chaque adversaire. Le benchmark exécute toujours un duel de
deux joueurs via `GameRunner`.

## Target Behavior

Chaque résultat conserve le nombre total de tours de la partie et expose `turns_per_player`, égal
au nombre total de tours divisé par le nombre de joueurs du runner. Les agrégats exposent les deux
statistiques avec moyenne, minimum et maximum. Le rapport affiche explicitement les deux valeurs
dans le résumé de chaque adversaire et dans le tableau comparatif.

## Non-Goals

- Modifier la définition d’un tour ou les transitions de `Game`.
- Ajouter un compteur par rôle (Neural versus adversaire) ; la métrique demandée est la moyenne
  par joueur.
- Modifier le format des campagnes, la distribution des adversaires ou les performances du moteur.

## Key Decisions

- `GameState.turn_number` est la source de vérité du nombre total de tours, comme dans le benchmark
  existant.
- Le dénominateur est `len(runner.players)`, et non une constante `2`, pour garder la métrique
  correcte si le runner devient configurable.
- Le nom JSON `turns_per_player` est conservé en anglais comme les autres clés du benchmark ; les
  libellés du rapport restent en français.
- La métrique est calculée par partie avant agrégation, puis agrégée avec la même fonction
  `numeric` que `turns`.

## Open Questions

Aucune question bloquante. Le benchmark actuel est un duel ; si des parties à plus de deux joueurs
sont ajoutées, la formule restera une moyenne de tours par joueur et non un décompte exact par rôle.

## Proposed Architecture

`play_game` capture `state.turn_number` et calcule `state.turn_number / len(runner.players)`.
`_summary` agrège `turns` et `turns_per_player`. `_render_report` utilise ces deux agrégats dans
le tableau et dans le paragraphe de synthèse de chaque adversaire. Aucun changement n’est requis
dans le moteur, l’IA ou la persistance des parties.

## Data Model

Les résultats JSON par partie ajoutent :

```json
{"turns": 333, "turns_per_player": 166.5}
```

Les résumés ajoutent `turns_per_player` avec `mean`, `min` et `max`, suivant le format de `turns`.
Les anciens JSON restent lisibles comme artefacts historiques ; seuls les nouveaux rapports
contiennent la nouvelle clé.

## Testing Strategy

- Vérifier l’agrégation de `turns_per_player` à partir d’un enregistrement synthétique.
- Vérifier la présence des libellés et valeurs de tours total et par joueur dans le HTML.
- Exécuter les tests du benchmark Neural existants et une validation syntaxique ciblée.

## Files Expected To Change

- `benchmarks/benchmark_neural_mix.py`
- `tests/ai/test_neural_benchmark.py`
- `doc/Current state/Neural player.md`
