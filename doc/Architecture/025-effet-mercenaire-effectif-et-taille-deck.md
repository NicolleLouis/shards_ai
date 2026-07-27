# Effet immédiat effectif des mercenaires et taille des decks — Architecture

## Objective

Corriger la valorisation des mercenaires joués immédiatement afin de mesurer l'effet réellement
obtenu dans l'état courant, puis enrichir le benchmark macro avec la taille des decks finaux.

Deux besoins sont liés à l'analyse de parties :

- `Clerc aux Spores` ne soigne effectivement que 2 PV lorsque le joueur est à 48/50 PV, mais
  l'heuristique le valorise actuellement comme un soin de 4 ;
- le rapport macro doit comparer la taille du deck Heuristic et Random, ainsi que la taille du deck
  Heuristic dans les victoires et les défaites.

## Current State

Les effets immédiats d'un `RecruitMercenary` passent par `_effect_features()` et utilisent le montant
nominal de chaque opération. Le moteur applique pourtant un plafond de santé à `Game.STARTING_HEALTH`
lors de la résolution réelle. Les valeurs prospectives des achats durables ne doivent pas être
réduites par la santé actuelle, car la carte sera jouée plus tard.

Le benchmark `scripts/benchmark_heuristic_report.py` conserve déjà les cartes finales par rôle et par
résultat, mais n'expose pas de résumé numérique de leur taille. Les snapshots excluent les cartes
bannies et la rivière ; la taille doit compter les cartes encore possédées dans les zones de jeu,
avec les champions inclus pour représenter toutes les cartes conservées.

## Target Behavior

Pour un effet immédiat de mercenaire :

```text
gain_health_effectif = min(gain_health_nominal, Game.STARTING_HEALTH - health_courante)
```

Le score de `RecruitMercenary` utilise ce gain effectif. Les autres effets restent nominaux tant
qu'aucune règle moteur ne les plafonne. Les effets joués depuis la main pourront réutiliser le même
paramètre dans une évolution ultérieure, mais ce correctif cible d'abord les mercenaires.

Le rapport macro ajoute :

- `deck_size_by_role.heuristic` et `.random` ;
- `heuristic_deck_size_by_result.heuristic_win` et `.heuristic_loss` ;
- les mêmes valeurs dans le HTML et un CSV synthétique.

## Non-Goals

- Modifier la résolution moteur des effets.
- Appliquer la santé courante aux achats durables ou aux effets futurs.
- Compter les cartes bannies, la rivière ou la pioche centrale dans le deck final.
- Changer la politique de sélection autrement que par la correction du signal fourni.

## Key Decisions

1. **Source du plafond.** Utiliser `Game.STARTING_HEALTH`, et non recopier la constante dans l'IA.
2. **Portée initiale.** Activer la capacité effective uniquement pour `_purchase_features(...,
   immediate=True)`, qui représente les mercenaires joués immédiatement.
3. **Effets multiples.** Appliquer la capacité restante séquentiellement aux opérations `gain_health`
   d'un même effet.
4. **Deck size.** Compter les instances dans `hand`, `draw_pile`, `discard_pile`, `play_zone` et
   `champions`. Les cartes bannies et consommées ne sont pas comptées.
5. **Comparaisons.** Les tailles Heuristic vs Random sont calculées par rôle ; les tailles Heuristic
   win vs Heuristic loss filtrent les snapshots sur le rôle Heuristic.

## Proposed Architecture

Ajouter un indicateur interne `effective_now` au calcul des features d'effets. Il est `True` pour
les mercenaires immédiats et `False` pour les cartes durables. Le calcul conserve une capacité de
santé locale pendant l'itération des opérations, sans modifier l'observation.

Ajouter `deck_size` à chaque snapshot du benchmark, puis produire les résumés numériques à partir
des snapshots déjà conservés. Cette agrégation ne crée pas de nouvelle partie ni de nouvelle copie
d'état.

## Data Model

Les événements et les règles du moteur ne changent pas. Les sorties JSON ajoutent :

```json
{
  "deck_size_by_role": {
    "heuristic": {"count": 1000, "mean": 12.3, "min": 8, "max": 20},
    "random": {"count": 1000, "mean": 15.1, "min": 10, "max": 24}
  },
  "heuristic_deck_size_by_result": {
    "heuristic_win": {"count": 930, "mean": 11.8},
    "heuristic_loss": {"count": 70, "mean": 14.2}
  }
}
```

## Testing Strategy

- mercenaire à 48 PV avec soin 4 : feature `health_gained == 2` ;
- mercenaire à 50 PV : feature de soin nulle ;
- même effet à 40 PV : feature de soin égale à 4 ;
- achat durable du même effet : projection non plafonnée par la santé courante ;
- snapshots incluant les champions dans `deck_size` ;
- agrégats séparés par rôle et par victoire/défaite ;
- rapport HTML/JSON/CSV contenant les nouvelles sections ;
- suite complète du projet.

## Rollout And Migration

Le correctif change uniquement l'évaluation heuristique des mercenaires. Une campagne de validation
indépendante devra mesurer l'impact avant une nouvelle optimisation globale. Le métrique de taille de
deck est rétrocompatible pour les anciens rapports, qui ne contiennent simplement pas les nouveaux
champs.

## Files Expected To Change

- `shards_ai/ai/heuristic_features.py` : effets immédiats effectifs ;
- `scripts/benchmark_heuristic_report.py` : taille des decks et rendu ;
- `tests/game/test_heuristic_player.py` ou tests de features : plafonds de soins ;
- `tests/analysis/test_heuristic_benchmark_report.py` : agrégats et fichiers ;
- `doc/Current state/Heuristic player.md` et `doc/Current state/Analysis.md`.
