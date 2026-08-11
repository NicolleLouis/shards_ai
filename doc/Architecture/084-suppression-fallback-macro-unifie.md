# Suppression du fallback macro unifié

## Objective

Supprimer le fallback nominal du joueur macro. Toute décision avec plusieurs actions possibles doit
être présentée au scoreur V4 unifié, soit comme branche `macro_play`, soit comme candidats atomiques
`atomic`. Une limite du solveur doit changer la granularité de la décision, pas sélectionner un autre
joueur, une autre politique ou une action implicite.

La promotion `v005` reste la base de travail. `v004` demeure un rollback explicite et ne doit plus
être appelé par le chemin nominal.

## Current State

`PlayTurnSolver.resolve()` renvoie `fallback=True` et aucun candidat lorsque ses limites sont
atteintes : cycle, mémoïsation, longueur de segment, budget d’expansion ou nombre de candidats.
`MacroNeuralPlayer` rejoue alors le préfixe puis utilise déjà le même scoreur V4 via
`_choose_unified_atomic()`. Le réseau est donc unifié, mais le contrat et l’observabilité portent
encore le nom fallback.

Les phases hors `PLAY` utilisent déjà des candidats atomiques de longueur 1. Le dataset V4 contient
les deux types `macro_play` et `atomic`.

## Target Behavior

- Une résolution retourne toujours zéro candidat si la partie est terminée, un candidat pour un
  replay déterministe, ou au moins deux candidats représentables pour une inférence.
- En cas de limite pendant l’exploration, le solveur conserve le préfixe automatique validé puis
  expose les actions légales de l’état courant comme candidats atomiques V4.
- Le joueur ne possède plus de branche `fallback`, de compteur `fallbacks` ni de chemin vers V004.
- Les catégories runtime deviennent `macro_choice`, `atomic_choice`, `macro_replay` et les actions
  automatiques sans décision exposée.
- Une trace partielle de branche n’est jamais présentée au réseau comme une macro complète.

## Non-Goals

- Modifier les budgets fixes du solveur.
- Modifier les règles du moteur ou l’information observable.
- Réentraîner ou requalifier `v005` dans cette évolution.
- Supprimer le profil stable V004 ou empêcher son utilisation explicite comme rollback.

## Key Decisions

1. Une limite de recherche produit une décision atomique du même scoreur, jamais une action par
   défaut et jamais un second checkpoint.
2. Les candidats atomiques sont construits depuis `working.legal_actions()` après le préfixe validé.
3. La représentation atomique V4 conserve `decision_kind="atomic"` et une trace d’une action.
4. Le champ structurel `fallback` est supprimé de la résolution ; une raison de limite peut rester
   dans un champ diagnostique non décisionnel si les rapports en ont besoin.
5. Les tests vérifient que chaque action légale est couverte lorsque la limite est atteinte et que
   le scoreur est appelé exactement une fois pour la décision atomique.

## Proposed Architecture

`PlayTurnSolver` centralise la transition et l’autorité des règles. Après une limite, il retourne
une `PlayTurnResolution` composée du préfixe automatique et de `PlayTurnCandidate` atomiques créés
à partir de l’état `working`. Le joueur applique le même protocole de représentation et de scoring
que pour les décisions atomiques BUY, combat et pending.

Le joueur ne distingue plus l’origine de la décision atomique. L’origine éventuelle de la limite est
seulement une donnée d’observabilité (`budget_boundary_reason`) et ne change ni l’action proposée ni
le checkpoint utilisé.

## Edge Cases

- Si le préfixe atteint une fin de partie, aucun candidat n’est nécessaire.
- Si une seule action légale reste, elle est rejouée automatiquement sans inférence.
- Si plusieurs actions restent après une limite, elles sont toutes candidates ; aucune troncature de
  la liste légale n’est autorisée.
- Une trace devenue illégale après application reste une erreur du moteur/solveur.
- L’absence de scoreur est réservée aux tests ou aux adaptateurs déterministes explicites ; elle ne
  constitue pas un fallback neural.

## Testing Strategy

- Adapter les tests de budget, cycle, mémoïsation et longueur de segment.
- Vérifier les candidats atomiques, leur représentation V4 et la couverture des actions légales.
- Vérifier le préfixe automatique et l’absence de trace macro partielle.
- Vérifier l’absence des compteurs et labels `fallback` dans le joueur et les rapports.
- Exécuter les tests unitaires macro, solver, entraînement et profils, puis `git diff --check`.
- Le panel complet et la validation de performance restent une validation ultérieure de la politique,
  pas une condition de correction du contrat.

## Rollout And Migration

Le changement est rétrocompatible avec les datasets V4 et le checkpoint `v005`. Les artefacts
historiques qui contiennent des métriques `fallback` restent lisibles comme historiques. Les nouveaux
rapports utilisent les catégories sans fallback. En cas de régression du solveur, le rollback de
profil vers V004 reste explicite, mais ne doit pas être invoqué automatiquement.

## Files Expected To Change

- `shards_ai/ai/play_turn_solver.py`
- `shards_ai/ai/macro_player.py`
- `benchmarks/benchmark_neural_mix.py`
- `scripts/compare_macro_vs_heuristic_actions.py`
- `tests/ai/test_play_turn_solver.py` et tests de benchmark associés
- `doc/Current state/Neural player.md`
