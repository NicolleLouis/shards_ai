# Entraînement d'imitation pondéré Architecture

## Objective

Réduire la perte de performance causée par un dataset d'imitation exclusivement rééquilibré en
entraînant sur la distribution naturelle du dataset brut et en donnant une importance modérée aux
décisions stratégiques.

## Current State

`scripts/train_neural_imitation.py` lit un seul JSONL et cherche à la fois les exemples de train et
de validation dans ce fichier. Un dataset normalisé ne contenant que le split train produit donc
une validation vide. `train_epoch()` applique actuellement le même poids à chaque décision.

## Target Behavior

- le dataset d'entraînement peut rester le JSONL naturel 1M ;
- un JSONL de validation séparé peut être fourni explicitement ;
- les actions stratégiques reçoivent un poids configurable dans la loss d'entraînement ;
- la validation reste non pondérée et conserve sa distribution naturelle ;
- les paramètres effectifs sont conservés dans les métadonnées du checkpoint.

## Key Decisions

- La première expérience utilisera le dataset naturel, pas le dataset pruné.
- Le poids stratégique par défaut reste `1.0` ; l'expérience ciblée utilisera `2.0` ou `3.0`.
- Les actions stratégiques sont configurables par répétition de `--strategic-action`.
- La pondération concerne uniquement la loss d'entraînement ; elle ne modifie pas les métriques de
  validation.
- Le checkpoint stable v001 reste la source de reprise et le checkpoint de travail reste la seule
  sortie mutable.

## Testing Strategy

- vérifier qu'un fichier de validation séparé produit des métriques non vides ;
- vérifier qu'une action stratégique reçoit le poids demandé ;
- conserver les tests existants de split et d'évaluation ;
- valider la compilation et un smoke test d'entraînement court.

## Files Expected To Change

- `shards_ai/ai/neural_training.py` ;
- `scripts/train_neural_imitation.py` ;
- `tests/ai/test_neural_training.py`.
