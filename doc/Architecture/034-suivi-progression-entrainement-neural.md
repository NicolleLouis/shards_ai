# Suivi de progression de l'entraînement neural

## Objective

Rendre chaque entraînement observable afin de déterminer si le modèle progresse, généralise sur
le split de validation et commence à surapprendre.

## Current State

`train_neural_imitation.py` entraîne le modèle et affiche uniquement une loss moyenne par epoch.
Le checkpoint conserve cette liste minimale, sans métriques de validation ni courbes.

## Target Behavior

Chaque epoch doit produire des métriques d'entraînement et de validation, être ajouté à un fichier
JSON structuré, et pouvoir être visualisé sans installer une stack graphique obligatoire.

## Key Decisions

- Le fichier de métriques est séparé du checkpoint et écrit à côté de lui par défaut.
- Le split de validation reste déterministe et séparé par `game_id`.
- Les métriques principales sont la loss, l'accord top-1, le rang normalisé de l'action choisie et
  la précision paire-à-paire.
- Le fichier JSON est la source de vérité ; un CSV est produit pour l'analyse tabulaire.
- Un SVG simple est produit avec la bibliothèque standard afin d'être lisible immédiatement sans
  ajouter Matplotlib comme dépendance obligatoire.
- L'évaluation peut être limitée à un nombre de décisions reproductible pour contrôler la durée.

## Non-Goals

- Déterminer automatiquement le meilleur modèle ou arrêter automatiquement l'entraînement.
- Remplacer l'évaluation en parties complètes.
- Ajouter TensorBoard, Weights & Biases ou une base de données d'expériences.

## Proposed Architecture

`neural_training.py` expose une évaluation sans gradient qui calcule les métriques d'une partition.
Le script entraîne sur `train`, évalue sur `validation` à chaque epoch, puis écrit les métriques
JSON/CSV et un SVG avec trois panneaux : loss, top-1/rang et précision paire-à-paire.

## Edge Cases

- Une décision avec une seule action ne produit pas de paires et est ignorée pour cette métrique.
- Des scores heuristiques égaux ne produisent pas de contrainte d'ordre.
- Une décision sans action choisie valide est comptée mais exclue du top-1 et du rang.
- Une partition vide produit des métriques nulles explicites, sans faire échouer le run.

## Testing Strategy

- vérifier les métriques sur des décisions synthétiques connues ;
- vérifier la sérialisation JSON/CSV ;
- vérifier que le SVG est produit sans dépendance graphique externe ;
- conserver les tests complets du modèle et de l'entraînement.

## Files Expected To Change

- `shards_ai/ai/neural_training.py` — métriques et évaluation ;
- `scripts/train_neural_imitation.py` — export des métriques et graphiques ;
- `tests/ai/test_neural_training.py` — validation des métriques ;
- `doc/Current state/Neural player.md` — artefacts disponibles.
