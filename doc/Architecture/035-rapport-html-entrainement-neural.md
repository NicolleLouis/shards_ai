# Rapport HTML de progression neural

## Objective

Permettre une lecture visuelle simple des métriques d'entraînement et de validation sans dépendre
d'un notebook, d'un serveur ou d'un outil externe.

## Current State

Le training produit déjà des métriques JSON/CSV et un SVG global, mais le SVG est limité pour
comparer plusieurs indicateurs et consulter les valeurs exactes par epoch.

## Target Behavior

Générer un fichier HTML autonome à côté des métriques, contenant les graphiques, les meilleurs
résultats observés et un tableau détaillé. Le fichier doit être ouvrable directement depuis le
système de fichiers.

## Key Decisions

- Le JSON reste la source de vérité ; le HTML est un artefact de présentation régénérable.
- Le HTML ne charge aucune ressource distante et utilise des SVG inline.
- Les courbes affichées sont : loss train/validation, top-1/rang normalisé, précision paire-à-paire
  et volume de décisions/paires.
- Le rapport est généré automatiquement par le script d'entraînement et peut aussi être régénéré
  depuis un fichier metrics existant.

## Non-Goals

- Ajouter une interface interactive complète ou une base d'expériences.
- Remplacer les métriques structurées par des valeurs embarquées uniquement dans le HTML.

## Proposed Architecture

`shards_ai/ai/neural_reporting.py` lit la liste d'epochs, construit des graphiques SVG inline et
produit le document HTML avec un résumé et un tableau. Le script `train_neural_imitation.py` appelle
ce module après l'écriture du JSON. Un script séparé permet de régénérer le rapport sans réentraîner.

## Testing Strategy

- rapport valide sur zéro, un et plusieurs epochs ;
- absence de ressource externe dans le HTML ;
- présence des séries et valeurs principales ;
- smoke-test de génération depuis les métriques d'un entraînement court.

## Files Expected To Change

- `shards_ai/ai/neural_reporting.py` ;
- `scripts/train_neural_imitation.py` ;
- `scripts/generate_neural_training_report.py` ;
- `tests/ai/test_neural_reporting.py` ;
- `doc/Current state/Neural player.md`.
