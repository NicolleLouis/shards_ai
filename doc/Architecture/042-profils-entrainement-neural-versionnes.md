# Profils d'entraînement neural versionnés

## Objective

Rendre chaque entraînement neural reproductible, identifiable et comparable, comme les profils
heuristiques. Le dépôt doit versionner la recette d'entraînement et ses décisions importantes sans
versionner les datasets volumineux ni les checkpoints binaires.

## Current State

`scripts/train_neural_imitation.py` reçoit actuellement le dataset, le checkpoint de sortie, le
nombre d'epochs, le learning rate, les seeds et les limites principalement par arguments CLI. Le
checkpoint conserve le modèle, l'optimiseur et quelques métriques, mais pas une identité de run ni
la recette complète qui l'a produit. Le `Makefile` pointe directement vers `baseline.pt`.

Les profils heuristiques sont des YAML sous `configs/heuristic_profiles/`, chargés et validés par
`shards_ai/ai/heuristic_profiles.py`. Les datasets neural actuels font plusieurs gigaoctets et les
checkpoints sont des artefacts de run ; ils ne doivent pas être ajoutés au dépôt.

## Target Behavior

Un profil YAML sous `configs/neural_training_profiles/` décrit une campagne complète : son
identifiant, son parent éventuel, sa méthode, le dataset, son checkpoint d'initialisation, les
hyperparamètres, le split, la configuration du modèle et les chemins d'artefacts. Le script accepte
`--profile` et permet des overrides explicites pour les essais courts.

Le checkpoint produit contient l'identifiant du profil, sa configuration résolue et un fingerprint
de cette configuration. Les métriques JSON contiennent les mêmes métadonnées. Une commande Makefile
utilise une version active explicite, analogue à `HEURISTIC_VERSION`.

## Non-Goals

- versionner les datasets JSONL, checkpoints `.pt`, rapports ou sorties de benchmark ;
- implémenter le reinforcement learning ou modifier le modèle neural ;
- gérer un registre distant de modèles ;
- rendre obligatoire un parent existant sur disque ;
- supprimer la possibilité d'utiliser le script neural directement avec des arguments.

## Key Decisions

- Les profils sont des recettes committables ; les artefacts restent sous `artifacts/` et sont
  ignorés par Git.
- Le profil est immuable conceptuellement : une modification significative reçoit un nouvel
  `profile_id` et peut déclarer `parent_profile_id`.
- La configuration du modèle est stockée dans le profil et convertie explicitement en
  `NeuralModelConfig`. Les clés inconnues ou les valeurs invalides provoquent une erreur claire.
- Les chemins du profil sont relatifs à la racine du dépôt lorsqu'ils sont relatifs au fichier de
  configuration. Le script les résout depuis le répertoire de travail, avec une erreur si une entrée
  requise n'existe pas.
- Les overrides CLI sont destinés aux expériences temporaires et sont inscrits dans les métadonnées
  du checkpoint ; ils ne modifient jamais le YAML.
- Le fingerprint SHA-256 de la configuration résolue permet de détecter qu'un checkpoint et un
  profil ne correspondent pas.
- Le profil `v001` décrit le baseline d'imitation actuel. Le profil actif est sélectionné dans le
  `Makefile` par `NEURAL_VERSION`, sans basculer automatiquement vers un candidat.

## Open Questions

- Blocking: aucune pour l'imitation supervisée actuelle.
- Non-blocking: le futur profil PPO pourra réutiliser le même namespace ou disposer d'un dossier
  distinct lorsque l'architecture RL sera définie.

## Proposed Architecture

Ajouter `shards_ai/ai/neural_training_profiles.py` avec un dataclass immuable
`NeuralTrainingProfile`, `load_training_profile` et `save_training_profile`. Le chargeur valide la
structure YAML, les types, la méthode (`imitation` pour cette première version), les paramètres
positifs et la correspondance avec `NeuralModelConfig`.

Le script d'entraînement charge le profil avant de construire le modèle. Les arguments CLI présents
restent compatibles ; `--profile` fournit les valeurs par défaut et les arguments explicitement
fournis les remplacent. Le fingerprint est calculé après résolution des overrides.

Le checkpoint devient le résultat d'un run traçable : profil, fingerprint, configuration modèle,
configuration d'entraînement, dataset et checkpoint parent. Le chargement depuis `--resume-from`
conserve le profil existant par défaut et refuse un profil différent sans override explicite.

## Data Model

Le YAML contient notamment :

```yaml
schema_version: 1
profile_id: v001
parent_profile_id: null
method: imitation
dataset: artifacts/imitation_dataset/v007-v008.jsonl
output: artifacts/neural_imitation/v001.pt
seed: 50
split_seed: 0
epochs: 5
learning_rate: 0.001
torch_threads: 1
max_records: null
max_validation_records: 2000000
model: {}
```

Les champs dérivés (fingerprint, overrides, epoch, métriques) sont générés dans les artefacts et ne
sont pas ajoutés manuellement au profil.

## Observability And Operations

Chaque run affiche son profil et son fingerprint avant l'entraînement. Les métriques et checkpoints
permettent de retrouver la recette, le parent et le dataset utilisés. Les fichiers lourds sont
exclus du versionnement ; le YAML, le code et les tests sont les éléments à committer.

## Edge Cases

- profil absent, YAML non-mapping ou identifiant vide ;
- méthode inconnue ou modèle contenant une clé inconnue ;
- dataset requis absent ;
- checkpoint d'initialisation absent ;
- reprise d'un checkpoint produit par un autre profil ;
- output identique à input lors d'une reprise, autorisé ;
- override de valeur négative ou nulle pour les paramètres nécessitant une valeur positive.

## Testing Strategy

Ajouter des tests de chargement, de validation, de fingerprint stable, de résolution des chemins et
de sauvegarde/rechargement. Ajouter des tests du script ou de ses fonctions de configuration pour
vérifier les overrides et les métadonnées du checkpoint. Exécuter les tests neural existants et la
suite complète.

## Rollout And Migration

Créer `v001.yaml` en reprenant la configuration baseline. Le `Makefile` utilise ce profil par
défaut. Les commandes CLI historiques restent utilisables pendant la transition. Les artefacts
existants ne sont pas réécrits ; un nouvel entraînement produit un checkpoint explicitement associé
à `v001`.

## Files Expected To Change

- `configs/neural_training_profiles/v001.yaml`
- `shards_ai/ai/neural_training_profiles.py`
- `shards_ai/ai/__init__.py`
- `scripts/train_neural_imitation.py`
- `Makefile`
- `tests/ai/test_neural_training_profiles.py`
- `tests/ai/test_neural_training.py`
- `doc/Current state/Neural player.md`
