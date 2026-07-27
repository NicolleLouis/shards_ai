# Checkpoints neural versionnés

## Objective

Conserver dans le dépôt les checkpoints neural suffisamment légers pour reproduire et comparer les
versions (`v001` contre `v002`, etc.), tout en laissant les datasets, rapports et sorties de campagnes
hors Git. Le workflow doit rester simple à reprendre lors d'une future session Codex.

## Current State

Les profils YAML sont versionnés, mais leur champ `output` pointe vers `artifacts/`, qui est ignoré
par Git. La promotion d'un candidat copie uniquement le profil et laisse le checkpoint à son chemin
temporaire. `NeuralPlayer` accepte déjà un chemin de checkpoint explicite, mais le benchmark existant
ne propose que Random et les heuristiques comme adversaires.

Le checkpoint baseline actuel est petit (moins d'un mégaoctet), tandis que les datasets JSONL font
plusieurs gigaoctets. Le stockage Git classique est donc adapté pour les premiers checkpoints ; Git
LFS reste une option ultérieure si la taille ou le nombre de versions augmente fortement.

## Target Behavior

Les checkpoints promus sont stockés sous `checkpoints/neural/vNNN.pt` et suivis par Git. Chaque profil
`configs/neural_training_profiles/vNNN.yaml` pointe vers son checkpoint canonique. Une promotion
copie le checkpoint candidat, met à jour ses métadonnées de profil et crée le nouveau profil.

Le benchmark neural accepte un checkpoint neural comme adversaire. La documentation décrit le cycle
complet : entraîner ou reprendre une version, valider un candidat, promouvoir, puis comparer deux
versions.

## Non-Goals

- versionner les datasets, métriques détaillées, rapports HTML ou logs de campagnes ;
- mettre en place Git LFS immédiatement ;
- modifier le format des observations ou le modèle ;
- rendre les checkpoints actifs automatiquement sans validation.

## Key Decisions

- `checkpoints/neural/` est le stockage Git canonique des checkpoints promus.
- `artifacts/` reste réservé aux checkpoints candidats/temporaires, datasets et résultats non
  versionnés.
- Un checkpoint promu conserve les poids, l'état optimiseur, les métriques et les métadonnées de run
  afin de permettre à la fois l'inférence et la reprise du training.
- La promotion recopie le fichier vers le nom versionné et synchronise `profile_id` et le fingerprint
  du nouveau profil dans le checkpoint.
- Les profils historiques et leurs checkpoints ne sont jamais écrasés ; une nouvelle version reçoit
  un nouvel identifiant.
- Le benchmark neural-vs-neural prend deux chemins explicites, ce qui évite de dépendre du profil
  actif pour les comparaisons historiques.
- Le profil actif pointe vers le dernier profil promu, mais les benchmarks historiques utilisent les
  checkpoints versionnés directement.

## Open Questions

- Non-blocking: basculer vers Git LFS si les futurs modèles dépassent une taille raisonnable ou si le
  nombre de versions rend le clone Git pénalisant.

## Proposed Architecture

Le profil d'entraînement reste la source de vérité de la recette. Le checkpoint est son artefact
versionné et contient une copie de traçabilité de cette recette. Le script de validation devient le
seul chemin de promotion : il vérifie le résultat, copie le checkpoint, met à jour ses métadonnées,
écrit le profil versionné et déplace le pointeur actif.

`benchmarks/benchmark_neural_players.py` ajoute le choix `--opponent neural` et
`--opponent-checkpoint`, en conservant les modes Random et heuristique existants.

## Data Model

```text
configs/neural_training_profiles/v001.yaml
configs/neural_training_profiles/v002.yaml
configs/neural_training_profiles/active.yaml
checkpoints/neural/v001.pt
checkpoints/neural/v002.pt
```

Les chemins des profils promus sont relatifs à la racine du dépôt. Le `.gitignore` ne doit pas
exclure `checkpoints/`.

## Observability And Operations

Le checkpoint et le profil indiquent leur version, parent et fingerprint. Le benchmark affiche les
deux checkpoints et son rapport JSON conserve leur chemin. La taille des checkpoints doit être
surveillée avant d'adopter Git LFS.

## Edge Cases

- checkpoint candidat absent ou invalide : aucune promotion ;
- destination de version déjà existante : erreur, aucune écriture destructive ;
- profil actif sans checkpoint : chargement par défaut refusé explicitement ;
- reprise d'un checkpoint promu : le profil et les métadonnées doivent rester cohérents ;
- comparaison d'une version avec elle-même : autorisée pour diagnostic, mais non promotable par la
  règle de validation.

## Testing Strategy

Tester la copie et la synchronisation des métadonnées lors d'une promotion, le refus si la destination
existe, le chargement d'un checkpoint actif, et le benchmark neural-vs-neural. Exécuter la suite
complète sans lancer de campagne longue.

## Rollout And Migration

Copier le baseline actuel dans `checkpoints/neural/v001.pt`, mettre à jour `v001.yaml`, puis utiliser
le chemin canonique pour les futurs trainings. Les anciens artefacts sous `artifacts/` restent
disponibles localement mais ne deviennent pas la référence versionnée.

## Files Expected To Change

- `checkpoints/neural/v001.pt`
- `configs/neural_training_profiles/v001.yaml`
- `shards_ai/ai/neural_training_profiles.py`
- `scripts/validate_neural_profile.py`
- `benchmarks/benchmark_neural_players.py`
- `Makefile`
- `README.md`
- `doc/Current state/Neural player.md`
- tests neural associés
