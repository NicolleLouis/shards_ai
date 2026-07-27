# Profil du joueur neural actif

## Objective

Séparer la recette d'entraînement du joueur neural livrable. Les training profiles décrivent comment
reproduire un run ; les neural profiles contiennent les checkpoints versionnés utilisables en partie
et indiquent la version active, comme les profils heuristiques indiquent une version de stratégie.

## Current State

Les recettes sont sous `configs/neural_training_profiles/`. Les checkpoints promus sont actuellement
sous `checkpoints/neural/`, tandis que `NeuralPlayer` sans chemin explicite résout le profil actif de
training pour trouver son checkpoint. Les informations de recette ne sont pas nécessaires à
l'inférence : le checkpoint contient les poids et `model_config`.

## Target Behavior

Créer `configs/neural_profiles/` comme espace canonique des joueurs neural livrables :

```text
configs/neural_profiles/v001.pt
configs/neural_profiles/active.yaml
```

`active.yaml` contient l'identifiant de la version active. `NeuralPlayer(checkpoint_path=None)` charge
le checkpoint correspondant à ce pointeur. Un chemin explicite continue de permettre les benchmarks
historiques et les expériences candidates.

Les training profiles restent sous `configs/neural_training_profiles/` et pointent vers le checkpoint
livrable lorsqu'ils décrivent une version promue. Les checkpoints candidats restent dans `artifacts/`.

## Non-Goals

- dupliquer les poids dans un YAML ;
- déplacer les datasets ou métriques dans `configs/` ;
- supprimer la possibilité de charger un checkpoint explicite ;
- modifier le format du modèle ou les règles du jeu.

## Key Decisions

- Le checkpoint `.pt` est le neural profile versionné ; il contient les poids, `model_config`, les
  métriques et les métadonnées de training.
- `configs/neural_profiles/active.yaml` est le pointeur de production pour le joueur neural par
  défaut.
- La promotion copie le checkpoint candidat dans `configs/neural_profiles/vNNN.pt` et met à jour ce
  pointeur seulement après validation.
- `configs/neural_training_profiles/vNNN.yaml` reste la recette historique associée et son `output`
  pointe vers le checkpoint neural canonique.
- Les checkpoints promus sont suivis par Git ; les checkpoints candidats, datasets et rapports restent
  sous `artifacts/`.

## Open Questions

- Non-blocking: adopter Git LFS si le volume des checkpoints futurs rend Git classique pénalisant.

## Proposed Architecture

Ajouter un chargeur `load_active_neural_profile` dans le module de profils. Il lit `active.yaml`,
résout `vNNN.pt` dans le même dossier et vérifie l'existence du fichier. `NeuralPlayer` l'utilise
uniquement lorsque `checkpoint_path` et `scorer` sont absents.

Le script de validation utilise le dossier neural profile comme destination de promotion. Il conserve
le training profile séparé pour les paramètres de run et synchronise le `profile_id` et le fingerprint
du checkpoint promu.

## Data Model

```yaml
# configs/neural_profiles/active.yaml
schema_version: 1
active_profile_id: v001
```

Le fichier `configs/neural_profiles/v001.pt` est un checkpoint PyTorch autonome.

## Testing Strategy

Tester la résolution du pointeur actif, le refus d'un checkpoint actif absent, le chargement par
défaut de `NeuralPlayer`, la promotion vers le nouveau dossier et le benchmark avec chemins
explicites.

## Rollout And Migration

Déplacer `checkpoints/neural/v001.pt` vers `configs/neural_profiles/v001.pt`, créer `active.yaml`,
mettre à jour `v001.yaml`, puis changer les defaults du validateur et du `NeuralPlayer`. Les anciens
chemins candidats sous `artifacts/` restent inchangés.

## Files Expected To Change

- `configs/neural_profiles/v001.pt`
- `configs/neural_profiles/active.yaml`
- `configs/neural_training_profiles/v001.yaml`
- `shards_ai/ai/neural_training_profiles.py`
- `shards_ai/ai/neural_player.py`
- `scripts/validate_neural_profile.py`
- `README.md`
- `doc/Current state/Neural player.md`
- tests neural associés
