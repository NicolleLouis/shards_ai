# Promotion Neural V006 par fine-tuning PPO

## Décision

Le profil neural actif passe de V005 à V006 après un fine-tuning PPO du joueur macro/atomique
unifié. V006 conserve l'architecture `structured_semantic_v5_macro_tactical_action_v1` et le
contrat `PlayTurnSolver` de V005 ; il ne constitue pas une nouvelle représentation.

Le fine-tuning est parti de `configs/neural_profiles/v005.pt` avec `learning_rate=0.0005`,
`gamma=1`, `gae_lambda=1` et une récompense uniquement terminale. Sur 2 304 parties de training,
la validation paired de 100 parties par adversaire a mesuré un gain pondéré de `+1,125 point` contre
V005. Le candidat a été accepté par la gate et promu avec un digest identique entre le checkpoint
mutable et `configs/neural_profiles/v006.pt`.

## Cycle de vie

- `configs/neural_profiles/v006.pt` est stable et ne doit plus être entraîné en place.
- `configs/neural_training_profiles/v006.yaml` décrit la recette promue et son parent V005.
- `artifacts/neural_training/checkpoint.pt` reste le seul checkpoint mutable.
- `configs/neural_profiles/active.yaml` et `configs/neural_training_profiles/active.yaml`
  pointent vers `v006`.

## Évaluation

Le panel de qualité actif comprend V007, V008 et Neural V001 à V006. Random reste un benchmark
diagnostique, hors gate. La gate exige une moyenne pondérée strictement positive ; aucun adversaire
individuel n'est une contrainte dure.
