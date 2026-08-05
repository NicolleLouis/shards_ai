# Contexte global des alternatives pour le réseau neural

## Objective

Tester une évolution intermédiaire du scoreur neural : chaque action légale doit être évaluée en
tenant compte d’un résumé de l’ensemble des alternatives légales disponibles dans la même décision.
L’objectif prioritaire est d’améliorer l’imitation de `recruit_mercenary`, `pass_play_phase` et
`stop_buying`, sans introduire immédiatement un Transformer complet.

## Current State

`NeuralActionScorer` encode l’observation une fois et chaque `ActionRepresentation` séparément,
puis calcule `score(observation, action_i)`. La liste complète est passée au forward, mais les
embeddings d’actions ne communiquent pas entre eux. `NeuralPlayer` choisit ensuite le score maximal.

Le dataset d’imitation contient déjà la liste variable d’actions, leurs scores heuristiques et le
choix teacher. Le PPO utilise encore directement le backbone indépendant et n’est pas inclus dans
cette première expérience.

## Target Behavior

Le nouveau scoreur encode toutes les actions, calcule leur moyenne d’embeddings comme contexte global,
transforme ce contexte, puis concatène ce vecteur à l’état et à chaque action avant le score final :

```text
actions -> embeddings -> pooling global -> contexte
                              ↘ état + action_i + contexte -> score_i
```

Le modèle reste permutation-invariant vis-à-vis de l’ordre de la liste. L’API conserve la
correspondance `legal_actions[i] ↔ representations[i] ↔ scores[i]`; le classement est produit par
le joueur en triant les logits.

## Non-Goals

- Aucun Transformer ni attention action-vers-action dans cette expérience.
- Aucun changement de `Game.legal_actions()` ou `Game.apply()`.
- Aucun réentraînement PPO avant validation offline de l’imitation.
- Aucun remplacement automatique du checkpoint stable v001.

## Key Decisions

- Créer un scoreur distinct `ContextualNeuralActionScorer`; le scoreur baseline reste disponible.
- Ajouter `candidate_context_dim` à `NeuralModelConfig` avec une valeur par défaut stable.
- Identifier l’architecture dans les métadonnées du checkpoint afin que `NeuralPlayer` et les
  scripts d’analyse chargent le bon scoreur.
- Réentraîner le modèle : les couches de contexte et la nouvelle tête de score n’existent pas dans
  les checkpoints baseline. Un warm-start partiel pourra être étudié ultérieurement, mais ne sera
  pas confondu avec une comparaison à architecture et initialisation contrôlées.
- Comparer le candidat et le baseline sur le même split hors entraînement, le même dataset et les
  mêmes métriques par action.
- Une amélioration offline ne suffit pas à promouvoir le modèle : un benchmark de parties reste
  obligatoire.

## Open Questions

- Non bloquant : après validation offline, faut-il adapter le PPO à cette architecture ou conserver
  le scoreur contextualisé uniquement comme expérience d’imitation ?
- Non bloquant : comparer ultérieurement moyenne, somme et pooling avec attention.

## Proposed Architecture

Le scoreur contextualisé réutilise les encodeurs de cartes, d’état et d’actions du baseline. Il
calcule un pooling moyen des action embeddings, passe ce vecteur dans `candidate_context_encoder`,
puis ajoute le contexte à l’entrée de la tête `scorer`. Aucun encodage de position n’est utilisé.

Le CLI d’entraînement sélectionne l’architecture par profil. Le checkpoint conserve `architecture`,
`model_config` et les poids. Le chargement du joueur et du rapport offline devient compatible avec
les deux architectures.

## Training And Evaluation

La loss paire-à-paire existante reste utilisable car les logits sont toujours alignés sur les
actions légales. La loss d’action choisie reste également active. Une future expérience listwise
pourra être ajoutée séparément.

Le candidat doit être évalué avec `scripts/analyze_neural_imitation.py` sur `--split non_train`,
notamment pour `recruit_mercenary`, `pass_play_phase`, `stop_buying`, puis dans des parties contre
Random, v007 et v008.

## Performance And Risks

Le pooling est en `O(n)` et ne nécessite ni padding ni masque d’attention. Il coûte une couche
supplémentaire par décision et conserve le hot path CPU actuel. En contrepartie, le contexte moyen
perd l’identité précise des alternatives et peut être insuffisant pour des comparaisons fines entre
deux cartes ou deux cibles.

## Files Expected To Change

- `shards_ai/ai/neural_model.py`
- `shards_ai/ai/neural_player.py`
- `shards_ai/ai/neural_training.py` et `scripts/train_neural_imitation.py`
- `scripts/analyze_neural_imitation.py`
- `configs/neural_training_profiles/candidates/v004.yaml`
- `doc/Ideas.md` et `doc/Current state/Neural player.md`
- tests neural associés
