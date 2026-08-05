# Architecture d'embedding des cartes pour NeuralPlayer V003

## Objectif

Introduire une architecture expérimentale de représentation des cartes donnant davantage de
capacité aux caractéristiques sémantiques et relationnelles, sans rendre les checkpoints V001 et
V002 illisibles. V003 est un candidat de recherche : il ne devient le profil actif qu'après une
validation indépendante contre les profils existants.

## État courant

`NeuralActionScorer` encode actuellement chaque carte par la concaténation d'un embedding
d'identité (`card_definition_id`) et d'un encodage sémantique structuré, fusionnés en une
représentation de dimension `card_embedding_dim` (32 par défaut). V001 utilise
`independent_action`. V002 utilise `global_candidate_context` et ajoute un résumé permutation
invariante des actions légales.

Le checkpoint contient déjà `architecture`, `model_config`, `card_ids` et `model_state_dict`.
`NeuralPlayer`, l'imitation et le PPO sélectionnent leur classe à partir de cette métadonnée, avec
un fallback historique vers `independent_action` lorsque la métadonnée est absente.

## Comportement cible

- Un checkpoint V001 doit continuer à charger `NeuralActionScorer` avec exactement sa configuration.
- Un checkpoint V002 doit continuer à charger `ContextualNeuralActionScorer` avec exactement sa
  configuration.
- Un checkpoint V003 doit porter l'architecture `semantic_identity_v3` et charger une classe
  dédiée, sans heuristique sur le nom du fichier.
- Le moteur, les actions légales, l'observation masquée et le contrat de décision de
  `NeuralPlayer` ne changent pas.
- Une expérience V003 doit pouvoir comparer plusieurs tailles sans changement de code : identité
  16 ou 24, couche sémantique 48 ou 64, représentation finale de carte 64.

## Hors périmètre

- Modifier les règles du jeu ou la visibilité de l'observation.
- Réentraîner ou remplacer V001/V002.
- Promouvoir automatiquement V003.
- Ajouter une attention entre cartes ou entre actions : ce document ne valide pas encore le coût
  et le bénéfice d'un Transformer.

## Décisions clés

1. Le choix d'architecture est une métadonnée explicite du checkpoint, et non une inférence à
   partir de `profile_id`, du chemin ou de la taille des tenseurs.
2. Les classes historiques sont conservées. Leur défaut de configuration et leurs noms de clés
   restent compatibles avec les checkpoints existants.
3. V003 réutilise le contrat action-conditionnel et le pooling permutation-invariant des zones,
   mais utilise une voie d'identité configurable (16/24), une voie sémantique configurable
   (48/64), puis une représentation fusionnée de carte configurable et fixée à 64 dans le profil
   candidat initial.
4. La factory de modèles est la seule résolution canonique de l'identifiant d'architecture ;
   elle est utilisée par l'inférence, l'imitation, le PPO et les analyses.
5. La promotion suit la validation existante : mêmes seeds et panel indépendant contre Random,
   v007, v008 et les profils neural disponibles ; aucune mise à jour de `active.yaml` pendant
   l'expérimentation.

## Architecture proposée

`SemanticIdentityNeuralActionScorer` reprend l'encodeur d'état, l'encodeur d'action et le scorer
action-conditionnel de la baseline. Pour chaque carte, il calcule :

```text
features sémantiques (46) -> Linear(semantic_hidden_dim) -> ReLU -> Linear(card_embedding_dim)
card_definition_id -> Embedding(card_id_embedding_dim)
concaténation -> Linear(card_embedding_dim) -> ReLU -> représentation finale
```

Le profil candidat V003 utilise `card_id_embedding_dim=24`, `semantic_hidden_dim=64` et
`card_embedding_dim=64`. Les deux premiers paramètres peuvent être abaissés à 16 et 48 dans une
expérience contrôlée ; ils sont enregistrés dans `model_config` pour rendre le checkpoint
reproductible.

La dimension finale de 64 augmente le coût des onze pools de cartes et des forwards d'actions.
Cette hausse est acceptée pour l'expérience, mais doit être mesurée avec le benchmark neural
existant avant toute décision de promotion.

## Compatibilité et rollout

Les profils stables restent `configs/neural_profiles/v001.pt` et `v002.pt`. Le candidat initial
est décrit sous `configs/neural_training_profiles/candidates/v003.yaml` et écrit dans le seul
checkpoint mutable `artifacts/neural_training/checkpoint.pt`. Une promotion, si elle est validée,
créera le prochain fichier stable ; elle ne modifiera pas les poids V001/V002.

## Tests et validation

- vérifier la résolution explicite des trois architectures ;
- vérifier qu'un scorer V003 produit des scores finis et respecte l'équivariance à l'ordre des
  actions ;
- charger réellement V001 et V002 avec `NeuralPlayer.load_scorer` ;
- vérifier que `NeuralActorCritic` construit et restaure V003 ;
- exécuter les tests IA ciblés puis la suite disponible ;
- mesurer V003 contre les profils de référence avec un workload et des seeds indépendants avant
  toute promotion.

## Questions ouvertes

- La meilleure combinaison identité 16/24 et sémantique 48/64 reste empirique (non bloquant pour
  le branchement architectural).
- La représentation sémantique doit-elle intégrer davantage de relations entre cartes, ou le
  pooling actuel suffit-il ? À décider uniquement après analyse offline et benchmarks en parties.

## Fichiers attendus

- `shards_ai/ai/neural_model.py` : classe V003 et factory d'architecture.
- `shards_ai/ai/neural_player.py`, `shards_ai/ai/rl_training.py` et scripts d'entraînement/analyse :
  résolution via la factory.
- `configs/neural_training_profiles/candidates/v003-embedding.yaml` : recette expérimentale.
- `tests/ai/test_neural_model.py`, `tests/ai/test_neural_player.py`, `tests/ai/test_rl_training.py` :
  couverture de compatibilité et de restauration.
