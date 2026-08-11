# Training imitation macro et état du deck — Architecture

## Objective

Entraîner un scoreur neural capable de choisir entre les conséquences stratégiques produites par
`PlayTurnSolver`, puis de rejouer la trace atomique sélectionnée. Le modèle doit conserver les
cardinalités de zones et la composition factionnelle déjà définies par `deck_state_v1`.

Le premier dataset cible les démonstrations de Heuristic V8 contre Heuristic V7 et Neural V4.

## Current State

Le dataset macro est produit par `generate_macro_imitation_dataset.py`. Ses décisions `macro_play`
contiennent une observation, une liste de candidats macro et `chosen_candidate_index`. Ses décisions
`atomic` couvrent les choix stratégiques qui ne passent pas par le solveur PLAY.

Le trainer historique `train_neural_imitation.py` attend des actions atomiques et ne peut pas lire
ce schéma. Le scoreur V004 utilise `structured_semantic_v5_fusion_experiment` avec le feature set
baseline. Le feature set `deck_state_v1` existe déjà et ajoute sept cardinalités de zones et quatre
comptes factionnels actifs.

## Target Behavior

Ajouter un scoreur et un trainer dédiés :

1. l’observation est encodée par le même encodeur structuré que V004, avec `deck_state_v1` activé ;
2. chaque candidat macro est encodé par ses types de trace, son résultat résumé et sa longueur ;
3. le modèle produit un score par candidat ;
4. la loss combine classement pair-à-pair du teacher et cross-entropy sur le candidat choisi ;
5. les splits restent groupés par `game_id` ;
6. le checkpoint de travail reste `artifacts/neural_training/checkpoint.pt`.

## Non-Goals

- modifier le moteur ou les actions atomiques ;
- modifier le solveur ou ses budgets fixes ;
- faire choisir au modèle l’ordre des actions dans une trace ;
- modifier V004 ou un checkpoint stable ;
- entraîner le modèle macro avec le trainer atomique historique ;
- ajouter les cardinalités dans le JSONL : elles sont dérivées de l’observation existante.

## Key Decisions

1. **Architecture séparée.** Le scoreur est identifié par `structured_semantic_v5_macro_deck_state_v1`.
   Il ne surcharge pas l’architecture V004 active.
2. **Réutilisation de l’état du deck.** `NeuralModelConfig.observation_feature_set` vaut
   `deck_state_v1`, ce qui active les sept cardinalités et quatre factions dans l’état du modèle.
3. **Transfert contrôlé.** Le trainer peut initialiser les modules communs depuis V004 via la
   migration `baseline -> deck_state_v1`; les nouvelles colonnes d’état et le nouvel encodeur macro
   sont initialisés à zéro ou selon l’initialisation PyTorch standard documentée.
4. **Représentation macro bornée.** Elle utilise les types d’actions de la trace, les trois états
   terminaux, la phase, sept scalaires de conséquence et la longueur de trace. Aucun identifiant
   d’instance caché n’est introduit.
5. **Budget stable.** Le profil candidat fixe `epochs`, `max_records`, `max_validation_records`,
   seed et threads. Les valeurs ne sont pas ajustées automatiquement à chaque exécution.
6. **Validation séparée.** Une validation externe peut être fournie comme JSONL ; sinon le split
   `game_id` du dataset est utilisé.

## Open Questions

1. La promotion du scoreur macro nécessitera un adapter de benchmark qui joue les traces macro ;
   elle est hors de ce premier trainer.
2. Les décisions atomiques du dataset sont conservées pour une phase ultérieure ; le premier
   trainer cible uniquement `macro_play`, afin de ne pas mélanger deux espaces de sortie.

## Proposed Architecture

### Scoreur

`MacroActionScorer` réutilise `encode_observation()` du scoreur structuré V005 deck-state, puis
encode chaque `MacroActionRepresentation` dans un vecteur fixe :

- histogramme des types d’actions de la trace ;
- type terminal et phase ;
- compteurs normalisés de ressources, zones et longueur.

Un MLP partagé transforme chaque candidat et un second MLP produit son score avec l’état observé.
Le nombre de candidats reste variable et l’ordre des candidats reste celui fourni par le solveur.

### Trainer

`train_macro_imitation.py` charge le profil, stream le JSONL, filtre `decision_kind=macro_play`,
construit les observations avec `observation_from_dict`, applique la loss ranking + chosen-index,
écrit métriques et checkpoint atomiquement selon les conventions existantes.

## Data Model

Le dataset existant conserve son schéma macro. Aucun champ de jeu n’est ajouté. Le checkpoint
contient l’architecture, `NeuralModelConfig`, les dimensions de représentation et l’identifiant du
profil. Toute incompatibilité de feature set ou d’architecture doit être refusée au chargement.

## Performance And Risks

Le coût par record est proportionnel au nombre de candidats, plafonné par le budget du solveur. Le
dataset est lu en streaming ; les observations ne sont pas chargées en mémoire. Le principal risque
est la différence de distribution entre les candidats générés par le solveur et ceux rencontrés par
le futur joueur macro, à mesurer au benchmark.

## Testing Strategy

- dimensions et valeurs de la représentation macro ;
- forward fini avec un nombre variable de candidats ;
- présence des cardinalités et factions dans la dimension d’état ;
- loss et métriques sur un record synthétique ;
- filtrage des décisions atomiques ;
- sauvegarde/rechargement du scoreur ;
- smoke training sur quelques records ;
- vérification que V004 reste chargeable par son architecture historique.

## Rollout And Migration

Le profil macro reste candidat. Le trainer écrit uniquement le checkpoint mutable canonique. Aucun
fichier `configs/neural_profiles/vNNN.pt` n’est modifié. La promotion exigera ensuite un adapter de
joueur macro, un benchmark complet et le protocole qualité existant.

## Files Expected To Change

- `shards_ai/ai/macro_model.py` ;
- `shards_ai/ai/macro_training.py` ;
- `scripts/train_macro_imitation.py` ;
- `configs/neural_training_profiles/candidates/exp00109-macro-v8-deck-state.yaml` ;
- `tests/ai/test_macro_model.py` ;
- `tests/ai/test_macro_training.py` ;
- exports dans `shards_ai/ai/__init__.py`.
