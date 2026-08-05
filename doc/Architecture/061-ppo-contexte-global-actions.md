# PPO avec contexte global des actions Architecture

## Objective

Préparer un fine-tuning PPO du scoreur `global_candidate_context` après validation de son
réentraînement d’imitation. La séquence expérimentale est obligatoire :

1. mesurer l’imitation hors entraînement par phase et par action ;
2. mesurer le comportement en parties contre Random, v007 et v008 ;
3. lancer PPO seulement si le candidat est au moins décent et comparable au baseline.

## Current State

Le scoreur contextualisé v004 moyenne les embeddings de toutes les actions légales et injecte ce
contexte dans la tête de score. Le PPO actuel construit toujours un `NeuralActorCritic` avec le
backbone indépendant, même si un checkpoint contextualisé est fourni. Il faut donc propager le
type d’architecture dans l’actor-critic et dans les checkpoints PPO.

Le checkpoint mutable unique reste `artifacts/neural_training/checkpoint.pt`. Le checkpoint v004
d’imitation doit être la base du PPO contextualisé, puis sera remplacé par les états PPO successifs.

## Target Behavior

`NeuralActorCritic` doit construire le même backbone que le checkpoint source. Pour l’architecture
contextualisée, les logits actor sont calculés avec le contexte global de chaque liste d’actions,
tandis que le critic conserve une valeur dépendant de l’observation seule. La distribution
`Categorical(logits=...)`, les rollouts, la loss PPO et la KL de référence restent inchangés dans
leur contrat mathématique.

Le checkpoint PPO conserve `architecture: global_candidate_context` et reste directement chargeable
par `NeuralPlayer` pour l’évaluation gloutonne.

## Non-Goals

- Ne pas lancer le PPO avant la validation offline et en parties du v004.
- Ne pas modifier le reward terminal ou ajouter un nouveau shaping dans cette étape.
- Ne pas adapter un Transformer complet.
- Ne pas créer un deuxième checkpoint mutable.

## Key Decisions

- Le profil candidat v005 reprend les hyperparamètres conservateurs v002/v003 : `gamma=1.0`,
  `gae_lambda=1.0`, learning rate `1e-4`, entropie `0.001`, KL de référence `0.02`.
- La référence gelée du PPO est le checkpoint contextualisé v004 lui-même ; la copie de référence
  et le modèle entraîné utilisent exactement la même architecture.
- L’évaluation gloutonne reconstruit `ContextualNeuralActionScorer` depuis l’actor exporté, afin de
  ne pas mesurer par erreur un actor contextualisé avec un scoreur indépendant.
- La policy loss PPO utilise les logits contextualisés ; la valeur du critic ne reçoit pas la liste
  des actions, afin de rester une estimation de l’état et non un artefact de l’ensemble légal.
- La décision de poursuivre dépendra des métriques ciblées et des parties réelles, pas seulement de
  la loss PPO ou du top-1 offline.

## Open Questions

- Blocking avant exécution : confirmer que le checkpoint v004 d’imitation est terminé et disponible
  au chemin canonique avant de démarrer v005.
- Non bloquant : adapter ensuite le reward shaping v003 si le PPO contextualisé progresse sans
  préserver le deckbuilding.

## Proposed Architecture

`NeuralActorCritic(architecture=...)` sélectionne `NeuralActionScorer` ou
`ContextualNeuralActionScorer`. Son actor encode le contexte candidat une fois par décision et
concatène le contexte à chaque couple état/action avant la policy head. Le critic réutilise seulement
`encode_observation` et sa value head.

`from_checkpoint` déduit l’architecture depuis la métadonnée du checkpoint. `inference_state_dict`
exporte les couches contextualisées et la tête actor ; `evaluate_greedy_model` sélectionne le même
scoreur contextualisé pour les parties de validation.

## Rollout And PPO Flow

```text
checkpoint v004 imitation
        ↓
NeuralActorCritic contextualisé + référence gelée
        ↓
rollout contre Random/v007/v008
        ↓
PPO update avec logits contextualisés et critic d’état
        ↓
évaluation gloutonne périodique
        ↓
checkpoint mutable v005
```

## Risks And Validation

Le risque principal est de confondre une amélioration de l’imitation avec une amélioration de
performance réelle. Le protocole doit donc conserver les mêmes seeds, adversaires et limites de
parties que la baseline. Il faut aussi tester que les logits et probabilités restent cohérents
quand la liste des actions est permutée.

## Files Expected To Change

- `shards_ai/ai/rl_training.py`
- `scripts/train_neural_rl.py`
- `configs/neural_training_profiles/candidates/v005.yaml`
- `tests/ai/test_rl_training.py`
- `doc/Current state/Neural player.md`
