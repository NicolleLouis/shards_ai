# DAgGER étape 2 : pondération ciblée et fusion des datasets

## Objective

Produire un second dataset DAgGER à partir des états visités par le Neural
actuel, en donnant plus de poids aux familles identifiées comme coûteuses :
`play_card`, `recruit_mercenary`, `assign_power` et
`choose_pending_decision`. Les labels doivent rester exclusivement ceux de
Heuristic V8. Une étape de fusion réunira ensuite le dataset historique,
DAgGER 1 et DAgGER 2 avant le prochain fine-tuning.

## Current State

`scripts/collect_dagger_dataset.py` collecte les décisions du Neural et calcule
les scores de toutes les actions légales avec le profil passé par `--profile`.
Le contrat d'entraînement utilise `chosen_action_index` et `heuristic_scores`;
ces champs doivent donc toujours désigner l'action et les scores du teacher,
avec l'action Neural conservée séparément pour l'analyse.

`scripts/sample_dagger_dataset.py` sait échantillonner un dataset DAgGER avec
des priorités de regret, de rang et de divergence, mais ne permet pas encore
de pondérer explicitement les types d'action ni de fusionner plusieurs cycles.

## Target Behavior

Le cycle 2 utilise le checkpoint Neural courant pour visiter de nouvelles
trajectoires. Chaque décision est étiquetée par Heuristic V8, y compris les
décisions hors `PLAY`. Le sampler applique en plus des multiplicateurs de
priorité par `teacher_action_type` :

- `play_card` : 1.5 ;
- `recruit_mercenary` : 3.0 ;
- `assign_power` : 3.0, uniquement utile lorsque le regret est élevé ;
- `choose_pending_decision` : 2.0 ;
- autres actions : 1.0.

Ces multiplicateurs se combinent avec les indicateurs existants de divergence,
regret, rang hors Top-3 et divergence stratégique. Ils ne changent pas le label
teacher et ne dupliquent pas mécaniquement les lignes.

Un script séparé fusionne les datasets historiques, DAgGER 1 et DAgGER 2, en
ajoutant la provenance de chaque ligne et en vérifiant que les datasets DAgGER
portent bien `teacher_profile_id=v008`.

## Non-Goals

- Remplacer Heuristic V8 par l'action du Neural comme label.
- Réentraîner pendant la collecte ou le sampling.
- Modifier la définition des actions légales ou les règles du moteur.
- Promouvoir automatiquement le checkpoint du cycle 2.

## Key Decisions

- Le teacher est validé explicitement comme profil `v008` lors de la collecte.
- `chosen_action` et `chosen_action_index` restent l'action v008 ;
  `neural_action` et `neural_action_index` décrivent séparément le Neural.
- La fusion conserve toutes les lignes disponibles et ajoute `dataset_source`
  et `dagger_stage` sans réécrire les scores.
- Les holdouts historiques et DAgGER sont fusionnés séparément du dataset
  d'entraînement.
- Le nouveau training sera lancé depuis le checkpoint v004 pré-DAgGER sauvegardé,
  avec un learning rate contrôlé.

## Open Questions

- Non bloquant : les multiplicateurs peuvent être ajustés après la distribution
  du cycle 2 sans modifier le collecteur ni les labels.
- Non bloquant : la fusion complète peut produire plus d'un million de lignes ;
  un sampler global pourra être ajouté ensuite si le temps d'entraînement devient
  limitant.

## Proposed Architecture

Le workflow est :

1. collecter `dagger_cycle_2_raw.jsonl` avec v008 comme teacher ;
2. produire `dagger_cycle_2_train.jsonl` et son holdout avec les pondérations ;
3. fusionner l'historique, DAgGER 1 et DAgGER 2 ;
4. vérifier les manifests et les labels ;
5. entraîner depuis la baseline v004 sauvegardée.

Le moteur et `GameRunner` restent inchangés.

## Data Model

Les datasets fusionnés conservent le schéma de décision existant et ajoutent :

- `dataset_source` : chemin logique de la source ;
- `dagger_stage` : `historical`, `dagger_1` ou `dagger_2` ;
- `dagger_priority_weight` : poids appliqué au sampling, informatif seulement.

Le manifest indique les comptes par source, étape, phase, action teacher et
profil teacher. Une erreur de profil teacher invalide la fusion.

## Testing Strategy

- vérifier que le collecteur refuse un profil teacher différent de v008 ;
- vérifier que le label d'entraînement est toujours `teacher_action_index` ;
- tester les multiplicateurs par type d'action ;
- tester la fusion et la provenance sur de petits JSONL ;
- vérifier les manifests avant toute campagne lourde.

## Files Expected To Change

- `shards_ai/analysis/dagger_dataset.py` ;
- `scripts/collect_dagger_dataset.py` ;
- `scripts/sample_dagger_dataset.py` ;
- `scripts/merge_dagger_datasets.py` ;
- `Makefile` ;
- tests d'analyse et d'IA ;
- `doc/Current state/Neural player.md`.
