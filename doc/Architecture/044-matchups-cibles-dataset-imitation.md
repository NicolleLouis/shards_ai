# Matchups ciblés pour le dataset d'imitation

## Objective

Permettre de générer un dataset où certains profils heuristiques produisent les labels et où les
profils adverses servent uniquement à faire évoluer les parties. Le cas principal est un dataset
avec des décisions v008 uniquement, dans des parties contre RandomPlayer et v007.

## Current State

`generate_imitation_dataset.py` accepte plusieurs `--profile`, mais `default_matchups` construit
tous les matchups entre ces profils et le générateur enregistre les décisions de tous les joueurs
heuristiques. Il est donc impossible de demander v008 comme unique teacher tout en utilisant v007
comme adversaire sans ajouter également les labels v007.

## Target Behavior

Lorsque `--opponent-profile` est fourni, chaque `--profile` devient un profil teacher : le générateur
crée un matchup teacher contre RandomPlayer et un matchup teacher contre chaque profil adverse. Les
décisions des teachers sont écrites ; les décisions des adversaires heuristiques sont jouées mais
ignorées pour le dataset.

Sans `--opponent-profile`, le comportement historique reste inchangé : les profils fournis forment
le pool de teachers et les décisions de tous les joueurs heuristiques sont enregistrées.

## Non-Goals

- modifier les règles ou les stratégies heuristiques ;
- changer le format des observations ou des labels ;
- retirer les décisions adverses des parties, uniquement du dataset ciblé ;
- modifier les campagnes existantes qui n'utilisent pas le nouveau mode.

## Key Decisions

- `--profile` conserve le sens de profil producteur de labels dans le mode ciblé.
- `--opponent-profile` est répétable et ajoute toujours un matchup contre RandomPlayer en plus des
  adversaires demandés.
- `target_decisions` compte uniquement les enregistrements effectivement écrits, donc uniquement les
  décisions des teachers en mode ciblé.
- Le manifest conserve les profils, les matchups et les profils enregistrés afin de rendre la
  campagne vérifiable.
- Les décisions ignorées ne sont pas sérialisées puis filtrées après coup : elles ne consomment pas
  de mémoire dataset inutile et évitent les pipelines temporaires fragiles.

## Open Questions

Aucune question bloquante pour cette évolution.

## Proposed Architecture

Ajouter `record_profile_ids` à `DatasetCampaignConfig`. Le générateur vérifie ce filtre au moment du
callback de décision. Ajouter une construction de `MatchupSpec` dans le CLI lorsque des profils
adverses sont fournis ; la configuration générale du générateur reste réutilisable par les tests et
les appels Python.

## Data Model

Le schéma de chaque ligne reste inchangé. Le manifest ajoute `record_profile_ids` lorsque le mode
ciblé est utilisé. Les champs `heuristic_profile_id` et `opponent_profile_id` restent présents sur
les lignes conservées.

## Testing Strategy

Tester que le mode ciblé construit teacher/random et teacher/adversaire, que seules les décisions du
teacher sont enregistrées, et que `target_decisions` s'applique à ces seules lignes. Tester que le
mode historique conserve les décisions de tous les profils.

## Rollout And Migration

Le nouveau mode est opt-in via `--opponent-profile`. Les datasets existants et leurs manifests ne
sont pas modifiés.

## Files Expected To Change

- `shards_ai/ai/imitation_dataset.py`
- `scripts/generate_imitation_dataset.py`
- `tests/ai/test_imitation_dataset.py`
- `README.md`
- `doc/Current state/Neural player.md`
