# Analyse offline de l’imitation neural Architecture

## Objective

Mesurer, sur un dataset hors entraînement imitant l’heuristique v008, la distance entre un
checkpoint neural et les décisions du teacher. Le rapport HTML doit fournir au minimum :

- l’accord top-1 et top-3 ;
- le regret heuristique moyen et le score heuristique moyen du choix teacher ;
- ces indicateurs par phase, par famille de décision et par `action_type`.

Le calcul doit fonctionner sur le dataset 1M en streaming et produire un artefact autonome sous
`artifacts/analysis/`.

## Current State

Le dataset `artifacts/imitation_dataset/v008_vs_random_v007_1m.jsonl` contient déjà, pour chaque
décision, l’observation masquée, les représentations des actions légales, les scores heuristiques
et `chosen_action_index`. `shards_ai.ai.neural_training.iter_jsonl_records` le lit ligne par ligne.

`NeuralActionScorer` est action-conditionné : il doit scorer toutes les représentations légales
d’une décision. Les checkpoints enregistrent `model_config` et `model_state_dict`; le checkpoint
mutable canonique est `artifacts/neural_training/checkpoint.pt`, mais un profil stable peut être
passé explicitement.

## Target Behavior

Un script CLI charge un checkpoint et un JSONL, infère sans gradient, classe les actions par score
neural et agrège les métriques globales, par phase, par famille de décision et par action exacte.
Il écrit un JSON de synthèse optionnel et un HTML autonome lisible dans un navigateur.

Les groupes demandés sont : Achat (`buy_card`), Attaque (`assign_power`), Recrutement
(`recruit_free_card`, `recruit_mercenary`) et Ciblage (action dont `target` est renseigné). Le
ciblage est explicitement non exclusif avec les autres groupes. Le rapport inclut aussi le détail
exact par `action_type` pour éviter de masquer ce recouvrement.

## Non-Goals

- Aucun entraînement, ajustement de checkpoint ou modification du dataset.
- Aucun benchmark de parties et aucune conclusion sur le taux de victoire.
- Aucun recalcul de l’heuristique : les scores présents dans le dataset sont la référence.

## Key Decisions

- Le calcul est streaming et utilise `torch.inference_mode()` avec un seul modèle en mémoire.
- Le top-1 est vrai si l’action teacher est première dans le classement neural ; le top-3 est vrai
  si elle est dans les trois premières, bornées par le nombre d’actions légales.
- Le regret utilise le score heuristique de l’action teacher moins celui de l’action neural top-1.
  Les égalités de score neural sont départagées de façon déterministe par l’ordre du dataset.
- Les décisions invalides sont rejetées explicitement plutôt que comptées silencieusement.
- Les sorties générées restent sous `artifacts/analysis/` et ne sont pas ajoutées à `doc/`.

## Open Questions

- Blocking: aucune pour cette première analyse ; les catégories de décision sont documentées et
  pourront être ajustées dans une future architecture si une taxonomie métier plus fine est
  souhaitée.

## Proposed Architecture

`shards_ai/analysis/neural_imitation.py` porte les agrégateurs réutilisables et le rendu HTML.
`scripts/analyze_neural_imitation.py` porte uniquement le CLI, le chargement du checkpoint et le
parcours JSONL. Le module n’accède jamais au moteur mutable : il utilise les observations et
représentations sérialisées du dataset.

Chaque ligne produit un enregistrement de métriques avec le rang neural, le score teacher de
référence, le regret et les dimensions de regroupement. Les agrégats conservent seulement des
compteurs et sommes ; la mémoire est donc indépendante du nombre de décisions.

## Data Model

Le JSON de sortie contient les métadonnées d’entrée, le nombre de lignes analysées et des groupes
`overall`, `by_phase`, `by_decision_type`, `by_action_type`. Chaque groupe expose `records`,
`top1_agreement`, `top3_agreement`, `mean_heuristic_score` et `mean_heuristic_regret`.

## Observability And Operations

Le CLI affiche le nombre de lignes analysées. `--max-records`, `--split` et `--split-seed`
permettent un contrôle rapide ou une analyse d’un split indépendant. Par défaut, `--split non_train`
inclut validation et test et exclut le train selon le même hash de `game_id` que l’entraînement.
Le HTML rappelle les chemins, le checkpoint et la définition du regret.

## Edge Cases

Les lignes sans actions, avec des longueurs incohérentes, ou avec un index teacher hors limites
sont des erreurs. Une ligne à une seule action a un top-1 et un top-3 définis, et un regret nul.
Les catégories sans décision sont affichées comme `n/a` plutôt que comme zéro.

## Testing Strategy

Ajouter des tests unitaires sur le rang, le regret, les regroupements non exclusifs, les erreurs de
format et le rendu HTML. Exécuter ces tests ciblés puis le test suite disponible sans toucher aux
artefacts existants.

## Rollout And Migration

Aucune migration. Le script est utilisable avec le checkpoint mutable courant ou un profil stable
et son résultat est régénérable à partir des mêmes entrées.

## Files Expected To Change

- `shards_ai/analysis/neural_imitation.py`
- `scripts/analyze_neural_imitation.py`
- `tests/analysis/test_neural_imitation.py`
- `Makefile` pour une cible d’analyse reproductible
