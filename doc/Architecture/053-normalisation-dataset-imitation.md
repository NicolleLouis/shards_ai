# Normalisation du dataset d'imitation

## Objective

Produire un dataset d'entraînement d'imitation rééquilibré à partir d'un dataset brut de parties
heuristiques, afin que les décisions d'achat et de deckbuilding soient suffisamment représentées
sans modifier la distribution naturelle utilisée pour la validation et le test.

## Current State

`scripts/generate_imitation_dataset.py` produit un JSONL contenant une ligne par décision et un
manifest associé. Le dataset brut 1M est distribué selon les décisions naturelles de v008. Les
champs `observation.phase`, `chosen_action.action_type` et `game_id` permettent respectivement le
rééquilibrage par phase, l'échantillonnage par action et la séparation sans fuite entre parties.

## Target Behavior

Un nouveau script lit le JSONL brut et produit :

- un train normalisé à 100 000 décisions par défaut ;
- une validation naturelle ;
- un test naturel ;
- un manifest détaillant les quotas, les disponibilités, les sélections et les manques éventuels.

Les quotas par défaut du train sont 60 % `buy`, 10 % `attack` et 30 % `play`. Dans `play`,
`banish_card` et `skip_banish` représentent ensemble 5 % du train ; leur ratio est dérivé du
dataset de référence.

## Non-Goals

- modifier les lignes du dataset ou les observations ;
- créer des décisions synthétiques ou dupliquer des exemples ;
- rééquilibrer validation et test ;
- changer le moteur, la stratégie heuristique ou le trainer neural.

Cette normalisation par pruning reste un outil d'analyse et de génération de candidats. Le chemin
de training recommandé conserve le dataset naturel et utilise une pondération modérée de la loss ;
un dataset normalisé ne doit pas devenir automatiquement le nouveau dataset de référence.

## Key Decisions

- La séparation train/validation/test est faite par `split_for_game_id`, jamais ligne par ligne.
- Le dataset de référence 100k fournit les proportions internes des actions dans chaque phase.
- La sélection est déterministe grâce à un hash de la seed, du bucket, du `game_id`, de l'index de
  décision et de la ligne source.
- L'algorithme fonctionne en deux passes et conserve uniquement des numéros de ligne sélectionnés,
  pas les 1M observations en mémoire.
- Une catégorie insuffisamment présente est signalée dans le manifest ; elle n'est pas compensée par
  duplication implicite.

## Proposed Architecture

`normalize_imitation_dataset.py` effectue :

1. comptage des actions du dataset de référence ;
2. calcul des quotas cibles par phase et action ;
3. première lecture du dataset brut pour compter les candidats train et sélectionner les meilleurs
   hash par bucket ;
4. seconde lecture pour écrire les lignes train retenues dans l'ordre source ;
5. écriture des lignes validation/test naturelles et du manifest.

Les buckets de sélection sont les couples `(phase, action_type)`. Les quotas `play` de bannissement
sont donc garantis séparément pour `banish_card` et `skip_banish` tout en conservant leur ratio de
référence.

## Testing Strategy

- tester l'allocation exacte des quotas et les arrondis ;
- vérifier la reproductibilité avec une seed identique ;
- vérifier que les parties ne sont pas partagées entre splits ;
- vérifier les fréquences et les shortfalls sur un petit JSONL synthétique ;
- exécuter le script sur le dataset 1M et comparer le manifest aux quotas attendus.

## Files Expected To Change

- `scripts/normalize_imitation_dataset.py` ;
- `tests/ai/test_imitation_dataset_normalization.py` ;
- `doc/Architecture/053-normalisation-dataset-imitation.md`.
