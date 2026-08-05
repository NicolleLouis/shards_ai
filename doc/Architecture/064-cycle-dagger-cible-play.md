# Cycle DAgGER ciblé sur la politique Neural

## Objective

Améliorer le NeuralPlayer actuel sur les états qu'il rencontre réellement en
partie, avec une priorité sur les décisions de la phase `PLAY`, sans perdre les
acquis du dataset d'imitation historique. Le cycle doit produire un dataset
reproductible, un fine-tuning depuis le checkpoint courant et des mesures avant
et après sur données anciennes, états on-policy et parties réelles.

## Current State

Le générateur `shards_ai.ai.imitation_dataset.generate_dataset` produit des
décisions d'Heuristic V8 avec toutes les actions légales, leurs représentations,
les scores heuristiques et l'issue finale. Le benchmark
`scripts/benchmark_neural_visited_states.py` observe les décisions du
NeuralPlayer pendant des parties contre v008 et calcule déjà accord, rang et
regret contrefactuels, mais ne sérialise pas encore un dataset d'entraînement
DAgger.

Le modèle actuel est chargé depuis `artifacts/neural_training/checkpoint.pt`.
Le script `scripts/train_neural_imitation.py` sait reprendre un checkpoint,
réduire le learning rate et réutiliser les représentations existantes. Les
datasets et rapports expérimentaux restent sous `artifacts/`.

## Target Behavior

### Collecte on-policy

Le NeuralPlayer courant joue des parties contre un mélange configurable de :

- Heuristic V8 ;
- Heuristic V7 ;
- checkpoints Neural précédents explicitement fournis ;
- lui-même.

La répartition par matchup et la permutation des côtés sont déterministes. À
chaque décision du NeuralPlayer, le collecteur conserve toutes les décisions,
pas uniquement `PLAY` :

- observation masquée et phase ;
- toutes les actions légales et leurs représentations ;
- scores v008 de toutes les actions ;
- action choisie par le Neural et action choisie par v008 ;
- rang v008 de l'action Neural, regret heuristique et indicateurs top-1/top-3 ;
- type exact de décision, numéro de tour et première divergence éventuelle ;
- issue finale de la partie.

Le résultat brut est distinct du dataset historique et possède un manifest avec
les checkpoints, profils, seeds, matchups et compteurs.

### Priorisation et assemblage

Un second outil construit le dataset de fine-tuning avec une composition par
défaut :

- 45 % d'anciens exemples hors politique ;
- 35 % de nouveaux exemples on-policy de phase `PLAY` ;
- 20 % d'autres décisions on-policy.

Les exemples on-policy `PLAY` sont tirés avec priorité décroissante sur :

1. divergences non équivalentes ;
2. regret élevé ;
3. action v008 hors top-3 du Neural ;
4. états juste après une première divergence ;
5. phases dont l'état de fin diffère réellement.

Les exemples ne sont pas dupliqués sans limite : la sélection est pondérée,
déterministe par seed et conserve les compteurs de chaque bucket. Les décisions
hors `PLAY` restent présentes pour éviter une distribution artificiellement
tronquée.

### Équivalence de trajectoire

Une divergence d'action n'est pas automatiquement une erreur stratégique. Pour
les phases `PLAY`, le collecteur enregistre une comparaison de trajectoire :

- l'état de début de phase est cloné ;
- une branche exécute le Neural jusqu'au passage de phase ;
- une branche exécute v008 sur les mêmes états ;
- les états de fin sont comparés sur les informations observables et les
  conséquences de jeu.

Une phase est considérée équivalente si elle conserve les mêmes ressources,
cartes restantes/jouées, champions, effets déclenchés, santé, maîtrise et
ensemble de décisions disponibles à la sortie. Les identifiants d'instances et
les différences d'ordre sans conséquence sont normalisés. La comparaison doit
utiliser une copie contrôlée de l'état et de la source aléatoire afin de ne pas
modifier la partie principale.

Le rapport distingue donc `action_divergence`, `equivalent_play_phase` et
`strategic_divergence`. L'échantillonnage prioritaire utilise la dernière
catégorie, tout en conservant les autres pour audit.

## Non-Goals

- Réécrire immédiatement l'architecture du réseau ou passer à PPO.
- Supprimer les exemples anciens ou remplacer le dataset historique.
- Entraîner uniquement sur `play_card` et oublier les autres décisions.
- Considérer chaque différence d'ordre comme une erreur de politique.

## Key Decisions

- Le teacher est toujours Heuristic V8 pour les labels du cycle, quel que soit
  l'adversaire ayant produit l'état.
- Les adversaires servent à produire des trajectoires variées, pas à fournir
  des labels concurrents.
- Le dataset DAgger brut et le dataset assemblé sont deux artefacts distincts,
  avec manifests et seeds propres.
- Le fine-tuning reprend les poids et l'architecture du checkpoint courant avec
  un learning rate réduit, par défaut `0.1 ×` le learning rate d'imitation
  précédent et une à trois époques. L'état de l'optimiseur sera comparé en
  réinitialisation et en reprise contrôlée si la première mesure le justifie.
- Le checkpoint de travail reste exclusivement
  `artifacts/neural_training/checkpoint.pt`. Le cycle écrit d'abord ses métriques
  et résultats de validation ; aucune promotion automatique n'est effectuée.
- Le split de validation historique reste indépendant et sert à détecter
  l'oubli catastrophique. Les métriques on-policy ont leur propre split par
  `game_id`.
- Les mesures minimales avant/après sont : top-1 historique, top-1 on-policy,
  top-1 `play_card` on-policy, accord de phase équivalente, regret on-policy,
  winrate contre v007 et winrate contre v008.

## Open Questions

- Non bloquant : le volume initial peut être fixé à 1 000 parties par matchup
  ou à une cible de décisions ; les deux modes seront supportés.
- Non bloquant : le seuil exact de regret et la définition de l'état
  "juste après divergence" doivent être des paramètres du sampler, avec des
  valeurs par défaut documentées après une première distribution.
- Non bloquant : les actions dont l'ordre est équivalent peuvent rester dans
  l'évaluation top-1, mais elles ne doivent pas recevoir la même priorité
  d'entraînement qu'une divergence stratégique.
- Non bloquant : pour le premier fine-tuning, je recommande de réinitialiser
  l'optimiseur afin de ne pas conserver un momentum lié à l'ancien dataset ; la
  reprise de son état reste disponible comme variante expérimentale.

## Proposed Architecture

Le cycle est séparé en trois étapes exécutables :

1. `collect_dagger_dataset` joue les parties et sérialise les décisions ainsi
   que les labels v008 ;
2. `sample_dagger_dataset` mélange l'historique et l'on-policy suivant les
   quotas et priorités ;
3. `train_neural_imitation.py --resume-from` réentraîne le modèle courant,
   puis les benchmarks existants mesurent le résultat.

La comparaison de phases est un composant d'analyse indépendant du moteur. Le
moteur continue de valider toutes les actions ; aucune stratégie n'est codée
dans `Game`.

## Data Model

Le schéma DAgger étend le schéma de décision existant sans supprimer ses clés.
Les champs supplémentaires proposés sont :

- `dagger_cycle`, `teacher_profile_id`, `teacher_action_index` ;
- `neural_action_index`, `neural_action_type`, `teacher_action_type` ;
- `teacher_raw_ranks`, `teacher_scores`, `regret`, `teacher_top3` ;
- `first_divergence`, `decision_after_first_divergence` ;
- `play_phase_id`, `play_phase_start`, `play_phase_end` ;
- `play_phase_equivalent`, `strategic_divergence` ;
- `final_outcome` et les métadonnées du matchup.

Les sorties proposées sont :

- `artifacts/imitation_dataset/dagger_cycle_1_raw.jsonl` ;
- `artifacts/imitation_dataset/dagger_cycle_1_train.jsonl` ;
- manifests JSON associés ;
- `artifacts/analysis/dagger_cycle_1.html` et JSON de distribution ;
- métriques de fine-tuning sous `artifacts/neural_training/`.

## Backend Flow

Le collecteur construit un `GameRunner` avec le NeuralPlayer courant et un
opposant issu du calendrier. Le `decision_observer` ne conserve que les
décisions du joueur neural évalué, puis demande à une instance indépendante de
Heuristic V8 de classer les mêmes actions légales avant toute transition.

Pour une phase `PLAY` candidate à la comparaison, le collecteur capture un
snapshot sérialisable et utilise des branches de simulation isolées. Une erreur
de comparaison ne doit pas invalider la partie principale ; elle est comptée
dans le manifest et rend l'exemple non prioritaire.

Le sampler charge l'ancien dataset en streaming, réserve son split de validation,
calcule les buckets on-policy, puis écrit un dataset assemblé dans un fichier
temporaire avant remplacement atomique. Il doit pouvoir reprendre ou être
relancé à seed identique sans créer de doublons implicites.

## Observability And Operations

Chaque campagne imprime et écrit : parties tentées/terminées, décisions,
matchups, phases, action types, divergences, équivalences, regrets, erreurs de
branche et distributions effectivement sélectionnées. Les rapports séparent
les métriques d'action des métriques de phase complète.

## Edge Cases

- Une partie arrêtée par `max_actions` est exclue ou marquée explicitement,
  jamais silencieusement considérée comme une victoire.
- Une phase sans divergence ne produit pas d'exemple prioritaire artificiel.
- Une phase avec plusieurs ordres équivalents reste conservée mais porte
  `strategic_divergence=false`.
- Les checkpoints Neural de versions précédentes doivent fournir la même
  architecture, le même vocabulaire de cartes et un chemin explicite.
- Les actions légales vides, scores non finis ou snapshots non comparables sont
  des erreurs de collecte visibles dans le manifest.

## Testing Strategy

- Reproductibilité du collecteur sur une petite campagne et seed identique.
- Vérification que toutes les actions légales et les labels v008 sont conservés.
- Tests du sampler sur les quotas 45/35/20 et les priorités de regret/divergence.
- Tests de l'équivalence sur une permutation sans conséquence et sur une phase
  qui change les ressources ou les cartes.
- Smoke training avec learning rate réduit et checkpoint temporaire contrôlé.
- Suite complète avant de lancer la campagne lourde.

## Rollout And Migration

L'implémentation sera ajoutée sans modifier le dataset historique. Après
validation du collecteur et du sampler, une campagne DAgger sera générée. Le
fine-tuning sera lancé depuis le checkpoint courant avec métriques avant/après.
Le checkpoint ne sera considéré comme meilleur que si le gain on-policy ne
s'accompagne pas d'une régression inacceptable sur l'ancien holdout ou les
benchmarks v007/v008.

## Files Expected To Change

- `scripts/collect_dagger_dataset.py` ;
- `shards_ai/analysis/dagger_dataset.py` ou module équivalent ;
- `scripts/sample_dagger_dataset.py` ;
- `scripts/train_neural_imitation.py` et profils d'entraînement ;
- `Makefile` ;
- `tests/analysis/` et `tests/ai/` ;
- `doc/Current state/Neural player.md` et `doc/Current state/Analysis.md`.
