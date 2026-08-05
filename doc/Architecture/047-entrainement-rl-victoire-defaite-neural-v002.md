# Entraînement RL victoire/défaite du NeuralPlayer v002

## Objective

Permettre de fine-tuner le checkpoint neural actuel par renforcement dans de vraies parties de
Shards of Infinity, afin de produire un candidat v002 évalué puis éventuellement promu comme
checkpoint stable.

Le joueur neural doit apprendre uniquement à partir de l'issue de la partie : victoire, défaite ou
nul. Les adversaires d'entraînement sont `RandomPlayer`, `HeuristicPlayer` v007 et
`HeuristicPlayer` v008. Aucun potentiel d'état, score heuristique ou autre reward shaping ne doit
entrer dans la récompense RL.

Critères de succès :

- les parties sont rejouables à seed identique et utilisent uniquement des actions légales du
  moteur ;
- le candidat peut reprendre son état depuis `artifacts/neural_training/checkpoint.pt` ;
- les performances sont suivies séparément contre Random, v007 et v008 ;
- la promotion v002 reste soumise à la validation existante, sans modifier v001 stable avant la
  décision.

## Current State

Le modèle actuel est un `NeuralActionScorer` action-conditionnel, entraîné par
`scripts/train_neural_imitation.py` sur un dataset JSONL de décisions heuristiques. Il reçoit une
observation neural masquée et une représentation de chaque action légale, puis renvoie un score par
action.

`NeuralPlayer` utilise ces scores en mode glouton (`argmax`) et le `GameRunner` orchestre déjà une
partie complète, valide l'action choisie et expose des callbacks de décision et de transition. Le
moteur possède déjà un état terminal explicite (`GameStatus`, `winner`) et une limite de tours qui
produit un nul.

La boucle d'imitation raisonne en epochs sur un dataset fini. Elle ne sait pas collecter des
transitions online, échantillonner une action selon une politique, calculer une valeur d'état ni
reprendre un optimiseur RL. `RewardShapingTracker` calcule des récompenses de transition basées sur
un potentiel d'état ; il est volontairement hors du chemin de cette architecture.

Les checkpoints stables sont sous `configs/neural_profiles/`. Le seul checkpoint mutable est
`artifacts/neural_training/checkpoint.pt`, utilisé par les cibles neural du `Makefile`.

## Target Behavior

Un entraînement RL v002 suit ce cycle :

1. charger le checkpoint de travail, ou initialiser sa copie depuis le checkpoint stable v001 ;
2. sélectionner un adversaire parmi Random, v007 et v008 selon un mélange configuré ;
3. jouer des parties complètes avec un NeuralPlayer d'entraînement stochastique ;
4. attribuer une récompense terminale à chaque transition du joueur neural ;
5. calculer les avantages par épisode et effectuer plusieurs passes PPO sur les transitions
   collectées ;
6. écrire métriques et checkpoint au même chemin canonique ;
7. évaluer périodiquement le candidat sur un panel fixe, puis le soumettre à
   `scripts/validate_neural_profile.py` pour une éventuelle promotion v002.

Le côté du joueur neural alterne de manière déterministe entre les parties. Le seed de chaque
partie est dérivé du seed de campagne et de son index ; le choix de l'adversaire et le choix du côté
ne doivent pas dépendre de l'horloge ou d'une source aléatoire globale non contrôlée.

## Non-Goals

- implémenter le self-play ou un adversaire neural dans cette première boucle RL ;
- modifier les règles du jeu ou contourner `Game.legal_actions()` ;
- réutiliser les scores heuristiques comme labels ou récompenses ;
- ajouter une récompense intermédiaire de santé, maîtrise, achats, dégâts ou durée de partie ;
- réentraîner un checkpoint stable situé sous `configs/neural_profiles/` ;
- remplacer la validation de promotion par une seule métrique d'entraînement ;
- supprimer le pipeline d'imitation, qui reste nécessaire pour initialiser de futurs modèles.

## Key Decisions

- **Algorithme : PPO actor-critic.** Le modèle action-conditionnel existant fournit les logits de
  politique sur les actions légales et un nouveau head de valeur fournit `V(observation)`. PPO est
  préféré à un REINFORCE brut car le signal victoire/défaite est très retardé et que le clipping
  limite les mises à jour destructrices du checkpoint d'imitation.
- **Récompense terminale :** `+1` si le joueur neural gagne, `-1` s'il perd, `0` en cas de nul. Les
  transitions non terminales reçoivent `0`. La récompense est calculée selon l'identité du joueur
  neural, jamais selon le joueur actif au moment où la partie se termine.
- **Pas de shaping :** `RewardShapingTracker`, `state_potential` et les scores
  `HeuristicPlayer` ne sont pas appelés par le trainer RL.
- **Adversaires :** le mélange initial est équilibré, un tiers Random, un tiers v007 et un tiers
  v008. Le mélange est une configuration du profil et doit rester observable dans les métriques.
  Une autre pondération ne doit pas être introduite implicitement dans le code.
- **Unité principale de budget :** le nombre de parties complètes (`total_games`). Une partie est
  un épisode RL ; sa longueur en actions est variable et ne doit pas être assimilée à une partie
  d'epoch.
- **Unité d'optimisation :** les parties sont regroupées en rollouts de `games_per_update`, puis
  les transitions de ces parties sont parcourues `optimization_epochs` fois par PPO. Les epochs PPO
  sont donc des passes sur un rollout déjà collecté, pas des parties supplémentaires.
- **Checkpoint :** tous les updates écrivent `artifacts/neural_training/checkpoint.pt`. Le
  checkpoint contient le réseau actor-critic, l'optimiseur, les compteurs `games_seen` et
  `updates_seen`, l'état RNG, les métriques et l'empreinte du profil. Un checkpoint stable promu
  dans `configs/neural_profiles/v002.pt` n'est plus une sortie d'entraînement.
- **Promotion :** le candidat v002 est accepté uniquement après la campagne de validation
  existante, avec comparaison à v001 et rapport par adversaire. Une progression moyenne masquant une
  régression contre un adversaire ne suffit pas.

## Open Questions

- **Non-blocking :** confirmer les pondérations finales du mélange si l'objectif métier privilégie
  un adversaire particulier. L'architecture propose `1/3, 1/3, 1/3` pour éviter un biais initial.
- **Non-blocking :** fixer après un smoke test les valeurs finales de `games_per_update`, de la
  taille de minibatch et du nombre de passes PPO ; elles doivent être calibrées sur la mémoire et
  le débit local, pas choisies uniquement par convention.
- **Blocking for promotion, not for implementation :** définir le budget total v002 et le seuil de
  validation avant de lancer une campagne longue. Le trainer peut être implémenté et testé avec un
  petit budget déterministe.

## Proposed Architecture

### Profil de training

Étendre `NeuralTrainingProfile` pour accepter `method: ppo` et les paramètres RL. Le profil
reproductible v002 décrit notamment :

```yaml
schema_version: 2
profile_id: v002
parent_profile_id: v001
method: ppo
output: artifacts/neural_training/checkpoint.pt
initial_checkpoint: configs/neural_profiles/v001.pt
seed: 52
total_games: 100000
games_per_update: 128
optimization_epochs: 4
minibatch_size: 2048
gamma: 0.995
gae_lambda: 0.95
clip_epsilon: 0.2
value_loss_coefficient: 0.5
entropy_coefficient: 0.01
opponents:
  random: 0.333333
  v007: 0.333333
  v008: 0.333334
```

Les valeurs numériques sont des valeurs de départ à valider par un smoke test. Le profil doit
également conserver les chemins des profils heuristiques, les limites `max_actions` / `max_turns`,
le nombre de jeux d'évaluation et le seed d'évaluation.

### Politique actor-critic

Créer un module RL qui réutilise l'encodeur et les représentations de cartes du modèle existant :

- un actor produit un logit pour chaque action légale fournie, sans générer d'action illégale ;
- une distribution catégorielle choisit l'action en training ;
- le critic produit une valeur scalaire pour l'observation courante ;
- l'inférence de `NeuralPlayer` reste gloutonne et ne change pas pour v001 ;
- le modèle conserve la compatibilité de vocabulaire et de représentation avec le checkpoint v001.

La politique d'entraînement doit exposer l'action, son log-probability et sa valeur estimée afin que
le collector n'ait pas à refaire un forward incohérent après l'action.

### Collector d'épisodes

Ajouter un collector dédié qui construit un `GameRunner` par partie avec :

- le joueur RL d'un côté ;
- `RandomPlayer`, v007 ou v008 de l'autre ;
- un `GameRandom` dérivé par partie et par joueur ;
- une alternance déterministe du côté du joueur RL.

Le collector utilise la frontière de décision du `GameRunner` et capture uniquement les décisions du
joueur RL : observation neural, actions légales et représentations, action choisie, log-probability,
valeur, récompense après transition et indicateur terminal. Il ne sérialise pas les observations
dans un dataset durable : le rollout reste en mémoire puis est libéré après l'update.

À la fin de la partie, le collector transforme l'issue en récompense terminale. Les nuls causés par
`max_turns` restent explicitement des récompenses nulles et sont comptés séparément.

### Mise à jour PPO

Pour chaque épisode, calculer les retours et les avantages avec GAE sans franchir une frontière de
partie. Normaliser les avantages au niveau du rollout, puis optimiser :

- ratio de politique `exp(new_log_prob - old_log_prob)` ;
- objectif clipped PPO ;
- perte de valeur sur le retour terminal ;
- entropie comme métrique et coefficient configurable pour éviter un effondrement prématuré de la
  politique.

Après `optimization_epochs` passes et les minibatches, incrémenter `updates_seen`, `games_seen` et
écrire le checkpoint atomiquement au chemin unique.

## Data Model

### Transition en mémoire

Une transition RL contient :

```text
episode_id, game_seed, opponent_id, neural_player_id, turn_number
observation, legal_action_representations, chosen_action_index
old_log_probability, value_estimate, reward, done
```

Les actions concrètes peuvent rester en mémoire jusqu'à l'appel à `Game.apply`; le checkpoint ne
contient pas le rollout.

### Checkpoint

Le checkpoint mutable contient :

```text
model_state_dict(actor + critic)
optimizer_state_dict
profile_id, parent_profile_id, profile_fingerprint
update_index, games_seen, transitions_seen
torch/random/game seed state
training_metrics, opponent_mix, algorithm settings
```

Les résultats de campagne et les rapports restent sous `artifacts/neural_benchmark/` ou
`artifacts/neural_validation/`. Aucun dataset de transitions RL géant ne doit être ajouté à `doc/`.

## Training And Evaluation Flow

Le flux recommandé est :

```text
v001 stable
    -> checkpoint mutable unique
    -> rollout de N parties
    -> PPO sur les transitions du rollout
    -> checkpoint mutable mis à jour
    -> évaluation périodique Random / v007 / v008
    -> validation de promotion
    -> configs/neural_profiles/v002.pt si accepté
```

Le training est donc piloté par `total_games`, avec une optimisation intermédiaire par
`games_per_update` et `optimization_epochs`. Les métriques doivent toujours afficher les deux
notions séparément : `games_seen`, `transitions_seen`, `updates_seen` et `optimization_epochs`.

## Observability And Operations

Chaque update journalise au minimum :

- parties et transitions collectées par adversaire ;
- taux de victoire, défaite et nul par adversaire ;
- longueur moyenne et percentile des parties ;
- récompense terminale moyenne ;
- policy loss, value loss, entropie, ratio de clipping et KL approximative ;
- valeur moyenne, avantage moyen et proportion d'actions illégales (attendue à zéro) ;
- durée de collecte, durée d'optimisation, débit de parties et débit de transitions.

Les rapports de benchmark séparent obligatoirement Random, v007 et v008. Une moyenne agrégée peut
être affichée en complément, jamais à la place des trois lignes.

Le checkpoint est écrit via un fichier temporaire puis `os.replace`, comme la promotion actuelle.
Une interruption entre deux updates doit laisser le dernier checkpoint cohérent et reprenable.

## Scalability And Performance Challenge

Le nombre de parties ne fixe pas le coût réel : une partie longue produit beaucoup plus de
transitions qu'une partie courte. Le budget principal doit donc être `total_games` pour contrôler
la couverture des matchups, mais le trainer doit aussi imposer `max_transitions_per_update` ou une
limite mémoire pour éviter qu'un rollout de parties anormalement longues ne remplisse la RAM.

Un entraînement équilibré contre trois adversaires coûte environ trois fois une campagne mono-
adversaire. Le premier run doit être un smoke test court, puis une comparaison à budget identique
entre seeds et configurations. Les benchmarks de promotion restent séparés du budget de training.

Un signal terminal seul peut être trop sparse pour progresser rapidement ou favoriser une stratégie
qui exploite un adversaire faible. C'est pourquoi la validation par adversaire, l'entropie, les
nuls et la longueur des parties sont des garde-fous obligatoires. Si aucun progrès n'apparaît après
un budget mesuré, la prochaine architecture devra d'abord examiner le crédit temporel, le taux
d'exploration et la composition des adversaires avant de réintroduire du shaping.

## Edge Cases

- partie nulle : récompense `0` pour toutes les transitions de l'agent ;
- limite `max_actions` atteinte : erreur d'intégrité, pas une victoire artificielle ;
- action illégale : erreur immédiate et rollout rejeté, jamais un gradient ;
- exception pendant une partie : partie exclue et campagne arrêtée en mode strict ;
- joueur neural placé en Player 2 : signe de la récompense calculé par rapport à son identité ;
- observation ou action non sérialisable : erreur de collector avant l'update ;
- reprise avec un profil, vocabulaire ou modèle incompatibles : refus explicite ;
- checkpoint stable fourni comme sortie d'entraînement : erreur de configuration ou garde-fou avant
  écriture ;
- résultat d'un seul adversaire manquant : validation de promotion incomplète et non promotable.

## Testing Strategy

Tester sans campagne longue :

- récompenses `win=+1`, `loss=-1`, `draw=0`, y compris lorsque le neural est Player 2 ;
- collector déterministe à seed identique, alternance des côtés et sélection des adversaires ;
- zéro reward shaping appelé dans le chemin RL ;
- conservation des actions légales et des log-probabilities ;
- calcul GAE qui ne traverse pas deux épisodes ;
- clipping PPO, perte de valeur, entropie et arrêt sur valeurs non finies ;
- checkpoint/resume avec compteurs et états RNG ;
- smoke test de quelques parties contre Random, v007 et v008 ;
- rapport séparé par adversaire et validation qui refuse un résultat incomplet ;
- compatibilité de chargement du checkpoint stable v001 par l'actor-critic.

Une campagne de référence de promotion doit utiliser des seeds fixes et le même nombre de parties
pour v001 et le candidat v002, conformément à `043-validation-promotion-profils-neural.md`.

## Rollout And Migration

1. Ajouter le profil et les composants RL sans modifier le comportement de `NeuralPlayer` v001.
2. Charger le checkpoint stable v001 comme initialisation et écrire uniquement dans le checkpoint
   mutable canonique.
3. Exécuter un smoke test déterministe avec un très petit `total_games` et vérifier les métriques,
   les limites et la reprise.
4. Lancer une campagne courte à budget identique par seed pour calibrer le mélange et le coût.
5. Lancer le budget v002 retenu, avec évaluations intermédiaires conservées sous `artifacts/`.
6. Valider le candidat sur Random, v007 et v008 ; promouvoir uniquement si la règle de validation
   est satisfaite.
7. Après promotion, laisser v001 intact et enregistrer v002 sous `configs/neural_profiles/v002.pt`.

Rollback : supprimer le candidat mutable ou le remplacer par une copie de travail ; le pointeur
stable `configs/neural_profiles/active.yaml` ne change qu'après promotion atomique.

## Files Expected To Change

- `shards_ai/ai/neural_training_profiles.py` — paramètres et validation du profil PPO ;
- `shards_ai/ai/neural_model.py` ou nouveau module actor-critic — tête policy et tête value ;
- nouveau `shards_ai/ai/rl_training.py` — rollout, récompense terminale, GAE et PPO ;
- nouveau `scripts/train_neural_rl.py` — CLI et checkpoint/resume ;
- `Makefile` — cibles RL utilisant `NEURAL_CHECKPOINT` et budgets en parties ;
- `scripts/validate_neural_profile.py` — compatibilité avec les profils PPO ;
- `tests/ai/` — tests unitaires et smoke tests RL ;
- `configs/neural_training_profiles/v002.yaml` ou profil candidat temporaire ;
- `doc/Current state/Neural player.md` et `README.md` — workflow final et métriques.
