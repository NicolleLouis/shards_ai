# PPO deckbuilding du joueur Hybrid V003

## Objectif

Entraîner par PPO uniquement la politique de deckbuilding de `hybrid-v003`, en lui
faisant jouer des parties complètes. La reward est strictement terminale, depuis le
joueur apprenant : `+1` victoire, `-1` défaite, `0` partie nulle. Aucun shaping ne
doit être ajouté.

Composition fixe pendant l'expérience :

```text
acquisition / deckbuilding -> PPO entraînable
PLAY                       -> Heuristic V008 fixe
BANISH                     -> deterministic_blaster_crystal fixe
```

Le deckbuilding couvre les décisions déjà routées par `HybridPlayer` : `BuyCard`,
`RecruitMercenary`, `RecruitFreeCard` et `StopBuying`.

## État actuel

Le dépôt contient `HybridPlayer` et `configs/hybrid_profiles/hybrid-v003.yaml`, qui
référencent respectivement `neural_v006`, `heuristic_v008` et
`deterministic_blaster_crystal`. Il contient aussi `NeuralActorCritic`,
`PPOTrainingPlayer`, `PPOTrainingMacroPlayer`, `terminal_reward` et `gae_returns`.

Le collecteur PPO actuel joue des parties complètes, mais le chemin macro V4 expose
les décisions macro/atomiques du joueur neural général. Il ne respecte donc pas encore
la frontière d'apprentissage `HybridPlayer` : ses transitions peuvent concerner PLAY.
Le besoin est un nouveau contrat de collecte spécialisé, pas un simple changement de
nom de profil.

Le moteur reste l'autorité pour les actions légales et les transitions. Les politiques
fixes sont exécutées dans la même partie réelle et leurs effets sont l'environnement
du learner.

## Comportement cible

Pour chaque partie :

1. construire exactement la composition `hybrid-v003` ;
2. injecter une politique PPO uniquement dans le composant acquisition ;
3. laisser V008 et la politique banish choisir toutes les autres actions du learner ;
4. jouer jusqu'à la terminaison normale ;
5. enregistrer une transition uniquement pour chaque décision PPO acquisition ;
6. mettre `0.0` sur toutes ces transitions sauf sur la dernière, qui reçoit la reward
   terminale et `done=True` ;
7. calculer GAE sans traverser les frontières d'épisodes.

PLAY et banish sont donc des actions de l'environnement : elles influencent le résultat
mais ne produisent ni transition, ni log-probabilité, ni valeur à optimiser.

## Hors périmètre

- entraîner PLAY ou banish ;
- ajouter du shaping sur achats, deck, ressources, maîtrise, dégâts ou banishment ;
- modifier les règles ou la légalité du moteur ;
- entraîner plusieurs têtes de politique dans cette première expérience ;
- modifier `configs/neural_profiles/v006.pt` ;
- promouvoir automatiquement le checkpoint ;
- considérer une loss PPO ou un smoke test comme preuve de qualité compétitive.

## Décisions clés

### 1. La frontière est la famille `acquisition`

L'adaptateur PPO reçoit uniquement les actions que `HybridPlayer` identifie comme
acquisition. Il ne reçoit jamais `PlayCard`, `ActivateChampion`, `GainMastery`,
`PassPlayPhase`, `BanishCard` ou `SkipBanish`.

Les actions doivent rester choisies parmi les actions légales et être validées par
`Game.apply()`. Les actions fixes ne sont pas transformées en transitions artificielles.

### 2. La partie complète est l'environnement

Le learner ne s'arrête pas après la phase d'achat. Les achats sont évalués par leur
effet sur la partie complète avec PLAY et banish V003. La reward terminale est attachée
à la dernière décision acquisition ; `gamma=1.0` et `gae_lambda=1.0` font remonter ce
résultat aux décisions acquisition antérieures.

### 3. Initialisation et checkpoint

Le modèle part de `configs/neural_profiles/v006.pt`. Le checkpoint de travail unique
reste `artifacts/neural_training/checkpoint.pt`, avec état actor-critic et optimiseur.
Une version stable n'est créée qu'après validation, dans `configs/neural_profiles/vNNN.pt`.

Le profil doit déclarer explicitement `decision_family=acquisition`, la composition
`hybrid-v003` et `reward_contract=terminal_win_loss_only`.

Si le candidat passe la gate officielle, la promotion ne remplace pas `hybrid-v003`.
Elle produit le prochain profil neural stable pour l'acquisition PPO, par exemple
`v007.pt` si ce numéro est bien le prochain emplacement disponible, puis
`configs/hybrid_profiles/hybrid-v004.yaml`. Ce profil V4 référence le nouveau neural
stable et conserve PLAY V008 ainsi que le banish déterministe de V3.

V3 reste donc le contrôle historique complet et V4 devient la composition
`acquisition PPO + PLAY V008 fixe + banish fixe`. La promotion technique du checkpoint
neural et la promotion qualité du joueur hybride sont deux étapes liées mais distinctes.

### 4. L'environnement fixe est versionné

Chaque campagne charge les chemins exacts du profil V003 : V006 pour l'acquisition
initiale, `configs/heuristic_profiles/v008.yaml` pour PLAY et
`configs/player_policies/banish/v001.yaml` pour banish. Le fingerprint du profil
hybride doit être conservé dans les métadonnées. Une modification de PLAY ou banish
constitue une nouvelle expérience.

### 5. Le shaping est refusé explicitement

Le profil doit exiger `reward_shaping: {}`. Le chargeur et le collecteur doivent
refuser toute configuration non vide, ainsi que toute tentative de calcul de delta
de transition. Les champs `card_values`, `beta` et `clip` ne doivent pas pouvoir
activer silencieusement un shaping.

### 6. Critic partagé, décisions spécialisées

Le critic reste une fonction de l'observation d'état et n'est évalué que lors des
décisions acquisition conservées. Il n'y a pas de critic ou de head PPO pour PLAY ou
banish dans cette expérience.

## Protocole d'évaluation et d'entraînement

Le panel de promotion reprend la gate officielle courante, définie dans
`scripts/validate_neural_profile.py::QUALITY_OPPONENT_WEIGHTS` :

```yaml
v007: 1.0
v008: 1.5
hybrid:v002: 0.5
hybrid:v003: 1.5
neural:v002: 0.5
neural:v005: 1.0
neural:v006: 1.0
```

Random reste hors de la gate officielle. La validation finale joue **200 parties par
adversaire**, avec les mêmes seeds candidat/référence, puis applique la règle actuelle
de moyenne pondérée strictement positive ; aucun adversaire individuel n'est une garde
dure de non-régression. La validation doit utiliser le chemin batch et être terminée
pour tous les adversaires avant d'interpréter la décision.

Le validateur actuel construit toutefois un `NeuralPlayer` générique pour le candidat.
Il devra être étendu ou complété pour construire le candidat comme `HybridPlayer` avec
la politique PPO acquisition et PLAY/banish V003 ; sinon la gate mesurerait une autre
composition que celle entraînée. Le pool et les poids de la gate restent inchangés.

Le mélange d'adversaires de l'entraînement et du contrôle reprend le même pool et les
mêmes poids que cette gate. Il ne doit pas être redéfini séparément dans un script.

Pour le premier essai, la campagne est limitée à un budget de **2 à 3 heures maximum**.
Elle commence par un smoke test, puis un batch court permettant de mesurer le débit réel
de parties, de transitions acquisition et d'updates PPO. Le nombre de parties total et
le nombre d'updates seront déterminés à partir de cette mesure ; aucune extrapolation
de la seule durée d'une partie ne suffit.

Le learning rate du premier candidat est fixé à `0.0005`, conformément à la recette PPO
récente retenue dans le dépôt. Le smoke test et le batch initial utiliseront cette valeur
depuis V006 ; toute comparaison ultérieure de learning rate devra repartir du même
checkpoint parent et des mêmes seeds.

La validation complète représente 1 400 parties pour le candidat et 1 400 pour la
référence, soit 2 800 exécutions de parties. Elle doit être planifiée comme une étape
distincte du batch d'entraînement ou son coût doit être inclus explicitement dans le
budget global.

Les statistiques PLAY/banish ne sont pas conservées dans le rapport d'apprentissage :
elles ne sont ni optimisées, ni nécessaires pour la décision de promotion. Seuls les
compteurs utiles au contrat PPO sont conservés : parties, transitions acquisition,
types de décisions acquisition, rewards terminales, résultats par adversaire et
épisodes tronqués.

## Architecture proposée

```text
GameRunner -> HybridPlayer -> DecisionRouter
                              |
             +----------------+----------------+
             |                                 |
   actions acquisition                    autres actions
             |                         +--------+--------+
   PPOAcquisitionPolicy              PLAY V008       BANISH fixe
             |                         |                  |
             +-------------------------+------------------+
                                      |
                                  Game.apply()
                                      |
                              partie complète
                                      |
                     reward terminale sur la dernière
                     transition acquisition seulement
```

### Adaptateur PPO acquisition

Ajouter un adaptateur dans `shards_ai/ai/rl_training.py` ou dans un module dédié :

```python
choose_action(observation: GameState, legal_actions: Sequence[Action]) -> Action
pop_last_decision() -> AcquisitionDecisionPayload
```

Il doit construire `Game.neural_observation_for(player_id)`, représenter les candidats
acquisition avec le contrat neural existant, échantillonner une `Categorical`, puis
conserver observation, candidats, index choisi, ancienne log-probabilité, valeur et
numéro de tour. Un appel PLAY ou banish ne doit pas remplir la file de payloads.

### Collecteur hybride

Créer une variante de `_collect_episode`/`collect_rollout` qui construit le learner
avec `build_hybrid_player(..., profile="hybrid-v003")`, en injectant la politique PPO
dans l'acquisition seulement. Le collecteur consomme un payload uniquement si le
`DecisionDiagnostic` indique `decision_family == "acquisition"` et le `policy_id`
PPO attendu.

`RolloutResult` doit distinguer transitions acquisition, parties et résultats par
adversaire. Les transitions doivent porter `decision_family="acquisition"` pour rendre
une mauvaise route détectable ; les actions fixes PLAY/banish n'ont pas à être comptées.

### Reward et épisodes

Toutes les transitions intermédiaires ont reward `0.0`. À la fin d'une partie terminée,
la dernière transition acquisition reçoit `terminal_reward(final_state, learner_id)`
et `done=True`. `gae_returns` doit réinitialiser son état sur chaque `episode_id`.

Une partie sans décision acquisition est une erreur de contrat pour l'entraînement.
Une partie interrompue par `max_actions` ou `max_turns` doit être signalée comme
tronquée et ne doit pas être convertie implicitement en victoire, défaite ou draw.

## Configuration et modèle de données

Ajouter `configs/neural_training_profiles/candidates/ppo-deckbuilding-hybrid-v003.yaml`
avec au minimum :

```yaml
schema_version: 1
profile_id: ppo-deckbuilding-hybrid-v003
parent_profile_id: v006
method: ppo
learning_rate: 0.0005
output: artifacts/neural_training/checkpoint.pt
initial_checkpoint: configs/neural_profiles/v006.pt
composition_profile: configs/hybrid_profiles/hybrid-v003.yaml
decision_family: acquisition
gamma: 1.0
gae_lambda: 1.0
reward_shaping: {}
opponents:
  v007: 1.0
  v008: 1.5
  hybrid:v002: 0.5
  hybrid:v003: 1.5
  neural:v002: 0.5
  neural:v005: 1.0
  neural:v006: 1.0
metadata:
  objective: ppo_deckbuilding_only_full_game_terminal_outcome
  reward_contract: terminal_win_loss_only
  fixed_play_policy: heuristic_v008
  fixed_banish_policy: deterministic_blaster_crystal
```

Les champs `composition_profile` et `decision_family` doivent être validés par
`NeuralTrainingProfile` plutôt que rester des métadonnées libres. Le dataset
d'imitation n'est pas obligatoire : PPO collecte des trajectoires en ligne.

## Pré-estimation par smoke test

Le chemin PPO macro actuellement implémenté a été mesuré avec V005, un thread PyTorch,
2 parties et 1 époque d'optimisation : 3,964 s de collecte pour 161 transitions,
7,684 s d'update, soit 11,647 s au total hors évaluation. Le script d'entraînement
complet a pris 18,63 s avec son évaluation initiale réduite à une partie par adversaire.

Cette mesure est un ordre de grandeur du moteur et du PPO, pas une validation de la
future collecte Hybrid V003. La nouvelle collecte peut être plus rapide ou plus lente
car elle ne score que l'acquisition, mais elle joue toujours les mêmes parties
complètes. Le premier batch réel doit donc commencer avec une durée bornée, mesurer
le débit après un update, puis choisir le nombre de parties restant sous le budget de
2 à 3 heures. Les tests PPO existants passent actuellement : 22 tests ciblés réussis.

## Flux d'entraînement

1. charger et valider le profil PPO, le checkpoint initial et le fingerprint V003 ;
2. jouer `games_per_update` parties avec seeds, adversaires et limites reproductibles ;
3. conserver seulement les transitions acquisition ;
4. attribuer une seule reward terminale par épisode ;
5. calculer GAE puis exécuter l'update PPO ;
6. sauvegarder atomiquement le checkpoint mutable ;
7. reprendre depuis ce checkpoint après interruption ;
8. évaluer périodiquement l'acteur glouton avec la même composition fixe.

La collecte est le chemin chaud : elle exécute aussi les décisions PLAY/banish, mais
seules les transitions acquisition et le temps total de collecte sont nécessaires au
rapport PPO.

## Reproductibilité, observabilité et opérations

Le workload de référence fixe seeds, mélange d'adversaires, `torch_threads`, parties
par update, limites du runner et checkpoint initial. Chaque rollout conserve au minimum
`episode_id`, seed, adversaire, joueur, tour, observation masquée, candidats, choix et
ancienne log-probabilité.

Les métadonnées conservent aussi le fingerprint V003, le profil PPO, le checkpoint de
départ, les transitions par type acquisition, les résultats par adversaire et les
épisodes tronqués. Les métriques offline et PPO restent diagnostiques;
la promotion dépend de parties complètes reproductibles contre le panel de référence.

## Cas limites

- singleton acquisition : une vraie décision PPO est conservée ; seul un replay
  automatique est exclu ;
- banishment présent pendant PLAY : routage exclusif vers banish fixe ;
- partie nulle : reward terminale `0.0` ;
- terminaison par limite : épisode tronqué, sans label implicite ;
- absence de décision acquisition : erreur explicite ;
- reprise : vérifier profil, architecture, fingerprint, dimensions et optimizer state ;
- changement de V008 ou banish : nouvelle campagne et nouveau fingerprint.

## Tests et validation

Ajouter ou adapter des tests pour vérifier :

- construction exacte de V003 et injection de PPO dans acquisition seulement ;
- payload sur `BuyCard`, `RecruitMercenary`, `RecruitFreeCard`, `StopBuying` ;
- absence de payload sur PLAY et banish ;
- application réelle des politiques fixes dans une partie complète ;
- une seule reward terminale sur la dernière transition acquisition ;
- GAE sans fuite inter-épisode avec `gamma=1`, `lambda=1` ;
- rejet de tout `reward_shaping` non vide ;
- reproductibilité à configuration et seeds identiques ;
- sauvegarde/rechargement du checkpoint et des métadonnées.

Avant toute campagne longue : smoke test de quelques parties, puis un update complet
avec contrôle du nombre de transitions, de leur famille et de la position de la reward.
La validation compétitive est ensuite menée sur le panel complet, séparément de cette
implémentation.

## Déploiement, migration et rollback

Il n'y a pas de migration moteur ni de données. L'ordre est : implémenter l'adaptateur
et le collecteur, ajouter le profil candidat, exécuter les tests et le smoke test,
initialiser `NEURAL_CHECKPOINT` depuis V006, puis lancer les updates avec sauvegarde
atomique. En cas de rejet, V006 et `hybrid-v003` restent inchangés et le checkpoint
candidat n'est pas actif. En cas d'acceptation, copier le checkpoint vers le prochain
profil neural stable, créer `hybrid-v004` en conservant PLAY/banish V3, puis valider
séparément les pointeurs actifs.

## Fichiers attendus à modifier

- `shards_ai/ai/rl_training.py` ou nouveau module PPO acquisition ;
- `shards_ai/ai/composed_player.py` si l'injection de politique doit y être exposée ;
- `shards_ai/ai/neural_training_profiles.py` pour valider les nouveaux champs ;
- `scripts/train_neural_rl.py` et `Makefile` pour sélectionner le collecteur sans
  créer de checkpoint parallèle ;
- `configs/neural_training_profiles/candidates/ppo-deckbuilding-hybrid-v003.yaml` ;
- tests PPO, composition hybride et profil.

La documentation de `doc/Current state/` sera mise à jour uniquement après
implémentation et validation ; cette architecture reste historique.

## Critères de réussite

L'implémentation est conforme lorsque seules les décisions acquisition entrent dans
PPO, PLAY et banish restent exactement ceux de V003, chaque épisode complet produit
une unique reward terminale, aucun shaping n'est accepté, la collecte est reproductible
et la qualité est jugée par un panel de parties complètes sans promotion automatique.
