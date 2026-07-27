# NeuralPlayer et benchmark en jeu

## Objective

Intégrer le modèle d'imitation dans le cycle de décisions atomiques du moteur afin de mesurer si
les bonnes métriques offline observées sur le batch de 10 000 décisions se traduisent en parties
réelles contre `RandomPlayer` et `HeuristicPlayer`.

Le premier objectif est la validation fonctionnelle et comportementale du checkpoint
`artifacts/neural_imitation/baseline.pt`. L'entraînement sur le dataset complet et le
reinforcement learning restent des étapes ultérieures.

## Current State

- `GameRunner` appelle `player.choose_action(observation, legal_actions)` à chaque décision atomique.
- Les joueurs existants reçoivent un `GameState` et déclarent `observation_is_read_only`.
- `Game.neural_observation_for(player_id)` construit déjà l'observation masquée.
- `NeuralActionScorer` score des `ActionRepresentation` et le checkpoint contient le modèle, le
  vocabulaire de cartes, la configuration et les métriques.
- `representation_for_action(action, state)` dépend encore du `GameState` complet et ne peut donc
  pas être appelé directement dans le `NeuralPlayer` sans risquer d'exposer des informations cachées.

## Target Behavior

À chaque décision où le joueur neural est actif :

1. le moteur calcule les actions légales ;
2. le runner fournit au neural player une `NeuralObservation` masquée ;
3. le joueur convertit chaque action légale en représentation depuis cette observation publique ;
4. le modèle score toutes les actions en batch sous `torch.inference_mode()` ;
5. le joueur retient le score maximal ;
6. en cas d'égalité numérique, `GameRandom` choisit uniformément parmi les meilleures actions ;
7. le runner conserve la validation finale de l'action et applique la transition moteur.

Le joueur doit pouvoir être placé indifféremment comme joueur 1 ou joueur 2 et jouer contre Random,
heuristique ou un autre neural player.

## Non-Goals

- modifier les règles ou `Game.legal_actions()` ;
- révéler le `GameState` complet au réseau ;
- ajouter du reinforcement learning, du self-play ou de la recherche ;
- remplacer automatiquement le profil heuristique de référence ;
- optimiser l'architecture du réseau avant d'avoir une baseline en jeu.

## Key Decisions

- Le `NeuralPlayer` implémente le même protocole `choose_action` que les joueurs existants.
- Le runner utilise un mode d'observation explicite et générique (`game_state` ou `neural`) au lieu
  de faire dépendre le réseau d'un accès au moteur.
- Une nouvelle conversion `representation_for_neural_action(action, observation)` sera la source
  de vérité pour résoudre les cartes publiques d'une action sans consulter les zones cachées.
- Le modèle est chargé depuis un checkpoint validé ; une incompatibilité de vocabulaire, de schéma
  ou de configuration provoque une erreur explicite au chargement.
- L'inférence utilise CPU par défaut, `torch.inference_mode()` et le nombre de threads configuré.
- Le départage des scores utilise le flux aléatoire injectable du joueur, dérivé par le runner comme
  pour `RandomPlayer`, afin de rester reproductible.
- Les décisions légales originales restent alignées positionnellement avec les scores ; le réseau
  ne retourne jamais une action nouvellement construite.
- Les traces détaillées sont optionnelles afin de ne pas ralentir les benchmarks. Le benchmark
  agrégé conserve au minimum les parties, résultats, seeds, décisions et erreurs.

## Open Questions

- La tolérance exacte pour considérer deux scores égaux est non bloquante ; première proposition :
  `1e-6` sur les scores CPU.
- Le checkpoint sera initialement exécuté sur CPU ; le support explicite GPU est non bloquant et
  pourra être ajouté après mesure de la latence.
- La métrique de victoire sera d'abord le taux de victoire et de match nul ; les métriques par phase
  et type d'action pourront être ajoutées au rapport de benchmark.

## Proposed Architecture

### Observation bridge

Ajouter au runner une sélection de représentation d'observation basée sur une capacité déclarée par
le joueur. Les joueurs existants gardent leur comportement actuel et reçoivent le `GameState` ou sa
copie détachée. Le neural player reçoit exclusivement `game.neural_observation_for(player_id)`.

### Public action representation

Résoudre les actions à partir de `NeuralObservation` : main, défausse et champions actifs, champions
adverses, rivière et candidats publics. Les actions qui ciblent une carte adverse cachée doivent
échouer explicitement. La conversion conserve l'ordre de la liste légale et ne modifie aucune action.

### NeuralPlayer

Responsabilités :

- charger et valider le checkpoint ;
- garder le modèle en mode évaluation ;
- encoder et scorer les actions d'une décision ;
- départager les maxima avec le RNG injecté ;
- exposer éventuellement la dernière trace de décision hors du hot path.

Il ne doit pas connaître les transitions de règles ni construire de `GameState`.

### Benchmark

Créer un benchmark dédié paramétrable par :

- checkpoint ;
- adversaire (`random`, profil heuristique ou neural) ;
- nombre de parties ;
- seed de départ ;
- alternance du joueur qui commence ;
- nombre maximal d'actions/tours.

Le rapport doit indiquer le taux de victoire de chaque joueur, les matchs nuls, les parties
terminées, les erreurs illégales, le temps total, les parties par seconde et le temps moyen par
décision neural. Les mêmes seeds et règles de terminaison seront utilisés pour comparer les
adversaires.

## Data Model

Pas de nouvelle donnée persistante moteur. Le checkpoint existant est lu en lecture seule. Les
rapports de benchmark sont des artefacts hors `doc/`, par exemple sous
`artifacts/neural_benchmark/`, et contiennent le hash/chemin du checkpoint, sa configuration, le
dataset d'origine si disponible, les seeds et les versions de schémas.

## Backend Flow

`GameRunner.run()` conserve l'ordre : observation, actions légales, décision, validation, transition.
Une exception de chargement ou une action illégale arrête le benchmark avec la seed concernée ; elle
ne doit pas être transformée silencieusement en choix aléatoire.

## Observability And Operations

Le mode benchmark agrégé est activé par défaut. Le mode diagnostic peut enregistrer pour quelques
décisions : phase, nombre d'actions, type choisi, score maximal, égalité et durée d'inférence. Les
logs détaillés ne doivent pas être activés sur les campagnes longues sans demande explicite.

## Edge Cases

- checkpoint absent, corrompu ou produit avec un vocabulaire de cartes différent ;
- observation et actions vides ;
- action sans carte publique résoluble ;
- scores `NaN` ou infinis : échec explicite du joueur ;
- plusieurs scores maximaux ;
- partie terminée par limite de tours/actions ;
- erreur de joueur ou action illégale retournée au runner.

## Testing Strategy

- tests du bridge `GameState`/`NeuralObservation` dans `GameRunner` ;
- tests de conversion publique de toutes les familles d'actions ;
- test d'interdiction d'accès à une carte cachée adverse ;
- test de chargement et de compatibilité du checkpoint ;
- test de sélection du maximum et du tie-break reproductible ;
- test d'une partie courte NeuralPlayer contre Random ;
- benchmark reproductible sur des seeds fixes, avec comparaison des résultats et de la latence.

## Rollout And Migration

1. intégrer le bridge et le joueur sans modifier les joueurs existants ;
2. valider le checkpoint baseline sur des parties courtes ;
3. lancer le benchmark Neural vs Random puis Neural vs heuristique ;
4. analyser les erreurs et la vitesse ;
5. seulement ensuite décider d'un entraînement sur le dataset complet ou d'un fine-tuning.

## Files Expected To Change

- `shards_ai/game/runner.py` — sélection d'observation ;
- `shards_ai/game/players.py` — contrat/capacité d'observation si nécessaire ;
- `shards_ai/ai/action_representation.py` — conversion depuis observation masquée ;
- `shards_ai/ai/neural_player.py` — joueur et chargement du checkpoint ;
- `shards_ai/ai/__init__.py` — export ;
- `benchmarks/benchmark_neural_players.py` — parties et rapport ;
- `tests/game/test_runner.py` et `tests/ai/test_neural_player.py` ;
- `doc/Current state/Neural player.md` — comportement disponible après implémentation.
