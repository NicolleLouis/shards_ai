# Observation partielle du réseau neural

## Objectif

Définir une représentation sérialisable, déterministe et sans fuite d'information pour le futur
réseau neural. Cette observation doit être exprimée du point de vue du joueur actif et contenir les
informations qu'un joueur humain peut connaître au moment de chaque décision atomique.

Le step couvre la construction et les tests de l'observation. Il ne construit pas encore le card
embedding, l'encodage des actions ni le modèle neural.

## État actuel

`Game.observation_for(player_id)` retourne actuellement une copie détachée de `GameState` complète.
Cette copie contient notamment :

- la main et la pioche des deux joueurs ;
- les défausses, zones de jeu et champions des deux joueurs ;
- l'ordre du deck central et l'ordre des pioches personnelles ;
- les identifiants de cartes et les décisions internes.

Cette observation est adaptée aux joueurs heuristiques actuels, qui utilisent l'état complet du
moteur, mais elle ne peut pas être transmise au réseau neural.

Les structures sources sont `GameState`, `PlayerState`, `CardInstance` et `CardDefinition`. Le
catalogue contient actuellement cinq valeurs de faction, dont `neutral`. Le masque demandé pour le
réseau comporte quatre positions, une par faction jouable (`maquis`, `spectra`, `homodeus`, `order`) ;
`neutral` n'en fait pas partie car elle ne contribue pas aux conditions de faction concernées.

## Comportement cible

Ajouter une observation dédiée au réseau, construite à partir de l'état réel mais ne partageant pas
ses listes mutables. Elle sera toujours normalisée ainsi :

```text
joueur observé = joueur actif
adversaire = l'autre joueur
```

L'observation contiendra :

### État global

- phase courante ;
- statut et indicateur de victoire si la partie est terminée ;
- numéro de tour si utile à l'apprentissage ;
- composition du deck central restant, sous forme de quantités par `card_definition_id` ;
- cartes de la rivière, avec leurs slots et leur identité de définition ;
- décisions en attente sous forme de contexte public : type, candidates visibles et contraintes
  applicables.

Le deck central ne sera jamais exposé dans son ordre.

### Joueur actif

- santé, maîtrise, Gems et Power ;
- carte en main, individuellement et avec leur `card_definition_id` ;
- pioche sous forme de quantités par définition, sans ordre ;
- défausse sous forme de quantités par définition, sans ordre ;
- zone de jeu visible ;
- champions visibles, leurs PV imprimés, leur capacité et leur état activé ;
- composition totale des cartes possédées et non bannies, dérivée des zones connues ;
- masque des factions déjà jouées pendant le tour courant.

### Adversaire

- santé et maîtrise ;
- composition globale des cartes possédées et non bannies, sans séparation entre main, pioche,
  défausse et autres zones ;
- défausse sous forme de quantités par définition ;
- champions visibles, leurs PV imprimés, leur capacité et leur état public ;
- aucune carte de la main adverse individuellement ;
- aucune carte de la pioche adverse individuellement ;
- aucun ordre de main, pioche ou défausse.

La composition globale adverse permet de connaître les cartes potentiellement en main ou en
pioche sans savoir dans quelle zone elles se trouvent. La défausse est donnée séparément car elle
est publiquement observable.

## Hors périmètre

- déterminer les features numériques finales du réseau ;
- entraîner ou choisir les dimensions des embeddings ;
- encoder les actions légales ;
- modifier les règles de déplacement des cartes ;
- introduire l'information cachée probabiliste ou la recherche de croyances ;
- supprimer immédiatement `Game.observation_for()` ou casser les joueurs existants.

## Décisions clés

- L'observation neural est un type dédié, distinct de `GameState`.
- `Game.observation_for()` reste compatible avec les joueurs existants jusqu'à l'intégration
  explicite du joueur neural.
- Une méthode dédiée, proposée sous le nom `Game.neural_observation_for(player_id)`, construit la
  représentation masquée ; le nom final devra rester explicite sur sa confidentialité.
- Les zones dont l'ordre n'est pas connu du joueur sont représentées par des comptages déterministes
  triés par `card_definition_id`.
- Le format canonique des comptages dans l'observation est un tuple trié de paires
  `(card_definition_id, quantity)`. Un vecteur aligné sur le catalogue sera une transformation
  ultérieure de l'encodeur neural, pas un format du moteur.
- La main active reste représentée carte par carte, car le joueur connaît ses cartes et les actions
  légales ciblent des instances concrètes.
- La rivière conserve ses slots, car les actions d'achat et de recrutement ciblent un slot.
- Les identifiants de définition restent visibles dans l'observation ; les identifiants d'instance
  servent seulement à relier une action à une carte visible et ne seront pas utilisés comme signal
  stratégique indépendant.
- Les champions n'ont pas de PV courants distincts de leur définition : une attaque légale doit
  atteindre leurs PV imprimés, puis le champion est détruit en une seule transition. L'observation
  expose donc `champion_health` comme caractéristique publique de la carte, et non comme une jauge
  de dégâts persistante.
- Les cartes jouées lors des tours précédents ne sont pas exposées comme historique.
- `turn_number` est inclus comme métadonnée publique de progression ; il ne constitue pas une
  information cachée et ne doit pas être interprété comme un historique des cartes jouées.
- `played_card_ids_this_turn` reste une donnée interne du moteur pour les règles. L'observation
  neural reçoit uniquement un masque de présence des factions jouées pendant le tour courant.
- Le masque comporte quatre positions, dans un ordre documenté et stable, pour les factions jouables
  `maquis`, `spectra`, `homodeus` et `order`. Les cartes `neutral` sont ignorées par ce masque.
- Toute information absente de l'observation doit être impossible à reconstruire à partir d'un
  identifiant d'instance, d'un ordre de liste ou d'un champ auxiliaire.

## Architecture proposée

Créer des types d'observation immuables ou effectivement non mutables, sérialisables et composés de
valeurs simples :

```text
NeuralObservation
  global_state
  active_player
  opponent
  central_deck_counts
  river

NeuralPlayerObservation
  health, mastery, gems, power
  hand
  draw_pile_counts
  discard_counts
  play_zone
  champions
  owned_card_counts
  played_faction_mask
```

Les cartes visibles seront des références structurées contenant au minimum `card_definition_id` et
les attributs publics nécessaires à la suite de l'encodage. Les comptages seront normalisés dans un
ordre stable et ne dépendront jamais de l'ordre interne des listes du moteur.

La construction devra appliquer les règles suivantes :

1. identifier le joueur actif à partir de `state.active_player`, sans faire confiance à un
   `player_id` fourni par l'appelant pour déterminer le point de vue ;
2. copier uniquement les champs autorisés ;
3. agréger les zones cachées dans les comptages autorisés ;
4. exclure les cartes bannies et les identifiants d'instance non observables ;
5. retourner une structure détachée dont aucune mutation ne peut modifier `GameState` ;
6. vérifier par tests que deux états ne différant que par une information cachée produisent la même
   observation neural.

La méthode pourra être utilisée par le futur `NeuralPlayer` et par le générateur de dataset. Le
runner ne changera de mode d'observation qu'au moment où un joueur déclarera explicitement utiliser
cette représentation.

## Masque d'information et invariances

Les tests devront établir les invariances suivantes :

- permuter l'ordre de la pioche adverse ne change pas l'observation ;
- permuter l'ordre de la main adverse ne change pas l'observation ;
- permuter l'ordre de la défausse adverse ne change pas l'observation ;
- changer les `instance_id` de cartes adverses sans changer leurs définitions ne révèle rien de
  nouveau ;
- une carte ajoutée à la main adverse est seulement reflétée dans la composition globale, jamais
  comme carte visible individuellement ;
- les cartes de la main active restent distinctes et accessibles ;
- la rivière garde ses slots ;
- le masque des factions jouées ne dépend pas de l'ordre des cartes du tour.

## Performance et évolutivité

La construction de l'observation intervient à chaque décision atomique et pendant la génération du
dataset. Elle ne doit donc pas effectuer de `deepcopy` du jeu complet ni recalculer plusieurs fois
les mêmes zones.

Les comptages pourront être construits en un seul parcours par zone. Si le profilage montre que cela
devient un goulet d'étranglement, des compteurs maintenus dans `PlayerState` ou `GameState` pourront
être ajoutés, mais seulement après mesure et avec des tests d'invalidation lors de chaque déplacement
de carte.

Le masque de factions jouées pourra être calculé comme un entier bit-à-bit ou comme un tuple de
booléens. Le format public de l'observation restera stable et explicite ; l'optimisation interne ne
doit pas exposer une dépendance à l'ordre d'un `set`.

## Stratégie de test

Ajouter des tests qui vérifient :

- présence des champs publics attendus pour les deux joueurs ;
- normalisation correcte du point de vue du joueur actif ;
- absence de main et pioche adverses carte par carte ;
- agrégation correcte des cartes adverses possédées et de la défausse ;
- absence d'ordre dans les zones cachées ;
- présence des slots de rivière et des cartes visibles ;
- masque des factions jouées sur le tour courant ;
- réinitialisation du masque au changement de tour ;
- détachement complet de l'observation ;
- sérialisation stable pour deux états équivalents du point de vue de l'information ;
- conservation de `Game.observation_for()` pour les joueurs actuels.

## Fichiers attendus

- `shards_ai/game/observation.py` — nouveau modèle d'observation neural proposé ;
- `shards_ai/game/game.py` — méthode de construction de l'observation masquée ;
- `shards_ai/game/state.py` — modification seulement si un type de données auxiliaire est nécessaire ;
- `shards_ai/game/enums.py` — source canonique des factions ;
- `tests/game/test_neural_observation.py` — tests du masque, des zones et des invariances ;
- `doc/Current state/Game engine.md` — mise à jour après implémentation ;
- `doc/Roadmap.md` — passage du step 8.b à `DONE` après validation.

## Validation attendue

Exécuter les tests ciblés de l'observation, les tests du moteur et la suite complète. Vérifier que
les benchmarks du joueur heuristique restent inchangés, puisque `Game.observation_for()` demeure
compatible pendant cette étape.
