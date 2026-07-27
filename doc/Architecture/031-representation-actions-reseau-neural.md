# Représentation des actions pour le réseau neural

## Objectif

Définir une représentation sémantique et stable de chaque action légale afin que le futur modèle
puisse évaluer `score(observation, action)` sur une liste de taille variable.

La représentation doit conserver le type exact de l'action, son contexte de phase et ses paramètres
publics, tout en reliant une carte ciblée à son `card_definition_id` pour réutiliser le card embedding
du step 8.c.

Ce step ne construit pas encore le réseau, la fonction de score ou le dataset d'imitation.

## État actuel

Les actions publiques sont des dataclasses dans `shards_ai/game/actions.py` et sont générées par
`Game.legal_actions()` :

- `PlayCard(card_id)` ;
- `ActivateChampion(champion_id)` ;
- `BanishCard(card_id)` ;
- `SkipBanish()` ;
- `RecruitFreeCard(river_slot, card_instance_id)` ;
- `PassPlayPhase()` ;
- `GainMastery()` ;
- `BuyCard(river_slot, card_instance_id)` ;
- `RecruitMercenary(river_slot, card_instance_id)` ;
- `StopBuying()` ;
- `AssignPower(amount, target)` ;
- `ChoosePendingDecision(choice_id)`.

Les actions ne portent pas toujours leur définition de carte. La représentation devra donc être
construite avec l'état ou l'observation correspondant à la décision afin de résoudre les identités
de cartes de manière contrôlée.

## Comportement cible

Chaque action légale reçoit exactement un descripteur indépendant de la position dans la liste des
actions. Le nombre de descripteurs varie avec le nombre d'actions légales.

Exemple conceptuel :

```text
observation
    ├── action_representation(PlayCard, aspirant_maquis)
    ├── action_representation(ActivateChampion, additri)
    ├── action_representation(GainMastery)
    └── action_representation(PassPlayPhase)
```

Le réseau consommera ensuite l'observation et chaque représentation séparément, puis choisira le
meilleur score parmi les actions légales originales. La représentation ne remplace jamais l'objet
`Action` utilisé par `Game.apply()`.

## Hors périmètre

- modifier les classes d'actions existantes ;
- ajouter des actions non reconnues par le moteur ;
- valider les règles d'une action à la place de `Game.legal_actions()` et `Game.apply()` ;
- inclure les cartes cachées adverses ;
- produire directement des tenseurs ou entraîner un embedding d'action ;
- choisir la fonction de perte du réseau.

## Décisions clés

- La représentation est action-conditionnelle et compatible avec une liste variable d'actions.
- Le type d'action est un identifiant stable explicite, indépendant du nom Python de la classe.
- La phase courante est toujours incluse, même si elle est partiellement redondante avec le type.
- Une carte ciblée expose son `card_definition_id` et, si nécessaire pour relier l'action, son
  `card_instance_id` technique.
- Le `card_instance_id` ne doit pas devenir un signal stratégique appris ; le modèle doit surtout
  utiliser le type et la définition de carte.
- Un slot de rivière est conservé pour les actions qui ciblent la rivière.
- Une cible, un montant et un `choice_id` sont conservés seulement lorsqu'ils existent.
- `GainMastery`, `PassPlayPhase`, `StopBuying` et `SkipBanish` restent des types d'action
  distincts, même s'ils n'ont pas de carte ciblée.
- Le `card_definition_id` est résolu uniquement pour une carte publique et légalement ciblable.
  L'adversaire ne sera jamais parcouru dans sa main ou sa pioche pour enrichir une action.
- `Game.legal_actions()` reste la source de vérité de l'ensemble à scorer ; le descripteur ne rend
  pas une action illégale légale.

## Schéma proposé

```text
ActionRepresentation
  schema_version
  action_type
  phase
  card_definition_id: str | None
  card_instance_id: str | None
  river_slot: int | None
  target: str | None
  amount: int | None
  choice_id: str | None
```

Les valeurs recommandées de `action_type` sont :

```text
play_card
activate_champion
banish_card
skip_banish
recruit_free_card
pass_play_phase
gain_mastery
buy_card
recruit_mercenary
stop_buying
assign_power
choose_pending_decision
```

Le schéma reste volontairement plat à ce stade : les embeddings de la carte ciblée et les
embeddings catégoriels du type, de la phase et de la cible seront produits par le futur encodeur.

## Résolution des cartes ciblées

La fonction de conversion recevra l'action et le contexte de la décision. Elle pourra être appelée
avec l'état complet interne pendant la génération du dataset, mais elle appliquera une whitelist de
zones publiques :

| Action | Zone autorisée pour résoudre la définition |
|---|---|
| `PlayCard` | main active |
| `ActivateChampion` | champions actifs |
| `BanishCard` | main ou défausse actives |
| `BuyCard` | slot ciblé de la rivière |
| `RecruitMercenary` | slot ciblé de la rivière |
| `RecruitFreeCard` | slot ciblé de la rivière |
| `AssignPower` | champion adverse public si la cible n'est pas `opponent` |
| `ChoosePendingDecision` | candidat public de la décision en attente |

Les actions sans carte ciblée ne recherchent aucune carte. Une incohérence entre l'action et la zone
attendue doit produire une erreur explicite plutôt qu'un descripteur incomplet.

Pour `ChoosePendingDecision`, le `choice_id` est toujours conservé. Le `card_definition_id` est
ajouté seulement si ce choix correspond à une carte publique, par exemple un champion adverse ou une
carte de la défausse active.

## Architecture proposée

Créer `shards_ai/ai/action_representation.py` avec :

- `ActionRepresentation`, dataclass immuable ;
- une table explicite classe d'action → `action_type` ;
- `representation_for_action(action, state)` ou une API équivalente recevant le contexte ;
- des résolveurs privés par famille d'action ;
- une sérialisation déterministe versionnée ;
- des erreurs dédiées ou `ValueError` explicites pour les actions incompatibles avec leur zone.

Le convertisseur ne doit pas appeler `Game.apply()` et ne doit pas modifier l'état. Il pourra utiliser
les mêmes définitions immuables que le card representation du step 8.c, mais il ne dupliquera pas le
contenu complet du card embedding : il produira seulement l'identité nécessaire pour le joindre au
futur encodeur de carte.

Le joueur neural conservera une correspondance positionnelle :

```text
legal_actions[i] ↔ action_representations[i] ↔ predicted_scores[i]
```

Le choix final retournera `legal_actions[argmax(scores)]`, avec le départage aléatoire des égalités
au niveau du joueur neural.

## Confidentialité et données cachées

Le convertisseur ne doit pas rendre visible une information que l'observation ne contient pas. En
particulier :

- une action légale ne peut pas cibler la main ou la pioche adverses ;
- un `card_definition_id` ne doit pas être recherché dans ces zones ;
- les actions adverses ne sont pas encodées, seul le joueur actif décide ;
- les cartes adverses visibles dans les champions ou la rivière peuvent être résolues ;
- les cartes actives en main restent résolubles individuellement ;
- les cartes de la défausse active peuvent être résolues lorsqu'une action les cible, même si la
  défausse est agrégée dans l'observation, car sa composition est publique.

La génération du dataset devra vérifier que le descripteur d'une action ne contient pas de champ
provenant d'une zone cachée.

## Performance et évolutivité

La conversion doit être linéaire dans la taille des zones publiques nécessaires à l'action, et non
dans la taille complète du deck. Les actions de rivière utilisent directement leur slot ; les
actions de main utilisent une recherche ciblée ou un index temporaire de la main active.

La représentation est petite et plate. Les attributs structurés de la carte seront récupérés par le
card embedding, évitant de recopier toutes les opérations dans chaque action candidate.

Le format doit accepter l'ajout d'un nouveau type d'action via une modification explicite de la
table des types et des tests. Une action inconnue doit échouer explicitement pour éviter de produire
un dataset silencieusement incomplet.

## Cas particuliers

- `AssignPower(amount, target="opponent")` contient un montant et une cible sans carte ;
- `AssignPower` ciblant un champion contient le `card_definition_id` du champion visible ;
- `BanishCard` peut cibler la main ou la défausse active ;
- `SkipBanish` ne contient ni carte ni cible ;
- `RecruitFreeCard` porte à la fois le slot et l'identité d'instance de la rivière ;
- `ChoosePendingDecision` conserve son choix même si la cible n'est pas une carte ;
- l'alias `AssignDamage` utilise le même `action_type` que `AssignPower` ;
- une action légale dont la carte a disparu entre génération et conversion doit échouer, car cela
  indique un état incohérent ou un mauvais moment de capture.

## Stratégie de test

Ajouter des tests vérifiant :

- conversion de chaque classe d'action publique ;
- stabilité des `action_type` et de la version de schéma ;
- résolution correcte des cartes de la main, de la rivière, de la défausse et des champions ;
- conservation des slots, cibles, montants et `choice_id` ;
- correspondance positionnelle entre actions légales et représentations ;
- absence de recherche dans la main ou la pioche adverse ;
- erreur explicite pour une action inconnue ou une carte absente ;
- indépendance du résultat vis-à-vis des identifiants d'instance lorsqu'on compare les champs
  sémantiques ;
- sérialisation stable.

## Questions ouvertes

Aucune question bloquante. La manière de transformer les catégories et champs plats en tenseurs
sera décidée lors du step 8.f, après validation du contrat de représentation.

## Fichiers attendus

- `shards_ai/ai/action_representation.py` — représentation et résolution des actions ;
- `shards_ai/ai/__init__.py` — exports publics éventuels ;
- `tests/ai/test_action_representation.py` — tests par type et par zone ;
- `doc/Roadmap.md` — passage du step 8.d à `DONE` après validation.

## Validation attendue

Exécuter les tests ciblés d'actions, les tests de l'observation et du moteur, puis la suite complète.
Vérifier que la représentation n'ajoute aucune règle et que la paire
`legal_actions[i]` / `representation[i]` reste stable avant l'entraînement du dataset.
