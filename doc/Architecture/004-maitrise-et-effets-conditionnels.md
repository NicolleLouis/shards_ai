# Maîtrise et effets conditionnels — Architecture

**Statut : livré** — Implémenté et validé par 35 tests ; benchmark engine-only configurable,
comparaisons officielles à 10 000 parties et profiling à 1 000 parties.

## Objective

Étendre le moteur de duel afin d'implémenter la maîtrise, l'action de gain de maîtrise et les premières cartes dont l'effet dépend d'un seuil, tout en préparant l'implémentation progressive de toutes les cartes de Shards of Infinity et la consommation des descriptions par un futur joueur neuronal.

Critères de réussite :

- le premier joueur commence à 0 maîtrise et le second à 1 ;
- la maîtrise est persistante, bornée entre 0 et 30, et n'est jamais dépensée ;
- pendant `PLAY`, `GainMastery` dépense 1 Gem et ajoute 1 maîtrise, au plus une fois par tour ;
- `RandomPlayer` choisit aléatoirement entre les cartes jouables et `GainMastery`, puis passe uniquement lorsqu'aucune de ces actions n'est disponible ;
- le deck initial devient 7 `Cristal`, 1 `Blaster`, 1 `Réacteur d'éclat` et 1 `Éclat de l'infini` ;
- le Réacteur produit 2 Gems, 3 à partir de 5 maîtrise et 4 à partir de 15 ;
- l'Éclat produit 2 Power, 3 à partir de 10 et 5 à partir de 20 ; à 30 maîtrise, il fait gagner immédiatement ;
- les seuils sont évalués au moment où la carte est jouée ;
- les cartes ont une représentation exécutable et structurée utilisable par un futur réseau neuronal.

## Current State

Le moteur est mémoire, synchrone et déterministe.

- `shards_ai/game/state.py` contient `PlayerState` (santé, Gems, `pending_damage` temporaire et zones) et `GameState`. `pending_damage` sera migré vers `power`.
- `shards_ai/game/actions.py` contient `PlayCard`, le passage, les achats et l'attaque.
- `shards_ai/game/game.py` est l'autorité sur légalité, transitions et effets.
- `shards_ai/game/cards/model.py` ne représente actuellement que `Effect(gems, damage)` ; ce champ représente en pratique le Power produit par les cartes.
- `cards/definitions/` contient une définition Python par type de carte.
- `RandomPlayer` reçoit observation et actions légales et joue toutes les cartes avant de passer ; il ne gère pas encore la maîtrise.
- Les tests sont dans `tests/game/`.

Le modèle plat `Effect` ne permet ni seuil, ni victoire, ni effets ordonnés. Le deck de départ est encore composé de 8 Cristal et 2 Blaster.

## Target Behavior

Le moteur choisit le joueur qui commence comme aujourd'hui. Ce joueur reçoit 0 maîtrise et l'autre 1, indépendamment de la valeur de `PlayerId`. `GameState` conserve explicitement `starting_player` pour les observations et replays.

`GainMastery` est une action de `PLAY` :

1. vérifier au moins 1 Gem, une maîtrise inférieure à 30 et une utilisation non consommée ce tour ;
2. dépenser 1 Gem ;
3. augmenter la maîtrise de 1 ;
4. marquer l'action comme utilisée.

Le flag est réinitialisé au cleanup ; la maîtrise ne l'est jamais. Cette capacité est limitée à une fois par tour, conformément à Focus dans les règles officielles.

Les seuils sont inclusifs et sélectionnés du plus haut au plus bas : `>=15` avant `>=5` pour le Réacteur ; `>=30`, `>=20`, `>=10` pour l'Éclat. L'Éclat déclenche sa victoire dès que la maîtrise est `>=30` au moment où il est joué. Le nom affiché n'est jamais utilisé par le moteur.

## Non-Goals

- implémenter toutes les cartes, champions, Shields, dégâts directs complets ou multijoueur ;
- construire le réseau neuronal ;
- parser du texte libre pour exécuter une carte ;
- modifier la phase d'achat ;
- ajouter une persistance ou une interface graphique ;
- autoriser une maîtrise hors de l'intervalle 0..30.

## Key Decisions

1. **Maîtrise dans `PlayerState`.** C'est une ressource permanente du joueur, distincte des Gems et du Power temporaire.
2. **Premier joueur explicite.** `starting_player` est enregistré après le tirage initial et sert à initialiser 0/1 de manière déterministe.
3. **Gain de maîtrise limité à une fois par tour.** `GainMastery` est l'équivalent de Focus : son coût est de 1 Gem, son gain est de 1 maîtrise et son flag est réinitialisé au cleanup.
4. **Moteur autoritaire.** `legal_actions()` expose `GainMastery` uniquement quand elle est possible ; `apply()` revalide toutes les préconditions avant mutation.
5. **Effets déclaratifs.** Les cartes décrivent des conditions et opérations typées, plutôt que des lambdas ou des branches `if card_id == ...` dans `Game`.
6. **Résolution ordonnée.** Un `EffectProgram` est évalué dans l'ordre avec un contexte (joueur, adversaire, maîtrise au moment du jeu). Une carte peut produire plusieurs opérations.
7. **Description structurée comme source de vérité.** Chaque définition possède des champs structurés (coût, type, conditions, opérations, timing, tags). Le futur agent consomme un DTO sérialisable normalisé ; il ne parse ni nom ni texte libre. Un rendu humain sera ajouté plus tard uniquement pour le debug.
8. **Compatibilité progressive.** `CardDefinition.gems` et `.damage` peuvent rester pour les cartes simples, mais l'exécution passe progressivement par le même résolveur. `PlayerState.pending_damage` et `GameState.pending_damage` deviennent des propriétés d'alias vers `power`.
9. **Pas de mutation directe par les effets.** Les opérations passent par des primitives du moteur afin de centraliser bornes, victoire et invariants.
10. **Power distinct des dégâts.** Les cartes produisent du Power ; l'attaque transforme le Power assigné en dégâts contre une cible. Les dégâts directs futurs auront une opération distincte.
11. **Migration progressive du nommage.** `power` devient le nom canonique. `pending_damage` et `Effect.damage` restent des alias de compatibilité temporaires, avec des tests empêchant leur divergence.

## Open Questions

Aucune question ouverte bloquante ou non bloquante ne subsiste pour cette architecture.

## Proposed Architecture

### État et actions

Ajouter :

```text
PlayerState
  mastery: int = 0                 # invariant 0..30
  mastery_action_used: bool = False

GameState
  starting_player: PlayerId

Action
  GainMastery
```

Définir dans un module de règles partagé : `MIN_MASTERY = 0`, `MAX_MASTERY = 30`, `MASTERY_COST = 1`, `MASTERY_GAIN = 1`. Une fonction de clamp unique sera réutilisée par les cartes futures qui gagnent ou perdent de la maîtrise.

### Modèle d'effets

Cible conceptuelle pour `cards/model.py` :

```text
CardDefinition
  card_id, name, cost, card_type
  effect: EffectProgram
  tags: tuple[str, ...]

EffectProgram
  steps: tuple[EffectStep, ...]

EffectStep
  when: Condition = always
  operations: tuple[Operation, ...]

Condition
  mastery_at_least: int | None

Operation
  gain_gems(amount)
  gain_power(amount)
  deal_damage(target, amount)       # dégâts directs, distincts du Power
  gain_mastery(amount)
  win()
  mastery_at_least: int | None      # condition évaluée au moment de l'opération
```

Le résolveur évalue chaque condition au moment où l'étape est atteinte, avec l'état courant de la
maîtrise, puis applique les opérations dans l'ordre. Ainsi, une maîtrise gagnée par une opération
précédente de la même carte peut compter pour une étape suivante. Les étapes alternatives doivent
être déclarées explicitement comme exclusives et ordonnées du seuil le plus haut au plus bas.
`win()` termine la partie avec le joueur actif comme gagnant, via le même statut que la victoire
par santé. Les opérations nécessitant un choix futur devront produire une action ou une demande de
choix explicite, pas modifier directement le moteur.

### Catalogue et decks

| ID               | Nom               | Coût | Effet                                               | Quantité de départ |
| ---------------- | ----------------- | ---: | --------------------------------------------------- | -----------------: |
| `crystal`        | Cristal           |    0 | +1 Gem                                              |                  7 |
| `blaster`        | Blaster           |    0 | +1 Power                                            |                  1 |
| `shard_reactor`  | Réacteur d'éclat  |    0 | +2 Gems ; +3 à >=5 ; +4 à >=15                     |                  1 |
| `infinity_shard` | Éclat de l'infini |    0 | +2 Power ; +3 à >=10 ; +5 à >=20 ; victoire à >=30 |                  1 |
| `void_assassin`  | Assassins du vide |    2 | +5 Power                                            |            central |

Ajouter `shard_reactor.py` et `infinity_shard.py`, les exporter dans `definitions/__init__.py`, puis les enregistrer dans `catalog.py`. Les IDs sont stables ; les noms restent informatifs.

## Data Model

```text
PlayerState
  player_id
  health: int
  mastery: int                 # permanent, 0..30
  mastery_action_used: bool    # tour courant
  gems: int                    # ressource de tour
  power: int                    # Power produit pendant PLAY, consommé en ATTACK
  hand, draw_pile, discard_pile, play_zone

GameState
  starting_player: PlayerId
  active_player, phase, status, winner, turn_number
```

`CardInstance` reste une identité physique qui référence une définition immuable. La description normalisée, sérialisable pour le futur réseau neuronal, est mise en cache par définition et ne doit pas être reconstruite à chaque décision. Aucun texte humain n'est requis dans le modèle ; un renderer de debug pourra être ajouté ultérieurement.

## Backend Flow

Pendant `PLAY` :

```text
PlayCard(instance_id) -> resolve EffectProgram with current mastery
GainMastery              -> spend 1 Gem, mastery += 1, mark used
PassPlayPhase             -> BUY
```

La carte doit être validée et retirée de la main avant sa mise en zone de jeu, puis son programme doit être résolu sans laisser de mutation partielle en cas d'erreur de définition. Le cleanup remet Gems, Power et `mastery_action_used` à zéro, pioche 5 cartes et conserve la maîtrise.

### Migration Power / dégâts

Le champ actuel `pending_damage` est renommé conceptuellement en `power`, car les cartes ne
causent pas encore directement des dégâts : elles accumulent du Power. La phase d'attaque utilise
ensuite ce Power pour réduire la santé adverse. L'action canonique est désormais `AssignPower` et
vérifie puis consomme `PlayerState.power`. `AssignDamage` reste uniquement un alias de compatibilité
déprécié.

La migration se fait en deux temps :

1. ajouter `power` et faire de l'opération `gain_power` la voie canonique ;
2. conserver `PlayerState.pending_damage`, `GameState.pending_damage` et `Effect.damage` comme propriétés d'alias dépréciées, puis les
   supprimer lorsque les tests, benchmarks et consommateurs externes utilisent tous `power`.

Les dégâts directs restent une opération différente, `deal_damage(target, amount)`, pour les futures
cartes qui blesseraient directement un joueur ou un champion sans produire de Power.

Le runner reste ignorant de la règle : il transmet simplement les actions légales au joueur. Après une victoire par carte, `legal_actions()` est vide et aucune action supplémentaire n'est demandée.

## Frontend Flow

Aucun frontend n'est prévu. Les observations doivent néanmoins exposer maîtrise, plafond, flag d'action utilisé et descriptions des cartes visibles, afin de ne pas casser le futur contrat de l'agent neuronal.

## Authorization And Feature Gates

Sans objet pour l'exécution locale. Si des replays sont sérialisés, ajouter un `ruleset_version` ; un replay du nouveau deck ne doit pas être interprété silencieusement avec l'ancien catalogue.

## Observability And Operations

Les traces optionnelles doivent inclure seed, ruleset, `starting_player`, maîtrises initiales/finales, actions `GainMastery`, `card_id`, maîtrise observée, étape d'effet sélectionnée, cause de fin et gagnant. Elles restent désactivées par défaut pour conserver le débit des simulations.

## Edge Cases

- à 0, aucune opération ne peut rendre la maîtrise négative ;
- à 29, le gain porte à 30 ; à 30, l'action n'est plus légale ;
- 0 Gem ou action déjà utilisée : aucun état ne change ;
- seuil exactement atteint : le bonus est actif ;
- seuils multiples : seul le plus haut applicable est utilisé ;
- l'Éclat joué à 29 ne gagne pas automatiquement ;
- un gain réalisé avant l'Éclat dans le même tour est pris en compte ;
- une victoire par carte interdit toute action ultérieure ;
- une observation détachée ne peut pas modifier l'état réel ;
- une définition incohérente est rejetée avant mutation de la partie.

## Testing Strategy

### Moteur

Tester l'initialisation 0/1 et `starting_player`, les bornes, le coût et la limite de `GainMastery`, son reset au cleanup, sa présence dans `legal_actions()`, le deck `7/1/1/1`, et les seuils du Réacteur aux maîtrises 0/4/5/14/15/30 et de l'Éclat aux valeurs 0/9/10/19/20/29/30. Tester la victoire, l'ordre des opérations et l'absence de rétroactivité.

Tester aussi les IDs du catalogue, le DTO structuré sérialisable, les refus atomiques et la compatibilité des effets simples.

Tester que Blaster, Réacteur, Éclat et Assassins produisent du Power, que l'attaque convertit le
Power en dégâts, que le cleanup le remet à zéro et que les alias `pending_damage`/`damage` restent
strictement cohérents pendant la migration.

### RandomPlayer et performance

Le RandomPlayer doit choisir uniformément parmi cartes et `GainMastery` légales, puis passer seulement quand aucune des deux n'est disponible. Tester seeds identiques, parties terminées, actions légales et victoire par maîtrise. Ajouter un micro-benchmark : le coût du résolveur doit dépendre du nombre d'étapes de la carte, et aucune description textuelle ne doit être reconstruite à chaque décision.

## Rollout And Migration

Aucune migration de base n'est nécessaire. Ordre recommandé :

1. introduire le modèle déclaratif avec les cartes simples ;
2. ajouter maîtrise, `starting_player` et `GainMastery` ;
3. migrer le deck initial et ajouter les deux définitions ;
4. adapter RandomPlayer, runner, tests et benchmark ;
5. publier observation et description structurées ;
6. ajouter les cartes suivantes une par une avec tests de seuil.

Les benchmarks doivent identifier explicitement le ruleset, car la durée des parties changera. Les replays doivent versionner le catalogue et les règles.

## Files Expected To Change

- `shards_ai/game/state.py` : maîtrise, flag et joueur initial ;
- `shards_ai/game/actions.py` : `GainMastery` et future migration `AssignPower` ;
- `shards_ai/game/game.py` : initialisation, légalité, application, cleanup et résolution ;
- `shards_ai/game/cards/model.py` : effet déclaratif et compatibilité ;
- `shards_ai/game/cards/starter_deck.py` : composition `7/1/1/1` ;
- `shards_ai/game/cards/catalog.py` et exports ;
- nouveaux `cards/definitions/shard_reactor.py` et `infinity_shard.py` ;
- `shards_ai/ai/random_player.py` ;
- tests du moteur, des cartes, du RandomPlayer et du runner ;
- `benchmarks/benchmark_game.py` et les deux documents `doc/Current state/`.

Les chemins exacts du résolveur peuvent évoluer pendant l'implémentation, mais aucune carte ne doit injecter sa logique métier directement dans `Game`.
