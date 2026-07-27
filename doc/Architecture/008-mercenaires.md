# Mercenaires — Architecture

**Statut : DONE / livré** — Implémenté, testé et documenté dans l'état courant du moteur.

## Objective

Ajouter la règle des mercenaires au moteur de duel existant. Un mercenaire est une carte déjà
connue du catalogue, marquée par un type supplémentaire. Lorsqu'elle apparaît dans la rivière,
son propriétaire peut soit l'acheter normalement, soit la recruter :

- achat normal : la carte rejoint la défausse du joueur et pourra être jouée plus tard ;
- recrutement : le coût est payé, la carte est jouée immédiatement et son effet est résolu ;
- au cleanup, une carte recrutée ne rejoint pas la défausse : elle retourne dans le deck central.

Le succès se mesure par la conservation des règles existantes pour les achats normaux, la
résolution complète des effets mercenaires et l'absence de perte ou de duplication d'instances
entre la rivière, le deck central et les zones des joueurs.

Les types de cartes actuellement marqués mercenaires sont : Valkyrie des Landes, Saule Vengeur,
Racine de la Forêt, Chevalier Le'Shai, Ermite Fongique, Clerc aux Spores, Assassins du vide,
Apôtre des Ombres, Fléau des Ombres, Brise-Éther, Héritier du Néant, Zara Ra, Écorcheur d'âme,
Le Grand Architecte, Omnius l'érudit, Pirate Hérétique, Prophète de l'éclat et Garde Mémoire.

## Current State

Le moteur est mémoire, synchrone et déterministe. Les frontières d'autorité sont
`Game.legal_actions()` et `Game.apply(action)`.

- `CardDefinition` et `CardInstance` vivent dans `shards_ai/game/cards/model.py`.
- Le catalogue est indexé par `card_id` dans `cards/catalog.py` ; les cartes du deck central sont
  déclarées dans `cards/definitions/` puis assemblées par `central_deck.py`.
- `GameState` possède le deck central et les six slots de `river`.
- `PlayerState` possède `hand`, `draw_pile`, `discard_pile`, `play_zone` et `champions`.
- `Game._buy_card()` paie la carte, la défausse ou pose immédiatement un champion Homodeus déjà
  armé, puis remplit le slot de rivière.
- `Game._cleanup_and_start_next_turn()` défausse `play_zone` et `hand`, remet à zéro l'état de
  tour, pioche cinq cartes et passe au joueur suivant.
- `RecruitFreeCard` existe déjà pour le recrutement gratuit du Moine du portail ; il ne doit pas
  être réutilisé pour confondre une action gratuite avec le recrutement payant d'un mercenaire.
- `Sentinelle des ténèbres` possède actuellement un effet de Power ; sa nouvelle opération devra
  permettre de récupérer un mercenaire depuis la défausse vers la main.

Il n'y a ni migration de données, ni API, ni frontend à maintenir.

## Target Behavior

### Achat dans la rivière

Pour chaque carte de la rivière dont le coût est payable, `legal_actions()` expose toujours
`BuyCard(slot, instance_id)`. Si la définition de la carte est mercenaire, il expose également une
action explicite de recrutement payant.

Les deux actions valident le même slot, le même `instance_id` et le même coût. Elles consomment la
carte de la rivière et remplacent le slot avec la prochaine carte du deck central.

### Achat normal d'un mercenaire

`BuyCard` conserve exactement son comportement actuel, y compris pour un mercenaire : Gems
dépensées, carte ajoutée à la défausse, rivière remplie. La carte pourra ensuite être jouée comme
n'importe quelle autre carte et sera défaussée au cleanup.

### Recrutement d'un mercenaire

Une nouvelle action, nommée `RecruitMercenary`, paie le coût puis déplace l'instance de la rivière
vers la `play_zone` du joueur. L'effet de la carte est résolu immédiatement dans le même ordre que
lors d'un `PlayCard` normal. La phase reste `BUY`, afin que le joueur puisse continuer ses achats
après la résolution.

Une carte recrutée ne peut pas être recrutée une seconde fois : elle n'est plus dans la rivière et
son action ne figure plus dans la liste légale. Son identité physique est conservée pendant toute
la résolution et jusqu'au cleanup.

### Retour au deck central

Au cleanup, les cartes mercenaires jouées par recrutement sont séparées de `play_zone` et remises
dans `GameState.central_deck`. Les cartes achetées normalement et les cartes non mercenaires
suivent le chemin actuel vers la défausse.

Le deck central n'est pas mélangé au retour. Le moteur pioche actuellement avec
`central_deck.pop()`, donc le bas de la pile est représenté par l'index `0` : chaque mercenaire
retourné y est inséré avec `central_deck.insert(0, card)`. Le slot de rivière n'est jamais réservé
pour le mercenaire retourné : il revient au bas du deck central et pourra être révélé lors d'un
remplacement ultérieur, après les cartes déjà présentes dans la pile.

## Non-Goals

- Ajouter une nouvelle famille d'effets ou modifier les effets propres aux cartes mercenaires.
- Changer le coût, la faction, le bouclier, le statut de champion ou le texte d'une carte par le
  seul fait de la recruter.
- Permettre le recrutement depuis la main, la défausse ou une zone de jeu.
- Rendre le recrutement gratuit ; les Gems sont toujours dépensées.
- Ajouter un deck central séparé, une défausse centrale ou une persistance.
- Modifier le comportement des cartes achetées normalement.

## Key Decisions

1. **Le mercenaire est une propriété de définition.** Ajouter `is_mercenary: bool = False` à
   `CardDefinition`. Les instances réutilisent la définition immuable et ne portent pas de flag
   mutable susceptible de diverger entre copies.
2. **Le recrutement est une action publique distincte.** Ajouter `RecruitMercenary` plutôt qu'un
   paramètre ambigu à `BuyCard`. Les politiques de joueur, les replays et les tests distinguent
   clairement achat lent et jeu immédiat.
3. **L'instance reste dans `play_zone`.** Le moteur réutilise la résolution normale des cartes et
   le cleanup devient le seul endroit qui décide de son retour. Il ne faut pas créer une zone
   temporaire spéciale.
4. **La destination dépend du mode d'achat, pas uniquement de `is_mercenary`.** Un mercenaire
   acheté avec `BuyCard` est une carte du deck du joueur ; un mercenaire recruté est une instance à
   retourner au deck central.
5. **Le mode de recrutement doit être traçable jusqu'au cleanup.** Ajouter une collection d'IDs
   d'instances recrutées ce tour dans `PlayerState`, par exemple
   `recruited_mercenary_ids_this_turn: set[str]`. Le statut ne doit pas être déduit de la seule
   définition : un achat normal d'un mercenaire serait sinon renvoyé à tort au deck central.
6. **Le coût est payé avant la résolution.** La validation et la déduction des Gems précèdent tout
   effet. Une action invalide ne mute aucun état ; une définition d'effet invalide doit être
   rejetée au chargement du catalogue, comme aujourd'hui.
7. **Le recrutement s'appuie sur le même résolveur.** `RecruitMercenary` doit appeler la même
   logique que `PlayCard` après avoir placé la carte dans `play_zone`, afin de préserver l'ordre
   des opérations, les seuils, les factions, la maîtrise et les effets conditionnels.
8. **Les décisions doivent être indépendantes de la phase.** Un effet mercenaire peut produire
   une décision obligatoire ou optionnelle pendant `BUY`. Le garde-fou prioritaire de
   `legal_actions()`/`apply()` doit autoriser la résolution de cette décision avant de proposer un
   nouvel achat, y compris pour `BanishCard`, `SkipBanish`, `RecruitFreeCard` et
   `ChoosePendingDecision` lorsque leur état est armé.
9. **Le retour ne mélange pas le deck central.** Comme `central_deck.pop()` retire la carte au
   sommet représenté par la fin de la liste, les mercenaires retournés sont ajoutés au bas avec
   `central_deck.insert(0, card)`. Aucun flux aléatoire supplémentaire n'est consommé.
10. **Les mercenaires sont exclusivement non champions.** `CardDefinition` doit refuser ou les
    validations du catalogue doivent empêcher la combinaison `is_mercenary=True` et
    `is_champion=True`. Le flux mercenaire reste ainsi entièrement borné à `play_zone` et au
    cleanup du tour.
11. **La récupération de la Sentinelle est une décision du moteur.** Ajouter une opération
    déclarative de récupération de mercenaire. Lorsqu'au moins une cible existe dans la défausse,
    le moteur crée une décision avec les `instance_id` concernés ; `RandomPlayer` choisit une cible
    uniformément au hasard via son RNG. Lorsque la défausse ne contient aucun mercenaire,
    l'opération est ignorée sans décision ni erreur.

## Open Questions

Aucune question ouverte. L'action livrée est `RecruitMercenary`, les effets utilisent les
décisions existantes et `is_mercenary` est exposé via la définition immuable de la carte.

## Proposed Architecture

### Modèle de carte et catalogue

Faire évoluer `CardDefinition` :

```python
@dataclass(frozen=True, slots=True)
class CardDefinition:
    ...
    is_mercenary: bool = False
```

Les fichiers de définitions marqueront uniquement les types concernés. Les quantités de copies
restent dans `definitions/*_CARDS` et `central_deck.py`. Une carte conserve donc exactement le
même `effect`, `cost`, `faction`, `shield` et éventuel statut de champion.

La table de correspondance des IDs existants est :

| Carte | `card_id` |
|---|---|
| Valkyrie des Landes | `valkyrie_des_landes` |
| Saule Vengeur | `saule_vengeur` |
| Racine de la Forêt | `racine_de_la_foret` |
| Chevalier Le'Shai | `chevalier_le_shai` |
| Ermite Fongique | `ermite_fongique` |
| Clerc aux Spores | `clerc_aux_spores` |
| Assassins du vide | `void_assassin` |
| Apôtre des Ombres | `apotre_des_ombres` |
| Fléau des Ombres | `fleau_des_ombres` |
| Brise-Éther | `brise_ether` |
| Héritier du Néant | `heritier_du_neant` |
| Zara Ra, Écorcheur d'âme | `zara_ra` |
| Le Grand Architecte | `le_grand_architecte` |
| Omnius l'érudit | `omnius_l_erudit` |
| Pirate Hérétique | `pirate_heretique` |
| Prophète de l'éclat | `prophete_de_leclat` |
| Garde Mémoire | `garde_memoire` |

Les 17 définitions doivent recevoir `is_mercenary=True`. Les autres définitions restent à
`False`, notamment `sentinelle_des_tenebres`, qui est une carte normale pouvant récupérer un
mercenaire mais qui n'est pas elle-même mercenaire.

### Effet de Sentinelle des Ténèbres

Ajouter une nouvelle valeur d'`OperationKind`, par exemple `recover_mercenary`, sans encoder la
règle dans `Game` à partir du nom de la carte. L'effet de la Sentinelle conserve son opération
actuelle et lui ajoute cette opération.

La résolution suit ce flux :

```text
Sentinelle jouée
  ├─ aucun mercenaire en défausse → aucun effet supplémentaire
  └─ au moins une cible            → décision(instance_id)
                                      → carte retirée de la défausse
                                      → carte ajoutée à la main
```

Le résolveur filtre les cibles sur `card.definition.is_mercenary`, et non sur une liste de noms ou
d'IDs codée dans `Game`. La décision générique existante (`PendingDecision` /
`ChoosePendingDecision`) peut porter les IDs des cartes candidates. `RandomPlayer` choisit une
`ChoosePendingDecision` parmi ces candidates avec son flux aléatoire dédié.

### Actions et état

```python
@dataclass(frozen=True, slots=True)
class RecruitMercenary(Action):
    river_slot: int
    card_instance_id: str
```

Ajouter à `PlayerState` :

```python
recruited_mercenary_ids_this_turn: set[str] = field(default_factory=set)
```

Le champ doit être copié par `observation_for()` et `clone()`, puis vidé au cleanup après avoir
traité la zone de jeu. Les IDs sont plus sûrs qu'un index de zone : les cartes se déplacent et les
effets peuvent piocher ou créer des décisions durant le tour.

### Flux d'achat et de remplacement

```text
BUY
 ├─ BuyCard(slot, id)          → Gems -= cost → discard → refill river
 ├─ RecruitMercenary(slot, id) → Gems -= cost → play_zone → resolve → refill river
 └─ StopBuying                 → Gems = 0 → ATTACK
```

`_recruit_mercenary()` doit vérifier la phase, le slot, l'identité, `card.definition.is_mercenary`
et le coût avant mutation. Le helper de validation d'achat peut être partagé avec `_buy_card`, mais
la destination et le marquage doivent rester dans les deux handlers afin que les chemins soient
lisibles et atomiques.

Le remplissage de rivière doit rester commun aux deux actions. Il ne doit pas mélanger le deck à
chaque remplacement ni au retour d'un mercenaire : le mélange reste réservé à la construction
initiale du deck central.

### Résolution immédiate

Le handler place la carte dans `play_zone`, ajoute son `instance_id` à
`played_card_ids_this_turn` et à `recruited_mercenary_ids_this_turn`, puis appelle le même
résolveur que `_play_card`. Cela rend les bonus `Union`, `Domination`, `Echo`, les opérations
conditionnelles et les décisions cohérents avec une carte jouée depuis la main.

Le fait que la phase soit `BUY` ne doit pas empêcher la résolution d'une décision déclenchée par le
mercenaire. `legal_actions()` doit traiter les décisions en attente avant la branche de phase ; les
handlers doivent vérifier qu'une action d'achat ne peut pas contourner une décision obligatoire.

### Cleanup

Le cleanup doit appliquer un déplacement partitionné, conceptuellement :

```text
play_zone
 ├─ instance_id marqué recruté → bas du central_deck
 └─ autre carte                → discard_pile
hand → discard_pile
reset des états temporaires
draw 5 → joueur suivant
```

Le retour doit avoir lieu avant le tirage suivant et avant la remise à zéro de l'ensemble des
marqueurs. En cas de plusieurs mercenaires recrutés dans le même tour, ils sont tous insérés au
bas de la pile, sans mélange. En ajoutant les cartes dans l'ordre de `play_zone` avec
`insert(0, card)`, la carte rencontrée en dernier se retrouve au bas absolu.

## Data Model

```text
CardDefinition
  ... champs existants ...
  is_mercenary: bool

OperationKind
  recover_mercenary

CardInstance
  instance_id: str
  definition: CardDefinition

PlayerState
  ... zones existantes ...
  recruited_mercenary_ids_this_turn: set[str]

GameState
  central_deck: list[CardInstance]
  river: list[CardInstance | None]
```

Il n'y a pas de nouvelle table, migration, index ou format persistant. Le deck central peut
temporairement augmenter lorsqu'un ou plusieurs mercenaires y retournent ; il n'existe donc plus
de raison de supposer que sa taille ne fait que décroître pendant une partie.

## Backend Flow

1. `Game.legal_actions()` expose `BuyCard` pour toute carte abordable et `RecruitMercenary` en
   plus lorsque `card.definition.is_mercenary` vaut `True`.
2. `Game.apply()` route la nouvelle action vers `_recruit_mercenary()` et interdit tout achat si
   une décision obligatoire est en attente.
3. Le handler valide tout avant de modifier les Gems, la rivière ou les zones.
4. Il paie, retire l'instance de la rivière, remplit immédiatement le slot, pose la carte, puis
   résout l'effet. Les décisions éventuelles sont donc créées après que la rivière représente déjà
   l'état courant.
5. Au cleanup, le moteur partitionne `play_zone`, retourne les mercenaires marqués au bas du deck
   central, défausse les autres cartes et ne mélange pas le deck central.

Lors de la résolution de `recover_mercenary`, les candidats sont calculés au moment de l'effet,
depuis la défausse courante. La décision est obligatoire seulement lorsqu'il existe au moins une
cible ; son application déplace exactement l'instance sélectionnée de la défausse vers la main.

Le flux reste synchrone : aucun job, retry ou mécanisme d'idempotence externe n'est nécessaire.

## Frontend Flow

Aucun frontend n'existe. Toute future observation de carte peut lire `definition.is_mercenary`. Une
interface devra afficher deux choix distincts sur une carte mercenaire abordable, avec le coût dans
les deux cas, et signaler les décisions d'effet avant de permettre un nouvel achat.

## Authorization And Feature Gates

Il n'existe ni autorisation ni feature flag dans ce dépôt. La légalité est exclusivement contrôlée
par le moteur. Si un déploiement progressif est introduit plus tard, le flag devra affecter la
composition ou l'usage des définitions, jamais permettre à un joueur de fabriquer une action
`RecruitMercenary` illégale.

## Observability And Operations

Le moteur n'a pas de télémétrie opérationnelle. Les tests et les outils d'analyse doivent toutefois
pouvoir vérifier :

- le nombre de mercenaires dans le deck central avant et après chaque cleanup ;
- les déplacements d'une instance par `instance_id` ;
- la position des mercenaires retournés au bas du deck central ;
- l'absence de doublons entre deck central, rivière et zones de joueurs.

Les erreurs de règle doivent rester des `InvalidActionError` sans mutation partielle.

## Edge Cases

- mercenaire abordable acheté normalement : il reste dans la défausse du joueur au cleanup suivant ;
- mercenaire à coût nul : les deux actions sont légales, et le recrutement ne rend pas de Gems ;
- deck central vide au moment du recrutement : le slot devient `None`, puis le mercenaire revient au
  deck central au cleanup ;
- plusieurs mercenaires recrutés dans un même tour : tous retournent au bas du deck central, sans
  mélange ;
- mercenaire déjà marqué mais absent de `play_zone` : état incohérent à détecter par une validation
  interne ou un test, pas à corriger silencieusement ;
- action avec slot valide mais `instance_id` obsolète : rejet sans mutation ;
- effet déclenchant bannissement, recrutement gratuit ou choix obligatoire pendant `BUY` : la
  décision doit être résolue avant tout nouvel achat ;
- Sentinelle jouée avec plusieurs mercenaires en défausse : une seule cible est proposée et
  récupérée ;
- Sentinelle jouée sans mercenaire en défausse : aucune décision n'est créée et l'effet continue ;
- cible de récupération devenue obsolète avant la décision : action rejetée sans mutation ;
- victoire pendant l'effet immédiat : le joueur ne doit pas atteindre un cleanup normal après une
  partie terminée ;
- définition invalide combinant mercenaire et champion : rejet au chargement du catalogue ;
- retour au bas du deck central : aucun tirage aléatoire supplémentaire ne doit être consommé ;

## Testing Strategy

Ajouter des tests dans `tests/game/test_game.py` et, si les cartes concrètes le justifient, dans un
fichier dédié aux mercenaires :

- validation du flag `is_mercenary` et conservation de tous les autres champs de définition ;
- validation de la liste exacte des 17 `card_id` mercenaires et de l'absence de mercenaire champion ;
- présence de `RecruitMercenary` uniquement pour les cartes mercenaires abordables ;
- achat normal d'un mercenaire vers la défausse ;
- recrutement payant : coût, retrait de la rivière, remplacement et effet immédiat ;
- impossibilité de recruter une carte non mercenaire ou une instance obsolète ;
- décisions d'effet résolues pendant `BUY` sans achat contournant la décision ;
- Sentinelle : récupération d'un mercenaire choisi depuis la défausse ; absence de décision sans
  cible ; plusieurs cibles ; cible obsolète ; choix aléatoire de `RandomPlayer` ;
- retour au deck central au cleanup et non à la défausse ;
- achat normal d'un mercenaire non retourné au deck central ;
- plusieurs retours, ordre d'insertion au bas, deck central initialement vide et rivière
  partiellement vide ;
- absence de doublons et conservation de l'`instance_id` ;
- observation détachée, clone, seed identique et politiques `RandomPlayer` ;
- régression de tous les tests d'achat, de cleanup, de cartes à effets conditionnels et de champions.

Le test de retour ne doit pas dépendre d'une permutation aléatoire : il doit vérifier précisément
que les instances retournées sont au bas du deck central, que l'ordre d'insertion est respecté et
qu'aucun mélange ni tirage aléatoire supplémentaire n'est effectué.

## Rollout And Migration

Il n'y a pas de migration de données. L'implémentation livrée a suivi cet ordre :

1. ajouter le flag de définition et marquer les cartes après la liste métier validée ;
2. ajouter l'action, l'état copié et la génération des actions légales ;
3. implémenter le handler d'achat et la résolution immédiate ;
4. adapter le cleanup et les décisions en attente pendant `BUY` ;
5. adapter `RandomPlayer` pour choisir entre achat normal, recrutement et `StopBuying` ;
6. ajouter les cartes mercenaires concrètes et leurs tests ;
7. mettre à jour `doc/Current state/Game engine.md` et `Random player.md`.

Le rollback est un rollback de code. Les seeds et parties existantes peuvent changer dès qu'une
carte est marquée mercenaire ; cette rupture de comportement doit être acceptée dans les benchmarks
et analyses concernés, ou les scénarios de référence doivent désactiver ces cartes.

## Files Changed

- `shards_ai/game/cards/model.py` — ajout de `CardDefinition.is_mercenary`.
- `shards_ai/game/cards/model.py` — ajout de `recover_mercenary` à `OperationKind`.
- `shards_ai/game/cards/definitions/*.py` — marquage des 17 cartes mercenaires et ajout de la
  règle à `sentinelle_des_tenebres`.
- `shards_ai/game/cards/definitions/__init__.py` — enregistrement inchangé ou adapté selon les
  définitions retenues.
- `shards_ai/game/actions.py` — ajout de `RecruitMercenary`.
- `shards_ai/game/state.py` — IDs des mercenaires recrutés pendant le tour.
- `shards_ai/game/game.py` — actions légales, handler de recrutement, décisions pendant `BUY`,
  observation/clone, insertion en bas du deck central et cleanup.
- `shards_ai/ai/random_player.py` — choix d'une action de recrutement valide.
- `tests/game/test_game.py` ou `tests/game/test_mercenaries.py` — couverture de la règle et
  régressions.
- `doc/Current state/Game engine.md` — mise à jour après implémentation, pas pendant la conception.
- `shards_ai/game/cards/central_deck.py` — seulement si la liste de copies mercenaires nécessite
  une configuration ou une validation explicite.
