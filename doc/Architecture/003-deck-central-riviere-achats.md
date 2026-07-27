# Deck central, rivière et achats — Architecture

## Objective

Étendre le moteur de duel afin d'introduire les cartes à coût et effet, les Gems, le deck de
départ différencié, le deck central partagé, la rivière de six cartes et une phase d'achat. Cette
étape doit permettre de simuler le cycle de base suivant : jouer des cartes, produire des Gems et
des dégâts, acheter éventuellement des cartes de la rivière, attaquer, puis nettoyer le tour.

Critères de réussite :

- chaque joueur commence avec 8 `Cristal` et 2 `Blaster` ;
- `Cristal` coûte 0 et produit 1 Gem ; `Blaster` coûte 0 et produit 1 dégât ;
- le deck central contient 20 `Assassins du vide`, coût 2, produisant 5 dégâts ;
- le deck central est mélangé et six cartes sont révélées dans la rivière au démarrage ;
- une carte jouée applique son effet et incrémente les ressources du joueur ;
- une carte de rivière achetée rejoint la défausse du joueur, la dépense est déduite et la rivière
  est remplacée si le deck central contient encore des cartes ;
- les Gems restantes sont remises à zéro à la sortie de la phase d'achat ;
- deux `RandomPlayer` peuvent terminer une partie de manière déterministe, avec 10 % de chance de
  s'arrêter à chaque décision d'achat.

## Current State

Le moteur actuel est un moteur mémoire et synchrone, sans persistance ni API réseau.

- `shards_ai/game/cards.py` contient `CardDefinition` (id, nom, dégâts) et `CardInstance`.
- `Game.new()` crée dix cartes identiques par joueur, mélange la pioche et pioche cinq cartes.
- `PlayerState` possède main, pioche, défausse et zone de jeu, mais aucune ressource.
- `GameState` contient déjà `phase=BUY` dans l'enum, mais le cycle réel est `PLAY → ATTACK →
  CLEANUP → PLAY`.
- Les actions disponibles sont `PlayCard`, `PassPlayPhase` et `AssignDamage`.
- `Game.legal_actions()` et `Game.apply()` sont les frontières d'autorité du moteur.
- `RandomPlayer` joue toutes les cartes puis assigne tous les dégâts ; la logique d'achat est
  explicitement prévue comme extension dans `002-joueur-random.md`.

Il n'y a pas de migration de données ni de compatibilité de format à assurer : les changements
concernent les objets de simulation et leurs tests.

## Target Behavior

### Mise en place

1. Construire pour chaque joueur un deck de 10 instances : 8 `Cristal` et 2 `Blaster`.
2. Mélanger chaque deck avec le flux aléatoire du moteur.
3. Conserver la règle actuelle de main initiale de 5 cartes, sauf décision ultérieure contraire.
4. Construire le deck central avec 20 instances d'`Assassins du vide`, puis le mélanger.
5. Piocher six cartes du deck central et les placer dans six emplacements de rivière ordonnés.

Les instances du deck central doivent avoir des identifiants distincts, même si elles partagent la
même définition, afin que l'observation, les tests et les replays puissent suivre une carte précise.

### Phases et ressources

Le cycle devient :

```text
PLAY → BUY → ATTACK → CLEANUP → PLAY
```

Pendant `PLAY`, chaque `PlayCard` retire une carte de la main, la place dans la zone de jeu et
applique son effet. Pour cette étape, l'effet est un montant de Gems ou un montant de dégâts ; les deux valeurs peuvent être représentées dans un type extensible même si les cartes actuelles n'en utilisent qu'une seule.

Pendant `BUY`, le joueur peut acheter zéro ou plusieurs cartes. Une carte est achetable si son coût
est inférieur ou égal au nombre de Gems courant et si elle est présente dans la rivière. L'achat
retire la carte de la rivière, l'ajoute à la défausse du joueur, décrémente les Gems et remplit
immédiatement l'emplacement avec la prochaine carte du deck central. Si le deck central est vide,
l'emplacement reste vide ; la rivière peut donc contenir moins de six cartes en fin de deck.

`StopBuying` termine toujours la phase d'achat, même s'il n'existe aucun achat légal. À cette
transition, les Gems non dépensées sont remises à zéro, puis la partie passe à `ATTACK`. L'attaque
utilise les dégâts accumulés pendant `PLAY` ; la règle V0 d'assignation de la totalité des dégâts en
une action est conservée.

## Non-Goals

- champions, maîtrise, Shields, soins et bannissement ;
- effets conditionnels, ciblage complexe ou une mini-langue de règles ;
- achats de cartes hors rivière ou achats simultanés ;
- monnaie persistante d'un tour à l'autre ;
- modification de la taille de main, toujours fixée à 5 dans cette étape ;
- parties à plus de deux joueurs ;
- interface graphique, persistance ou environnement RL ;
- optimisation par cache prématuré des observations complètes.

## Key Decisions

1. **Le moteur reste l'autorité.** Le joueur propose `BuyCard` ou `StopBuying`, mais seul `Game`
   vérifie la phase, l'emplacement, le coût, les Gems et les transitions de zones.
2. **Coût légal inclusif.** Une carte est achetable lorsque `cost <= gems`. Cette formulation rend
   explicites les achats à coût nul et correspond à la notion de coût payable.
3. **Effets structurés et composables.** `CardDefinition` porte un coût et un effet structuré,
   par exemple `Effect(gems=1, damage=0)`. Le moteur applique ces effets génériques ; il ne déduit
   pas le comportement depuis le nom de la carte.
4. **ID stable, nom non fonctionnel.** `card_id` est unique et stable pour retrouver la définition
   dans le catalogue. `name` est informatif et réservé au debug/UI future ; aucune règle ne compare
   les noms.
5. **Catalogue centralisé avec index O(1).** Les définitions sont déclarées dans un fichier par
   carte puis exposées par un registre validé au démarrage. Les instances
   ne copient pas leurs règles et ne portent qu'une référence à une définition immuable.
6. **Rivière ordonnée et adressable.** Les actions ciblent un emplacement et portent le
   `card_instance_id` attendu, jamais un nom. L'emplacement est préférable pour l'UI ; le moteur
   vérifie que la carte ciblée est toujours celle attendue.
7. **Défausse immédiate à l'achat.** La carte achetée ne peut pas être jouée pendant le même tour.
   Elle rejoint la défausse du joueur conformément au flux deck-building demandé.
8. **Ressources de tour dans `PlayerState`.** `gems` et `pending_damage` appartiennent au joueur
   actif plutôt qu'à `GameState` global. Cela prépare le multijoueur et évite une ressource globale
   ambiguë ; un alias temporaire de `GameState.pending_damage` pourra être conservé uniquement si
   nécessaire pour une migration de tests.
9. **Deck central dans `GameState`.** Il est partagé et appartient à la partie, avec `central_deck`
   et `river`. Il n'est pas la propriété d'un joueur.
10. **Aléatoire séparé et reproductible.** Le mélange des decks, de la rivière et les choix du
    `RandomPlayer` utilisent les flux déterministes déjà dérivés ; ajouter la rivière ne doit pas
    rendre les choix d'un joueur dépendants d'un flux partagé imprévisible.
11. **Nommage du code en anglais.** Les phases, classes, actions, champs et symboles du code
    restent en anglais (`BUY`, `BuyCard`, `StopBuying`, `gems`, `damage`, etc.). Les noms français
    sont réservés aux noms affichés des cartes, comme `Cristal`, `Blaster` et `Assassins du vide`.
12. **Effets composables.** Le modèle d'effet autorise plusieurs composantes simultanées, notamment
    des Gems et des dégâts, même si les cartes de cette première version n'en utilisent qu'une
    seule.
13. **Rivière épuisable.** Lorsque le deck central est vide, les cartes achetées ne sont plus
    remplacées. La rivière diminue donc progressivement ; lorsqu'elle est vide, aucun achat n'est
    légal et le joueur doit terminer la phase avec `StopBuying`.
14. **Une définition par carte.** Chaque type de carte distinct possède son propre fichier de
    configuration, sans créer un fichier par copie physique. Cette organisation est retenue pour
    garder les cartes lisibles et faciles à modifier lorsque le catalogue dépassera 50 cartes.
15. **Pas de limite de tours dans `Game`.** Le moteur ne termine pas une partie sur un nombre de
    tours. `GameRunner` conserve une limite de 10 000 actions par défaut pour interrompre une partie
    bloquée ou anormalement longue.

## Open Questions

- Il ne reste pas de question bloquante pour cette extension.

## Proposed Architecture

### Catalogue de cartes

Faire évoluer `shards_ai/game/cards.py` vers un petit package si cela ne casse pas les imports
publics :

```text
shards_ai/game/cards/
  __init__.py          # exports publics et validation du catalogue
  model.py             # CardDefinition, CardInstance, Effect
  catalog.py           # index card_id -> CardDefinition
  starter_deck.py      # composition du deck de départ
  central_deck.py      # composition du deck central
  definitions/
    __init__.py        # agrégation des définitions
    crystal.py         # configuration de Cristal
    blaster.py         # configuration de Blaster
    void_assassin.py   # configuration d'Assassins du vide
  zones.py             # helpers de deck/rivière si nécessaires
```

Une nouvelle carte doit ajouter un fichier dans `definitions/` puis être enregistrée dans le
catalogue. Le registre refuse les IDs dupliqués et fournit l'accès O(1) par `card_id`. Les fichiers
de cartes ne doivent contenir ni logique de tour, ni mutation d'état, ni règles spécifiques
réutilisables : ils déclarent uniquement `card_id`, `name`, `cost` et `effect`.

À 50 cartes, un registre explicite dans `definitions/__init__.py` reste préférable à une
auto-découverte magique : les dépendances sont visibles, les erreurs d'import sont immédiates et
la reproductibilité du catalogue est plus simple à tester. Si le catalogue devient beaucoup plus
grand, l'enregistrement pourra être généré ou chargé depuis des données, sans changer le contrat
`card_id -> CardDefinition`.

Le registre refuse les IDs dupliqués, les coûts négatifs, les valeurs d'effet négatives et les noms
vides. Les factories de decks produisent ensuite des instances avec des IDs d'instance propres au
joueur ou au deck central.

### État et actions

Ajouter :

```python
@dataclass(frozen=True, slots=True)
class BuyCard(Action):
    river_slot: int
    card_instance_id: str

@dataclass(frozen=True, slots=True)
class StopBuying(Action):
    pass
```

`PlayerState` reçoit `gems: int = 0` et conserve `pending_damage` ou un champ de dégâts de tour
équivalent. `GameState` reçoit une représentation de la pioche centrale et de la rivière. Une
rivière sous forme de liste de six `CardInstance | None` rend les emplacements stables et les
emplacements vides explicites.

### Moteur

`Game.new()` délègue la construction aux factories de decks, initialise le deck central et la
rivière, puis conserve le tirage initial existant. `Game.legal_actions()` expose :

- en `PLAY`, les `PlayCard` de la main et `PassPlayPhase` ;
- en `BUY`, les `BuyCard` dont le coût est payable et `StopBuying` ;
- en `ATTACK`, `AssignDamage` avec le montant exact ;
- aucune action en `CLEANUP` observable par un joueur.

`Game._play_card` applique l'effet de la définition après validation de la carte. La transition
`_pass_play_phase` devient `PLAY → BUY`. `Game._buy_card` valide le slot et l'ID d'instance,
puis applique atomiquement le déplacement, le coût et le remplacement de rivière.
`Game._stop_buying` remet les Gems à zéro et passe à `ATTACK`. Le cleanup continue de défausser
zone de jeu et main, remet à zéro les dégâts de tour, pioche cinq cartes et change de joueur.

Le remplissage initial et les remplacements passent par le même helper afin de garantir un seul
comportement de mélange, de consommation du deck central et de gestion de rivière incomplète.

### Joueur random

En `BUY`, le joueur :

1. retourne `StopBuying` avec une probabilité de 10 % lorsqu'au moins un achat est légal ;
2. sinon choisit uniformément une action `BuyCard` légale ;
3. recommence après chaque achat avec la nouvelle observation et la nouvelle liste d'actions.

S'il n'y a plus d'achat légal, il retourne `StopBuying`. Cette règle appartient à
`RandomPlayer`, pas à `Game`, et le tirage de 10 % doit utiliser son flux dédié.

## Data Model

```text
CardDefinition
  card_id: str                 # clé stable
  name: str                    # debug uniquement
  cost: int
  effect: Effect

Effect
  gems: int = 0
  damage: int = 0

CardInstance
  instance_id: str
  definition: CardDefinition

PlayerState
  gems: int
  pending_damage: int
  hand, draw_pile, discard_pile, play_zone: list[CardInstance]

GameState
  central_deck: list[CardInstance]
  river: list[CardInstance | None]  # six slots
```

Le premier catalogue contient exactement :

| ID proposé | Nom | Coût | Gems | Dégâts | Quantité de deck |
|---|---|---:|---:|---:|---:|
| `crystal` | Cristal | 0 | 1 | 0 | 8 par joueur |
| `blaster` | Blaster | 0 | 0 | 1 | 2 par joueur |
| `void_assassin` | Assassins du vide | 2 | 0 | 5 | 20 central |

Les IDs proposés deviennent contractuels dès leur introduction ; ils devront donc rester stables.
Les quantités de decks sont des factories/configurations de partie, pas une propriété mutable des
définitions.

## Backend Flow

Le flux est synchrone et sans job :

```text
play cards → gems/damage on player → BUY actions
                         │
                         ├─ BuyCard(slot, instance_id) → discard + river refill + gems -= cost
                         └─ StopBuying  → gems = 0 → ATTACK
```

La validation d'un achat doit précéder toute mutation. La transition peut être testée comme une
opération atomique : si le slot est invalide, vide, le coût est trop élevé ou la phase est
incorrecte, ni la défausse, ni la rivière, ni les Gems ne changent. Le `GameRunner` reste inchangé
dans son rôle : il transmet les nouvelles actions au joueur et continue de vérifier que l'action
retournée appartient à `legal_actions()`.

## Frontend Flow

Aucun frontend n'existe. Les observations futures devront exposer les six slots de rivière, les
coûts, les effets, les Gems et les actions d'achat légales sans faire dépendre une interface du nom
de carte. Les IDs et les définitions pourront servir de contrat pour un mode debug, mais aucune API
réseau ou sérialisation n'est ajoutée ici.

## Authorization And Feature Gates

Sans objet pour le moteur local. Un feature flag n'est pas nécessaire tant que l'extension remplace
la V0 dans `Game.new()`. Si la compatibilité de la V0 est souhaitée, introduire une configuration
de ruleset explicite plutôt qu'un test dispersé sur la présence de la rivière.

## Observability And Operations

Conserver la seed dans `GameState` et ajouter, au besoin, un état de debug compact contenant :

- contenu/ordre des IDs de la rivière et du deck central ;
- Gems, dégâts et action d'achat ;
- slot acheté, coût payé et ID d'instance ;
- nombre de cartes restantes dans le deck central.

Ne pas journaliser chaque action par défaut dans les simulations. Une trace optionnelle doit
permettre de reproduire une partie à partir de la seed et d'identifier les erreurs de catalogue ou
de transition.

## Edge Cases

- deck central vide lors d'un remplissage : le slot devient `None` ;
- rivière partiellement vide : seules les cartes présentes peuvent être achetées ;
- coût exactement égal aux Gems : achat légal ;
- coût nul : achat légal sans diminuer les Gems ;
- achat d'un slot après une observation devenue obsolète : le moteur compare aussi
  `card_instance_id` et refuse sans mutation si le slot ne contient plus cette instance ;
- `StopBuying` sans achat légal : transition normale vers `ATTACK` et remise à zéro des Gems ;
- cartes restantes en main : défaussées au cleanup comme actuellement ;
- zéro dégât : `AssignDamage(0)` reste l'action d'attaque valide ;
- ID de définition inconnu ou dupliqué : erreur de configuration au démarrage, jamais fallback sur
  le nom ;
- coût ou effet négatif : erreur de validation du catalogue ;
- partie très longue ou bloquée : `Game` continue sans limite de tours, mais `GameRunner` l'arrête
  lorsque `max_actions` est atteint ;
- seed identique : mêmes decks, même rivière, mêmes décisions et même résultat.

## Testing Strategy

Ajouter ou adapter des tests pytest pour :

- composition exacte des deux decks de départ et de la main initiale de 5 ;
- composition exacte du deck central et six slots initiaux ;
- reproductibilité du mélange central et de la rivière ;
- application de l'effet Gems et de l'effet dégâts ;
- transition `PLAY → BUY → ATTACK` ;
- liste des achats légaux selon les Gems, y compris coût égal, coût nul et coût trop élevé ;
- achat : défausse, déduction, remplacement de rivière et épuisement du deck central ;
- arrêt volontaire et remise à zéro des Gems ;
- refus d'un slot invalide, vide ou non payable sans mutation partielle ;
- exclusion des cartes achetées de la main et de la zone de jeu du même tour ;
- politique random : arrêt à 10 % avec un RNG contrôlé, achat aléatoire légal, arrêt quand aucun
  achat n'est possible ;
- partie complète random reproductible et limite d'actions toujours efficace ;
- validation du catalogue : IDs uniques, coûts/effets valides et recherche par ID.

Les tests existants de V0 devront être adaptés aux valeurs `Cristal`/`Blaster` et au nombre de
phases, sans affaiblir les garanties de mélange, de cleanup et d'actions illégales.

## Rollout And Migration

Il n'y a pas de migration de stockage. L'ordre d'implémentation recommandé est :

1. introduire le modèle d'effet, le catalogue et les factories de decks ;
2. migrer les cartes V0 et préserver les exports publics ;
3. ajouter les champs de ressources et la rivière à l'état ;
4. ajouter les actions et transitions du moteur ;
5. adapter `RandomPlayer` et `GameRunner` ;
6. adapter les tests et le benchmark.

La migration de `cards.py` vers un package doit conserver des imports de compatibilité dans
`shards_ai/game/__init__.py`. Un rollback code peut revenir au ruleset V0 si une configuration de
ruleset est introduite ; sinon le changement est considéré comme une nouvelle version de moteur.

## Files Expected To Change

- `shards_ai/game/cards.py` ou migration vers `shards_ai/game/cards/` — modèle, catalogue et
  factories de cartes ;
- `shards_ai/game/actions.py` — `BuyCard`, `StopBuying` ;
- `shards_ai/game/state.py` — Gems, dégâts de tour, deck central et rivière ;
- `shards_ai/game/game.py` — mise en place, effets, achats et nouvelles transitions ;
- `shards_ai/game/enums.py` — aucune modification attendue, `BUY` reste le nom de code ;
- `shards_ai/game/__init__.py` — exports publics ;
- `shards_ai/ai/random_player.py` — politique d'achat à 10 % ;
- `tests/game/test_game.py` — règles de cartes, rivière et achats ;
- `tests/game/test_random_player.py` — comportement random en phase d'achat ;
- `benchmarks/benchmark_game.py` — workload avec catalogue et rivière ;
- `doc/Current state/Game engine.md` — état effectivement implémenté après livraison.

Les chemins du package `cards/` sont indicatifs jusqu'à la décision de migration du module
existant ; la séparation fonctionnelle du catalogue, du modèle et des factories est obligatoire.
