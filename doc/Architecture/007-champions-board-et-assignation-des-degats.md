# Champions, board persistant et assignation des dégâts — Architecture

**Statut : DONE** — Architecture implémentée et validée. Les champions listés, leurs effets et la
nouvelle assignation du Power sont disponibles dans le moteur.

## Objective

Ajouter les champions comme un type de carte persistant sur le board du joueur qui les joue.
Contrairement aux cartes normales, jouer un champion ne résout pas un effet immédiat : la carte
reste en jeu, possède des points de vie et fournit une capacité passive ou active.

Critères de réussite :

- un champion joué rejoint le board de son propriétaire et y reste après le cleanup ;
- une capacité active peut être activée une fois par tour, y compris pendant le tour où le champion
  est posé ;
- pendant `ATTACK`, le Power peut être attribué au joueur adverse ou à plusieurs champions adverses
  successivement ;
- un champion ne peut être ciblé que si le Power restant suffit à lui retirer tous ses PV ; il est
  alors défaussé, sans état de blessure partielle ;
- le `RandomPlayer` choisit aléatoirement parmi l'ennemi direct et les champions dont les PV sont
  inférieurs ou égaux au Power disponible ;
- les parties restent déterministes à seed et les anciennes cartes sans champion conservent leur
  comportement.

## Current State

Le moteur est synchrone, en mémoire et piloté par `Game.legal_actions()` / `Game.apply(action)`.
L'architecture décrite ici est désormais livrée ; les points ci-dessous rappellent les frontières
effectivement présentes dans le code.

- `PlayerState.play_zone` contient les cartes jouées pendant le tour et est entièrement déplacée en
  défausse au cleanup (`shards_ai/game/game.py`) ; `PlayerState.champions` persiste entre les tours.
- `CardDefinition` porte les PV, les effets de pose, les capacités actives et les passifs des
  champions catalogués.
- `Domination` peut compter un champion posé pendant le tour via
  `played_card_ids_this_turn`.
- `AssignPower(amount, target)` cible explicitement le joueur adverse ou un champion létal ; le
  Power restant est conservé après une destruction de champion.
- Les Boucliers sont calculés uniquement depuis la main du défenseur et réduisent les dégâts au
  joueur ; ils ne s'appliquent pas implicitement aux champions.
- `RandomPlayer` choisit parmi les cibles d'attaque et les décisions exposées par le moteur.
- Les decks et cartes sont déclarés explicitement sous `shards_ai/game/cards/definitions/`.
- Il n'y a ni frontend, ni persistance, ni autorisation réseau à modifier.

## Target Behavior

### Première tranche — faction Maquis

Les cartes existantes conservent leurs effets actuels ; les compétences suivantes s'y ajoutent :

| Carte | Modification | Quantité |
|---|---|---:|
| Zélote des Épines | Union : détruire directement un champion adverse, quel que soit son nombre de PV | 2 |
| Saule Vengeur | À partir de 15 maîtrise : détruire directement tous les champions adverses | 1 |
| Additri, Gaïamancienne | Coût 5, champion, 5 PV, capacité active : gagner 2 Power + 2 Power par carte Maquis jouée ce tour | 1 |

Zélote conserve sa pioche 1 et son Bouclier 3. Saule conserve son gain de 4 Power. Additri ne
produit pas d'effet immédiat à la pose ; sa capacité est activable dès le tour de pose et au plus
une fois par tour.

Les destructions directes de Zélote et Saule ignorent la règle d'assignation létale du Power : elles
retirent directement les champions ciblés du board et les placent dans la défausse de leur
propriétaire. Elles ne consomment pas le Power de l'attaquant.

### Deuxième tranche — faction Spectra

Les cartes Spectra existantes conservent leurs effets actuels. Les nouvelles cartes sont :

| Carte | Coût | Quantité | Type | Capacité |
|---|---:|---:|---|---|
| Li Hin, la Brisée | 3 | 1 | Champion, 1 PV | Passif : ne peut pas être ciblée par une assignation de Power ; actif : +1 Power |
| Zen Chi Set, Fléau des dieux | 7 | 1 | Champion, 5 PV | Actif : +3 Power, puis récupérer une carte Spectra de la défausse vers la main |

Li Hin est donc exclue des cibles d'attaque par Power, même lorsque le Power disponible est
suffisant. Sa destruction nécessite une opération de destruction directe autorisée par les règles
de la carte concernée. Son activation reste possible dès sa pose et une fois par tour.

L'activation de Zen Chi Set résout d'abord le gain de 3 Power, puis crée une décision de sélection
parmi les cartes Spectra de la défausse du propriétaire. Si aucune carte Spectra n'est disponible,
le gain de Power est conservé et aucune décision n'est créée.

### Troisième tranche — faction Ordre

Les cartes Ordre existantes conservent leurs effets actuels. Les nouvelles cartes sont :

| Carte | Coût | Quantité | Type | Capacité |
|---|---:|---:|---|---|
| I.A. Systema | 3 | 1 | Champion, 4 PV | Actif : +1 maîtrise ; puis, si la maîtrise est au moins 20, pioche 2 |
| Giga, Adepte de la Source | 2 | 1 | Champion, 4 PV | Pose : pioche 1 pendant le tour de pose uniquement ; actif : Domination → +3 maîtrise |
| Zetta, l'encodeuse | 5 | 1 | Champion, 5 PV, Bouclier 5 | Passif : Zetta devient l'unique cible légale d'attaque parmi les champions |

I.A. Systema et Giga sont activables dès leur pose et une fois par tour. Pour Systema, la maîtrise
gagnée est appliquée avant de vérifier le seuil de 20. L'effet de pose de Giga est immédiat, mais ne
se déclenche que lors du tour où il est joué.

Lorsque Zetta est présente sur le board, les autres champions du même joueur ne sont pas des cibles
d'attaque par Power ; Zetta reste elle-même ciblable. La valeur de Bouclier 5 doit être rattachée au
comportement actuel : elle réduit les dégâts au joueur uniquement lorsque Zetta est encore dans la
main. Une fois Zetta posée sur le board, son Bouclier ne contribue plus au total défensif.

### Quatrième tranche — faction Homodeus

Les cartes Homodeus existantes conservent leurs effets actuels. Les ajouts et nouvelles cartes sont :

| Carte               | Coût | Quantité | Type            | Capacité                                                                                                       |
| ------------------- | ---: | -------: | --------------- | -------------------------------------------------------------------------------------------------------------- |
| Valkyrie des Landes |    4 |        1 | Carte existante | Inspiration : l'adversaire perd 2 maîtrise                                                                     |
| Légionnaire Korvus  |    3 |        3 | Carte existante | Récupérer un champion de sa défausse vers sa main                                                              |
| Primus Pilus        |    2 |        1 | Champion, 6 PV  | Actif : si au moins 3 champions Homodeus sont en jeu, pioche 2                                                 |
| Drones Numeri       |    3 |        2 | Champion, 5 PV  | Actif : +1 Gem ; armer le prochain recrutement d'un champion Homodeus pour le poser et l'activer immédiatement |
| Evokatus            |    4 |        2 | Champion, 2 PV  | Pose : pioche 1 ; actif : +1 santé par champion Homodeus en jeu                                                |
| Broyeu Optio        |    5 |        2 | Champion, 4 PV  | Actif : +3 Power ; à 10 maîtrise, +2 Power supplémentaires                                                     |
| Drakonarius         |    6 |        1 | Champion, 2 PV  | Passif : protégé de l'attaque par Power si Général Décurion est en jeu ; actif : +6 Power                      |
| Général Décurion    |    7 |        1 | Champion, 7 PV  | Actif : +3 Gems ; à 20 maîtrise, recopier chaque effet d'une carte Homodeus non champion jouée ce tour         |

`Inspiration` est une clause d'activation : elle est satisfaite dès que le joueur actif possède au
moins un champion sur son board. Elle ne crée aucune cible ni décision. Pour Valkyrie des Landes,
l'adversaire perd alors 2 maîtrise. Les effets de pose d'Evokatus et de recrutement armé par Drones
Numeri sont résolus immédiatement selon les règles de leur carte.

Drones Numeri ne pose directement que le prochain champion Homodeus effectivement recruté pendant
le tour ; l'activation immédiate de ce champion est incluse. Le flag d'armement expire au cleanup si
aucun recrutement compatible n'a eu lieu.

### Jouer un champion

`PlayCard` retire le champion de la main et le place dans une zone persistante `champions` du
propriétaire. La pose déclenche uniquement l'éventuel effet de pose propre au champion, comme la
pioche de Giga ou d'Evokatus. Elle ne déclenche jamais son effet actif : celui-ci nécessite ensuite
une action `ActivateChampion`. La capacité passive est consultable par les règles du moteur dès que
le champion est présent sur le board.

Un champion actif posé pendant `PLAY` devient activable immédiatement. L'action d'activation reste
dans `PLAY` et peut apparaître après la pose du champion. Elle n'est possible qu'une fois par
champion et par tour.

### Cycle de vie

```text
main --PlayCard--> champions --destruction par attaque--> discard_pile
                         |
                         `--cleanup--> champions (reste en jeu)
```

Les champions ne sont donc jamais déplacés automatiquement en défausse au cleanup. Lorsqu'un
champion est détruit, son instance physique complète est ajoutée à la défausse de son propriétaire
et pourra être repiochée lors d'un remélange normal.

### Assignation des dégâts

Pendant `ATTACK`, le Power restant est une ressource temporaire de l'attaquant. Chaque décision
choisit une cible :

1. un champion adverse dont les PV sont inférieurs ou égaux au Power restant ; le moteur consomme
   exactement ses PV, le défausse, puis conserve le Power excédentaire ; ou
2. le joueur adverse ; le Power restant est consommé contre lui, après application du Bouclier de sa
   main, puis l'attaque se termine.

Tant que du Power reste après la destruction d'un champion, une nouvelle décision d'attaque est
requise. Il est possible de choisir un autre champion ou le joueur. Si aucun champion n'est ciblé,
le joueur reste toujours une cible valide et reçoit tout le Power restant.

Le moteur doit refuser une cible champion si le Power restant ne suffit pas. Il ne doit jamais
conserver de PV partiels sur un champion. La victoire par réduction de la santé du joueur conserve
son comportement actuel et termine immédiatement la partie.

Pour le `RandomPlayer`, les cibles candidates sont l'ennemi direct et les champions dont les PV
sont inférieurs ou égaux au Power disponible, conformément à la règle demandée. Le tirage est
uniforme entre ces cibles ; après une destruction, il est recalculé avec le Power restant.

## Non-Goals

- Ajouter de nouveaux champions ou de nouvelles capacités non listés dans cette tranche.
- Définir ici les déclencheurs précis des capacités passives (début de tour, attaque, destruction,
  etc.).
- Autoriser des dégâts partiels, des marqueurs de dégâts persistants ou des soins de champions tant
  que les cartes ne l'exigent pas.
- Modifier la composition des decks de départ sans décision explicite.
- Ajouter une interface graphique, une API, une base de données ou un système de permissions.
- Introduire un historique complet des activations dans la boucle de simulation.

## Key Decisions

1. **Zone dédiée.** Ajouter `PlayerState.champions: list[CardInstance]`. `play_zone` reste la zone
   temporaire des cartes normales du tour.
2. **PV portés par la définition.** La définition immuable d'un champion porte ses PV maximum ; une
   instance sur le board est donc vivante à ses PV maximum. Comme les dégâts sont obligatoirement
   létaux, aucun PV courant n'est nécessaire dans la première version.
3. **Capacité persistante séparée de l'effet immédiat.** `is_champion` ne doit pas être utilisé
   comme unique branche métier à long terme. Le modèle devra distinguer type de champion, capacité
   passive et capacité active, tout en réutilisant les `Operation` pour les effets simples.
4. **Activation explicite.** Ajouter une action `ActivateChampion(champion_id)` et un état de tour
   `activated_champion_ids` par joueur. Cet état est réinitialisé au cleanup ; l'ID d'instance évite
   toute ambiguïté entre deux copies du même champion. Une activation est légale à n'importe quel
   moment de `PLAY`, y compris avant d'avoir joué toutes les cartes de la main et pendant le tour de
   pose du champion. Une décision obligatoire déjà en attente conserve toutefois la priorité du
   modèle actuel.
5. **Mémoire du tour de pose.** Ajouter `played_card_ids_this_turn: set[str]` dans `PlayerState`.
   Toute carte jouée y est enregistrée, y compris un champion déplacé immédiatement vers le board.
   Cette mémoire est réinitialisée au cleanup. Les règles comme `Domination` peuvent ainsi distinguer
   un champion posé ce tour d'un champion qui était déjà présent au début du tour.
6. **Activation immédiate des passifs.** Une capacité passive prend effet dès que le champion est
   posé, sans attendre le tour suivant. Les passifs listés sont évalués tant que le champion est
   présent sur le board, selon leur portée définie ; leur retrait du board désactive immédiatement
   la protection ou la condition correspondante.
7. **Destruction ciblée de Zélote.** L'Union de Zélote crée une décision de ciblage lorsque plusieurs
   champions adverses sont présents. Le joueur choisit librement la cible ; s'il n'y a aucun champion
   adverse, l'opération est sans effet et ne crée aucune décision en attente. La destruction ignore
   les PV et ne consomme pas de Power.
8. **Destruction globale ordonnée de Saule.** À partir de 15 maîtrise, Saule ne produit aucun effet
   si le board adverse est vide. Sinon, le joueur choisit l'ordre de destruction des champions.
   Chaque destruction est résolue séparément, y compris ses éventuels déclencheurs « lorsqu'un
   champion est détruit », avant de proposer la destruction suivante.
9. **Power produit par Additri.** La capacité active d'Additri ajoute du `Power` pendant `PLAY` ;
   elle ne provoque pas de dégâts directs et ce Power sera assignable normalement pendant `ATTACK`.
10. **Décompte Maquis d'Additri.** La formule compte les cartes Maquis présentes dans
   `played_card_ids_this_turn`, qu'elles soient encore visibles dans une zone ou non. Additri est
   enregistré dès sa pose et se compte donc lui-même : activé au premier tour sans autre carte
   Maquis jouée, il produit `2 + 2 = 4 Power`.
11. **Immunité d'attaque de Li Hin.** L'immunité de Li Hin porte uniquement sur les assignations de
    Power pendant `ATTACK`. Les destructions directes de Zélote et les destructions globales de Saule
    peuvent la cibler et la détruire, indépendamment de son unique PV.
12. **Récupération de Zen Chi Set.** L'activation ajoute d'abord 3 Power, puis propose une carte
    Spectra, championne ou non, de la défausse du propriétaire à déplacer dans sa main. L'absence de
    carte Spectra ne retire pas le Power et ne crée pas de décision vide.
13. **Activation et pose de Giga.** Le tirage de Giga est un effet de pose ponctuel, disponible
    uniquement le tour de sa pose. Sa capacité active utilise la même règle `Domination` que les
    autres cartes Ordre et, si elle est valide, ajoute 3 maîtrise.
14. **Passif de Zetta.** Tant que Zetta est sur le board, elle filtre les cibles champion adverses
    légales : les autres champions sont protégés contre l'assignation de Power, tandis que Zetta
    reste la cible disponible. Cette restriction ne s'applique pas automatiquement aux destructions
    directes de Zélote ou Saule.
15. **Inspiration.** Une opération `Inspiration` est une précondition évaluée sur le board du joueur
    actif. Elle ne cible personne et ne crée aucune décision. Pour Valkyrie des Landes, l'opération
    conditionnelle qui suit cible toujours l'adversaire et lui retire 2 maîtrise ; sans champion sur
    le board, aucun effet Inspiration n'est résolu.
16. **Recrutement Homodeus armé.** Drones Numeri ajoute un flag temporaire de joueur qui transforme
    le prochain `BuyCard` ou `RecruitFreeCard` d'un champion Homodeus en pose sur le board suivie de
    son activation immédiate. Le flag est consommé par ce recrutement et réinitialisé au cleanup.
17. **Décisions en attente dans `PlayerState`.** Les effets qui nécessitent un choix créent une
    `pending_decision` typée dans le joueur actif. Tant qu'elle existe, `legal_actions()` n'expose
    que les actions capables de la résoudre et `Game.apply()` refuse toute action incompatible.
    Cette décision couvre notamment la sélection Spectra de Zen Chi Set, le champion de la défausse
    de Korvus, la cible de Zélote, l'ordre séquentiel de Saule et le choix des copies de Général
    Décurion.
18. **Protection conditionnelle de Drakonarius.** La présence de Général Décurion sur le board
    filtre Drakonarius hors des cibles d'attaque par Power. Les destructions directes suivent une
    règle indépendante, comme pour Li Hin.
19. **Copie de Général Décurion.** Lorsque le seuil de maîtrise est atteint, l'effet actif recopie
    chaque carte Homodeus non champion jouée ce tour. Le joueur choisit l'ordre de résolution des
    copies ; le `RandomPlayer` choisit aléatoirement parmi les cartes restantes. Les copies
    réutilisent le résolveur d'effets existant et réévaluent les conditions avec l'état courant ;
    elles ne recopient jamais un champion. Un `set` d'IDs suffit pour mémoriser l'appartenance au
    tour, l'ordre étant porté par les décisions successives.
20. **Choix de Korvus.** La récupération cible un champion de la défausse du propriétaire. Le joueur
    choisit lorsqu'il y en a plusieurs ; le `RandomPlayer` choisit aléatoirement. Sans cible, aucun
    effet n'est produit.
21. **Modèle d'aptitudes unifié.** Les cartes fournies couvrent désormais les besoins métier du
    modèle : effet à la pose, actif limité par tour, passif continu, précondition Inspiration,
    conditions de maîtrise/faction, choix de cible, récupération depuis une zone, recrutement armé,
    protections d'attaque et copie d'effets. La structure technique doit couvrir ces cas sans créer
    une classe de règles spécifique par champion.
22. **Attaque par décisions successives.** Remplacer l'hypothèse « une action attribue tout au
   joueur » par une cible d'attaque explicite. Après destruction d'un champion, le moteur reste en
   `ATTACK` jusqu'à l'attribution de tout le Power restant au joueur ou à des champions.
23. **Power canonique.** Conserver `AssignPower` comme nom canonique et `AssignDamage` comme alias de
   compatibilité. Les anciens scénarios sans champions pourront être adaptés par une cible joueur
   implicite, sans maintenir deux résolveurs de règles.
24. **Bouclier limité au joueur.** Le calcul actuel des Boucliers de la main adverse reste attaché à
   la cible joueur. Aucun champ `shield` de champion n'est déduit pendant une cible champion sans
   règle de carte explicite. Le Bouclier 5 de Zetta suit cette règle et ne s'applique que tant que
   Zetta est dans la main.
25. **RandomPlayer sans logique de règles.** `Game.legal_actions()` expose les cibles autorisées ;
   `RandomPlayer` filtre les actions champion/joueur selon la politique aléatoire puis tire avec son
   flux `GameRandom` dérivé.
26. **Compatibilité des observations et clones.** `observation_for()` et `clone()` copient les
   champions et l'état d'activation comme les autres zones, sans partager de listes mutables.
27. **Pas de migration de données.** Le moteur ne persiste pas les parties. Les replays futurs
    devront toutefois versionner le ruleset car la forme des actions d'attaque change.

## Open Questions

Aucune question métier bloquante ne subsiste pour les cartes listées. La structure technique des
aptitudes sera choisie pendant l'implémentation, en respectant les décisions ci-dessus et sans
modifier les comportements validés.

## Implemented Architecture

```text
CardDefinition
  ├── is_champion
  ├── champion_health
  └── champion_ability (passive | active, programme déclaratif)

PlayerState
  ├── play_zone                  # cartes temporaires du tour
  ├── champions                  # board persistant du joueur
  └── activated_champion_ids     # reset au cleanup

Game.legal_actions()
  ├── PLAY     → PlayCard / ActivateChampion / ...
  └── ATTACK   → AssignPower(target=player|champion_instance_id)

Game.apply(action)
  ├── _play_card / _resolve_champion
  ├── _activate_champion
  └── _assign_power_to_target
```

Le moteur reste l'autorité unique sur la légalité, la consommation du Power, la destruction, la
victoire et les transitions de phase. Les définitions de cartes ne mutent jamais directement
`GameState`. Une capacité qui nécessite un choix futur devra produire une action légale dédiée,
sur le modèle de `BanishCard` et `RecruitFreeCard`.

## Data Model

Modèle effectivement implémenté :

```text
CardDefinition
  champion_health: int | None
  champion_ability: ChampionAbility | None

PlayerState
  champions: list[CardInstance]
  activated_champion_ids: set[str]
  played_card_ids_this_turn: set[str]
  pending_decision: PendingDecision | None
  pending_homodeus_champion_recruitment: bool

AttackTarget
  kind: PLAYER | CHAMPION
  champion_id: str | None

PendingDecision
  kind: SELECT_SPECTRA_DISCARD | SELECT_CHAMPION_DISCARD
        | SELECT_OPPONENT_CHAMPION | ORDER_CHAMPION_DESTRUCTION
        | SELECT_EFFECT_COPY
  candidates / remaining_candidates: tuple[str, ...]
```

`champion_health` est obligatoire et strictement positif lorsque `is_champion=True`, nul/absent pour
les autres cartes. `champion_ability` doit être immuable et sérialisable ; sa représentation doit
couvrir les variantes définies dans cette architecture : pose, activation, passif, préconditions,
choix, protections, recrutement et copie d'effets. Les `CardInstance.instance_id` restent les
identifiants de transport et de ciblage.

L'action d'attaque expose une cible via `AssignPower(amount, target)`. Le moteur exige que `amount`
soit exactement le Power disponible : il dérive ensuite la consommation (PV du champion ou totalité
du Power contre le joueur), ce qui empêche une allocation partielle ou une triche par action. La
forme historique `AssignPower(active.power)` reste compatible et cible le joueur adverse.

## Backend Flow

### Pose et activation

`_play_card` valide la carte, la retire de la main, puis :

- carte normale : l'ajoute à `play_zone` et résout son `Effect` immédiat ;
- champion : l'ajoute à `champions`, enregistre/active son passif et résout uniquement son éventuel
  effet de pose ; son effet actif n'est jamais exécuté automatiquement.

`_activate_champion` vérifie la phase, le propriétaire, la présence sur le board, le type actif et
l'absence de l'ID dans `activated_champion_ids`, puis résout le programme actif et ajoute l'ID à
l'ensemble seulement après validation des préconditions.

### Attaque

```text
ATTACK
  → legal_targets(remaining_power)
  → AssignPower(champion_id)
       → vérifier remaining_power >= champion_health
       → power -= champion_health
       → retirer le champion du board
       → discard_pile du défenseur
       → rester en ATTACK si power > 0
  → AssignPower(player)
       → dégâts = max(0, power - shields_in_opponent_hand)
       → réduire health, power = 0
       → victoire ou cleanup
```

Les vérifications de cible et les mutations doivent être atomiques : une cible invalide ne doit
modifier ni les zones ni le Power. Le champion détruit est déplacé vers la défausse du joueur
défenseur avant de recalculer les actions légales.

Le cleanup déplace seulement `play_zone` et la main vers la défausse, conserve `champions`, vide le
Power et réinitialise `activated_champion_ids`, `played_card_ids_this_turn`, `pending_decision` et
`pending_homodeus_champion_recruitment` avec les autres décisions temporaires. En fonctionnement
normal, un cleanup ne doit toutefois être atteint qu'après résolution d'une décision obligatoire.

Une action qui crée une décision résout d'abord sa partie sans choix, puis renseigne
`pending_decision`. L'action de sélection suivante applique la mutation choisie, déclenche les effets
associés et vide la décision, ou la remplace par l'étape suivante. Pour Saule et Général Décurion,
les candidats restants sont ainsi recalculés après chaque choix.

### Interaction avec Domination

La pose d'une carte ajoute son `instance_id` à `played_card_ids_this_turn` avant la résolution de
son effet. Pour `Domination`, une carte de faction peut donc satisfaire la condition si elle est :

- dans la main du joueur ; ou
- dans `play_zone` pendant le tour ; ou
- dans `champions` et présente dans `played_card_ids_this_turn`.

Un champion posé lors d'un tour précédent ne satisfait pas cette condition, sauf si une future règle
de carte demande explicitement de compter les champions persistants sur le board. La carte en cours
de résolution reste exclue de son propre test.

## Frontend Flow

Aucun frontend n'existe dans le dépôt. L'observation du `GameState` doit néanmoins exposer :

- les champions de chaque joueur et leurs PV maximum ;
- les capacités structurées nécessaires au futur joueur neuronal ;
- les champions actifs déjà utilisés ce tour ;
- les actions d'attaque légales avec leur cible et le Power restant implicite.

## Authorization And Feature Gates

Aucune autorisation applicative n'est concernée. Si un ruleset ou un mode de jeu est introduit pour
activer progressivement les champions, il doit être porté par la configuration de création de
`Game`, inclus dans les seeds/replays et visible par le benchmark. Le runner ne doit pas connaître
les règles de ciblage.

## Observability And Operations

Les traces optionnelles devront pouvoir enregistrer `card_id`, `instance_id`, propriétaire, pose,
activation, cible d'attaque, Power avant/après, PV du champion, destruction et cause de victoire.
Elles restent désactivées dans les benchmarks massifs. Ajouter un scénario de benchmark séparé avec
champions afin de mesurer le coût de génération des cibles et de résolution des capacités.

## Edge Cases

- champion posé puis détruit pendant le même tour : il rejoint la défausse et n'est pas déplacé une
  seconde fois au cleanup ;
- champion posé ce tour puis utilisé pour `Domination` : son `instance_id` reste marqué jusqu'au
  cleanup, même s'il est dans la zone `champions` ;
- champion présent depuis un tour précédent : il ne compte pas pour `Domination` selon la règle
  actuelle des cartes jouées pendant le tour ;
- plusieurs copies du même champion : ciblage et activation par `instance_id`, jamais par `card_id` ;
- Power inférieur aux PV de tous les champions : seul le joueur est légal ;
- Power égal aux PV : cible légale pour le moteur et pour le `RandomPlayer` ;
- zéro Power : l'attaque doit terminer proprement sans cible champion et sans dégâts ;
- victoire après ciblage du joueur : aucune action d'activation ou d'attaque supplémentaire ;
- tentative d'activation adverse, double activation ou activation hors `PLAY` : action refusée sans
  mutation ;
- observation/clone : les listes `champions` et ensembles d'activation sont indépendants ;
- effet actif/passif nécessitant un choix : état pending explicite, jamais une mutation implicite.
- décision en attente : aucune action de jeu concurrente n'est légale ; les candidats obsolètes ou
  les identifiants absents doivent être refusés sans mutation ;
- Drones Numeri activé sans recrutement compatible avant le cleanup : le flag expire sans effet ;

## Testing Strategy

Ajouter des tests moteur couvrant :

- validation des champs champion et refus d'une définition incohérente ;
- pose sans effet immédiat, appartenance au bon board et persistance après cleanup ;
- activation active le tour de pose, une seule fois par tour, reset au tour suivant ;
- activation active légale avant, entre ou après les autres cartes jouées pendant `PLAY`, avec les
  décisions obligatoires déjà en attente traitées en priorité ;
- `played_card_ids_this_turn` est alimenté pour les cartes normales et les champions, puis vidé au
  cleanup ; `Domination` accepte un champion posé ce tour et refuse un champion ancien ;
- le `RandomPlayer` inclut les champions dont les PV sont exactement égaux au Power restant ;
- passifs actifs dès la pose, puis appliqués selon le déclencheur propre à chaque champion ;
- Zélote : sélection d'une cible champion et destruction directe, sans dépendre du Power ;
- Saule : destruction de tous les champions adverses à partir de 15 maîtrise ;
- Saule : choix de l'ordre, destruction et résolution des déclencheurs champion par champion ;
- Additri : activation immédiate possible et formule de Power dépendant des cartes Maquis jouées
  dans le tour ;
- Additri : le décompte utilise `played_card_ids_this_turn`, inclut Additri et conserve les cartes
  Maquis jouées même si elles ne sont plus visibles dans une zone ;
- ciblage d'un champion avec Power exactement suffisant, insuffisant et excédentaire ;
- destruction, déplacement en défausse, conservation du Power restant et ciblage successif ;
- Saule avec zéro, un et plusieurs champions : absence d'effet sans cible, ordre choisi et
  déclenchement séparé après chaque destruction ;
- Ordre : Systema applique le seuil de pioche après le gain de maîtrise ; Giga pioche uniquement le
  tour de pose puis applique Domination sur son activation ; Zetta filtre les cibles d'attaque et
  conserve son passif tant qu'elle est sur le board ;
- Général Décurion : chaque effet Homodeus non champion du tour est proposé une fois, dans un ordre
  choisi par le joueur ou tiré aléatoirement par le `RandomPlayer`, avec réévaluation de l'état
  courant ;
- ciblage joueur avec Bouclier, victoire, zéro Power et absence de champions ;
- atomicité des actions invalides ;
- copie détachée, clone, reproductibilité à seed et compatibilité des cartes existantes ;
- `RandomPlayer` : ensemble de cibles valide, tirage reproductible et recalcul après chaque
  destruction.

Les tests de chaque champion seront ajoutés dans un fichier de famille ou de carte dédié, sans
coupler les tests de capacités aux détails internes de la boucle d'attaque.

## Rollout And Migration

Implémentation livrée dans l'ordre suivant :

1. introduire le modèle de champion, la zone persistante et les invariants sans modifier le
   catalogue existant ;
2. introduire l'action d'activation et le mécanisme de capacité avec un champion de test minimal ;
3. migrer l'attaque vers des cibles explicites en conservant l'alias `AssignDamage` ;
4. adapter `RandomPlayer`, `GameRunner`, observations, clones et tests de compatibilité ;
5. ajouter les champions réels, leurs decks et leurs tests un par un ;
6. mettre à jour `doc/Current state/Game engine.md`, le benchmark et la version du ruleset.

La livraison ne nécessite aucune migration persistante. Les parties et replays futurs doivent
toutefois conserver la version du ruleset, car la forme des actions d'attaque et l'état des
champions font partie de l'observation.

## Files Changed

- `shards_ai/game/state.py` : board persistant et IDs d'activation ;
- `shards_ai/game/actions.py` : activation et cible d'attaque ;
- `shards_ai/game/cards/model.py` : PV et capacité de champion ;
- `shards_ai/game/game.py` : pose, activation, cibles, destruction, cleanup, observation/clone ;
- `shards_ai/ai/random_player.py` : choix aléatoire des cibles ;
- `shards_ai/game/cards/definitions/` : fichiers des champions et éventuels decks de famille ;
- `shards_ai/game/cards/definitions/zelote_des_epines.py` : ajout de l'opération Union de
  destruction ciblée ;
- `shards_ai/game/cards/definitions/saule_vengeur.py` : ajout de l'opération conditionnelle de
  destruction globale ;
- `shards_ai/game/cards/definitions/additri_gaia_mancienne.py` :
  définition du premier champion Maquis ;
- `shards_ai/game/cards/definitions/ia_systema.py`, `giga_adepte_de_la_source.py` et
  `zetta_l_encodeuse.py` : champions Ordre ;
- `shards_ai/game/cards/definitions/primus_pilus.py`, `drones_numeri.py`, `evokatus.py`,
  `broyeu_optio.py`, `drakonarius.py` et `general_decurion.py` : champions Homodeus ;
  champions Homodeus ;
- `shards_ai/game/cards/definitions/homodeus_deck.py` : ajout des nouveaux champions Homodeus ; les
  quantités de Valkyrie des Landes (`×1`) et Légionnaire Korvus (`×3`) restent inchangées ;
- `shards_ai/game/cards/catalog.py` et `shards_ai/game/cards/definitions/__init__.py` : registre ;
- `tests/game/` : invariants moteur, attaque et politique aléatoire ;
- `doc/Current state/Game engine.md` : comportement livré après implémentation ;
- `benchmarks/benchmark_game.py` : benchmark engine-only du ruleset livré.
