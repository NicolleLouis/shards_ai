# État courant — Moteur et cartes

Cette page décrit le comportement réellement disponible dans le code. Les décisions
d'architecture historiques sont conservées dans [Architecture](../Architecture/).

## Vue d'ensemble

| Composant | État | Responsabilité actuelle |
|---|---|---|
| `Game` | Disponible | État, règles et transitions d'une partie duel |
| `Player` | Disponible | Interface d'un joueur recevant observation et actions légales |
| [`RandomPlayer`](Random%20player.md) | Disponible | Politique aléatoire valide pour jeu, maîtrise, achats et attaque |
| [`HeuristicPlayer`](Heuristic%20player.md) | Disponible | Politique pondérée déterministe fondée sur les features des actions légales |
| `GameRunner` | Disponible | Exécution complète d’une partie avec limites de tours et d’actions |
| Catalogue de cartes | Disponible | Définitions immuables indexées par `card_id` |

## Cartes et catalogue

Le modèle est dans `shards_ai/game/cards/`.

- `CardDefinition` contient `card_id`, `name`, `cost`, `faction`, `shield`, `effect` et
  `is_mercenary`, ainsi que pour les champions `champion_health`, une éventuelle capacité de pose,
  une capacité active et un passif.
- Les factions disponibles sont `neutral`, `maquis`, `spectra`, `homodeus` et `order`.
- Les cartes de départ `Cristal`, `Blaster`, `Réacteur d'éclat` et `Éclat de l'infini` ont la faction `neutral`.
- `Effect` peut produire des Gems, du Power, de la santé, de la maîtrise, piocher, copier un effet,
  proposer un bannissement, récupérer un mercenaire depuis la défausse ou calculer un bonus selon
  la défausse.
- `EffectStep` et `Operation` forment la représentation déclarative des effets ; les opérations
  supportent notamment `gain_gems`, `gain_power`, `gain_mastery`, les dégâts directs et la victoire.
- `CardInstance` représente une copie physique avec un `instance_id` distinct.
- Chaque type de carte possède son fichier dans `cards/definitions/`.
- `CARD_CATALOG` permet une recherche en temps constant par `card_id`.
- Les noms de cartes sont informatifs et ne participent pas aux règles.

Cartes disponibles :

| ID                        | Nom                       | Coût | Effet                                                                                      |
| ------------------------- | ------------------------- | ---: | ------------------------------------------------------------------------------------------ |
| `crystal`                 | Cristal                   |    0 | +1 Gem                                                                                     |
| `blaster`                 | Blaster                   |    0 | +1 Power                                                                                   |
| `shard_reactor`           | Réacteur d'éclat          |    0 | +2 Gems, +3 à 5 maîtrise, +4 à 15                                                          |
| `infinity_shard`          | Éclat de l'infini         |    0 | +2 Power, +3 à 10, +5 à 20, victoire à 30                                                  |
| `aspirant_maquis`         | Aspirant Maquis           |    1 | +3 santé ; Union : +5 Power                                                                |
| `clerc_aux_spores`        | Clerc aux Spores          |    2 | +4 santé ; Mercenaire                                                                      |
| `ermite_fongique`         | Ermite Fongique           |    3 | +1 maîtrise ; à 10 maîtrise : +5 santé ; Mercenaire                                        |
| `zelote_des_epines`       | Zélote des Épines         |    3 | Pioche 1 ; Union : destruction directe d'un champion ; Bouclier 3                          |
| `chevalier_le_shai`       | Chevalier Le'Shai         |    3 | +3 Power ; Union : +3 Power ; Mercenaire                                                   |
| `gardien_de_la_foret`     | Gardien de la Forêt       |    4 | +2 Power, pioche 1 ; Union : +6 santé                                                      |
| `saule_vengeur`           | Saule Vengeur             |    4 | +4 Power ; à 15 maîtrise, destruction ordonnée de tous les champions adverses ; Mercenaire |
| `ojas`                    | Ojas, druide de la genèse |    4 | Copie le dernier effet non champion joué ; deux copies à 20 maîtrise                       |
| `elemental_du_sillon`     | Élémental du Sillon       |    5 | +4 santé, pioche 1 ; à 50 santé : +6 Power                                                 |
| `racine_de_la_foret`      | Racine de la Forêt        |    7 | +10 santé ; Union : +10 Power ; Mercenaire                                                 |
| `eclaireur_spectral`      | Éclaireur spectral        |    1 | +2 Power ; Echo : +4 Power                                                                 |
| `void_assassin`           | Assassins du vide         |    2 | +5 Power ; Mercenaire                                                                      |
| `apotre_des_ombres`       | Apôtre des ombres         |    2 | +2 Power ; bannissement optionnel ; Mercenaire                                             |
| `sentinelle_des_tenebres` | Sentinelle des ténèbres   |    3 | +3 Power ; récupère un mercenaire de la défausse dans la main                              |
| `fleau_des_ombres`        | Fléau des Ombres          |    3 | +1 maîtrise ; bannissement optionnel ; Mercenaire                                          |
| `brise_ether`             | Brise-Éther               |    4 | +4 Power ; +4 à 10 maîtrise ; Mercenaire                                                   |
| `heritier_du_neant`       | Héritier du Néant         |    5 | +3 Power ; Echo : +2 par carte Spectra en défausse ; Mercenaire                            |
| `zara_ra`                 | Zara Ra, Écorcheur d’âme  |    5 | +4 Power, +1 maîtrise ; jusqu’à 2 bannissements à 10 maîtrise ; Mercenaire                 |
| `initie_de_l_ordre`       | Initié de l'Ordre         |    1 | +2 Gems ; Domination : +2 maîtrise                                                         |
| `garde_memoire`           | Garde Mémoire             |    2 | +1 maîtrise ; à 10 maîtrise, pioche 1 ; Mercenaire                                         |
| `prophete_de_leclat`      | Prophète de l'éclat       |    3 | +2 maîtrise ; Mercenaire                                                                   |
| `pirate_heretique`        | Pirate Hérétique          |    3 | Pioche 2 ; Mercenaire                                                                      |
| `moine_du_portail`        | Moine du portail          |    3 | Recrutement gratuit d'une carte de coût ≤ 6 ; à 15 maîtrise, dans la main                  |
| `voyante_de_volonte`      | Voyante de Volonté        |    4 | +2 Gems ; Bouclier 5                                                                       |
| `moine_cryptopoing`       | Moine Cryptopoing         |    5 | Pioche 1 ; Bouclier 8                                                                      |
| `omnius_l_erudit`         | Omnius l'érudit           |    6 | Pioche 2 ; Domination : +5 maîtrise ; Mercenaire                                           |
| `le_grand_architecte`     | Le Grand Architecte       |    7 | +5 maîtrise ; Mercenaire                                                                   |
| `drone_kiln`              | Drone Kiln                |    1 | +2 Gems                                                                                    |
| `drones_miniers`          | Drones Miniers            |    2 | +1 Gem, pioche 1                                                                           |
| `legionnaire_korvus`      | Légionnaire Korvus        |    3 | Bouclier 2, +2 Power ; récupère un champion de la défausse                                 |
| `drone_reacteur`          | Drone Réacteur            |    3 | +3 Gems                                                                                    |
| `valkyrie_des_landes`     | Valkyrie des Landes       |    4 | +4 Power ; Inspiration : l'adversaire perd 2 maîtrise ; Mercenaire                         |

Champions ajoutés au catalogue :

| Faction | Carte | Coût | PV | Capacité |
|---|---|---:|---:|---|
| Maquis | Additri, Gaïamancienne | 5 | 5 | Actif : +2 Power, puis +2 par carte Maquis jouée ce tour |
| Spectra | Li Hin, la Brisée | 3 | 1 | Immunisée à l'attaque par Power ; actif : +1 Power |
| Spectra | Zen Chi Set, Fléau des dieux | 7 | 5 | Actif : +3 Power, puis récupère une carte Spectra de la défausse |
| Ordre | I.A. Systema | 3 | 4 | Actif : +1 maîtrise ; à 20, pioche 2 |
| Ordre | Giga, Adepte de la Source | 2 | 4 | Pose : pioche 1 ; Domination active : +3 maîtrise |
| Ordre | Zetta, l'encodeuse | 5 | 5 | Bouclier 5 en main ; rend les autres champions non ciblables par Power |
| Homodeus | Primus Pilus | 2 | 6 | Actif : pioche 2 si au moins 3 champions Homodeus sont en jeu |
| Homodeus | Drones Numeri | 3 | 5 | Actif : +1 Gem ; prochain champion Homodeus recruté posé et activé |
| Homodeus | Evokatus | 4 | 2 | Pose : pioche 1 ; actif : +1 Power par champion Homodeus en jeu |
| Homodeus | Broyeur Optio | 5 | 4 | Actif : +3 Power ; +2 supplémentaires à 10 maîtrise |
| Homodeus | Drakonarius | 6 | 2 | Protégé par Général Décurion contre Power ; actif : +6 Power |
| Homodeus | Général Décurion | 7 | 7 | Actif : +3 Gems ; à 20, choix de copie des effets Homodeus non champions du tour |

Les cartes Maquis appartiennent à la faction `Maquis`. Une capacité Union est active si une
autre carte de la même faction est présente en main ou déjà jouée pendant le tour.

Les cartes Spectra utilisent `Echo`, actif uniquement lorsqu'une carte Spectra est dans la défausse.
Les effets de bannissement créent une décision optionnelle immédiate : une carte de la main ou de la
défausse peut être bannie, mais aucune carte déjà jouée ce tour ne peut être ciblée. `RandomPlayer`
choisit d'activer ce pouvoir dans 50 % des cas.

Les cartes Ordre utilisent `Domination`, active si le joueur a une carte `Homodeus`, `Maquis` et
`Spectra` en main ou dans sa zone de jeu ; un champion posé pendant le tour compte via
`played_card_ids_this_turn`. Le `Moine du portail` crée une décision obligatoire de
recrutement gratuit parmi les cartes de la rivière coûtant au maximum 6 ; la carte rejoint la
défausse, ou la main à partir de 15 maîtrise.

## Mise en place

Chaque joueur commence avec :

- 50 points de vie ;
- une maîtrise initiale de 0 pour le premier joueur et de 1 pour le second ;
- un deck de 10 cartes composé de 7 `Cristal`, 1 `Blaster`, 1 `Réacteur d'éclat` et 1 `Éclat de l'infini` ;
- une main initiale de 5 cartes après mélange ;
- Gems et Power de tour à 0.

La partie possède également un deck central de 87 cartes : 22 cartes Maquis, 21 cartes Spectra,
22 cartes Ordre et 22 cartes Homodeus, dont 3 `Assassins du vide`. Il est mélangé, puis six cartes
sont placées dans la rivière ; 81 cartes restent dans le deck central.

## État et zones

Chaque joueur possède une main, une pioche, une défausse, une zone de jeu temporaire et un board de
champions persistant. Les ressources et décisions de tour sont stockées dans `PlayerState` :

- `gems` : monnaie dépensable pendant `BUY` ;
- `mastery` : ressource permanente bornée entre 0 et 30 ;
- `mastery_action_used` : indique si `GainMastery` a déjà été utilisée ce tour ;
- `power` : ressource produite pendant `PLAY` et convertie en dégâts pendant `ATTACK` ;
- `health` : santé bornée entre 0 et 50 ;
- `pending_damage` : alias de compatibilité déprécié vers `power`.
- `pending_banishes` : nombre de décisions de bannissement encore à résoudre ;
- `pending_free_recruit_cost` et `pending_free_recruit_to_hand` : décision de recrutement gratuit
  du Moine du portail.
- `champions` : champions actuellement posés et associés au joueur ;
- `activated_champion_ids` : champions actifs déjà utilisés pendant le tour ;
- `played_card_ids_this_turn` : cartes jouées pendant le tour, utilisé par Domination et Additri ;
- `pending_decision` : choix courant pour les destructions, récupérations et copies ;
- `pending_homodeus_champion_recruitment` : flag temporaire armé par Drones Numeri.

`GameState.pending_damage` reste disponible comme vue de compatibilité vers le Power du joueur
actif.

La partie possède :

- `central_deck` : pioche partagée ;
- `river` : six slots contenant une carte ou `None`.

Lorsqu'une pioche personnelle est vide, sa défausse est mélangée pour la reconstituer. Les pioches
multiples continuent carte par carte après ce remélange et s'arrêtent lorsqu'il n'y a plus aucune
carte disponible : demander `X` cartes signifie piocher au maximum `X` cartes.

## Phases et actions

Le cycle réellement exécuté est :

```text
PLAY → BUY → ATTACK → CLEANUP → PLAY
```

Actions publiques :

- `PlayCard(card_id)` pendant `PLAY` ; une carte normale résout son effet, un champion rejoint le
  board et déclenche seulement son éventuel effet de pose ;
- `ActivateChampion(champion_id)` pendant `PLAY` ; active un champion une fois par tour, dès sa pose ;
- `PassPlayPhase()` pendant `PLAY` ; passe à `BUY` ;
- `BuyCard(river_slot, card_instance_id)` pendant `BUY` ;
- `RecruitMercenary(river_slot, card_instance_id)` pendant `BUY` pour une carte mercenaire ; paie
  le coût, joue immédiatement la carte et résout son effet ;
- `StopBuying()` pendant `BUY` ; remet les Gems à zéro et passe à `ATTACK` ;
- `GainMastery()` pendant `PLAY` ; dépense 1 Gem, ajoute 1 maîtrise et n'est utilisable qu'une fois par tour ;
- `BanishCard(card_id)` pendant `PLAY` ou `BUY` lorsqu'un effet de bannissement est en attente ;
  retire la carte de la main ou de la défausse ;
- `SkipBanish()` pendant `PLAY` ou `BUY` ; refuse le bannissement optionnel en attente ;
- `RecruitFreeCard(river_slot, card_instance_id)` pendant `PLAY` ou `BUY` lorsqu'un recrutement
  gratuit est en attente ; prend la carte sélectionnée sans dépenser de Gems ;
- `ChoosePendingDecision(choice_id)` lorsqu'une décision est en attente ;
- `AssignPower(amount, target)` pendant `ATTACK` ; cible le joueur ou un champion adverse dont les PV
  sont inférieurs ou égaux au Power disponible. Une cible champion est détruite intégralement et le
  Power restant peut être réassigné.

`AssignDamage` reste un alias de compatibilité déprécié vers `AssignPower`.

Lors d’une attaque contre le joueur, les valeurs `shield` de toutes les cartes Bouclier présentes
dans sa main sont additionnées et soustraites des dégâts reçus. Les cartes Bouclier restent en main.
Li Hin, les champions protégés par Zetta et Drakonarius sous Général Décurion ne sont pas des cibles
Power légales. Les destructions directes de Zélote et Saule peuvent toutefois les détruire.

Un achat est légal si la carte existe dans la rivière et si `cost <= gems`. Une carte mercenaire
abordable expose deux choix : `BuyCard` la place dans la défausse, tandis que
`RecruitMercenary` la place dans la zone de jeu et résout immédiatement son effet. La carte
recrutée retourne au bas du deck central au cleanup, sans mélange ; un mercenaire acheté
normalement reste dans le deck du joueur. La carte achetée est placée dans la défausse du joueur,
sauf si Drones Numeri a armé le recrutement d'un champion
Homodeus : elle rejoint alors le board et son activation est résolue immédiatement. Les Gems sont
diminuées, puis le slot est rempli avec la prochaine carte du deck central. Lorsque celui-ci est
vide, le slot devient `None` ; la rivière diminue progressivement et peut finir vide.

Pendant le cleanup, la zone de jeu et la main restante rejoignent la défausse ; les mercenaires
recrutés rejoignent le bas du deck central ; les champions restent sur le board. Les Gems, le Power,
les décisions temporaires, les activations de champions et les flags de tour sont réinitialisés,
cinq cartes sont piochées et le tour passe à l'autre joueur. La maîtrise est conservée.

## Validation et observation

`Game` est l'autorité sur les règles et valide toutes les actions. Un achat vérifie notamment :

- la phase courante ;
- l'index du slot ;
- la présence d'une carte ;
- la correspondance avec `card_instance_id` ;
- le coût payable.

Les validations ont lieu avant les mutations afin d'éviter les états partiellement modifiés.
`observation_for()` retourne une copie complète détachée pour rester compatible avec les joueurs
existants ; elle ne doit pas être transmise au réseau neural. `neural_observation_for(player_id)`
retourne une observation dédiée, détachée et normalisée du point de vue du joueur actif. Elle
contient les zones propres visibles, les comptages triés des zones sans ordre, la rivière et le deck
central restant, ainsi que les informations publiques de l'adversaire.

L'observation neural ne contient pas la main ou la pioche adverse individuellement. La composition
globale des cartes adverses non bannies et sa défausse sont exposées séparément. Les cartes visibles
individuellement portent leur `card_definition_id` et leur `instance_id` technique. Les champions
exposent leurs PV imprimés ; le moteur ne stocke pas de dégâts partiels sur un champion.

Les comptages sont des tuples triés de `(card_definition_id, quantity)`. Le masque
`played_faction_mask` contient quatre booléens pour les factions jouables (`maquis`, `spectra`,
`homodeus`, `order`) et ignore `neutral`. Il est calculé à partir des cartes jouées pendant le tour
courant uniquement. `played_champion_faction_mask` contient le même type de masque, mais uniquement
pour les champions joués pendant le tour courant ; un champion seulement activé n'y contribue pas.

## Limites d'exécution

`GameRunner` limite par défaut une partie à `100 × nombre de joueurs` tours. Si cette limite est
atteinte, la partie devient nulle (`GameStatus.DRAW`, sans gagnant), afin d’éviter les boucles
longues dues notamment aux soins. Il conserve également une limite de 10 000 actions par défaut
et lève `InvalidGameStateError` lorsque celle-ci est atteinte.

## Tests et performances

La suite pytest contient actuellement 365 tests couvrant :

- la composition des decks et de la rivière ;
- la reproductibilité des mélanges ;
- les effets Gems/Power et les seuils de maîtrise ;
- les cartes Maquis, Union, soins, copie d’effets et Bouclier ;
- l'initialisation, les bornes et l'action de gain de maîtrise ;
- les transitions et achats ;
- la couverture des décisions atomiques du runner ;
- le masquage, l'agrégation et les invariances de l'observation neural.
- la légalité des achats, les mutations atomiques et l'épuisement de la rivière ;
- l'index stable du catalogue de cartes ;
- les cartes Spectra, Echo, les bannissements optionnels et leurs limites de ciblage ;
- les cartes Ordre, Domination et le recrutement gratuit avec placement conditionnel ;
- les mercenaires, le recrutement immédiat, le retour au bas du deck central et la récupération
  par Sentinelle des Ténèbres ;
- les cartes Homodeus, leurs quantités, coûts, effets de ressources, de dégâts et de Bouclier ;
- les champions, leur board persistant, leurs activations, leurs effets de pose, leurs protections,
  les décisions en attente et l'assignation ciblée du Power ;
- le remélange des défausses ;
- les pioches partielles lorsque moins de cartes que demandé sont disponibles ;
- les actions invalides et la fin de partie ;
- le `RandomPlayer` et `GameRunner` ;
- la politique de gain de maîtrise et d'arrêt à 10 % du `RandomPlayer` ;
- la limite d'actions et la reproductibilité des parties random ;
- la limite de tours et la déclaration d'ex æquo.
- le clone détaché du jeu et la conservation de la position du flux aléatoire pour la recherche.

Le benchmark `benchmarks/benchmark_game.py` identifie le ruleset par son scénario et mesure par
défaut 10 000 parties engine-only avec les seeds `0..9999`. L'option `--games` permet un workload
plus court pour le profiling. Le cleanup possède un fast-path lorsque le joueur n'a recruté aucun
mercenaire ; il conserve exactement le comportement normal tout en évitant la partition de la
zone de jeu dans le cas courant. Les mesures de débit restent indicatives de l'environnement
d'exécution courant.

Le benchmark `benchmarks/benchmark_random_players.py` mesure 1 000 parties avec un
`HeuristicPlayer` contre un `RandomPlayer`, achats et recrutements inclus, sur les seeds `0..999`.
Le joueur heuristique alterne entre les deux positions selon la seed afin de répartir les rôles.
Les parties sont vérifiées
comme terminées ou nulles ; ce benchmark sert à isoler le coût des observations et des décisions
d'achat du coût du moteur engine-only.

`Game.clone()` utilise la copie détachée explicite des zones et une copie explicite de l'état du
flux `GameRandom`. Cette voie conserve la reproductibilité tout en évitant le `deepcopy` complet de
l'objet `Game`, notamment pour les recherches bornées du solveur de play turns.
