# Cartes de factions et mécaniques avancées — Architecture

**Statut : DONE / livré** — Cette architecture décrit les ajouts réalisés pendant la session : factions,
cartes Maquis, Spectra, Ordre et Homodeus, mécaniques associées, limite de partie et optimisation
du moteur.

## Objectif

Étendre le moteur duel avec les cartes hors champions du catalogue de Shards of Infinity, tout en
conservant un moteur déterministe, testable et indépendant des joueurs. Les cartes doivent rester
faciles à modifier lorsque le catalogue augmente.

## État de référence

Le moteur est mémoire, synchrone et déterministe à seed et actions identiques. Les frontières
restent `Game.legal_actions()` et `Game.apply(action)`, tandis que `RandomPlayer` ne contient pas de
logique de règles.

Le cycle de partie est :

```text
PLAY → BUY → ATTACK → CLEANUP → PLAY
```

Le catalogue contient 36 définitions : 4 cartes neutres du deck de départ et 32 cartes de factions.
Le deck central contient 72 instances, six étant placées dans la rivière au démarrage ; 66 restent
dans le deck central.

## Comportement livré

### Factions

`Faction` contient désormais :

- `NEUTRAL` — Cristal, Blaster, Réacteur d'éclat et Éclat de l'infini ;
- `MAQUIS` — 21 instances ;
- `SPECTRA` — 19 instances ;
- `ORDER` — 19 instances ;
- `HOMODEUS` — 13 instances.

Les quatre cartes neutres restent inchangées dans les decks de départ des joueurs. Les decks des
joueurs ne contiennent aucune carte de faction ; les cartes de faction sont uniquement dans le
deck central et la rivière.

### Cartes Maquis

| Carte | Coût | Copies | Effet principal |
|---|---:|---:|---|
| Aspirant Maquis | 1 | 3 | +3 santé ; Union : +5 Power |
| Clerc aux Spores | 2 | 3 | +4 santé |
| Ermite Fongique | 3 | 2 | +1 maîtrise ; à 10 maîtrise : +5 santé |
| Zélote des épines | 3 | 2 | Pioche 1 ; Bouclier 3 |
| Chevalier Le'Shai | 3 | 3 | +3 Power ; Union : +3 Power |
| Gardien de la Forêt | 4 | 3 | +2 Power, pioche 1 ; Union : +6 santé |
| Saule Vengeur | 4 | 1 | +4 Power |
| Ojas, druide de la genèse | 4 | 1 | Copie le dernier effet non champion ; deux copies à 20 maîtrise |
| Elemental du Sillon | 5 | 2 | +4 santé, pioche 1 ; à 50 santé : +6 Power |
| Racine de la Forêt | 7 | 1 | +10 santé ; Union : +10 Power |

### Cartes Spectra

| Carte | Coût | Copies | Effet principal |
|---|---:|---:|---|
| Éclaireur spectral | 1 | 3 | +2 Power ; Echo : +4 Power |
| Assassin du vide | 2 | 3 | +5 Power |
| Apôtre des ombres | 2 | 3 | +2 Power ; bannissement optionnel |
| Sentinelle des ténèbres | 3 | 2 | +3 Power |
| Fléau des Ombres | 3 | 3 | +1 maîtrise ; bannissement optionnel |
| Brise-Éther | 4 | 2 | +4 Power ; à 10 maîtrise : +4 Power |
| Héritier du Néant | 5 | 2 | +3 Power ; Echo : +2 par carte Spectra en défausse |
| Zara Ra, Écorcheur d'âme | 5 | 1 | +4 Power, +1 maîtrise ; jusqu'à 2 bannissements à 10 maîtrise |

### Cartes Ordre

| Carte | Coût | Copies | Effet principal |
|---|---:|---:|---|
| Initié de l'Ordre | 1 | 3 | +2 Gems ; Domination : +2 maîtrise |
| Garde Mémoire | 2 | 2 | +1 maîtrise ; à 10 maîtrise : pioche 1 |
| Prophète de l'éclat | 3 | 3 | +2 maîtrise |
| Pirate Hérétique | 3 | 3 | Pioche 2 |
| Moine du portail | 3 | 2 | Recrutement gratuit d'une carte coûtant au plus 6 ; main à 15 maîtrise |
| Voyante de Volonté | 4 | 2 | +2 Gems ; Bouclier 5 |
| Moine Cryptopoing | 5 | 2 | Pioche 1 ; Bouclier 8 |
| Omnius l'érudit | 6 | 1 | Pioche 2 ; Domination : +5 maîtrise |
| Le grand architecte | 7 | 1 | +5 maîtrise |

### Cartes Homodeus

| Carte | Coût | Copies | Effet principal |
|---|---:|---:|---|
| Drone Kiln | 1 | 3 | +2 Gems |
| Drones Miniers | 2 | 3 | +1 Gem, pioche 1 |
| Légionnaire Korvus | 3 | 3 | Bouclier 2, +2 Power |
| Drone Réacteur | 3 | 3 | +3 Gems |
| Valkyrie des Landes | 4 | 1 | +4 Power |

## Mécaniques

### Effets structurés

Une carte possède un `Effect` composé d'étapes et d'`Operation`. Les opérations actuellement
utilisées couvrent les Gems, le Power, les dégâts directs, la maîtrise, la santé, la pioche, la
copie d'effet, le bannissement, le recrutement gratuit, les bonus liés à la défausse et la victoire.
Les seuils sont évalués au moment où la carte est jouée, dans l'ordre de ses opérations.

### Union

Une opération `requires_union` s'active si une autre carte de la même faction est en main ou dans
la zone de jeu pendant le tour courant. La carte qui porte l'effet ne peut pas satisfaire seule la
condition.

### Echo

Une opération `requires_echo` s'active uniquement si la défausse du joueur contient une carte
Spectra. La main et la zone de jeu ne satisfont pas cette condition. Les bonus proportionnels
comptent les cartes Spectra présentes dans la défausse.

### Domination

Une opération `requires_domination` s'active si le joueur possède simultanément une carte
Homodeus, Maquis et Spectra en main ou dans sa zone de jeu. La carte en cours de résolution ne peut
pas satisfaire seule une faction manquante.

### Bouclier

Chaque carte possède une valeur `shield`. Pendant `ATTACK`, le moteur additionne les Boucliers des
cartes présentes dans la main du défenseur, puis calcule :

```text
dégâts reçus = max(0, Power assigné - Bouclier total)
```

Les cartes Bouclier restent en main après l'attaque.

### Bannissement

Les effets de bannissement créent une décision optionnelle immédiate. `BanishCard` peut cibler une
carte de la main ou de la défausse, mais jamais une carte déjà jouée pendant le tour. `SkipBanish`
refuse la décision. `RandomPlayer` choisit d'activer le bannissement avec une probabilité de 50 %.

### Recrutement gratuit

Le `Moine du portail` crée une décision obligatoire `RecruitFreeCard`. Le joueur choisit une carte
de la rivière coûtant au plus 6 sans payer de Gems. La carte rejoint la défausse, ou la main si le
joueur possède au moins 15 maîtrise. Le slot de rivière est immédiatement rempli si le deck
central contient encore une carte.

## Décisions d'architecture

1. **Une définition par type de carte.** Chaque carte possède son fichier dans
   `shards_ai/game/cards/definitions/`. Les compositions de faction sont regroupées dans un fichier
   `*_deck.py`.
2. **Le catalogue est explicite.** `CARD_CATALOG` est indexé par `card_id`, vérifie l'unicité des
   36 IDs au chargement et expose des définitions immuables.
3. **Les effets sont déclaratifs.** Les règles de carte ne sont pas dispersées dans des branches
   basées sur les noms de cartes dans `Game`.
4. **Le moteur reste l'autorité.** `legal_actions()` expose les décisions intermédiaires et
   `apply()` valide les actions avant mutation.
5. **Les décisions en attente appartiennent au joueur.** Les champs
   `pending_banishes`, `pending_free_recruit_cost` et `pending_free_recruit_to_hand` vivent dans
   `PlayerState` et sont réinitialisés au cleanup.
6. **Compatibilité Power.** `power` est le nom canonique ; `pending_damage`, `damage` et
   `AssignDamage` restent des alias de compatibilité.
7. **Pas de champions dans cette étape.** `CardDefinition.is_champion` prépare l'extension, mais
   aucun champion n'est ajouté et aucune règle de champion n'est supposée.
8. **Limite de sécurité dans l'orchestrateur.** `GameRunner` arrête par défaut une partie après
   `100 × nombre de joueurs` tours et la déclare nulle, afin d'éviter les boucles prolongées dues
   notamment aux soins. La limite d'actions par défaut reste 10 000.
9. **Optimisation sans changement de modèle d'exécution.** Une référence locale du joueur actif est
   réutilisée dans `Game.apply`. Aucun cache, parallélisme, multiprocessing ou multithreading n'a
   été introduit.

## Non-objectifs

- ajouter des champions ou des règles de champions ;
- modifier les decks de départ des joueurs ;
- ajouter une interface graphique, une API réseau ou une persistance ;
- implémenter un joueur stratégique ou économique ;
- remplacer le catalogue explicite par une découverte dynamique ;
- introduire une optimisation concurrente ou un cache de résolution sans mesure et décision dédiée.

## Flux principal

1. `Game.new()` construit les decks de départ, le deck central de 72 instances et les six slots de
   rivière.
2. `PlayCard` retire une instance de la main, la place dans `play_zone`, puis résout son
   `Effect`.
3. Les conditions sont évaluées sur l'état courant du joueur, ses zones et la défausse.
4. Une décision intermédiaire éventuelle bloque toute action incompatible jusqu'à son choix.
5. Les Gems peuvent servir à `GainMastery` ou aux achats pendant `BUY` ; le recrutement gratuit ne
   dépense pas de Gems.
6. `AssignPower` applique les Boucliers de la main adverse, puis le cleanup déplace les cartes et
   pioche cinq cartes.
7. `GameRunner` coordonne les joueurs et transforme l'atteinte de la limite de tours en `DRAW`.

## Performance et validation

Le benchmark de référence est `benchmarks/benchmark_game.py`, avec 10 000 parties et les seeds
`0..9999`. La comparaison contrôlée la plus récente a mesuré une médiane de 609,4 parties/s avant
l'optimisation de `Game.apply`, puis 670,9 parties/s après, soit +10,1 %.

La suite compte 68 tests. Elle couvre notamment :

- la composition du catalogue, des factions et du deck central ;
- les effets et seuils de chaque famille de cartes ;
- Union, Echo, Domination, Bouclier, bannissement et recrutement gratuit ;
- la légalité des actions, l'atomicité des achats et les décisions en attente ;
- le cleanup, le remélange, la reproductibilité, le joueur random et les limites d'exécution.

## Questions ouvertes

- Les noms affichés et les traductions officielles devront-ils être normalisés séparément des IDs
  techniques ?
- Les dégâts directs, les champions et les cibles multiples nécessiteront-ils des opérations et
  actions dédiées lorsqu'ils seront introduits ?
- Le catalogue explicite restera adapté tant que le nombre de définitions reste modéré ; une
  génération ou une validation externe pourra être étudiée si le volume devient beaucoup plus grand.

## Fichiers de référence

- `shards_ai/game/enums.py` — factions et états de phase ;
- `shards_ai/game/cards/model.py` — définitions, effets, opérations et instances ;
- `shards_ai/game/cards/definitions/` — un fichier par carte et compositions de faction ;
- `shards_ai/game/cards/catalog.py` — registre des définitions ;
- `shards_ai/game/cards/central_deck.py` — composition du deck central ;
- `shards_ai/game/game.py` — validation, transitions et résolution des effets ;
- `shards_ai/game/actions.py` et `shards_ai/game/state.py` — décisions et état en attente ;
- `shards_ai/game/runner.py` — limites de tours et d'actions ;
- `shards_ai/ai/random_player.py` — politique random des décisions ;
- `tests/game/test_*_cards.py` — tests des cartes et mécaniques ;
- `benchmarks/benchmark_game.py` — benchmark engine-only.
