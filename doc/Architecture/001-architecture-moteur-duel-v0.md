# Moteur de jeu duel V0 — Architecture

## Objectif

Construire un moteur Python capable de simuler rapidement et correctement une partie duel
simplifiée de **Shards of Infinity**.

Le moteur doit être :

- déterministe à seed et actions identiques ;
- indépendant de toute stratégie d'IA ou interface graphique ;
- suffisamment rapide pour exécuter de nombreuses parties en simulation ;
- observable et rejouable pour faciliter le debug ;
- extensible vers les règles complètes et un environnement RL.

## État actuel

Le dépôt ne contient encore aucun code Python, test ou configuration de dépendances. Il contient
uniquement la documentation de roadmap et ce document d'architecture.

## Outillage Python

Le projet utilise **Poetry** pour isoler ses dépendances, créer son environnement virtuel et
verrouiller les versions. Les commandes Python du projet doivent être exécutées via `poetry run`
ou dans l'environnement créé par Poetry.

- dépendance de développement initiale : `pytest` ;
- pas de dépendance RL dans le moteur V0 ;
- les dépendances d'entraînement et Gymnasium seront ajoutées à l'étape environnement RL ;
- le fichier `poetry.lock` devra être versionné pour rendre les installations reproductibles.

## Règles incluses dans la V0

- exactement deux joueurs ;
- chaque joueur commence avec 50 points de vie ;
- un joueur perd dès que ses points de vie atteignent 0 ;
- les points de vie ne peuvent pas remonter dans cette version ;
- le premier joueur est tiré aléatoirement ;
- chaque joueur possède une main, une pioche et une défausse ;
- chaque joueur commence avec un deck de 10 cartes, puis en pioche 5 pour former sa main initiale ;
- toutes les cartes sont identiques et infligent exactement 1 point de dégât ;
- aucune autre capacité de carte ;
- aucune rivière d'achat et aucun achat de carte ;
- aucun champion ;
- aucune maîtrise ;
- pendant sa phase de jeu, un joueur peut jouer ses cartes dans n'importe quel ordre ;
- le joueur peut passer volontairement ;
- les dégâts destinés à l'adversaire sont assignés en une seule fois pendant la phase d'attaque ;
- à la fin du tour, la main restante est défaussée puis le joueur pioche 5 cartes ;
- si la pioche est vide pendant une pioche, la défausse est mélangée pour reformer la pioche ;
- les cartes sont toujours piochées une par une, y compris lors d'une pioche multiple.

La formulation correcte est : chaque joueur possède 10 cartes dans son deck de départ. Après
mélange, il en pioche 5 pour former sa main initiale. La règle de pioche de 5 cartes s'applique
ensuite à la fin de chaque tour.

## Non-objectifs

Cette architecture ne couvre pas encore :

- les achats, la rivière et les coûts en ressources ;
- les cartes différentes ou les effets de cartes ;
- les champions et leurs capacités ;
- la maîtrise et les conditions de victoire associées ;
- les boucliers, soins, bannissements et autres règles avancées ;
- l'information cachée ;
- les parties de 3 ou 4 joueurs ;
- l'interface de debug, prévue à l'étape suivante ;
- l'entraînement RL et les réseaux de neurones.

## Décisions structurantes

### Moteur indépendant des joueurs

Le moteur possède la vérité sur l'état et les règles. Un joueur ne modifie jamais directement
l'état : il demande une action, puis le moteur la valide et l'applique.

Interface cible :

```python
observation = game.observation_for(player_id)
action = player.choose_action(observation)
game.apply(action)
```

La méthode `observation_for` peut retourner l'état complet en V0. Elle deviendra une observation
partielle lorsque l'information cachée sera ajoutée.

### État mutable contrôlé

L'état de simulation sera mutable en interne. Une partie RL exécute énormément de transitions et
une copie complète à chaque action créerait une pression mémoire et un coût d'allocation inutiles.

Avantages :

- moins d'allocations dans la boucle chaude ;
- meilleure vitesse pour les parties séquentielles ;
- modèle naturel pour déplacer des cartes entre zones ;
- intégration simple avec un environnement RL.

Inconvénients :

- risque d'effets de bord difficiles à suivre ;
- snapshots et recherche nécessitent une copie explicite ;
- tests de transitions moins purement fonctionnels.

Pour limiter ces risques, les mutations ne seront accessibles qu'au moteur et aux objets de jeu
internes. Les actions seront validées avant modification, et le moteur pourra exposer une méthode
`clone()` ou `snapshot()` pour le debug, les tests et les futurs algorithmes de recherche.

### Aléatoire injectable

Toutes les opérations aléatoires utiliseront un générateur fourni par la partie ou par sa
configuration. La seed devra reproduire au minimum :

- le tirage du premier joueur ;
- le mélange des decks ;
- le mélange d'une défausse lors d'une pioche ;
- les choix aléatoires des joueurs qui en ont besoin.

Le moteur ne devra pas utiliser directement l'état global de `random` dans ses règles.

### Phases explicites

Les phases seront représentées par un enum ou un type équivalent, même si certaines sont
temporairement sans action :

```text
PLAY → BUY (réservée, aucune action en V0) → ATTACK → CLEANUP → PLAY
```

La phase `BUY` sera présente comme point d'extension mais sera automatiquement traversée en V0.
Cela évite de réécrire la machine de jeu lorsque les achats seront ajoutés.

## Architecture proposée

### Package de jeu

Structure cible, à confirmer lors de l'implémentation :

```text
shards_ai/
└── game/
    ├── __init__.py
    ├── cards.py          # CardDefinition et CardInstance
    ├── enums.py          # PlayerId, Phase, zones et types simples
    ├── state.py          # GameState, PlayerState et zones de cartes
    ├── actions.py        # actions publiques et paramètres
    ├── rules.py          # validation et application des règles
    ├── game.py           # façade de partie et boucle de phases
    ├── random.py         # wrapper du générateur aléatoire injectable
    └── players.py        # protocole/interface de joueur, sans stratégie
tests/
└── game/
    ├── test_setup.py
    ├── test_draw.py
    ├── test_play_phase.py
    ├── test_attack_phase.py
    ├── test_cleanup.py
    └── test_reproducibility.py
```

Les noms pourront être adaptés aux conventions Python retenues, mais les responsabilités doivent
rester séparées.

### `CardDefinition` et `CardInstance`

Même si la V0 ne contient qu'une carte, distinguer :

- `CardDefinition` : description immuable d'une carte, réutilisable par plusieurs exemplaires ;
- `CardInstance` : exemplaire concret déplacé entre la main, la pioche, la défausse et la zone de
  jeu.

La définition V0 contiendra au minimum un identifiant stable, un nom et une valeur de dégât. Les
effets seront laissés derrière une extension future, plutôt que d'introduire dès maintenant un
interpréteur de cartes.

### Zones de cartes

Chaque `PlayerState` possède explicitement :

- `hand` ;
- `draw_pile` ;
- `discard_pile` ;
- `play_zone` pour les cartes jouées pendant le tour.

Les zones sont des collections ordonnées. La pioche doit permettre de retirer la carte du dessus
en O(1). La défausse doit pouvoir être mélangée puis transférée efficacement dans la pioche.

### `GameState`

Le state doit contenir au minimum :

- les deux `PlayerState` ;
- le joueur actif ;
- la phase courante ;
- le total de dégâts produits pendant la phase de jeu ;
- l'état de la partie (`RUNNING`, `FINISHED`) ;
- le gagnant ou l'identifiant du joueur éliminé ;
- le numéro du tour ;
- les informations nécessaires à la reproductibilité.

Les ressources inutiles en V0 ne doivent pas être ajoutées artificiellement. Le modèle devra
toutefois permettre d'ajouter des ressources sans changer l'interface des joueurs.

## Système d'actions

Les actions publiques sont des objets ou structures de données simples, sérialisables et validables.

Actions V0 proposées :

- `PlayCard(card_id)` : déplacer une carte de la main vers la zone de jeu et ajouter son dégât ;
- `PassPlayPhase()` : terminer volontairement la phase de jeu ;
- `AssignDamage(amount)` : assigner les dégâts accumulés à l'adversaire ;
- `ResolveCleanup()` ou transition interne équivalente : effectuer la défausse et la pioche ;
- `AdvancePhase()` : transition interne, non exposée aux joueurs si elle n'est pas nécessaire.

Le joueur ne devrait pas choisir librement `amount` en V0 : l'assignation doit normalement être
égale à tous les dégâts disponibles. L'action peut néanmoins être conçue pour accepter une
allocation explicite afin de rester compatible avec les futurs champions et plusieurs cibles.

Une action illégale doit produire une erreur métier explicite ou un résultat de validation
structuré. Elle ne doit jamais modifier partiellement l'état.

## Déroulement d'une partie

### Initialisation

1. Créer les deux joueurs à 50 PV.
2. Créer 10 instances de la carte V0 pour chaque joueur dans son deck de départ.
3. Mélanger chaque deck avec le générateur aléatoire de la partie.
4. Tirer 5 cartes de chaque deck pour former les mains initiales.
5. Tirer aléatoirement le premier joueur.
6. Démarrer en phase `PLAY` pour le premier joueur.

### Phase de jeu

Le joueur actif peut jouer zéro ou plusieurs cartes de sa main, dans n'importe quel ordre. Chaque
carte jouée rejoint sa zone de jeu et produit 1 dégât en réserve. Le joueur peut ensuite passer.

### Phase d'achat

La phase est traversée sans action en V0. Elle existe dans la machine de phases pour préparer une
future implémentation sans mélanger cette évolution avec la logique de combat.

### Phase d'attaque

Le joueur actif assigne tout son dégât en réserve à l'adversaire. Les PV adverses diminuent du
montant assigné. Si les PV atteignent 0, la partie se termine immédiatement et l'adversaire perd.

### Cleanup

1. Déplacer toutes les cartes de la zone de jeu vers la défausse.
2. Déplacer toutes les cartes restantes de la main vers la défausse.
3. Réinitialiser le dégât en réserve.
4. Piocher 5 cartes, une par une, en reconstituant la pioche depuis la défausse si nécessaire.
5. Passer le joueur actif à l'adversaire.
6. Revenir en phase `PLAY`, sauf si la partie est terminée.

## Pioche et mélange

La primitive `draw_one(player_id)` sera la seule opération faisant avancer une pioche. Elle devra :

1. reconstituer la pioche si elle est vide et que la défausse contient des cartes ;
2. tirer exactement une carte ;
3. la placer dans la main ;
4. retourner l'instance tirée.

La primitive `draw_many` appellera `draw_one` autant de fois que demandé. Ainsi, un deck vide au
milieu d'une pioche de 5 cartes est correctement géré.

Cas terminal à traiter explicitement : si la pioche et la défausse sont toutes deux vides alors
qu'une carte est demandée, la partie ne peut pas satisfaire la pioche. En V0 cela ne devrait pas
arriver avec les règles de défausse normales ; le moteur doit néanmoins lever une erreur métier
claire plutôt que boucler ou retourner une valeur invalide.

## Performance et RL futur

Le moteur ne dépendra pas de Gymnasium dans cette étape. Une couche d'adaptation ultérieure pourra
exposer `reset(seed)`, `step(action)`, `observation`, `reward`, `terminated` et `truncated`.

Cette séparation permet :

- de tester les règles sans dépendre du framework RL ;
- de lancer des parties directement en Python ;
- d'ajouter ensuite des wrappers d'observation, de récompense et de vectorisation ;
- de mesurer le moteur seul avant d'optimiser l'environnement RL.

Optimisations initiales attendues :

- collections simples en mémoire ;
- aucun I/O dans la boucle de jeu ;
- aucun log par défaut à chaque action ;
- identifiants entiers ou structures légères pour les instances ;
- copies uniquement sur demande ;
- benchmark séparé du moteur et des joueurs.

Il ne faut pas introduire prématurément de multiprocessing, de cache complexe ou de compilation
JIT. Ces options seront évaluées après un benchmark reproductible.

## Observabilité et replay

Le moteur doit pouvoir exposer :

- un état lisible pour les tests et le debug ;
- la liste des actions légales dans la phase courante ;
- un historique optionnel des actions et transitions ;
- une seed et une configuration de partie ;
- une méthode de snapshot ou clone pour reproduire une situation.

L'historique ne doit pas être activé ou sérialisé par défaut dans les simulations massives afin de
ne pas ralentir l'entraînement.

## Gestion des erreurs

Les erreurs métier à prévoir incluent notamment :

- jouer une carte absente de la main ;
- jouer une carte pendant une mauvaise phase ;
- agir lorsque ce n'est pas son tour ;
- assigner des dégâts avant la phase d'attaque ;
- assigner un montant différent des dégâts disponibles en V0 ;
- agir après la fin de la partie ;
- demander une carte alors qu'aucune zone ne peut en fournir.

Une erreur ne doit pas laisser l'état partiellement modifié.

## Stratégie de tests

Utiliser `pytest`, standard et largement adopté pour les projets Python et compatible avec les
futurs tests d'environnements RL.

Tests minimum :

- initialisation à 50 PV, avec un deck de départ de 10 cartes et une main initiale de 5 cartes ;
- tirage aléatoire du premier joueur avec seed ;
- jeu de zéro à dix cartes ;
- impossibilité de jouer une carte deux fois ;
- accumulation correcte des dégâts ;
- passage à la phase d'attaque ;
- assignation des dégâts et défaite à 0 PV ;
- déplacement des cartes jouées et non jouées vers la défausse ;
- pioche de 5 cartes ;
- reconstitution de la pioche depuis la défausse ;
- reconstitution pendant une pioche multiple ;
- impossibilité d'agir après la fin de partie ;
- deux parties identiques avec la même seed et les mêmes actions ;
- divergence contrôlée avec des seeds différentes.

Des tests de propriétés avec `Hypothesis` pourront être ajoutés après la première implémentation,
notamment pour vérifier qu'une carte appartient toujours à une seule zone.

## Fichiers attendus

### À créer pour le moteur

- `pyproject.toml` — configuration Poetry du projet et dépendances de développement ;
- `poetry.lock` — versions verrouillées des dépendances ;
- `shards_ai/game/` — moteur indépendant de l'IA ;
- `tests/game/` — tests unitaires et scénarios du moteur ;
- `benchmarks/` — benchmarks reproductibles, hors boucle de production.

### Documentation

- `doc/Architecture/001-architecture-moteur-duel-v0.md` — ce plan d'architecture ;
- `doc/feature-moteur-duel-v0.md` — état courant des fonctionnalités pendant l'implémentation ;
- `doc/regles-shards-of-infinity-base.md` — référence synthétique des règles officielles de la
  boîte de base, sans extensions.

## Questions ouvertes non bloquantes

- La forme exacte de la configuration de partie pourra évoluer lorsque les decks deviendront
  différents.
- Le format de sérialisation des snapshots sera choisi avec l'implémentation.
- La représentation numérique des observations RL sera définie à l'étape environnement RL.
- La sémantique Gymnasium sera ajoutée dans un adaptateur dédié, pas dans le cœur du moteur.

## Critères de réussite

Le step 1 est considéré comme terminé lorsque :

- une partie V0 complète peut être exécutée sans intervention manuelle ;
- toutes les règles V0 sont couvertes par des tests ;
- une partie peut être rejouée à l'identique avec sa seed et ses actions ;
- le moteur expose les actions légales et refuse les actions invalides ;
- un benchmark de référence est disponible ;
- aucune dépendance à une stratégie d'IA ou à une interface graphique n'existe.
