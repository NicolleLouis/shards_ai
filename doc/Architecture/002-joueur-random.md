# Joueur random et orchestrateur de parties — Architecture

## Objective

Ajouter un joueur artificiel indépendant du moteur, capable de sélectionner des actions
aléatoires valides, ainsi qu’un orchestrateur permettant d’exécuter automatiquement une partie
complète. L’exécution d’un grand nombre de parties sera une fonctionnalité ultérieure.

Cette première version prépare l’ajout des Gems, du Power et des achats de cartes. En V0, le
joueur sera réellement aléatoire uniquement sur l’ordre de jeu des cartes : il jouera toujours
toutes les cartes de sa main et assignera toujours tous les dégâts disponibles.

Critères de réussite :

- deux joueurs random peuvent terminer une partie V0 sans intervention humaine ;
- chaque joueur joue toutes les cartes de sa main, dans un ordre aléatoire ;
- chaque joueur assigne tous les dégâts disponibles ;
- les mêmes seeds et les mêmes configurations reproduisent les mêmes parties ;
- l’orchestrateur peut exécuter une partie complète sans dépendance à une interface graphique ;
- les stratégies restent remplaçables sans modifier le moteur.

## Current State

Le moteur expose déjà :

- `Game.observation_for(player_id)` ;
- `Game.legal_actions()` ;
- `Game.apply(action)` ;
- les actions `PlayCard`, `PassPlayPhase` et `AssignDamage` ;
- le protocole `Player.choose_action(observation)` dans `shards_ai/game/players.py` ;
- un générateur aléatoire injectable dans `shards_ai/game/random.py`.

Le protocole actuel ne transmet pas encore la liste des actions légales au joueur et aucun joueur
concret ni orchestrateur n’existe. Le benchmark actuel joue toujours la dernière carte de la main
et ne constitue pas un joueur random.

## Target Behavior

À chaque décision, l’orchestrateur :

1. identifie le joueur actif ;
2. demande au moteur une observation pour ce joueur ;
3. demande au moteur la liste des actions légales ;
4. transmet observation et actions légales au joueur concerné ;
5. applique l’action retournée par le moteur ;
6. répète jusqu’à la fin de la partie.

Le joueur ne reçoit jamais de référence au moteur et ne modifie jamais directement l’état.
L’action retournée reste soumise à la validation du moteur.

### Comportement V0

- `PLAY` : extraire toutes les actions `PlayCard`, les mélanger avec la source aléatoire, les
  retourner une par une, puis retourner `PassPlayPhase` ;
- `ATTACK` : retourner `AssignDamage` avec exactement le montant de dégâts disponibles ;
- phase inattendue : lever une erreur explicite plutôt que choisir une action arbitraire.

Le joueur ne doit donc jamais sélectionner `PassPlayPhase` avant d’avoir joué toutes les cartes,
même si cette action figure dans la liste légale fournie par le moteur.

### Extension achat

Lorsque les achats seront implémentés, les actions d’achat seront incluses dans la liste fournie
par le moteur. À chaque décision d’achat :

- avec une probabilité de 10 %, le joueur s’arrête prématurément, même si un achat reste possible ;
- sinon, il choisit aléatoirement parmi les actions d’achat légales ;
- il recommence jusqu’à choisir de s’arrêter ou jusqu’à ce qu’aucun achat ne soit légal.

Cette politique est une politique de référence, pas une règle du moteur. Le moteur reste seul
responsable de la légalité, des coûts et de la résolution des achats.

## Non-Goals

- ajouter les Gems, le Power, la rivière ou les achats dans cette tâche ;
- créer une stratégie heuristique ou une IA d’apprentissage ;
- donner au joueur accès à `Game`, `GameState` mutable ou aux méthodes privées du moteur ;
- introduire une observation partielle avant la gestion de l’information cachée ;
- ajouter un historique complet ou un système de replay ;
- optimiser prématurément les copies d’observation ou ajouter du multiprocessing ;
- modifier les règles V0 pour rendre le joueur random plus complexe.

## Key Decisions

1. **Contrat indépendant du moteur.** Le joueur reçoit une observation et une liste d’actions
   légales. Il ne connaît pas l’implémentation de `Game` et ne peut pas muter l’état.
2. **Le moteur reste l’autorité.** L’orchestrateur transmet les actions au moteur, qui les valide
   encore. Une action illégale ne doit jamais être acceptée par confiance dans le joueur.
3. **Aléatoire reproductible.** L’orchestrateur reçoit une seed racine de partie. Les choix random
   et les mélanges doivent être reproductibles avec cette configuration.
4. **Flux aléatoires dérivés.** La seed racine produit des flux indépendants et déterministes :
   un flux pour le moteur et un flux par joueur. La convention de dérivation est stable et utilise
   les identifiants `engine`, `player-1`, `player-2`, etc. Les tirages du moteur ne décalent donc
   pas la suite de décisions d’un joueur lorsqu’une nouvelle règle aléatoire est ajoutée.
5. **Orchestrateur sans règle.** Il coordonne observation, actions et transitions, mais ne décide
   pas de la légalité ni de la résolution des cartes.
6. **Exécution bornée.** L’orchestrateur doit compter les actions et permettre une limite maximale
   configurable afin de détecter une boucle ou une stratégie défectueuse au lieu de bloquer une
   partie.
7. **V0 déterministe dans ses obligations.** Le random porte sur l’ordre des cartes, pas sur le
   fait de jouer ou non une carte ni sur le montant des dégâts.
8. **Observation V0.** Le joueur reçoit un `GameState` détaché en V0. Une observation dédiée sera
   introduite lorsque l’information cachée deviendra une contrainte réelle.
9. **Nom de l’orchestrateur.** Le composant d’exécution d’une partie s’appelle `GameRunner`.

## Open Questions

Il ne reste pas de question bloquante pour l’implémentation de cette architecture.

## Proposed Architecture

### `Player` protocol

Faire évoluer le contrat vers une signature conceptuelle équivalente à :

```python
def choose_action(
    self,
    observation: GameState,
    legal_actions: Sequence[Action],
) -> Action:
    ...
```

La liste doit être considérée comme non mutable par le joueur. Le contrat devra préciser qu’un
joueur doit retourner une action de cette liste, tandis que le moteur conserve la validation finale.

### `RandomPlayer`

Responsabilités :

- conserver son identifiant et sa source aléatoire ;
- reconnaître la phase à partir de l’observation ;
- maintenir uniquement l’état décisionnel local nécessaire à l’ordre random du tour courant ;
- sélectionner une action selon la politique V0 ;
- ne jamais appliquer l’action lui-même.

Pour éviter de re-mélanger la liste à chaque appel et de sélectionner accidentellement `Pass`, le
joueur pourra préparer une file d’actions `PlayCard` pour le tour courant. Cette optimisation
reste locale et réinitialisable au changement de tour ou de phase. Si l’observation montre que la
main ou la phase ne correspond plus à cette file, le joueur doit la reconstruire.

### Orchestrateur

Responsabilités :

- associer chaque `PlayerId` à une implémentation de `Player` ;
- créer ou recevoir une partie ;
- fournir observation et actions légales au joueur actif ;
- appliquer l’action choisie ;
- compter les actions et interrompre proprement en cas de limite ;
- retourner le résultat de la partie.

`GameRunner` reçoit une seed racine et dérive les sources aléatoires du moteur et des joueurs avec
les labels documentés (`engine`, `player-1`, `player-2`, etc.). Il peut recevoir une partie déjà
créée pour les tests, mais la construction standard doit centraliser cette dérivation afin que la
reproductibilité ne dépende pas de l’appelant.

Il ne doit pas :

- accéder aux zones pour décider à la place du moteur ;
- contourner `legal_actions()` ou `Game.apply()` ;
- contenir de logique spécifique aux cartes ou aux achats.

Flux cible :

```text
Game + seed
   │
   ▼
Orchestrator ── observation + legal actions ──▶ Player
   ▲                                             │
   └────────────── action choisie ──────────────┘
                     │
                     ▼
                 Game.apply
```

## Data Model

Aucune table, migration ou dépendance externe n’est nécessaire.

Types à faire évoluer ou ajouter :

- `Player.choose_action(observation, legal_actions)` ;
- `RandomPlayer` dans un module d’IA ou de joueurs artificiels, selon l’organisation retenue ;
- un résultat de partie minimal pour `GameRunner`, potentiellement `GameState` final en première
  version ;
- une limite d’actions dans la configuration de `GameRunner` si nécessaire ;
- une capacité de dériver un `GameRandom` nommé depuis la seed racine.

Les actions restent les structures sérialisables existantes. Les futures actions `BuyCard` et
`StopBuying` devront être ajoutées par la fonctionnalité d’achat, pas anticipées artificiellement
dans cette tâche.

## Backend Flow

Il n’y a pas de backend réseau. Le flux mémoire est synchrone : un choix, une validation et une
transition à la fois.

L’orchestrateur doit traiter explicitement :

- partie déjà terminée ;
- joueur manquant pour le joueur actif ;
- liste d’actions légales vide alors que la partie est encore en cours ;
- action retournée absente des actions légales ;
- erreur métier remontée par `Game.apply()` ;
- dépassement de la limite maximale d’actions.

Les erreurs doivent conserver suffisamment de contexte pour identifier la seed, le joueur, la
phase et le numéro d’action.

## Frontend Flow

Aucun frontend n’est prévu dans cette tâche. L’orchestrateur doit toutefois exposer une API assez
simple pour être réutilisée plus tard par le mode debug.

## Authorization And Feature Gates

Sans objet. Les joueurs artificiels sont des composants locaux du processus de simulation.

## Observability And Operations

Par défaut, aucune sortie par action afin de préserver les performances des simulations futures.

Prévoir au minimum :

- le retour de la seed de partie ;
- le gagnant et le statut final ;
- le nombre de tours ou d’actions ;
- une option de trace ou de callback réservée au debug ;
- une erreur explicite en cas de blocage ou d’action illégale.

Le logging détaillé appartient au futur mode debug.

## Edge Cases

- main vide avant le passage : le joueur passe directement ;
- aucun dégât : assigner `AssignDamage(0)` si cette action est légale ;
- une seule action légale : la retourner sans tirage inutile ;
- partie terminée après une attaque : ne pas demander d’action supplémentaire ;
- actions légales incohérentes avec la phase : lever une erreur de stratégie ;
- action retournée non présente dans la liste : laisser l’orchestrateur ou le moteur la refuser ;
- seed identique : même ordre random et même résultat à configuration identique ;
- partie bloquée : remonter la seed et le contexte fautifs avec l’erreur.

## Testing Strategy

Ajouter des tests couvrant :

- le joueur random joue toutes les cartes présentes dans sa main ;
- toutes les cartes sont jouées une seule fois ;
- l’ordre varie avec des seeds différentes ;
- le joueur passe seulement après avoir épuisé les actions `PlayCard` ;
- le joueur assigne exactement tous les dégâts ;
- une action hors liste est détectée ;
- deux joueurs random terminent une partie V0 ;
- une seed identique reproduit l’ordre des décisions et le résultat ;
- la limite d’actions arrête une stratégie bloquée ;
- le résultat final d’une partie orchestrée est cohérent.

Les tests existants du moteur doivent continuer à passer. Le benchmark devra éventuellement être
étendu avec un scénario utilisant l’orchestrateur, en distinguant le débit du moteur seul et celui
du joueur random.

## Rollout And Migration

1. Faire évoluer le protocole `Player` et ses tests.
2. Implémenter `RandomPlayer` avec la politique V0.
3. Implémenter l’orchestrateur d’une partie.
4. Ajouter les tests de partie complète et de reproductibilité.
5. Ajouter un benchmark d’une partie pilotée par joueurs.
6. Ajouter ensuite les ressources, achats et décisions d’arrêt d’achat selon une architecture
   séparée ou une mise à jour dédiée.

La compatibilité avec l’ancien appel `choose_action(observation)` n’est pas nécessaire puisqu’il
n’existe aucun joueur concret dans le dépôt. Si un joueur externe apparaît avant la migration, il
faudra fournir un adaptateur temporaire plutôt que coupler le protocole au moteur.

## Files Expected To Change

- `doc/Architecture/002-joueur-random.md` — référence historique de cette décision.
- `doc/Current state/Game engine.md` — état vivant de l’implémentation.
- `agents.md` — convention architecture historique et current state.
- `shards_ai/game/players.py` — évolution du protocole, chemin probable.
- `shards_ai/ai/players.py` ou `shards_ai/game/random_player.py` — emplacement du joueur random,
  à confirmer selon la séparation moteur/IA.
- `shards_ai/game/runner.py` ou `shards_ai/simulation.py` — orchestrateur, chemin probable.
- `tests/game/test_random_player.py` — tests du joueur.
- `tests/game/test_runner.py` — tests de l’orchestrateur.
- `benchmarks/benchmark_game.py` — ajout éventuel d’un scénario orchestré.
