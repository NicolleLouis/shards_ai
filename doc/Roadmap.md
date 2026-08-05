# Roadmap macro

Cette roadmap décrit les grandes étapes de construction d'une IA capable de jouer à
**Shards of Infinity**. Chaque étape fera ensuite l'objet d'un document dédié dans ce vault.

L'ordre proposé suit une progression allant d'un moteur fiable et observable vers des agents
de plus en plus autonomes, puis vers la gestion de l'information cachée.

## 1. Moteur déterministe et rapide — ✅ DONE

Construire le moteur de jeu principal : état de partie, règles, tours, ressources, effets et
transitions d'état. Le moteur doit être déterministe, testable, reproductible et suffisamment
rapide pour exécuter de nombreuses parties pendant l'entraînement.

## 2. Mode debug avec un mini front — À FAIRE

Ajouter une interface minimale permettant d'observer et de piloter une partie humainement afin
d'analyser les comportements du moteur, de reproduire des situations et de diagnostiquer les
erreurs de règles ou de transitions.

## 3. Définition des actions et système d'action propre — ✅ DONE

Définir explicitement les actions possibles, leurs paramètres, leurs préconditions, leur
validation et leurs conséquences. Le système doit être utilisable à la fois par le front de
debug, les joueurs artificiels et le futur environnement RL.

## 4. Définition déclarative des cartes — ✅ DONE

Mettre en place une définition structurée et extensible des cartes afin de faciliter l'ajout des
cartes suivantes. Cette représentation devra également permettre, à terme, au réseau de neurones
de lire et d'exploiter les caractéristiques des cartes.

## 5. Construire un joueur aléatoire — ✅ DONE

Implémenter un joueur random valide, capable de sélectionner des actions légales. Il servira de
référence minimale pour tester le moteur et lancer les premières parties automatisées.

## 6. Construire un joueur heuristique simple — ✅ DONE

Implémenter une première stratégie heuristique déterministe ou contrôlable, fondée sur quelques
règles simples de priorité et d'évaluation des actions.

## 7. Construire un joueur heuristique pondéré — ✅ DONE

Étendre l'heuristique avec des critères pondérés, puis rechercher ou entraîner des pondérations
viables à partir de parties simulées et d'une fonction d'évaluation reproductible.

## 8. Première roadmap neural — imitation de l'heuristique — ✅ DONE

Cette phase a produit un joueur neural capable de prendre une décision à chaque action atomique et
de reproduire le classement des actions produit par les profils heuristiques validés.

### 8.a. Vérifier la couverture des décisions atomiques — ✅ DONE

Vérifier que toutes les décisions exposées par le moteur passent par `Game.legal_actions()` et
`Game.apply()` : cartes, champions, achats, mercenaires, bannissements, choix en attente, attaque,
passage de phase et `GainMastery` (paiement d'une gemme pour gagner une maîtrise).

### 8.b. Définir l'observation partielle et le masque d'information — ✅ DONE

Créer une observation dédiée au joueur neural, toujours normalisée du point de vue du joueur actif.
Elle doit exposer :

- la main, la pioche, la défausse, la zone de jeu et les champions actifs ;
- la rivière et les cartes restantes du deck central ;
- les cartes possédées par l'adversaire sous forme de quantités globales, sans ordre ni séparation
  main/pioche ;
- la défausse et les champions adverses ;
- la santé et la maîtrise des deux joueurs ;
- les gemmes et la puissance du joueur actif ;
- les quatre indicateurs de factions déjà jouées pendant le tour courant.

Les cartes des tours précédents ne doivent pas être exposées comme historique. L'état interne peut
conserver `played_card_ids_this_turn` pour les règles exactes ; l'observation neural utilise seulement
un masque compact des factions présentes ce tour.

Ajouter des tests garantissant qu'aucune carte de la main ou de la pioche adverse n'est identifiable
dans l'observation.

### 8.c. Définir la représentation des cartes — ✅ DONE

Construire un encodeur commun aux cartes ordinaires et aux champions. Il combinera :

- l'identifiant stable de la définition de carte, nécessaire aux effets ciblant une carte précise ;
- faction, coût, puissance, maîtrise et autres attributs numériques ;
- type de carte et rôle de champion ;
- effets, conditions et paramètres structurés.

L'identifiant ne doit pas être la seule information utilisée : les attributs structurés doivent
permettre une généralisation partielle lorsqu'une carte est ajoutée ou modifiée. Un réentraînement
minimal sera prévu dans ce cas.

### 8.d. Définir la représentation des actions — ✅ DONE

Le modèle doit scorer chaque action légale avec une fonction de la forme `score(observation, action)`.
L'encodage d'une action contiendra son type, sa phase, sa cible, sa carte ou son champion ciblé,
son slot de rivière, son montant éventuel et son identifiant de choix intermédiaire.

### 8.e. Générer le dataset d'imitation — ✅ DONE

Produire des exemples contenant l'observation masquée avant décision, toutes les actions légales,
le score heuristique de chaque action, l'action choisie, le profil utilisé, la seed et le résultat
final de la partie.

Le dataset mélangera :

- les profils heuristiques validés `v007` et `v008` ;
- des parties heuristique contre RandomPlayer ;
- des parties entre profils heuristiques différents.

Les scores bruts seront conservés. Les cibles d'entraînement pourront comparer plusieurs approches :
régression de score, normalisation par décision et préférence paire-à-paire.

### 8.f. Implémenter et entraîner le premier réseau — ✅ DONE

Créer un modèle capable de scorer une liste variable d'actions légales. Il ne doit pas dépendre d'un
ensemble statique d'actions. L'inférence choisira une action parmi celles ayant le meilleur score ;
les égalités seront départagées aléatoirement avec une source contrôlée par seed.

### 8.g. Intégrer le joueur neural et mesurer ses performances — ✅ DONE

Ajouter un joueur neural compatible avec l'interface existante des joueurs, puis mesurer :

- imitation de l'action et du classement heuristique ;
- vitesse de décision ;
- victoire contre RandomPlayer ;
- victoire contre le profil heuristique de référence `v008` ;
- comparaison avec les benchmarks reproductibles existants.

Une validation courte devra précéder les entraînements longs. Aucun profil ou modèle candidat ne
devra devenir la référence active sans validation explicite.

### Décisions validées pour cette phase

- Le réseau est appelé à chaque décision atomique, jamais une seule fois au début du tour.
- Le modèle score chaque action légale avec `score(observation, action)` et choisit le meilleur score.
- L'apprentissage initial est de l'imitation de l'heuristique ; le reinforcement learning est
  maintenant suivi à l'étape 10 et le self-play reste ultérieur.
- Le dataset contient toutes les actions légales et leurs scores heuristiques, pas uniquement
  l'action choisie.
- Les données mélangent les profils heuristiques validés (`v007` et `v008`) ainsi
  que des parties contre RandomPlayer et entre profils heuristiques.
- L'objectif initial privilégie le classement relatif des actions. Les scores bruts sont conservés
  pour permettre plusieurs fonctions de perte et normalisations.
- Les égalités du réseau sont départagées aléatoirement avec une source contrôlée par seed.
- L'observation est toujours exprimée du point de vue du joueur actif.
- La main et la pioche adverses ne sont jamais exposées individuellement. Le réseau reçoit la
  composition globale des cartes adverses non bannies, sans ordre ni séparation main/pioche, ainsi
  que la défausse et les champions adverses visibles.
- Les zones propres sont représentées par quantités ou cartes visibles ; l'historique des cartes
  jouées est limité au tour courant.
- Pour les effets du tour courant, l'observation utilise quatre booléens de factions déjà jouées.
  Le moteur peut conserver les identifiants exacts nécessaires à ses règles.
- L'identité de carte est conservée dans l'entrée du modèle pour les relations précises comme
  Drakonarius, mais elle est combinée avec les attributs et effets structurés afin de généraliser
  partiellement aux nouvelles cartes.
- Les cartes ordinaires et les champions utilisent le même encodeur de carte, avec le contexte
  d'action et les attributs de champion pour les distinguer.
- `GainMastery` est une action atomique distincte : payer une gemme pour gagner une maîtrise.
- L'observation destinée au réseau est une représentation masquée dédiée ; `GameState` complet ne
  doit pas être transmis directement au modèle.

## 9. Transformer le moteur en environnement RL — ✅ DONE

Le moteur fournit les transitions, récompenses terminales, états terminaux, resets, seeds et
interfaces nécessaires au training PPO, en réutilisant l'observation et l'interface d'action de la
phase d'imitation. La collecte RL peut être parallélisée, tandis que l'update PPO reste séquentiel.

## 10. Premier véritable agent RL avec PPO — 🔄 EN COURS

Le candidat v002 est entraîné avec PPO contre Random, v007 et v008, avec récompense terminale
victoire/défaite, régularisation KL vers v001 et sélection gloutonne périodique sans régression par
adversaire. Le checkpoint de travail est unique et les validations larges doivent précéder toute
promotion. Les prochaines expériences sont suivies dans `doc/Ideas.md`.

## 11. Ajouter le self-play — ⏳ À VENIR

Faire évoluer l'entraînement vers le self-play, avec une gestion des versions ou des snapshots
d'agents adverses afin d'éviter les régressions et les cycles de stratégies trop étroits.

## Dépendances macro

```text
1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11
```

Les étapes 2 à 7 peuvent être menées avec un certain recouvrement une fois les fondations du
moteur et du système d'actions stabilisées. L'étape 8 dépend du joueur heuristique, d'une interface
d'action complète et d'une observation partielle testée. Les étapes 10 et 11 dépendent de la
validation du joueur neural et de checkpoints reproductibles.

## Documents détaillés à venir

Un fichier Markdown dédié sera créé pour chacune des étapes lorsque son périmètre devra être
conçu ou implémenté en détail.
