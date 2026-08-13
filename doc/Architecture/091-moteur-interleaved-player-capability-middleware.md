# Moteur intercalable et middleware de capacités joueur — Architecture

## Objective

Faire évoluer le moteur pour permettre l'intercalation des décisions de jeu et d'acquisition,
notamment :

```text
PLAY moderne -> RecruitMercenary -> PLAY moderne -> GainMastery -> achat -> fin de tour
```

Conserver en parallèle les joueurs existants, qui utilisent le cycle historique `PLAY -> BUY` et
un espace d'actions plus petit. Ces joueurs ne contournent pas les règles : ils reçoivent une vue
volontairement censurée des actions légales produites par le moteur complet.

Le succès se mesure par :

- un moteur unique, autoritaire et déterministe pour les règles complètes ;
- des joueurs historiques inchangés dans leur comportement observable ;
- des joueurs modernes capables de recevoir et choisir toutes les actions de leur espace ;
- des comparaisons reproductibles entre joueur simplifié et joueur moderne dans le même moteur.

## Current State

Le moteur est dans `shards_ai/game/game.py` et expose le contrat :

```text
Game.legal_actions() -> list[Action]
Game.apply(action) -> None
```

Le cycle interne est actuellement `PLAY -> BUY -> ATTACK -> CLEANUP -> PLAY`.

- `PassPlayPhase` change `PLAY` en `BUY`.
- `BuyCard` et `RecruitMercenary` ne sont légaux qu'en `BUY`.
- `GainMastery` n'est légal qu'en `PLAY`.
- `StopBuying` change `BUY` en `ATTACK`.
- les décisions de bannissement et de recrutement gratuit peuvent déjà suspendre `PLAY` ou `BUY`.

Les joueurs actuels sont :

- `RandomPlayer`, qui classe les actions selon `observation.phase` ;
- `HeuristicPlayer`, dont les priorités et le seuil d'achat dépendent de la phase ;
- `NeuralPlayer`, qui score les actions fournies par le moteur à partir de `NeuralObservation` ;
- `MacroNeuralPlayer`, qui délègue `PLAY` à `PlayTurnSolver` et traite les autres actions de
  manière atomique ;
- `HybridPlayer` / `ComposedPlayer`, qui routent acquisition, play et bannissement vers des
  politiques différentes.

Le `GameRunner` attend actuellement qu'un joueur retourne une action directement applicable à
`Game.apply()`. Il ne possède pas de notion d'action consommée uniquement par une vue joueur.

## Target Behavior

Le moteur moderne possède une phase principale unique, nommée `MAIN` dans son état interne. Les
actions suivantes peuvent être disponibles pendant cette phase, selon les préconditions du moteur :

- `PlayCard` ;
- `ActivateChampion` ;
- `GainMastery` ;
- `BuyCard` ;
- `RecruitMercenary` ;
- les décisions pendantes de bannissement, recrutement gratuit ou choix d'effet.

`BuyCard` et `RecruitMercenary` traversent tous deux l'ancien découpage `PLAY/BUY` : le joueur
moderne peut acheter une carte normale ou recruter un mercenaire avant ou après des actions de jeu,
si les autres préconditions sont satisfaites.

Une action explicite `EndMainPhase` termine la phase principale et passe à `ATTACK`. Elle remplace
la responsabilité de `StopBuying` dans le moteur moderne.

Le joueur moderne reçoit directement cette liste complète. Le joueur historique reçoit une vue
filtrée :

```text
view_mode = PLAY
  PlayCard / ActivateChampion / GainMastery / PassPlayPhase

view_mode = BUY
  BuyCard / RecruitMercenary / StopBuying
```

`PassPlayPhase` est une action de contrôle de vue pour le joueur historique. Le middleware la
consomme et passe son `view_mode` virtuel de `PLAY` à `BUY`; elle n'est pas envoyée au moteur.

Les actions de jeu réelles sont toujours envoyées au moteur :

- `PlayCard`, `ActivateChampion`, `GainMastery`, `BuyCard` et `RecruitMercenary` sont validées par
  `Game.apply()` ;
- `StopBuying` est traduite par le middleware en `EndMainPhase` ;
- `PassPlayPhase` n'entraîne aucune mutation de `GameState`, uniquement une mutation du contexte
  privé du middleware.

## Non-Goals

- Ne pas permettre à un joueur historique de bénéficier implicitement des nouvelles actions.
- Ne pas dupliquer les règles de coût, de résolution de cartes ou de transition dans le middleware.
- Ne pas conserver deux moteurs de jeu parallèles.
- Ne pas modifier les anciens checkpoints neural ou prétendre qu'ils ont appris les nouvelles
  séquences.
- Ne pas faire de `Game.apply()` une méthode permissive qui accepterait des actions filtrées ou
  invalides.
- Ne pas modifier le current state avant que l'implémentation et ses tests soient terminés.

## Key Decisions

1. **Le moteur moderne reste l'unique source de vérité.** Le middleware filtre une liste d'actions
   et maintient un contexte de présentation ; il ne modifie jamais directement `GameState`.

2. **La phase moderne est `MAIN`.** Les valeurs historiques `PLAY` et `BUY` restent des valeurs de
   compatibilité pour les observations et les politiques historiques, pas des états internes
   concurrents du moteur.

3. **La compatibilité est attachée au joueur.** Une partie peut opposer un joueur moderne à un
   joueur historique. Le moteur est identique pour les deux ; seul l'espace de décision visible
   diffère.

4. **Le filtre est explicite et versionné.** Un joueur est construit avec un profil de capacité,
   par exemple `legacy_play_buy_v1` ou `full_main_v1`. Il est interdit de déduire silencieusement
   la compatibilité du nom de la classe ou du checkpoint.

5. **Les actions de vue sont distinguées des actions de règles.** `PassPlayPhase` est consommée par
   le middleware legacy. `StopBuying` est traduite en action moteur `EndMainPhase`, afin que la fin
   du tour reste observable et validée par le moteur.

6. **Les deux formes d'acquisition sont intercalables.** Dans `MAIN`, `BuyCard` et
   `RecruitMercenary` sont traitées comme des actions d'acquisition disponibles indépendamment de
   l'ancien `view_mode`. Le profil legacy les expose uniquement lorsqu'il présente sa vue `BUY`.

7. **Les décisions middleware sont observables.** Les observateurs peuvent enregistrer l'action
   visible (`PassPlayPhase`) et l'action moteur éventuelle (`EndMainPhase`) sans les confondre dans
   les transitions de jeu.

8. **Les joueurs actuels sont adaptés, pas réécrits.** `RandomPlayer`, `HeuristicPlayer`,
   `NeuralPlayer`, `MacroNeuralPlayer` et `ComposedPlayer` sont tous exécutés derrière un
   adaptateur legacy dans la première migration. Un nouveau joueur peut utiliser directement le
   contrat moderne.

## Open Questions

- **Non bloquante :** le nom public final de `Phase.MAIN` et de `EndMainPhase`.
- **Non bloquante :** faut-il conserver une représentation historique `phase = play/buy` dans les
  datasets, ou ajouter aussi `view_mode` pour distinguer phase moteur et phase joueur ?
- **Non bloquante :** faut-il appeler `PassPlayPhase` une action legacy purement virtuelle, ou
  introduire un type générique `ChangeViewMode` interne au middleware ?

## Proposed Architecture

### Vue d'ensemble

```text
                         GameRunner
                              |
              full GameState + full legal actions
                              |
                    PlayerDecisionMiddleware
                    /                         \
       LegacyCapabilityProfile             FullCapabilityProfile
              |                                      |
     view_mode PLAY / BUY                    NeuralObservation moderne
     actions censurées                       actions complètes
              |                                      |
       joueur existant                         joueur moderne
              |                                      |
       visible action                    visible action / engine action
              |                                      |
              +------------ Action translation --------+
                              |
                         Game.apply()
                              |
                         moteur MAIN
```

### `PlayerDecisionMiddleware`

Le middleware reçoit :

- le `Game` autoritaire ;
- le joueur sous-jacent ;
- un `CapabilityProfile` immuable ;
- un `view_mode` privé au joueur, réinitialisé au début de son tour.

Il expose au runner une méthode de décision qui peut produire :

```python
MiddlewareDecision(
    visible_observation=...,
    visible_actions=...,
    chosen_visible_action=...,
    engine_action=Action | None,
    consumed=True | False,
)
```

Le runner boucle si `engine_action is None` : il ne demande pas à `Game.apply()` d'appliquer une
action de vue. Il applique une seule action moteur réelle par transition de jeu.

### Profil legacy

Le profil legacy transforme l'observation moderne en une observation compatible avec la politique
existante :

- `MAIN + view_mode PLAY` devient `Phase.PLAY` ;
- `MAIN + view_mode BUY` devient `Phase.BUY` ;
- `ATTACK` et les décisions pendantes restent exposés avec leur contrat historique ;
- les champs inconnus des anciennes architectures neural ne sont pas ajoutés.

Le filtre conserve les actions pendantes prioritaires. Il ne doit jamais supprimer l'unique choix
qui permet de résoudre un bannissement ou un recrutement gratuit.

### Profil moderne

Le profil moderne transmet :

- l'observation complète autorisée par le moteur ;
- les actions de la phase `MAIN` ;
- la représentation `phase = main` et, si nécessaire, un champ `decision_mode` séparant jeu,
  acquisition et résolution pendante.

Les nouveaux scoreurs et solveurs doivent recevoir les mêmes candidats que le moteur afin de
rester comparables et de ne pas apprendre des actions impossibles.

## Data Model

### État moteur

Modifier `GameState.phase` pour accepter `Phase.MAIN`, tout en conservant `ATTACK` et `CLEANUP`.
Les champs existants de `PlayerState`, notamment `gems`, `mastery_action_used`, les cartes jouées
et les décisions pendantes, restent la source de vérité.

Ajouter, si nécessaire, un champ de diagnostic non-règle dans le runner ou le middleware, mais ne
pas stocker `view_mode` dans `GameState` : deux joueurs d'une même partie peuvent avoir des vues
legacy différentes et ce contexte n'est pas une propriété des règles.

Ajouter `EndMainPhase` dans `shards_ai/game/actions.py`. Les actions historiques restent
disponibles pour les politiques legacy et ne doivent pas être réutilisées comme actions moteur
modernes par ambiguïté.

### Observation et représentation

Mettre à jour :

- `observation.py` pour l'observation moderne et l'observation legacy ;
- `action_representation.py` pour représenter `main` et les vues legacy ;
- les schémas de dataset afin d'enregistrer séparément `engine_phase`, `view_mode`,
  `visible_action_type` et `engine_action_type` lorsque le middleware est actif.

Les anciens datasets restent lisibles sans migration destructive. Les nouveaux datasets doivent
indiquer le `capability_profile_id` et le `ruleset_id`.

## Backend Flow

1. `GameRunner` demande au middleware les actions complètes via `Game.legal_actions()`.
2. Le middleware résout les décisions pendantes avant d'appliquer son filtre de mode.
3. Il construit l'observation compatible du joueur et transmet uniquement les actions visibles.
4. Le joueur choisit une action parmi cette liste.
5. Le middleware vérifie que l'action appartient à sa liste visible.
6. Il consomme `PassPlayPhase` localement, change `view_mode` et redemande une décision.
7. Il traduit `StopBuying` en `EndMainPhase`.
8. Pour toute action réelle, il vérifie son appartenance aux actions complètes courantes et la
   retourne au runner.
9. `GameRunner` appelle `Game.apply(engine_action)`.
10. Les observers reçoivent la décision visible et la transition moteur avec des champs distincts.

Toute divergence entre les actions visibles et les actions complètes doit produire une erreur
explicite, avec le profil de capacité, la phase moteur, le `view_mode` et les actions concernées.

## Frontend Flow

Sans objet pour le moteur actuel. Les rapports et scripts d'analyse devront néanmoins afficher,
quand le middleware est actif, le profil de capacité et distinguer :

- décisions visibles du joueur ;
- actions réellement appliquées ;
- actions de vue consommées.

## Authorization And Feature Gates

Sans autorisation utilisateur. Le profil de capacité est une configuration expérimentale du joueur,
pas une permission de contourner le moteur.

Le constructeur de joueur doit exiger un profil explicite. Aucun profil ne doit être sélectionné
implicitement selon un checkpoint ou une classe historique.

## Observability And Operations

Chaque décision doit pouvoir enregistrer :

- `ruleset_id` ;
- `capability_profile_id` ;
- `engine_phase` ;
- `view_mode` ;
- action visible ;
- action moteur appliquée, si différente ;
- `middleware_consumed` ;
- joueur et seed.

Les benchmarks doivent comparer les profils sur les mêmes seeds, adversaires et rôles. Les résultats
ne doivent pas mélanger une action virtuelle `PassPlayPhase` avec une action consommant une action
moteur dans le nombre de transitions de jeu.

## Edge Cases

- `PassPlayPhase` alors qu'aucune action d'achat n'est possible : le middleware expose quand même
  `StopBuying` pour préserver la progression.
- Un effet de bannissement devient pending après une action réelle : le filtre de mode est suspendu
  et toutes les actions de résolution nécessaires restent visibles.
- Un mercenaire recruté produit un effet nécessitant une décision : le middleware ne repasse pas
  automatiquement en mode `PLAY` ; il expose d'abord la décision pendante.
- Une action visible devient invalide entre la décision et l'application : le moteur reste
  autoritaire et l'erreur est remontée ; le middleware ne rejoue pas silencieusement une autre
  action.
- Un joueur legacy reçoit une action moderne qu'il ne connaît pas : elle doit être filtrée, jamais
  sérialisée vers une ancienne représentation inconnue.
- Le middleware ne doit pas boucler indéfiniment entre `PLAY` et `BUY`. Les actions de vue
  consommées doivent être comptées et une limite de décisions middleware doit être vérifiée.
- Une partie opposant un joueur moderne et un joueur legacy doit conserver un seul `GameState` et
  une seule séquence de règles ; seules les observations et actions visibles diffèrent.

## Testing Strategy

### Moteur

- actions `PlayCard`, `BuyCard`, `RecruitMercenary` et `GainMastery` dans toutes les combinaisons
  légales de `MAIN` ;
- fin de phase principale et passage vers `ATTACK` ;
- coût, maîtrise, limite une fois par tour et reset au cleanup ;
- recrutements qui déclenchent bannissement, choix ou effet immédiat ;
- déterminisme et absence de mutation partielle lors d'une action invalide.

### Middleware

- `PassPlayPhase` change uniquement le `view_mode` et n'appelle pas `Game.apply()` ;
- `StopBuying` devient exactement un `EndMainPhase` ;
- un joueur legacy voit le même contrat d'actions qu'avant dans les scénarios historiques ;
- un joueur moderne voit les nouvelles séquences ;
- aucune action filtrée ne peut être appliquée par le middleware ;
- les actions pendantes ne sont jamais masquées ;
- limite anti-boucle et diagnostic complet.

### Joueurs

- Random, heuristique, neural atomique, neural macro et composé derrière le profil legacy ;
- chaque joueur retourne toujours une action visible et applicable après traduction ;
- tests de non-régression des choix déterministes historiques ;
- joueur moderne minimal choisissant chaque famille d'action dans un scénario contrôlé ;
- parties legacy contre moderne avec seeds identiques et comparaison des trajectoires ;
- chargement des anciens checkpoints avec l'observation legacy, sans réentraînement implicite.

### Validation expérimentale

Les nouveaux modèles doivent être évalués sur des parties complètes et des slices explicites
`capability_profile × phase × action_type`. Une précision offline sur les nouvelles actions ne
constitue pas une preuve de qualité en partie.

## Rollout And Migration

1. Écrire et valider le contrat de `Phase.MAIN`, `EndMainPhase` et `MiddlewareDecision`.
2. Implémenter le moteur moderne avec des tests de règles, sans modifier encore les joueurs.
3. Implémenter `LegacyActionMiddleware` et vérifier la trajectoire historique complète.
4. Adapter `GameRunner` pour les décisions consommées et les observations legacy.
5. Envelopper Random, Heuristic, Neural, MacroNeural et Composed avec le profil legacy par défaut.
6. Ajouter un joueur moderne de test et les scénarios `recruit -> play -> mastery`.
7. Mettre à jour les scripts de génération de datasets et les rapports pour le profil de capacité.
8. Générer un dataset moderne séparé ; ne pas mélanger silencieusement les trajectoires legacy et
   modernes.
9. Entraîner et évaluer les nouveaux modèles avec le checkpoint mutable canonique existant, selon
   une nouvelle recette et un nouveau fingerprint.
10. Ne promouvoir aucun nouveau checkpoint avant validation sur le panel complet apparié.

Le rollback consiste à sélectionner le profil legacy et l'ancien chemin de génération de données ;
il ne nécessite pas de modifier ou supprimer les checkpoints historiques.

## Files Expected To Change

### Moteur

- `shards_ai/game/enums.py` — ajout de `Phase.MAIN`.
- `shards_ai/game/actions.py` — ajout de `EndMainPhase`.
- `shards_ai/game/game.py` — actions légales, transitions et validations de `MAIN`.
- `shards_ai/game/state.py` — uniquement si un champ moteur est réellement nécessaire.
- `shards_ai/game/observation.py` — observations modernes et legacy.
- `shards_ai/game/runner.py` — boucle middleware et décisions consommées.

### Middleware et joueurs

- `shards_ai/ai/player_middleware.py` — profil de capacité, vue, filtrage et traduction.
- `shards_ai/ai/action_representation.py` — phase moteur et représentation legacy.
- `shards_ai/ai/random_player.py` — adaptation minimale si le contrat legacy ne suffit pas.
- `shards_ai/ai/heuristic_player.py` — conservation du comportement legacy et support moderne
  ultérieur.
- `shards_ai/ai/neural_player.py` — construction d'observation selon le profil.
- `shards_ai/ai/macro_player.py` et `shards_ai/ai/play_turn_solver.py` — frontière `MAIN` et
  compatibilité des traces.
- `shards_ai/ai/composed_player.py` — routage selon actions visibles et profil.

### Tests et expérimentation

- `tests/game/test_game.py`, `tests/game/test_mercenaries.py`, `tests/game/test_random_player.py`.
- tests dédiés `tests/game/test_main_phase.py` et `tests/ai/test_player_middleware.py`.
- `tests/ai/test_action_representation.py`, `tests/ai/test_play_turn_solver.py` et tests des
  joueurs neural/composés.
- scripts de génération, benchmark et analyse sous `scripts/`, `benchmarks/` et `tests/analysis/`.

### Documentation après implémentation

- `doc/Current state/Game engine.md`.
- `doc/Current state/Random player.md`.
- `doc/Current state/Heuristic player.md`.
- `doc/Current state/Neural player.md`.
- `doc/Current state/Player profiles.md`.
