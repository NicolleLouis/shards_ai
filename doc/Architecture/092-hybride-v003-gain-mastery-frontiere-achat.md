# Joueur hybride V003 avec conversion GainMastery à la frontière d’achat

## Objective

Créer le profil hybride `hybrid-v003`, fonctionnellement identique à `hybrid-v001` sauf sur
un cas précis : lorsque la politique neural d’acquisition choisit `StopBuying`, le joueur
doit d’abord jouer `GainMastery` si cette action est encore légale et qu’il possède au moins
une Gem.

Le comportement attendu est :

```text
achats neural
    -> StopBuying choisi par le neural BUY
    -> GainMastery si encore légal et Gems >= 1
    -> StopBuying choisi par le neural BUY
    -> fin de la phase principale
```

Le changement concerne l’orchestration d’une vue legacy PLAY/BUY au-dessus du moteur moderne
intercalable. Il ne modifie ni la règle de `GainMastery`, ni le scoreur neural, ni le comportement
du profil `hybrid-v001`.

## Current State

Le moteur expose actuellement, en mode moderne, les actions de la phase principale dans un espace
unique : `PlayCard`, `ActivateChampion`, `GainMastery`, `BuyCard`, `RecruitMercenary` et
`EndMainPhase`. `Game.legal_actions()` est la source de vérité de la légalité et
`Game.apply()` revalide les préconditions avant mutation.

Les joueurs historiques sont enveloppés par `LegacyActionMiddleware` :

- la vue virtuelle `PLAY` expose les actions de jeu et remplace `EndMainPhase` par
  `PassPlayPhase` ;
- `PassPlayPhase` est une décision virtuelle qui bascule la vue vers `BUY` sans appeler le moteur ;
- la vue virtuelle `BUY` expose les achats et remplace `EndMainPhase` par `StopBuying` ;
- `StopBuying` est actuellement traduit directement en `EndMainPhase` ;
- le `HybridPlayer` route les actions d’acquisition vers `NeuralAcquisitionPolicy` et les actions
  de jeu vers la politique heuristique ou algorithmique choisie.

`hybrid-v001` utilise :

- acquisition neural `neural_v006` ;
- jeu `heuristic_v008` ;
- bannissement déterministe `deterministic_blaster_crystal`.

Le profil V3 doit reprendre exactement cette composition. Le dépôt contient déjà des modifications
non commitées liées au moteur intercalable et au middleware ; l’implémentation devra les préserver.

## Target Behavior

V3 active une capacité supplémentaire uniquement dans le middleware legacy de ce joueur.

Lorsque `translate()` reçoit `StopBuying` :

1. vérifier que la vue virtuelle est `BUY` ;
2. consulter `game.legal_actions()` ;
3. si `GainMastery()` est présent dans cette liste et que le joueur actif possède au moins une
   Gem, retourner `GainMastery()` comme action moteur réelle ;
4. conserver la vue virtuelle `BUY` ;
5. au prochain choix, laisser la politique neural choisir à nouveau parmi les actions BUY ;
6. traduire le `StopBuying` suivant en `EndMainPhase` comme aujourd’hui.

Lorsque la condition n’est pas remplie, V3 doit suivre exactement le chemin actuel de V1 :
`StopBuying` devient `EndMainPhase` et la vue est réinitialisée pour le tour suivant.

### Schéma d’architecture

```mermaid
flowchart TD
    R[GameRunner] --> M[LegacyActionMiddleware]
    M -->|vue BUY : BuyCard / RecruitMercenary / StopBuying| H[HybridPlayer]
    H --> N[NeuralAcquisitionPolicy]
    N --> S[Neural V006]

    S -->|StopBuying| M
    M --> L{GainMastery dans
    Game.legal_actions() ?}
    L -->|non| E[EndMainPhase]
    L -->|oui + Gems >= 1| G[GainMastery]
    G --> A[Game.apply(GainMastery)]
    A --> B[Vue BUY conservée]
    B --> H
    E --> X[Game.apply(EndMainPhase)]

    H --> P[Heuristic V008 / Play policy]
    P -.->|inchangée| M
    M --> V[Game.legal_actions()]
    V --> M
```

### Séquence nominale

```text
GameRunner
  -> middleware.view_mode = BUY
  -> HybridPlayer choisit StopBuying
  -> middleware vérifie Game.legal_actions()
  -> middleware retourne GainMastery
  -> Game.apply(GainMastery)
  -> middleware.view_mode reste BUY
  -> HybridPlayer choisit StopBuying
  -> middleware retourne EndMainPhase
  -> Game.apply(EndMainPhase)
```

`GainMastery` est une action moteur réelle : elle consomme une action de jeu, dépense une Gem,
augmente la maîtrise et marque l’action comme utilisée pour le tour. Seul le changement de vue
est virtuel ; aucune action moteur ne doit être simulée ou appliquée deux fois.

## Non-Goals

- modifier la règle ou le coût de `GainMastery` ;
- faire choisir `GainMastery` par le scoreur neural lorsqu’il choisit `StopBuying` ;
- modifier le checkpoint `neural_v006` ;
- modifier `hybrid-v001` ou `hybrid-v002` ;
- permettre au joueur V3 de jouer des cartes supplémentaires après `StopBuying` ;
- changer la politique PLAY heuristique ou algorithmique ;
- modifier la représentation du moteur, le dataset ou l’entraînement neural ;
- créer une nouvelle action moteur spécifique à V3.

## Key Decisions

### 1. La capacité est déclarative dans le profil hybride

Le profil V3 déclare une capacité, par exemple `boundary_gain_mastery_v1`. Le middleware ne doit
pas tester directement `profile_id == "hybrid-v003"`. Cette séparation permet de réutiliser la
capacité dans une future composition sans coupler la logique à un numéro de version.

### 2. Le middleware est propriétaire de la compatibilité legacy

La transformation `StopBuying -> GainMastery` appartient à `LegacyActionMiddleware`, car
`StopBuying` est une action de la vue virtuelle BUY et non une action moderne produite par le
moteur. Le moteur reste indépendant des profils de joueurs.

### 3. La légalité vient exclusivement du moteur

Le middleware ne doit pas reconstruire les préconditions à partir de `gems`, `mastery` ou
`mastery_action_used`. Il doit vérifier la présence de `GainMastery()` dans la liste retournée par
`Game.legal_actions()`. Le contrôle explicite `gems >= 1` documente l’intention métier, tandis que
la présence dans la liste reste le garde-fou complet contre les évolutions de règles.

### 4. La vue BUY est conservée après GainMastery

Après l’application de `GainMastery`, le joueur neural d’acquisition doit recevoir à nouveau la
vue BUY. Il ne faut ni passer automatiquement en vue PLAY, ni appeler la politique PLAY. Cela
évite que V3 ne devienne accidentellement une politique intercalée complète.

### 5. Une seule conversion par frontière

La conversion ne s’applique qu’au `StopBuying` choisi par le joueur. Après `GainMastery`,
`Game.legal_actions()` ne doit normalement plus contenir `GainMastery` pour ce tour ; le prochain
`StopBuying` suit donc le chemin normal vers `EndMainPhase`. Une protection contre une répétition
inattendue doit reposer sur la légalité moteur, pas sur un compteur parallèle.

## Open Questions

- Non bloquante : faut-il exposer dans `DecisionDiagnostic` que `StopBuying` a été consommé comme
  décision virtuelle et que l’action moteur appliquée a été `GainMastery` ? La recommandation est
  oui, car les benchmarks doivent distinguer décision visible et action appliquée.
- Non bloquante : faut-il nommer la capacité `boundary_gain_mastery_v1` ou
  `stop_buy_gain_mastery_v1` ? Le nom doit décrire le contrat, pas la version du profil ;
  `boundary_gain_mastery_v1` est recommandé.

## Proposed Architecture

### Profil hybride

Ajouter `configs/hybrid_profiles/hybrid-v003.yaml` avec `hybrid-v001` comme parent et les mêmes
politiques d’acquisition, PLAY et bannissement. Ajouter uniquement une métadonnée ou un champ
typé de capacité au contrat `HybridProfile`.

La capacité doit être validée au chargement du profil afin d’échouer rapidement en cas de valeur
inconnue. Les profils existants doivent conserver une valeur désactivée par défaut.

### Middleware

Ajouter à `LegacyActionMiddleware` une configuration immuable de capacité, initialisée par le
constructeur. Le middleware applique la règle dans `translate(StopBuying)` après la validation de
la vue BUY et avant la traduction vers `EndMainPhase`.

Pseudo-flux :

```python
if isinstance(action, StopBuying):
    require(view_mode is Phase.BUY)
    if boundary_gain_mastery and GainMastery() in game.legal_actions():
        return GainMastery()
    view_mode = Phase.PLAY
    return EndMainPhase()
```

Le test sur la Gem peut être explicite pour rendre le contrat lisible, mais ne doit pas remplacer
la vérification de `legal_actions()`.

### Construction du joueur

`GameRunner` doit transmettre la capacité du joueur au middleware, ou utiliser une propriété
stable du joueur construite depuis `HybridProfile`. La construction ne doit pas modifier la
décision du `HybridPlayer.choose_action()` : le joueur continue à retourner `StopBuying`, et le
middleware traduit cette décision selon la capacité active.

Cette frontière conserve :

- la responsabilité du neural : choisir parmi les actions visibles BUY ;
- la responsabilité du middleware : adapter le contrat legacy ;
- la responsabilité du moteur : définir et appliquer les actions légales.

## Data Model

Aucune modification de `GameState`, `PlayerState` ou des actions moteur n’est nécessaire.

Les changements de configuration sont :

- un nouveau fichier `configs/hybrid_profiles/hybrid-v003.yaml` ;
- éventuellement un champ `capabilities` ou `capability_profile_id` dans `HybridProfile` ;
- un identifiant stable de capacité désactivé pour V1 et V2.

La capacité ne doit pas être persistée dans l’état d’une partie : elle appartient au profil du
joueur et peut différer entre deux joueurs utilisant le même `GameState`.

## Backend Flow

Il n’y a ni route, ni base de données, ni job asynchrone.

Le flux synchrone est :

1. `GameRunner` demande une décision au middleware ;
2. le middleware expose la vue BUY ;
3. `HybridPlayer` délègue au neural ;
4. le middleware traduit la décision visible ;
5. `GameRunner` applique l’action moteur retournée ;
6. l’action suivante est recalculée à partir de l’état réel.

En cas de configuration inconnue, le chargement du profil doit échouer. En cas d’action illégale,
le moteur et le middleware doivent conserver leurs erreurs actuelles ; aucune substitution
silencieuse d’une action différente ne doit être ajoutée.

## Frontend Flow

Sans objet. Les rapports et observateurs de partie sont toutefois concernés par la distinction
entre `StopBuying` visible et `GainMastery` réellement appliquée.

## Authorization And Feature Gates

Sans autorisation utilisateur. La capacité est un contrat expérimental de profil de joueur.

Le garde-fou de déploiement est le choix explicite de `hybrid-v003` par le benchmark ou le
constructeur. Aucun checkpoint ou profil actif ne doit sélectionner V3 implicitement.

## Observability And Operations

Les diagnostics devraient distinguer :

- l’action visible sélectionnée : `StopBuying` ;
- l’action moteur appliquée : `GainMastery` ;
- la raison : `boundary_gain_mastery` ;
- le nombre de décisions virtuelles et d’actions moteur ;
- le profil hybride et l’identifiant de capacité.

Les métriques de qualité doivent compter `GainMastery` comme une action moteur réellement jouée,
et non comme un second `StopBuying`. Les comparaisons V3/V1 doivent utiliser les mêmes seeds,
adversaires, rôles et checkpoints.

## Edge Cases

- `GainMastery` absent de `Game.legal_actions()` : traduire directement `StopBuying` en
  `EndMainPhase`.
- aucune Gem : ne pas convertir, même si une implémentation future expose incorrectement l’action ;
  la légalité moteur reste obligatoire.
- `GainMastery` déjà utilisée ce tour : ne pas convertir.
- maîtrise au plafond : ne pas convertir.
- action pendante de bannissement ou de décision : le middleware continue de donner priorité aux
  actions pendantes, comme actuellement.
- action `StopBuying` reçue en vue PLAY : conserver l’erreur actuelle.
- profil V1/V2 : capacité désactivée, trajectoire inchangée.
- après `GainMastery`, perte inattendue de la Gem ou mutation externe : `Game.apply()` reste
  autoritaire et l’erreur doit être remontée.
- répétition ou boucle de décisions virtuelles : respecter la limite existante de
  `GameRunner`; V3 ne doit introduire aucune boucle automatique non bornée.

## Testing Strategy

Ajouter des tests unitaires ciblés du middleware et du profil :

- chargement de `hybrid-v003` et parent `hybrid-v001` ;
- capacité absente ou désactivée pour V1 ;
- V3 convertit `StopBuying` en `GainMastery` lorsque l’action est légale ;
- V3 conserve `view_mode == BUY` après la conversion ;
- le second `StopBuying` devient `EndMainPhase` ;
- aucun appel moteur n’est effectué pour une décision virtuelle ;
- aucune conversion quand il n’y a plus de Gem, quand `GainMastery` est déjà utilisée ou quand
  l’action n’est pas légale ;
- une partie courte V3 applique effectivement la séquence
  `StopBuying visible -> GainMastery moteur -> StopBuying visible -> EndMainPhase` ;
- V1 conserve sa séquence historique sur le même scénario.

Les tests existants du moteur sur le coût, le plafond, la limite par tour et la légalité de
`GainMastery` doivent rester inchangés. Une validation de benchmark séparée devra comparer V3 à V1
sur des parties complètes ; les tests unitaires ne constituent pas une preuve de qualité de jeu.

## Rollout And Migration

1. Ajouter l’architecture et valider le contrat de capacité.
2. Ajouter la capacité déclarative et le profil V3.
3. Ajouter les tests unitaires et le scénario de séquence.
4. Implémenter la traduction dans le middleware.
5. Exécuter les tests ciblés puis la suite complète.
6. Exécuter un benchmark apparié V3 contre V1 sur les mêmes seeds.
7. Mettre à jour les current states après validation du comportement implémenté.

Le rollback consiste à sélectionner `hybrid-v001` ou `hybrid-v002`. Il ne nécessite ni migration
de données, ni modification du checkpoint neural, ni suppression de V3.

## Files Expected To Change

- `doc/Architecture/092-hybride-v003-gain-mastery-frontiere-achat.md` — ce document.
- `configs/hybrid_profiles/hybrid-v003.yaml` — nouveau profil.
- `shards_ai/ai/hybrid_profiles.py` — chargement et validation de la capacité, si le champ est
  ajouté au modèle typé.
- `shards_ai/ai/player_middleware.py` — traduction conditionnelle de `StopBuying`.
- `shards_ai/game/runner.py` — transmission de la capacité au middleware, seulement si nécessaire
  après inspection du constructeur actuel.
- `tests/ai/test_player_middleware.py` — tests de conversion et de conservation de la vue BUY.
- `tests/ai/test_composed_player.py` — chargement et contrat de V3.
- `tests/game/test_runner.py` — scénario de partie complète, si la couverture actuelle ne permet
  pas de vérifier la séquence.
- `doc/Current state/Player profiles.md` — mise à jour après implémentation validée.
- `doc/Current state/Game engine.md` — uniquement si le comportement observable du middleware ou
  la description de l’action appliquée doit être précisé.

Les fichiers `shards_ai/game/game.py`, `shards_ai/game/actions.py` et le checkpoint neural ne
devraient pas être modifiés pour cette évolution.
