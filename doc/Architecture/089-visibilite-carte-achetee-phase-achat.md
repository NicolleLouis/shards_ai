# Visibilité d'une carte achetée en phase achat — Architecture

## Objective

Calculer, pour chaque carte que le joueur peut acheter avec `BuyCard` pendant la phase achat, une classe de
visibilité indiquant si elle sera probablement vue :

```text
0 fois avant la fin de la partie
1 fois avant la fin de la partie
plus d'une fois avant la fin de la partie
```

Le calcul combine deux informations indépendantes :

1. le forecast de l'horizon de partie, exprimé en classes `T0` à `T5` et `T6+` tours futurs
   du joueur actif ;
2. la vitesse effective du deck du joueur après l'achat, en tenant compte de la pioche restante,
   du mélange de la défausse et des effets de pioche.

Ce calcul est fourni par un service appelé uniquement pendant la génération des datasets de
training. Il ne sera pas appelé automatiquement par un joueur réel et ne modifie pas le choix du
joueur neural.

## Current State

Le forecast d'horizon de l'architecture 088 produit une distribution sur sept classes :

```text
T0, T1, T2, T3, T4, T5, T6+
```

Il fournit donc une classe d'horizon, pas un nombre exact de tours.

Le moteur représente séparément `draw_pile`, `discard_pile`, `hand`, `play_zone` et les cartes
possédées. Une carte achetée normalement rejoint la défausse. La pioche est renouvelée lorsque la
pioche courante est vide, en mélangeant la défausse.

Les cartes qui piochent accélèrent le parcours du deck. La convention métier retenue est :

```text
carte Draw 1 -> -1 carte effective
carte Draw 2 -> -2 cartes effectives
carte Draw 3 -> -3 cartes effectives
```

Ainsi :

```text
17 cartes sans draw       -> 17 effectives -> 4 tours à 5 cartes
17 cartes avec 2 Draw 1   -> 15 effectives -> 3 tours
17 cartes avec 1 Draw 2   -> 15 effectives -> 3 tours
```

## Target Behavior

À chaque phase achat, pour chaque carte candidate, le calcul produit une valeur immuable :

```python
CardVisibilityForecast(
    visibility_class,
    selected_horizon_class,
    effective_deck_size,
    full_deck_cycle_turns,
    remaining_draw_cycle_turns,
)
```

Le service reçoit la classe d'horizon sélectionnée par l'architecture 088. Si cette classe est
`T6+`, la visibilité est directement classée `multiple`.

La carte achetée est ajoutée virtuellement à la défausse avant le calcul. Le calcul ne modifie pas
l'état du moteur et ne lance pas réellement l'action d'achat.

## Non-Goals

- modifier les règles de pioche ou de mélange ;
- modifier `Game.legal_actions()` ou `Game.apply()` ;
- appliquer directement une pénalité au score neural ;
- considérer qu'une carte achetée est immédiatement visible ;
- utiliser des cartes ou des zones adverses cachées ;
- réduire tous les draws conditionnels à leur valeur maximale certaine ;
- appeler le service depuis un joueur réel ou depuis le hot path d'inférence ;
- pénaliser `RecruitMercenary`, dont la valeur est immédiate.

## Key Decisions

1. **Le calcul est candidat-spécifique.** La taille effective du deck et le forecast de partie sont
   communs à la phase achat, mais la carte achetée peut elle-même modifier le deck, notamment si
   elle possède un effet de pioche.
2. **La convention effective est additive.** Pour une carte connue et activable, `draw_amount`
   réduit la taille effective du deck du même montant. Une carte Draw 1 réduit donc de 1 et non de
   0.
3. **La carte achetée contribue immédiatement à la taille effective.** Son effet de pioche est
   inclus dans `effective_deck_size` dès le calcul, même si la carte n'a pas encore été vue. Cette
   approximation est volontaire.
4. **La carte achetée reste différée.** Elle est ajoutée au deck futur, mais n'est pas considérée
   comme déjà vue au moment de l'achat.
5. **Le forecast de partie est discrétisé.** Le service utilise la classe d'horizon sélectionnée
   par le forecast (`argmax`) et ne combine pas plusieurs classes.
6. **La cible opérationnelle est `0`, `1`, `>1`.** Le service renvoie une seule classe de
   visibilité déterministe pour chaque carte candidate.
7. **Un draw supplémentaire accélère la consommation du deck.** La taille effective est calculée
   avant le cycle avec la convention `effective_deck_size = deck_size - total_draw_amount`.
8. **Deux cycles sont distingués.** `remaining_draw_cycle_turns` mesure les tours nécessaires pour
   épuiser la pioche actuelle et remélanger la défausse. `full_deck_cycle_turns` mesure les tours
   nécessaires pour parcourir le deck effectif complet après ce mélange.
9. **Le calcul est déterministe et sans Monte Carlo.** Le nombre de tours restant est comparé au
   nombre de tours nécessaires pour parcourir le deck effectif, avec une base de cinq cartes par
   tour.
10. **Le calcul est limité au dataset.** Il ne doit pas être exécuté dans `MacroNeuralPlayer`,
   `NeuralPlayer` ou un autre joueur réel.

## Open Questions

1. **Draw conditionnel — décision prise.** Les draws conditionnels ne sont pas comptés. Seuls les
   draws inconditionnels et connus sont inclus, sans probabilité d'activation.
2. **Classe T6+ — décision prise.** `T6+` est directement classée `multiple` : une partie avec au
   moins six tours actifs restants est considérée comme permettant au moins deux apparitions de la
   carte candidate dans le modèle déterministe.
3. **Tours partiels — décision prise.** Le résultat est exprimé en tours du joueur actif. Les
   décisions pendant la phase achat utilisent uniquement le nombre de futurs débuts de tour du
   joueur actif ; les actions restantes du tour courant sont exclues.
4. **Achat versus recrutement — décision prise.** Le calcul s'applique aux actions `BuyCard` et
   ne s'applique pas à `RecruitMercenary`. Le recrutement mercenaire conserve sa valeur immédiate
   et n'est pas pénalisé par la visibilité différée.

## Proposed Architecture

### 1. Snapshot du deck après achat virtuel

À partir de l'observation active et d'une action candidate :

1. calculer les zones du joueur après la fin du tour courant de façon non mutante ;
2. ajouter virtuellement la carte achetée à la défausse ;
3. obtenir la taille totale du deck futur ;
4. inclure l'effet de pioche de la carte candidate dans le total des draws ;
5. obtenir le contenu agrégé des cartes qui pourront contribuer aux draws ;
6. calculer les compteurs de draw effectifs.

La structure intermédiaire pourrait être :

```python
EffectiveDeckSnapshot(
    draw_pile_size,
    discard_size_before_shuffle,
    owned_card_count,
    effective_deck_size,
    total_draw_amount,
    total_draw_amount_in_current_draw_pile,
)
```

La taille effective de base est :

```python
effective_deck_size = max(
    1,
    owned_card_count + purchased_card_delta - total_draw_amount,
)
```

`total_draw_amount` est la somme des amounts de pioche inconditionnels et connus. Les draws
conditionnels ne sont pas inclus.

### 2. Deux temps de cycle

Le calcul doit produire deux valeurs différentes.

#### Cycle restant de la pioche actuelle

La carte achetée est ajoutée à la défausse et ne peut pas être vue avant le prochain mélange. On
calcule donc :

```python
remaining_draw_effective_size = max(
    0,
    current_draw_pile_size - draws_in_current_draw_pile,
)
remaining_draw_cycle_turns = ceil(remaining_draw_effective_size / 5)
```

L'effet de pioche de la carte candidate n'est pas inclus dans ce premier chiffre : la carte est
encore dans la défausse et ne peut pas accélérer la pioche avant d'être vue.

#### Cycle du deck complet

Après le mélange, la carte candidate fait partie du deck complet et son éventuel draw peut donc
accélérer le cycle futur :

```python
full_deck_cycle_turns = ceil(effective_deck_size / 5)
```

La taille effective est calculée avec la carte candidate incluse.

### 3. Classification conditionnelle à un horizon

Pour un horizon `h` :

Pour chaque valeur de l'horizon `h`, appliquer la règle déterministe suivante :

```python
if h <= remaining_draw_cycle_turns:
    visibility_class = "zero"
elif h > remaining_draw_cycle_turns + 2 * full_deck_cycle_turns:
    visibility_class = "multiple"
else:
    visibility_class = "once"
```

La règle signifie :

```text
pioche actuelle non épuisée : 0 apparition
après le mélange et jusqu'à deux cycles du deck complet : 1 apparition
après plus de deux cycles du deck complet : au moins 2 apparitions
```

Il n'y a pas de Monte Carlo dans cette architecture.

### 4. Utilisation de la classe d'horizon

Le service reçoit la classe d'horizon sélectionnée par le modèle 088. Pour `T0` à `T5`, `h` est la
valeur de la classe. Pour `T6+`, le service renvoie directement `multiple` :

```python
visibility_class = "multiple" if selected_horizon_class == "T6+" else deterministic_visibility_class(snapshot, h)
```

Le résultat comprend au minimum :

```python
visibility_class  # "zero", "once" ou "multiple"
```

### 5. Point d'intégration phase achat

Le calcul est appelé par le générateur de dataset après la génération des actions légales et
uniquement pour les candidats :

```text
BuyCard       -> visibilité différée calculée
RecruitMercenary -> exclu du calcul
RecruitFreeCard  -> hors périmètre initial
```

Dans un premier temps, le résultat est journalisé avec la décision et le candidat. Le score neural
reste inchangé. Cela permet de vérifier les classes sur des parties complètes avant de créer une
architecture de pénalité.

## Data Model

Ajouter un module dédié, par exemple :

```text
shards_ai/ai/card_visibility.py
```

Les résultats de diagnostic peuvent être sérialisés dans un dataset expérimental :

```json
{
  "action_type": "buy_card",
  "card_definition_id": "...",
  "selected_horizon_class": "T2",
  "effective_deck_size": 15.0,
  "full_deck_cycle_turns": 3,
  "remaining_draw_cycle_turns": 1,
  "visibility_class": "once"
}
```

Le schéma doit contenir la seed, le `game_id`, le tour, l'action candidate, le profil/checkpoint
du joueur et la version du calculateur. Les artefacts vont sous `artifacts/`, jamais dans `doc/`.

## Performance And Validation

Le calcul analytique doit être constant ou linéaire dans le nombre de définitions de cartes
possédées, et ne pas cloner un jeu complet pour chaque action par défaut. Les informations communes
à toute la phase achat doivent être calculées une seule fois ; seules les variations de la carte
candidate doivent être recalculées.

Validation obligatoire :

- decks synthétiques de 5, 15 et 17 cartes ;
- 17 cartes avec deux Draw 1 -> 15 effectives ;
- 17 cartes avec un Draw 2 -> 15 effectives ;
- carte achetée en défausse avant mélange ;
- horizon T0, T1, T2, T5 et T6+ ;
- pioche actuelle de 0, 1, 2 et plusieurs tours avant remélange ;
- distinction entre `remaining_draw_cycle_turns` et `full_deck_cycle_turns` ;
- zéro, une et plusieurs apparitions ;
- comparaison de la classe calculée aux exemples synthétiques ;
- coût du service par candidat lors de la génération du dataset.

## Testing Strategy

- tester le calcul `effective_deck_size` et les valeurs Draw 1/2/3 ;
- tester la non-mutation de l'état du jeu ;
- tester la sélection de la classe d'horizon et la convention `T6+ -> multiple` ;
- vérifier que la classe produite appartient à `zero`, `once` ou `multiple` ;
- tester que la carte candidate contribue immédiatement à `effective_deck_size` ;
- tester les cartes avec draws conditionnels et inconditionnels ;
- tester les transitions de reshuffle aux bornes exactes ;
- vérifier que le service n'est appelé que par le générateur de dataset ;
- exécuter les tests du moteur, du forecast et du joueur neural sans modifier V006/V005.

## Rollout And Migration

Cette architecture ne modifie pas encore la politique neural ni les checkpoints actifs.

Phase 1 : implémenter le calcul analytique et ses tests synthétiques.

Phase 2 : l'exécuter en diagnostic pendant les phases achat lors de la génération d'un dataset.

Phase 3 : comparer les classes prédites aux apparitions réelles des cartes achetées dans le dataset.

Phase 4 : rédiger une architecture séparée pour convertir `zero`, `once` et `multiple` en pénalité
douce des actions `BuyCard` dans le scorer neural, sans modifier la valeur des recrutements
mercenaires.

## Files Expected To Change

- `shards_ai/ai/card_visibility.py` : calculateur et types de résultat ;
- `shards_ai/ai/horizon_forecast.py` : lecture du forecast de classes, sans réintroduire de
  régression ;
- `shards_ai/game/game.py` ou un helper de snapshot : uniquement si la simulation doit reproduire
  précisément les transitions de pioche ;
- `shards_ai/game/runner.py` ou un observateur de phase achat : collecte diagnostic, chemin exact à
  confirmer ;
- `tests/ai/test_card_visibility.py` : formules et classes ;
- `tests/game/test_card_visibility_integration.py` : validation contre le moteur ;
- `scripts/measure_card_visibility.py` : génération du dataset de visibilité et contrôle des classes ;
- `doc/Current state/Neural player.md` : uniquement après intégration effective au runtime.
