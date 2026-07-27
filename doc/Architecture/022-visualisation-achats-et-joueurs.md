# Extension de la visualisation des achats et des joueurs — Architecture

## Objective

Rendre le rapport d'une partie détaillée plus lisible lors des phases d'achat. Pour chaque décision
d'achat d'un joueur heuristique, l'utilisateur doit voir la rivière disponible, le score d'achat de
chaque carte achetable et la conclusion effectivement retenue par le moteur. Les tours doivent
également identifier visuellement le joueur actif.

## Current State

`scripts/analyze_game_detail.py` capture déjà les actions légales, l'action choisie, les scores
heuristiques et les états avant/après. Le HTML regroupe les actions par numéro de tour, mais tous
les tours ont actuellement la même apparence et les alternatives d'achat sont enfouies dans le
bloc générique des actions légales.

Le joueur heuristique sait calculer les features d'une action `BuyCard`, `RecruitMercenary` ou
`RecruitFreeCard` à partir de l'observation courante. La rivière contient jusqu'à six cartes, dont
certaines peuvent être vides ou non achetables dans le contexte courant.

## Target Behavior

Lorsqu'une action est prise en phase `BUY` par un `HeuristicPlayer`, l'événement JSON et le rapport
HTML ajoutent une section `purchase_analysis` contenant :

- les cartes présentes dans chaque slot de la rivière au moment de la décision ;
- le ou les types d'achat légaux pour chaque carte et leur score heuristique ;
- les features et contributions principales de chaque option scorée ;
- l'option choisie par le moteur, ou l'action de fin d'achat si aucune carte n'est retenue ;
- une conclusion courte indiquant l'action sélectionnée et son score.

Dans le HTML, cette section est visible directement dans la ligne de décision d'achat, avec un
tableau compact. Les détails de features restent repliables. Les blocs de tours utilisent une
classe CSS distincte pour `PLAYER_1` et `PLAYER_2`, avec une légende explicite.

## Non-Goals

- Modifier la sélection d'achat ou les scores du moteur.
- Recalculer une stratégie différente de celle utilisée par `HeuristicPlayer.choose_action()`.
- Ajouter un affichage permanent ou ralentir les parties sans observateur.
- Exposer les cartes privées de l'adversaire dans les résumés destinés à l'analyse d'un joueur.

## Key Decisions

1. **Source unique du score.** Les scores d'achat sont calculés avec `features_for_action()` et les
   poids du joueur courant, comme les scores déjà affichés pour les autres actions.
2. **Rivière complète.** Les six slots sont conservés dans l'ordre, y compris les slots vides, afin
   de rendre visible le contexte exact de la décision.
3. **Score et légalité séparés.** Chaque carte visible reçoit le score de son achat théorique, même
   si elle est actuellement trop chère. Une colonne `légale` indique séparément si l'action faisait
   partie de `legal_actions`.
4. **Conclusion séparée.** Le tableau distingue les scores des candidats et la décision finale,
   notamment quand le moteur choisit `StopBuying`.
5. **Couleur sémantique stable.** `PLAYER_1` utilise une teinte bleue et `PLAYER_2` une teinte
   violette ; le nom du joueur reste affiché pour ne pas dépendre de la couleur seule.
6. **Coût nul en mode normal.** Le nouveau calcul ne s'exécute que dans le script de trace, via les
   observers déjà optionnels. Aucun état supplémentaire ni sérialisation n'est ajouté sans observer.

## Open Questions

- **Non bloquante — actions non heuristiques :** elles affichent la rivière et les options légales,
  mais pas de score ; elles indiquent que la sélection est aléatoire ou indisponible.
- **Non bloquante — choix de carte gratuite :** si une action `RecruitFreeCard` existe, elle est
  présentée comme une option distincte de l'achat normal avec son propre score.

## Proposed Architecture

Ajouter une fonction dédiée au script, `_purchase_analysis(observation, player, legal_actions,
chosen, player_id)`, appelée uniquement depuis `on_decision` lorsqu'une décision est en phase
d'achat. Elle parcourt `observation.river`, construit les options théoriques `BuyCard` et,
pour les cartes mercenaires, `RecruitMercenary`. Les scores sont calculés même si les options sont
trop chères, puis comparés à `legal_actions` pour indiquer leur légalité.

Chaque entrée contient `river_slot`, l'identité de la carte, `available`, `legal_options`, les
scores/features des options et `selected`. Le champ `conclusion` contient l'action choisie et son
score lorsqu'il est calculable.

Le renderer HTML spécialisé produit un tableau `slot`, `carte`, `type d'achat`, `score`, `légale`,
`choisie`.
Les features et contributions sont repliées par ligne. Le bloc de tour reçoit `player-1` ou
`player-2`, tandis que chaque action conserve aussi sa classe de joueur.

## Data Model

Aucun changement persistant. Le JSON reçoit seulement des champs optionnels :

```text
events[].purchase_analysis = { river: [...], conclusion: {...} }
events[].player_id = "PLAYER_1" | "PLAYER_2"
```

Les anciennes traces sans `purchase_analysis` restent rendables.

## Backend Flow

1. Le decision observer reçoit l'observation avant application.
2. Si la phase est `BUY`, il construit l'analyse de rivière sans modifier l'observation.
3. Le runner applique l'action normalement.
4. Le transition observer complète l'événement avec l'état après.
5. Le JSON et le HTML sont écrits comme auparavant.

Les erreurs d'un score individuel sont enregistrées sur l'option concernée et n'interrompent pas la
partie ni l'écriture du reste de la trace.

## Frontend Flow

Le HTML autonome affiche une légende `Joueur 1` / `Joueur 2`, des blocs de tour colorés et un
tableau d'achat immédiatement visible. Les détails volumineux utilisent `<details>`.

## Observability And Operations

La console reste inchangée. Le JSON est la source de diagnostic détaillé et les sorties restent
dans `analysis_output/game_detail/`.

## Edge Cases

- rivière avec slots vides ;
- aucune carte achetable mais `StopBuying` légal ;
- achats mercenaires et achats long terme proposés pour une même carte ;
- joueur Random sans scores heuristiques ;
- action gratuite ou banissement présent pendant la phase d'achat.

## Testing Strategy

- tester la rivière avec six slots et slots vides ;
- vérifier qu'une option non légale n'est pas candidate scorée ;
- vérifier que la conclusion correspond à l'action choisie ;
- vérifier les classes CSS distinctes des deux joueurs ;
- vérifier la compatibilité d'un événement sans `purchase_analysis` ;
- exécuter la suite complète et un benchmark sans observer.

## Rollout And Migration

Aucun changement de comportement de jeu. L'extension est rétrocompatible avec les anciennes traces
et activée uniquement par `analyze_game_detail.py`.

## Files Expected To Change

- `scripts/analyze_game_detail.py` : capture et rendu des analyses d'achat et couleurs ;
- `tests/analysis/test_game_detail.py` : assertions sur la rivière, la conclusion et le HTML ;
- `doc/Current state/Analysis.md` : documentation du nouveau rapport.
