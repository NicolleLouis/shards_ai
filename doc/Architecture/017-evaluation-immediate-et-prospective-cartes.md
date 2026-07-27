# Évaluation immédiate et prospective des cartes — Architecture

## Objective

Séparer les deux notions actuellement mélangées dans l’heuristique :

- la valeur d’une carte jouée depuis la main, qui doit représenter uniquement l’effet résolu
  maintenant ;
- la valeur d’une carte acquise depuis la rivière, qui peut prendre en compte son potentiel futur
  et le risque de ses contraintes.

Le cas observable qui motive cette évolution est `Éclat de l’infini`. À faible maîtrise, il produit
immédiatement 2 Power, mais l’heuristique additionne aussi les pénalités de ses seuils 10, 20 et 30.
Elle préfère alors passer son tour avec la carte en main.

## Current State

`HeuristicPlayer.choose_action()` compare les actions légales et délègue l’extraction à
`shards_ai/ai/heuristic_features.py`.

`_effect_features()` sélectionne déjà la branche active via `Effect.operations_for_mastery()`, mais
`_effect_constraint_penalty()` parcourt toutes les étapes de l’effet. Cette fonction est utilisée
par les chemins de jeu et d’acquisition ; les seuils futurs pénalisent donc une carte même lorsqu’ils
ne participent pas à l’effet joué.

Les actions concernées sont :

- `PlayCard` : effet immédiat depuis la main ;
- `BuyCard` et `RecruitFreeCard` : carte durable acquise depuis la rivière ;
- `RecruitMercenary` : effet immédiat, sans actif durable ;
- bannissement et décisions de sélection : arbitrages de conservation, qui peuvent continuer à
  utiliser une valeur prospective sans changer la valeur de `PlayCard`.

Les poids existants restent injectés par `HeuristicWeights`, `CardAcquisitionWeights` et
`CardConstraintWeights`. Aucun changement du moteur, des règles ou des actions légales n’est requis.

## Target Behavior

### Évaluation immédiate

Pour `PlayCard` et `RecruitMercenary` :

1. sélectionner la branche applicable à la maîtrise actuelle ;
2. valoriser uniquement les opérations de cette branche ;
3. appliquer uniquement les contraintes de cette branche et les conditions observables non
   satisfaites ;
4. ignorer complètement les branches et seuils futurs.

Une opération conditionnelle non satisfaite dans la branche actuelle est ignorée comme le serait le
jeu lui-même : elle ne fournit ni valeur ni pénalité. Une branche future ne peut jamais rendre
négatif le score d’une carte jouable maintenant.

### Évaluation prospective

Pour `BuyCard` et `RecruitFreeCard` :

1. valoriser l’effet actuellement disponible à pleine valeur ;
2. ajouter le potentiel positif des branches futures accessibles avec la progression de maîtrise ;
3. soustraire un malus de contrainte pour distinguer le potentiel d’un effet garanti d’un effet
   immédiatement assuré ;
4. appliquer ensuite le facteur de replay durable déjà existant.

Les branches sont des étapes successives d’un même actif durable : leur contribution potentielle
peut donc être cumulée pour une carte acquise. Cette agrégation ne doit toutefois pas être utilisée
pour un mercenaire recruté immédiatement, qui reçoit seulement son effet courant.

Le comportement prospectif reste une estimation déclarative : il ne simule ni ordre de pioche,
ni nombre réel de tours, ni probabilité d’atteindre une condition.

## Non-Goals

- modifier la résolution moteur des effets ou des seuils ;
- apprendre automatiquement la probabilité d’atteindre une contrainte ;
- simuler une partie future ou cloner `Game` pendant une décision ;
- recalibrer les profils `v005` dans cette évolution ;
- modifier les règles de bannissement ou la légalité des achats ;
- introduire un nouveau poids avant d’avoir mesuré le comportement obtenu.

## Key Decisions

1. **Deux contrats explicites.** L’extracteur expose un chemin immédiat et un chemin prospectif,
   plutôt qu’un unique calcul de pénalité partagé implicitement.
2. **Branche active pour le jeu.** `PlayCard` et `RecruitMercenary` ne consultent jamais les
   seuils futurs pour leur valeur immédiate. Les opérations conditionnelles inactives de la branche
   sélectionnée sont également ignorées, sans malus : seule la partie effectivement résolue est
   scorée.
3. **Potentiel pour l’acquisition durable.** `BuyCard` et `RecruitFreeCard` peuvent cumuler les
   effets futurs comme valeur d’actif, mais leurs contraintes non garanties restent pénalisées.
   Chaque contribution d’effet est bornée par `max(0, valeur - pénalité)` avant l’agrégation ; un
   effet conditionnel ne peut donc jamais dégrader la valeur d’un autre effet de la carte.
4. **Mercenaire immédiat.** `RecruitMercenary` n’utilise ni la valeur d’acquisition durable ni le
   facteur de replay ; il valorise seulement l’effet résolu maintenant.
5. **Pas de mutation.** Les projections utilisent des structures détachées et des fonctions pures.
6. **Compatibilité des poids.** Les types et profils YAML existants restent inchangés. La correction
   modifie l’interprétation des contraintes par contexte, pas leur configuration.
7. **Stabilité du classement.** Les priorités de victoire, létalité et égalité de score restent
   inchangées ; la correction agit uniquement sur les features des cartes.

## Open Questions

Aucune question bloquante. La valeur des branches futures reste volontairement une estimation
simple et devra être validée comportementalement avant toute campagne de calibration dédiée.

## Proposed Architecture

### Fonctions d’extraction

Dans `shards_ai/ai/heuristic_features.py` :

- conserver une primitive d’effet immédiat qui reçoit une branche d’opérations explicitement
  sélectionnée ;
- ajouter une primitive prospective qui parcourt les étapes déclaratives et agrège leur valeur
  potentielle avec les pénalités de contraintes ;
- faire appeler la primitive immédiate par `PlayCard` et `RecruitMercenary` ;
- faire appeler la primitive prospective par l’acquisition durable et les chemins de conservation
  qui utilisent déjà `_card_acquisition_value()`.

Les fonctions de pénalité ne doivent plus déduire seules le contexte depuis la carte entière. Le
caller fournit soit la branche active, soit l’étape prospective à analyser.

### Flux de décision

```text
legal_actions
      |
      +--> PlayCard --------------------> effet actif + contraintes actuelles
      |
      +--> RecruitMercenary ------------> effet actif + contraintes actuelles
      |
      +--> BuyCard / RecruitFreeCard ---> potentiel durable + malus de risque
      |
      +--> Banish / pending choice -----> valeur de conservation existante
```

Les signaux restent des `ActionFeatures` et le produit scalaire de `HeuristicWeights` reste le
seul classement final.

## Data Model

Aucun nouveau champ de profil, de moteur ou de sérialisation. Les mêmes
`ActionFeatures.card_acquisition_value` et `ActionFeatures.constraint_penalty` changent seulement
de source selon le contexte de l’action.

## Performance And Scalability

La rivière contient au plus six cartes et les mains restent de taille limitée. Parcourir les étapes
d’une définition pendant une décision d’achat reste borné et ne justifie ni cache mutable, ni
parallélisme, ni simulation de partie. Le chemin `PlayCard`, beaucoup plus fréquent, doit rester
une seule sélection de branche et ne doit pas parcourir les seuils futurs.

## Edge Cases

- effet plat sans étapes : comportement identique avant/après ;
- carte à seuil unique non atteint : aucune valeur immédiate si aucune branche active, mais potentiel
  possible dans l’acquisition durable ;
- plusieurs contraintes sur une opération : les malus applicables se cumulent ;
- seuil atteint : la branche sélectionnée est valorisée sans malus de seuil ;
- effet terminal `win` : la priorité terminale existante reste supérieure au score pondéré ;
- mercenaire avec effets conditionnels futurs : seuls les effets de son recrutement immédiat sont
  évalués ;
- profil historique : aucun changement de chargement YAML.

## Testing Strategy

- tester que `Éclat de l’infini` à maîtrise 1 reçoit uniquement la valeur de ses 2 Power et que
  `HeuristicPlayer` choisit `PlayCard` plutôt que `PassPlayPhase` ;
- tester les valeurs de l’Éclat à maîtrises 10, 20 et 30 ;
- tester qu’une carte à branches futures n’est pas pénalisée par ses seuils futurs lors de
  `PlayCard` ;
- tester que l’évaluation `BuyCard` conserve le potentiel futur et son malus de contrainte ;
- tester que `RecruitMercenary` n’utilise pas le potentiel durable ;
- préserver les tests existants d’acquisition, de contraintes, de `GainMastery` et de profils ;
- exécuter la suite complète du dépôt et une partie reproductible Heuristic contre Random.

## Rollout And Migration

La modification est rétrocompatible au niveau des API et des fichiers YAML. Les profils historiques
ne sont pas réécrits. Après validation des tests, comparer v005 avant/après sur des seeds identiques
et inspecter le taux de passages avec cartes en main avant toute nouvelle optimisation des poids.

## Files Expected To Change

- `shards_ai/ai/heuristic_features.py` — séparation immédiat/prospectif ;
- `tests/game/test_heuristic_player.py` et/ou `tests/game/test_heuristic_state_features.py` — tests
  de branche active et de classement ;
- `doc/Current state/Heuristic player.md` — comportement effectivement disponible après validation.
