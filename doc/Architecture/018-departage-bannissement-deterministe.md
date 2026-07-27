# Départage déterministe du bannissement — Architecture

## Objective

Supprimer le biais introduit par l’ordre de la main et de la défausse lorsque plusieurs cartes ont
exactement le même score de bannissement. Le joueur doit préférer une carte objectivement moins
coûteuse à conserver, puis utiliser un ordre stable indépendant de la construction du deck.

Le cas observé est le `Blaster`, présent très souvent mais rarement banni lorsqu’il est à égalité
avec des cartes plus tardives dans la liste des candidats.

## Current State

`HeuristicPlayer.choose_action()` classe les actions par victoire, létalité, score pondéré, priorité
de phase, puis `-index`. Pour les `BanishCard`, l’index vient directement de
`Game._banishable_cards()`, qui concatène `hand` puis `discard_pile`.

Le score de bannissement utilise `deck_thinning = max(0, 3 - card_acquisition_value)`. Tant que
plusieurs cartes ont une valeur d’acquisition nulle ou identique, elles sont donc équivalentes et
l’ordre des zones décide arbitrairement.

## Target Behavior

Le score principal reste inchangé. Uniquement en cas d’égalité entre actions de bannissement :

1. préférer la carte dont la valeur immédiate si elle était jouée est la plus faible ;
2. à valeur immédiate égale, préférer la carte au coût imprimé le plus faible ;
3. à coût égal, préférer l’identifiant de carte lexicographiquement le plus petit ;
4. conserver l’index courant comme dernier secours pour des instances strictement identiques.

Le `Blaster` sera donc préféré à l’`Éclaireur spectral` lorsqu’ils ont le même score, car son effet
immédiat (+1 Power) est moins valorisé que l’effet de base de l’Éclaireur (+2 Power). Ce départage
ne remplace pas la valorisation positive du Power et ne crée pas d’exception spécifique à une carte.

## Non-Goals

- ajouter un poids optimisable ou une campagne de calibration pour le tie-break ;
- modifier le score principal de bannissement ;
- modifier les actions légales ou les règles de bannissement ;
- utiliser des informations cachées ou la composition future de la pioche ;
- imposer un ordre global arbitraire aux actions qui ne sont pas des bannissements.

## Key Decisions

1. **Départage hors modèle.** Le tie-break est une règle déterministe de politique, pas un signal
   appris ; il ne peut donc pas compenser un mauvais poids de valeur.
2. **Valeur immédiate, coût puis identifiant.** La valeur immédiate réutilise l’évaluation de
   `PlayCard`, sans dépendre de la zone de la carte. Le coût et l’identifiant assurent une sortie
   stable lorsque cette valeur ne suffit pas.
3. **Portée limitée.** Le départage ne s’applique qu’entre `BanishCard` concurrents. Les égalités
   entre phases ou types d’actions gardent le classement existant.
4. **Valorisation séparée.** `card_acquisition_weights.power_produced` reste responsable de la
   valeur économique du Power. Le profil `v005` l’utilise temporairement à `0.25` ; le tie-break
   ne dépend pas de cette valeur.
5. **Protection terminale inchangée.** Les cartes à branche de victoire restent exclues des cibles
   de bannissement avant le classement.

## Open Questions

Aucune question bloquante. Si le coût s’avère un proxy insuffisant à long terme, une évolution
ultérieure pourra introduire une valeur de conservation dédiée, avec une architecture et une
calibration séparées.

## Proposed Architecture

Dans `HeuristicPlayer.choose_action()` :

1. conserver le calcul des `ActionFeatures` et du score principal ;
2. identifier les seules actions `BanishCard` présentes dans `actions` ;
3. construire un rang local trié par `(card.definition.cost, card.definition.card_id)` ;
4. utiliser l’opposé de ce rang comme dernier critère avant l’index d’origine.

Le rang est local à la liste d’actions légales : il ne dépend pas d’une constante globale, du nombre
de copies ou de la zone contenant la carte.

## Performance And Scalability

Une phase de bannissement contient généralement quelques cartes, avec une borne imposée par la main
et la défausse. Un tri local de cette petite liste est négligeable par rapport à l’extraction des
features et ne justifie ni cache, ni mutation persistante, ni parallélisme.

## Edge Cases

- une seule cible : le rang n’a aucun effet ;
- plusieurs cartes de même coût : l’identifiant stable départage ;
- cartes de même définition et instances différentes : l’index final reste déterministe ;
- `SkipBanish` : son score principal reste prioritaire, le rang ne s’applique pas à cette action ;
- carte de victoire conditionnelle : elle reste protégée avant le tie-break ;
- action hors bannissement : aucun changement de classement.

## Testing Strategy

- vérifier qu’à score égal un Blaster est choisi avant un Éclaireur spectral même si ce dernier
  apparaît en premier dans la défausse ;
- vérifier le départage par identifiant pour deux cartes de même coût ;
- vérifier que `SkipBanish` reste choisi ou rejeté uniquement selon son score principal ;
- vérifier que les cartes de victoire protégées restent exclues ;
- exécuter la suite complète du dépôt et une analyse de bannissement reproductible.

## Rollout And Migration

Aucune migration de profil ou de données. Le changement est immédiatement actif dans la politique
heuristique. Les profils historiques restent lisibles ; leur score principal ne change pas, seul le
départage des égalités change.

## Files Expected To Change

- `shards_ai/ai/heuristic_player.py` — classement local des bannissements ;
- `tests/game/test_heuristic_player.py` — cas d’égalité Blaster/Éclaireur et identifiant stable ;
- `doc/Current state/Heuristic player.md` — comportement disponible.
