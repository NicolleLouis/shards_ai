# Protection de bannissement des cartes remplacées — Architecture

## Objective

Empêcher l’HeuristicPlayer de bannir une carte qui se remplace effectivement par une ou plusieurs
pioches lorsqu’elle est jouée. Bannir une telle carte ne réduit pas réellement la taille du deck et
peut supprimer une carte dont l’effet immédiat reste utile.

Le problème observé concerne notamment `Drones Miniers`, `Pirate Hérétique`, `Evokatus` et d’autres
cartes qui piochent. Leur valeur d’acquisition actuelle rend parfois `BanishCard` préférable à
`SkipBanish`, même si la carte bannie est immédiatement remplacée.

## Current State

`Game.legal_actions()` propose toutes les cartes de la main et de la défausse comme cibles de
bannissement, ainsi que `SkipBanish`. `HeuristicPlayer` protège déjà les cartes comportant une
branche de victoire, mais les cartes qui se remplacent ne bénéficient d’aucune protection structurelle.

Le score actuel de bannissement combine `banish_threshold` et la valeur d’acquisition de la carte.
La pioche est valorisée comme une propriété de la carte conservée, mais le score ne modélise pas
explicitement le fait que la carte sera remplacée dans le deck. Une comparaison numérique seule est
donc insuffisante pour cette règle.

## Target Behavior

Avant le classement des actions de bannissement, l’HeuristicPlayer retire des candidats les cartes
qui possèdent une pioche effective dans leur effet jouable maintenant :

- effet direct de la carte avec une opération `draw_card` active ;
- effet `on_play_effect` actif d’un Champion joué depuis la main ;
- contrainte déjà satisfaite au moment de la décision.

Si toutes les cartes candidates sont protégées, `SkipBanish` reste l’action légale retenue.

Une pioche conditionnelle non satisfaite ne protège pas la carte. Une pioche future, potentielle ou
liée à une capacité qui n’est pas disponible dans l’action courante ne protège pas non plus la carte.

## Non-Goals

- Modifier `Game.legal_actions()` ou interdire la cible dans le moteur ; un joueur humain ou une autre
  IA doit conserver accès aux mêmes actions légales.
- Modifier la valeur numérique de `card_draw`, `banish_threshold` ou `SkipBanish`.
- Protéger une carte uniquement parce qu’elle possède une valeur d’acquisition élevée.
- Prédire les tirages futurs ou garantir qu’une pioche produira une carte utile.
- Traiter l’activation d’un Champion déjà posé comme une cible de bannissement : les Champions posés
  ne se trouvent pas dans les zones bannissables. Seul leur effet de pose est pertinent lorsqu’ils
  sont encore en main.

## Key Decisions

1. **Protection dans l’IA.** La règle est une préférence stratégique, pas une contrainte de règle du
   jeu ; elle appartient donc à `HeuristicPlayer`, pas au moteur.
2. **Source de vérité.** Réutiliser `_play_card_features()` afin d’évaluer la branche réellement
   jouable avec les contraintes et le niveau de maîtrise courants.
3. **Critère minimal.** Une feature `card_draw > 0` dans l’effet immédiat actif suffit à protéger la
   carte. Le nombre de cartes piochées n’a pas besoin d’être recalculé pour la décision binaire.
4. **Champions.** `_play_card_features()` sélectionne déjà `on_play_effect` pour un Champion ; la
   protection couvre donc automatiquement les Champions qui piochent à la pose.
5. **Priorité.** La protection est appliquée avant le tie-break et avant le classement numérique ;
   aucun poids ne peut la contourner.

## Proposed Architecture

Ajouter un prédicat d’IA, par exemple `is_replacement_card(...)`, qui reçoit l’observation, le joueur,
la carte et les poids de contraintes. Il appelle `_play_card_features()` et retourne vrai si la
pioche immédiate effective est strictement positive.

Dans `HeuristicPlayer.choose_action()` :

1. construire les actions `BanishCard` légales ;
2. retirer les cibles gagnantes protégées existantes ;
3. retirer les cartes de remplacement ;
4. conserver `SkipBanish` et les autres actions légales ;
5. appliquer le score et le tie-break aux seules cibles restantes.

Cette séquence évite de modifier les transitions ou de dupliquer la logique des contraintes.

## Data Model

Aucun changement de modèle de jeu, de sérialisation ou de profil YAML. La protection est dérivée à
chaque décision à partir de `GameState` et des définitions déclaratives des cartes.

## Performance

Le prédicat réutilise une évaluation déjà nécessaire au classement des bannissements. Il ne crée ni
partie simulée ni copie d’état et reste limité au nombre de cartes bannissables dans la main et la
défausse, donc négligeable dans le hot path.

## Edge Cases

- Une carte avec pioche conditionnelle inactive reste éligible au bannissement.
- Une carte avec plusieurs branches ne protège que si la branche active pioche maintenant.
- Une carte qui pioche zéro carte n’est pas protégée.
- Une carte avec pioche et d’autres effets négatifs reste protégée : la règle exprime que la carte
  se remplace, sans prétendre que son effet global est toujours optimal.
- Les cartes gagnantes restent protégées même si elles ne piochent pas.
- Lorsque seules des cartes protégées sont présentes, `SkipBanish` est sélectionné.

## Testing Strategy

- protéger une carte ordinaire avec pioche directe ;
- protéger `Drones Miniers` et `Pirate Hérétique` ;
- protéger un Champion avec pioche à la pose ;
- ne pas protéger une pioche conditionnelle dont la contrainte est inactive ;
- bannir une carte sans effet de remplacement ;
- conserver la protection existante des cartes à branche `win` ;
- exécuter la suite complète des tests.

## Rollout And Migration

Le changement est immédiatement actif pour toute instance d’`HeuristicPlayer`, sans migration de
profil. Une nouvelle campagne macro contre `RandomPlayer` devra mesurer l’effet sur le taux de
victoire, le nombre de bannissements et la taille finale des decks avant une éventuelle campagne
globale.

## Files Expected To Change

- `shards_ai/ai/heuristic_features.py` : prédicat de détection de remplacement effectif ;
- `shards_ai/ai/heuristic_player.py` : filtrage des cibles protégées ;
- `tests/game/test_heuristic_player.py` : scénarios directs, conditionnels et Champions ;
- `doc/Current state/Heuristic player.md` : comportement disponible après implémentation.
