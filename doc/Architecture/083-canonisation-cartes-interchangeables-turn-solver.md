# Canonisation des cartes interchangeables du turn solver — Architecture

## Objective

Réduire l’espace de candidats macro du `PlayTurnSolver` lorsque plusieurs actions
visent des copies physiques équivalentes d’une même carte. Deux cartes ayant le même
`card_definition_id` et appartenant à la même zone logique doivent produire une seule
candidate macro lorsque l’identité de l’instance ne modifie ni l’effet, ni la légalité,
ni les choix futurs.

Dans la main de référence — deux `crystal`, un `blaster`, un `pirate_heretique` et un
`apotre_des_ombres` — le solver doit exposer cinq branches au lieu de six. Le moteur
continue toutefois à recevoir une action atomique avec un `instance_id` réel.

Critères de réussite :

- les copies équivalentes ne créent plus de choix neural dupliqué ;
- le choix d’une candidate reste rejouable par une trace atomique légale ;
- l’identité physique est conservée pour l’exécution, les cibles et les diagnostics ;
- aucune branche non équivalente n’est fusionnée ;
- le nombre de candidates, les collisions sémantiques et les regroupements sont
  mesurables dans les tests et les rapports ;
- le comportement du moteur et les profils atomiques historiques restent inchangés.

## Current State

Le moteur est atomique et fait autorité via `Game.legal_actions()` et `Game.apply()`.
`PlayTurnSolver.resolve()` explore des clones et retourne des `PlayTurnCandidate` dont
`atomic_trace` contient les actions physiques. La clé `canonical_key` est actuellement
construite avec `_action_key()`, donc avec les `instance_id`, mais elle n’est pas utilisée
pour dédupliquer les candidates.

`MacroNeuralPlayer` reçoit les candidates, fait construire leur représentation selon le
schéma V1 à V4, demande au scorer un index, puis rejoue la trace atomique choisie. Le
scorer V2+ masque déjà `card_instance_id` dans la représentation racine ; la duplication
existe donc surtout dans l’espace de recherche et dans la cardinalité présentée au joueur.

Dans l’état actuel du moteur, `instance_id` sert principalement à résoudre une action
physique. Les informations individuelles qui changent la légalité sont stockées à côté
de l’instance : `activated_champion_ids`, `played_card_ids_this_turn` et
`recruited_mercenary_ids_this_turn`. Pour les cartes ordinaires en main, défausse ou
zone de jeu, aucune autre propriété par instance n’est actuellement définie.

## Target Behavior

À chaque point de décision, le solver forme des classes d’équivalence d’actions avant
d’explorer ou de présenter les branches au scorer.

Exemples :

```text
PlayCard(crystal-1) ┐
                    ├── classe (play_card, crystal, hand)
PlayCard(crystal-2) ┘
```

Une seule action représentative est explorée et conservée dans la trace, par exemple
`PlayCard(crystal-1)`. Le joueur ne fabrique pas de nouvelle action macro et le moteur
continue de valider cette action réelle.

La fusion est refusée si une différence de position, de zone, de cible, de paramètre,
d’état individuel ou de conséquence future est détectée. Les cartes identiques dans le
deck de pioche ne sont donc pas regroupées par simple `card_definition_id`, car leur
position détermine l’ordre des pioches. Les cartes du marché restent distinguées par
leur slot. Les champions activés et non activés restent distincts.

## Non-Goals

- modifier les règles ou les classes d’actions du moteur ;
- supprimer les `instance_id` des états, des actions atomiques ou des logs ;
- canoniser des cartes différentes sur la base d’un delta de ressources identique ;
- traiter la représentation masquée du réseau comme preuve suffisante d’équivalence ;
- changer les budgets fixes du solver ;
- modifier les profils, checkpoints ou architectures neuronales actives ;
- relancer un entraînement ou une promotion dans cette évolution ;
- dédupliquer les états internes de mémoïsation sans preuve de renumérotation sûre.

## Key Decisions

1. **L’équivalence est définie au niveau de la décision, pas de l’état physique.**
   Une candidate garde sa trace physique et son représentant concret. La classe
   d’équivalence sert uniquement à réduire le choix macro.

2. **Clé de base par zone.** Pour les cartes ordinaires en `hand`, `discard_pile` ou
   `play_zone`, la clé utilise :

   ```text
   action_type + card_definition_id + zone + paramètres non physiques
   ```

   Pour `draw_pile`, elle inclut la position. Pour `river`, elle inclut le slot. Pour
   `champions`, elle inclut les flags individuels qui affectent la légalité, au minimum
   l’état activé.

3. **Les états individuels sont explicites.** Une future propriété par instance qui
   modifie un effet, une condition ou une action légale doit être ajoutée à la clé
   d’équivalence avant de permettre une fusion. L’absence d’une propriété connue ne
   constitue pas une permission implicite de fusionner.

4. **La trace représentative est la source de vérité.** Le joueur rejoue exactement la
   trace choisie. Il ne reconstruit pas `PlayCard` à partir d’un `card_definition_id` et
   ne demande pas au moteur d’accepter une action abstraite.

5. **La déduplication précède le scoring.** Le solver renvoie déjà les candidates
   regroupées ; `MacroNeuralPlayer`, les builders V1–V4 et les datasets n’ont pas à
   filtrer indépendamment. Cela évite que le joueur et le dataset apprennent des
   cardinalités différentes.

6. **Le représentant est déterministe.** Dans chaque classe, le représentant est choisi
   selon l’ordre actuel de `_action_key()` ou une règle stable équivalente. Le résultat
   ne dépend ni du hash Python ni de l’ordre d’un `set`.

7. **`canonical_key` devient un contrat utilisé.** Elle doit identifier la classe
   sémantique de la candidate, tandis qu’une clé physique séparée reste disponible pour
   les diagnostics et la replayabilité. Une candidate fusionnée peut exposer le nombre
   de variantes physiques regroupées.

8. **Mémoïsation conservatrice.** La première livraison ne fusionne pas les états
   internes contenant des instances différentes. Elle réduit seulement les actions
   symétriques à un même point de décision. Une normalisation d’état pourra être étudiée
   séparément avec des invariants de permutation testés.

9. **Compatibilité du joueur.** Une resolution singleton issue de la canonisation est
   rejouée sans appel au scorer et sans `macro_decisions`, comme toute candidate unique
   actuelle. Les compteurs ajoutent seulement des métriques de variantes regroupées.

10. **Payload sans liste d’instances.** `MacroDecisionPayload` expose uniquement
    `physical_variant_count` et la trace du représentant. La liste des `instance_id`
    équivalents reste interne au diagnostic éventuel et ne rejoint ni le tenseur, ni le
    contrat normal du joueur.

11. **Bannissement canonisé par zone.** Les `BanishCard` visant des copies identiques
    dans une même zone sont regroupés avec la clé `(banish_card, card_definition_id,
    zone)`, sans jamais fusionner deux définitions différentes. Cette réduction conserve
    la décision stratégique de bannir ou de passer, ainsi que les distinctions entre
    cartes de valeurs différentes.

12. **Deux niveaux de comparaison dans les rapports.** Les diagnostics conservent à la
    fois la comparaison exacte incluant `instance_id` et la comparaison sémantique
    canonisée. Un écart physique entre deux copies équivalentes ne sera ainsi pas
    présenté comme un désaccord de politique.

## Open Questions

## Proposed Architecture

### 1. Équivalence des actions

Ajouter dans `shards_ai/ai/play_turn_solver.py` un résolveur de zone et une fonction
stable de clé sémantique, par exemple `equivalence_key_for_action(game, action)`.
Cette fonction résout la carte depuis l’état du clone, détermine sa zone et renvoie une
clé structurée. Elle doit lever une erreur ou retourner une clé non canonisable lorsqu’un
objet ne peut pas être résolu de manière sûre.

La fonction doit distinguer au minimum :

| Contexte | Clé minimale |
| --- | --- |
| Carte en main | type d’action + définition + paramètres sémantiques |
| Carte en défausse | type d’action + définition + paramètres sémantiques |
| Carte en zone de jeu | type d’action + définition + état individuel pertinent |
| Carte du deck de pioche | définition + position |
| Carte de la rivière | définition + slot + type d’action |
| Champion | définition + activation + état pertinent |
| Choix pending | type de décision + définition/choix sémantique |
| Action sans carte | type + paramètres |

La clé ne doit pas remplacer `_action_key()`, qui reste utile pour le tri déterministe
et la traçabilité physique.

### 2. Regroupement du solver

Au moment où `resolve()` obtient les actions légales d’un point de décision, il les
partitionne par clé d’équivalence. Le premier représentant de chaque groupe est exploré.
Après exploration, la candidate conserve :

```text
atomic_trace                 trace du représentant physique
canonical_key                clé sémantique stable
physical_variant_count       nombre d’actions physiques regroupées
physical_variant_ids         diagnostic optionnel hors tenseur
```

Pour la première version, le regroupement est appliqué aux actions racines. Les suffixes
de trace restent issus de l’exécution du représentant et ne sont jamais réécrits par
approximation. Cela limite le risque et suffit à supprimer les permutations de copies
identiques observées dans la main de référence.

Une assertion de test vérifiera qu’un groupe ne contient qu’un seul résultat sémantique
sur les champs de conséquence macro et les actions stratégiques légales suivantes. Si
l’assertion échoue, le groupe sera conservativement éclaté.

### 3. Adaptation de `MacroNeuralPlayer`

Le joueur continue à consommer `resolution.candidates` sans logique de regroupement
locale. Il doit cependant :

- accepter les traces représentant une classe physique ;
- journaliser `canonical_key` et `physical_variant_count` dans le payload de décision ;
- conserver le contrôle `action in legal_actions` avant chaque replay ;
- vider la trace et repasser par une résolution fraîche si elle devient illégale ;
- distinguer dans les diagnostics `macro_choice`, `macro_replay`, `canonical_grouped`
  et `atomic_fallback` si le reporting actuel le permet.

Les représentations V1–V4 ne reçoivent aucun `instance_id` supplémentaire. Le modèle
voit une seule action `play_card/crystal`, ce qui rend cohérente la représentation avec
la cardinalité du solver.

### 4. Dataset et entraînement

Le générateur de dataset macro doit enregistrer le nombre de variantes physiques
regroupées et le schéma de canonisation dans le manifeste. Le label reste l’index de la
candidate canonique, jamais un index d’instance.

Les datasets historiques restent lisibles. Un dataset généré avant cette évolution est
marqué avec une version de canonisation différente et ne doit pas être comparé sans
normalisation. Aucun checkpoint existant n’est modifié ou promu dans cette livraison.

### 5. Diagnostics et performance

Mesurer sur les mêmes seeds :

- candidates brutes versus candidates canonisées ;
- ratio de variantes physiques supprimées ;
- temps et expansions du solver ;
- appels scorer et décisions macro ;
- désaccords exacts versus sémantiques ;
- absence de changement dans les actions atomiques effectivement appliquées.

La réduction doit diminuer la cardinalité et le coût du scoring. Elle ne doit pas être
présentée comme une amélioration de force de jeu sans panel complet reproductible.

## Data Model

Aucune table ni modification du moteur n’est prévue. Les changements de données sont
des champs de diagnostic sérialisables dans les structures IA :

- `canonicalization_schema_version` dans le manifeste/dataset macro ;
- `canonical_key` sémantique ;
- `physical_variant_count` ;
- éventuellement `representative_instance_id` uniquement dans les rapports hors modèle.

Les anciens schémas de représentation restent désérialisables ; les nouveaux enregistrements
annoncent explicitement leur version.

## Backend Flow

Le flux devient :

```text
legal_actions()
  -> clés d’équivalence sémantiques
  -> représentants déterministes
  -> exploration Game.clone()/apply()
  -> candidates canonisées
  -> MacroNeuralPlayer choisit un index
  -> replay de la trace physique
  -> Game.apply() valide chaque action
```

En cas d’action non résoluble, de divergence de conséquence ou de trace illégale, le
solver abandonne le regroupement concerné et conserve les actions physiques séparées.
Le fallback global existant reste inchangé.

## Frontend Flow

Sans objet : le changement reste dans le moteur de décision IA et les artefacts de
recherche. Aucun écran ou contrat frontend n’est concerné.

## Authorization And Feature Gates

Sans objet. Aucun feature flag n’est prévu : après validation, la canonisation devient
le comportement par défaut du turn solver. Les checkpoints existants ne sont pas
automatiquement basculés.

## Observability And Operations

Ajouter aux rapports de résolution et aux diagnostics de benchmark :

- `raw_candidate_count` ;
- `canonical_candidate_count` ;
- `grouped_variant_count` ;
- distribution des groupes par action et `card_definition_id` ;
- nombre de groupes éclatés pour cause d’incertitude ;
- erreurs de replay après canonisation.

Les logs ne doivent pas être écrits dans `doc/`. Les artefacts restent dans les chemins
de benchmark/dataset existants.

## Edge Cases

- deux cartes identiques dans la main mais dont une est déjà marquée par un futur état
  individuel : elles restent séparées ;
- deux champions identiques, dont un est activé : ils restent séparés ;
- cartes identiques à des positions différentes dans la pioche : elles restent séparées ;
- cartes identiques dans des slots différents de la rivière : elles restent séparées ;
- pending choices avec des identités physiques différentes : fusion uniquement si les
  définitions, zones et conséquences sont équivalentes ;
- trace représentative devenue illégale : purge de la trace et résolution fraîche ;
- budget atteint : fallback atomique existant, sans fusion arbitraire ;
- deux branches ayant le même delta de ressources mais des cartes restantes différentes :
  elles ne sont pas fusionnées.

## Testing Strategy

Ajouter ou adapter des tests ciblés dans `tests/ai/test_play_turn_solver.py` et
`tests/ai/test_macro_model.py` :

1. deux Cristaux en main donnent une seule candidate `play_card/crystal` ;
2. deux cartes identiques en défausse sont regroupées pour `BanishCard` si la décision
   est canonisable ;
3. deux cartes différentes restent distinctes même avec le même delta de ressources ;
4. deux cartes identiques à des positions différentes de pioche restent distinctes ;
5. deux champions identiques activé/non activé restent distincts ;
6. la candidate conserve un `PlayCard` avec un `instance_id` réel et le joueur la rejoue ;
7. le scorer reçoit le nombre canonisé, sans doublon d’instance ;
8. un singleton canonisé ne déclenche pas d’appel au scorer ;
9. le replay refuse proprement une trace rendue illégale ;
10. les compteurs et payloads distinguent choix macro, replay canonique et fallback ;
11. le dataset et son manifeste enregistrent la version de canonisation ;
12. un scénario complet vérifie que le regroupement ne change pas les transitions
    atomiques finales pour un représentant donné.

Valider d’abord avec les tests IA ciblés, puis la suite complète et un benchmark court
à seeds fixes. Le screening complet de parties et toute décision de promotion restent
hors de cette architecture.

## Rollout And Migration

1. Implémenter les clés de zone et les tests unitaires sans modifier le joueur.
2. Ajouter le regroupement dans `PlayTurnSolver` et valider les traces atomiques.
3. Adapter `MacroNeuralPlayer`, les payloads et les métriques.
4. Mettre à jour le générateur de dataset, le manifeste et les tests de schéma.
5. Régénérer un petit dataset de contrôle pour vérifier la cardinalité, sans lancer
   d’entraînement long.
6. Comparer les benchmarks courts exacts et sémantiques.
7. Après validation de l’architecture complète, envisager un nouveau dataset et un
   entraînement depuis zéro dans le checkpoint mutable canonique.

Le rollback consiste à désactiver le regroupement dans la couche IA ou à revenir au
solver précédent ; le moteur, les actions publiques et les anciens checkpoints restent
compatibles.

## Files Expected To Change

- `shards_ai/ai/play_turn_solver.py` — clés d’équivalence, partitionnement, candidates
  canonisées et métadonnées de variantes.
- `shards_ai/ai/macro_player.py` — payloads, compteurs et diagnostics de replay.
- `shards_ai/ai/macro_imitation_dataset.py` — version et métriques de canonisation.
- `shards_ai/ai/macro_model.py` — probablement aucun changement de tenseur ; uniquement
  adaptation si le manifeste expose un champ obligatoire.
- `tests/ai/test_play_turn_solver.py` — invariants d’équivalence et cardinalité.
- `tests/ai/test_macro_model.py` — absence de doublons d’instance dans les candidates.
- `tests/ai/test_macro_imitation_dataset.py` — compatibilité manifeste/dataset.
- `doc/Current state/Neural player.md` — comportement final après implémentation.
- `doc/Current state/Game engine.md` — seulement si les limites d’identité d’instance
  du moteur doivent être documentées plus précisément.

Les fichiers actuellement modifiés ou non suivis dans le worktree sont hors périmètre de
cette architecture et doivent être préservés.
