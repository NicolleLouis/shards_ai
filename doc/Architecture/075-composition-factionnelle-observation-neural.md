# Composition factionnelle de l'observation neuronale — Architecture

## Objective

Ajouter au modèle neural un résumé combiné de l'état du deck : cardinalités de zones et comptes
factionnels globaux du joueur actif. Ce résumé doit fournir une information sur le recyclage, la
composition du deck, le potentiel futur de synergies et la valeur des achats ou bannissements qui
modifient cette composition.

## Current State

Le modèle actif `v004` utilise `structured_semantic_v5_fusion_experiment`. Une candidate récente
utilise le feature set opt-in `zone_cardinality_v1`, qui ajoute sept cardinalités à l'état. La
nouvelle expérience ne conservera pas cette candidate comme architecture indépendante : elle
partira directement du checkpoint V004 et ajoutera les cardinalités et les factions dans un seul
feature set versionné. Les checkpoints sont reconstruits depuis `model_config` et leur première
couche `state_encoder` dépend de la dimension exacte de l'observation.

`NeuralObservation.active_player.owned_card_counts` contient déjà les comptes de cartes possédées
par définition. Le catalogue fournit la faction de chaque définition. Aucune nouvelle information
de jeu n'est nécessaire.

Le scoreur action-conditionnel pool les zones et encode chaque action légale séparément. Le moteur
valide les actions et le masque d'information ne doit pas changer.

## Target Behavior

Le nouveau feature set combiné et unique est :

```text
deck_state_v1
```

Il ajoute aux scalaires historiques de V004 onze scalaires, dans l'ordre stable suivant :

### Cardinalités

Les sept cardinalités de zones sont conservées dans l'ordre défini par l'architecture 074 : pioche
active, défausse active, champions actifs, total possédé actif, total possédé adverse public,
défausse adverse publique et champions adverses publics.

### Composition factionnelle

| Ordre | Faction | Source | Normalisation |
|---:|---|---|---:|
| 1 | `maquis` | `active_player.owned_card_counts` + catalogue | `min(count, 100) / 100` |
| 2 | `spectra` | `active_player.owned_card_counts` + catalogue | `min(count, 100) / 100` |
| 3 | `homodeus` | `active_player.owned_card_counts` + catalogue | `min(count, 100) / 100` |
| 4 | `order` | `active_player.owned_card_counts` + catalogue | `min(count, 100) / 100` |

Les cartes neutres sont exclues des quatre comptes. Elles ne sont pas placées dans une faction
artificielle et ne sont pas comptées dans un cinquième groupe pour cette expérience.

Les comptes incluent toutes les zones possédées déjà agrégées dans `owned_card_counts` : main,
pioche, défausse, zone de jeu et champions. Ils décrivent le potentiel total du deck, pas son
activation immédiate dans la décision courante.

## Non-Goals

- ne pas ajouter de comptes factionnels adverses ;
- ne pas ajouter les indicateurs tactiques Union, Echo ou Domination ;
- ne pas modifier `NeuralObservation` ou sa sérialisation ;
- ne pas modifier `Game.legal_actions()` ou `Game.apply()` ;
- ne pas modifier les heuristiques V8 ;
- ne pas modifier le profil actif `v004` ni les checkpoints stables ;

## Key Decisions

1. **Le résumé du deck est une feature d'état unique.** Cardinalités et composition factionnelle
   sont toujours activées ensemble ; aucun toggle indépendant ne sera exposé.
2. **La composition est une feature d'état, pas une feature d'action.** Elle est identique pour
   toutes les actions candidates d'une même observation.
3. **La source de vérité est `owned_card_counts` et le catalogue.** Aucun nouveau champ métier ni
   recalcul depuis une zone cachée adverse n'est autorisé.
4. **Le feature set est versionné et explicite.** `zone_cardinality_v1` reste historique ; le
   nouveau set `deck_state_v1` représente la combinaison cardinalité + factions et rend les anciens
   checkpoints incompatibles par construction de dimension.
5. **Les architectures historiques restent isolées.** V004 et le scorer cardinalité historique ne
   doivent pas changer de dimension implicitement.
6. **Le code partagé doit être limité à l'encodage des features.** Un scorer versionné dédié porte
   l'identité de la nouvelle architecture ; un mixin ou un helper peut éviter de recopier la
   construction du `state_encoder`.
7. **La migration part directement de V004.** Les onze nouvelles colonnes de la première couche
   seront initialisées à zéro afin de préserver le comportement de V004 au chargement initial. La
   migration doit être explicite, testée et enregistrée dans les métadonnées ; elle ne doit pas être
   confondue avec un chargement strict d'un ancien checkpoint.

## Open Questions

1. **Nom du scorer dédié — bloquant avant code.** Confirmer le nom final de la classe et de
   l'architecture. Proposition : `StructuredSemanticV5DeckStateScorer` et
   `structured_semantic_v5_deck_state_v1`.
2. **Borne de 100 — non bloquant.** Elle est cohérente avec le clipping des cardinalités actuelles,
   mais devra être confirmée par la distribution réelle des decks observés.

## Proposed Architecture

### Modèles et séparation historique

Le modèle cible doit être identifié explicitement par sa classe/factory :

```text
NeuralActionScorer
  baseline historique

StructuredSemanticV5FusionScorer
  V004 historique, inchangé

StructuredSemanticV5DeckStateScorer
  candidate combinant cardinalités et composition factionnelle
```

Les scoreurs historiques et le nouveau scorer peuvent partager un encodeur de scalaires et une
fonction de calcul de dimension, mais leurs identifiants d'architecture et leurs contrats de
checkpoint restent distincts. Le modèle cible aura onze scalaires d'observation expérimentaux :
sept cardinalités puis quatre comptes factionnels, suivis du contexte phase/tour existant.

### Calcul des factions

À l'encodage :

1. parcourir `active_player.owned_card_counts` ;
2. retrouver la définition dans le catalogue ;
3. ignorer les cartes neutres ;
4. incrémenter le compte de la faction correspondante ;
5. clipper et normaliser selon l'ordre contractuel ;
6. concaténer les quatre scalaires à l'état.

Le calcul doit rester déterministe, indépendant de l'ordre des tuples et sans accès à
`opponent.hand` ou `opponent.draw_pile_counts`.

## Data Model

Aucune modification de `NeuralObservation`, de `OBSERVATION_SCHEMA_VERSION` ou du JSONL n'est
nécessaire. Le nouveau feature set est une propriété du modèle et doit figurer dans :

- la configuration effective ;
- le fingerprint du profil ;
- les métadonnées du checkpoint ;
- l'identifiant d'architecture.

Une observation historique reste désérialisable et peut être encodée par les architectures
historiques. Elle ne doit pas être interprétée silencieusement par le nouveau modèle si sa
configuration ne déclare pas le feature set attendu.

## Performance And Risks

Le coût du calcul factionnel est linéaire dans le nombre de types présents dans
`owned_card_counts`, donc faible par rapport aux embeddings et au forward du modèle. La première
couche du `state_encoder` augmente toutefois de quatre colonnes.

Le principal risque est la corrélation avec la composition des parties. Cette note se limite au
contrat d'encodage et de compatibilité ; les décisions d'utilisation du modèle sont hors périmètre.

## Edge Cases

- deck sans carte d'une faction : scalaire nul ;
- deck composé uniquement de neutres : quatre scalaires nuls ;
- définition inconnue dans `owned_card_counts` : utiliser la voie `UNK` existante sans attribuer une
  faction inventée ;
- total supérieur à 100 : clipping à 1 ;
- ordre différent des `CardCounts` : même sortie ;
- observation adverse masquée : aucun compte adverse calculé ;
- ancien checkpoint : chargement strict uniquement avec son feature set historique.

## Testing Strategy

- compte exact pour un deck contenant une faction et des neutres ;
- quatre factions distinguées dans des decks de même taille ;
- champions, défausse, pioche et main inclus via `owned_card_counts` ;
- permutation des comptes sans effet ;
- clipping à la borne 100 ;
- absence de lecture des zones cachées adverses ;
- dimension baseline/V004 inchangée ;
- dimension cardinalité inchangée ;
- nouvelle dimension avec exactement quatre scalaires supplémentaires ;
- forward fini avec le scorer dédié et toutes les actions légales ;
- sauvegarde/restauration des métadonnées d'architecture ;
- migration de `configs/neural_profiles/v004.pt` refusée en chargement strict et acceptée uniquement
  par la fonction de migration testée.

## Rollout And Migration

La candidate combinée reste hors du profil actif. Aucun fichier sous `configs/neural_profiles/` ne doit
être remplacé.

La migration part de `configs/neural_profiles/v004.pt` et produit un checkpoint de travail avec
l'architecture combinée explicite. Elle copie les poids historiques, initialise les onze nouvelles
colonnes de façon déterministe et enregistre le checkpoint source, le feature set cible et la méthode
de migration.

Un smoke test doit vérifier que le modèle migré produit exactement le même forward que V004 lorsque
les onze nouvelles features sont neutralisées.

## Files Expected To Change

- `shards_ai/ai/neural_model.py` ou un nouveau module de scorer versionné : classe et factory ;
- `shards_ai/ai/structured_v005.py` ou un module dédié de scorer deck state ;
- `tests/ai/test_neural_model.py` : comptes, dimensions, factory et restauration ;
- `doc/TODO/002-composition-factionnelle-observation-neural.md` : statut et protocole réel ;
- un profil de candidate séparé, seulement après validation de cette architecture.
