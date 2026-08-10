# Cardinalités de l'observation neuronale — Architecture

## Objective

Tester une extension minimale de l'entrée d'état du scoreur neural afin de conserver la taille
absolue des zones de cartes après leur pooling moyen.

La candidate doit exploiter uniquement les cardinalités déjà dérivables de `NeuralObservation`.
Elle ne doit modifier ni les règles, ni les actions légales, ni le masque d'information, ni le
checkpoint stable actif.

## Current State

`NeuralActionScorer.encode_observation()` agrège actuellement onze groupes de cartes dans
`shards_ai/ai/neural_model.py`. Plusieurs groupes sont représentés par une moyenne d'embeddings :
une zone homogène de trois cartes et une zone homogène de vingt cartes produisent donc le même
pooling.

Les scalaires actuels décrivent les ressources, les PV/maîtrise, les factions jouées ce tour, les
champions et le contexte de phase/tour. Les cardinalités détaillées des zones ne sont pas
transmises séparément.

`NeuralObservation` contient déjà les données nécessaires : tuples visibles pour le joueur actif,
comptes agrégés pour les zones comptées et comptes publics disponibles pour l'adversaire.

## Target Behavior

Un feature set explicite `zone_cardinality_v1` ajoute sept scalaires à l'état :

| Ordre | Information | Source | Borne de normalisation |
|---:|---|---|---:|
| 1 | taille de la pioche active | `active_player.draw_pile_counts` | 100 |
| 2 | taille de la défausse active | `active_player.discard_counts` | 100 |
| 3 | nombre de champions actifs | `active_player.champions` | 20 |
| 4 | total possédé actif | `active_player.owned_card_counts` | 100 |
| 5 | total possédé adverse public | `opponent.owned_card_counts` | 100 |
| 6 | taille de la défausse adverse publique | `opponent.discard_counts` | 100 |
| 7 | nombre de champions adverses publics | `opponent.champions` | 20 |

Chaque valeur est `min(max(count, 0), bound) / bound`. Le clipping est explicite et déterministe.
La taille de la main est exclue car son contenu et les actions légales sont déjà visibles par le
scoreur action-conditionnel. La taille de `play_zone` est exclue car elle est fortement redondante
avec le total possédé et les autres cardinalités actives.
Les cardinalités de la rivière et du deck central sont exclues : elles caractérisent l'offre
publique de la table et non la taille des zones possédées par un joueur.

## Non-Goals

- ajouter des comptes de faction ;
- ajouter les activations Union, Echo ou Domination ;
- remplacer le pooling moyen ;
- modifier `Game.legal_actions()` ou `Game.apply()` ;
- exposer la main, la pioche ou les instances cachées de l'adversaire ;
- modifier les heuristiques ou le masque d'information ;
- modifier le PPO avant validation de l'imitation ;
- entraîner ou écraser `configs/neural_profiles/v004.pt`.

## Key Decisions

1. Le feature set est opt-in via `NeuralModelConfig.observation_feature_set`, dont la valeur par
   défaut reste `baseline`.
2. Les configurations historiques sans ce champ restent donc compatibles et conservent leur
   dimension d'état historique.
3. `zone_cardinality_v1` augmente la dimension du premier `state_encoder`, ce qui rend les poids
   incompatibles avec un checkpoint baseline. La candidate doit utiliser une configuration
   explicitement fingerprintée et un nouvel entraînement ou une initialisation contrôlée.
4. Les datasets historiques restent lisibles : aucune modification de `NeuralObservation`, de sa
   sérialisation ou de `observation_from_dict()` n'est nécessaire.
5. Les cardinalités sont calculées à partir des comptes complets, et non de la limite de 20 utilisée
   par `_pool_counts()`.
6. La gate de promotion est celle du panel courant et de sa moyenne pondérée. V008 reste un
   adversaire pondéré, mais n'est pas une garde dure indépendante.

## Proposed Architecture

Ajouter dans `NeuralModelConfig` :

```python
observation_feature_set: str = "baseline"
```

Valeurs acceptées : `baseline` et `zone_cardinality_v1`.

`NeuralActionScorer` expose un calcul interne des scalaires d'état. Le chemin baseline conserve
exactement ses scalaires actuels. Le chemin `zone_cardinality_v1` concatène les sept valeurs après
les scalaires existants et avant le contexte phase/tour, puis construit un `state_encoder` de la
dimension correspondante.

La factory d'architecture n'est pas dupliquée pour cette étape : l'identité du feature set est
portée par la configuration sérialisée du checkpoint. Un chargeur reconstruit le même réseau à
partir de `model_config`; l'absence du champ signifie `baseline` pour les checkpoints historiques.

## Data Model

Aucune nouvelle donnée métier ni aucun champ de sérialisation n'est ajouté. Les comptes sont dérivés
à l'encodage depuis les champs existants. Le feature set est inclus dans `asdict(NeuralModelConfig)`
et donc dans le fingerprint de la recette et les métadonnées du checkpoint.

## Performance

Le coût ajouté par décision est constant et négligeable devant les onze poolings et les embeddings
de cartes : sept additions de comptes, sept divisions/clips et sept valeurs dans le premier MLP.
Le calcul ne parcourt pas les cartes individuelles lorsque la source est déjà un `CardCounts`.
Pour les zones visibles, seules les longueurs de tuples sont lues.

La latence doit néanmoins être mesurée séparément avant/après, car l'ajout de sept entrées change
les multiplications de la première couche du `state_encoder`.

## Edge Cases

- zone vide : valeur normalisée nulle ;
- comptes absents d'un dataset ancien : aucun impact, puisque les cardinalités sont dérivées des
  champs déjà présents ;
- valeur supérieure à la borne : clipping à 1 ;
- valeur négative impossible dans une observation valide : protection par `max(count, 0)` ;
- observation adverse masquée : aucune tentative d'accès à `hand` ou `draw_pile_counts` adverse ;
- configuration inconnue : `ValueError` explicite à la construction du modèle.

## Testing Strategy

- vérifier la validation des deux feature sets et le rejet d'un nom inconnu ;
- vérifier que baseline conserve la dimension historique du `state_encoder` ;
- vérifier que `zone_cardinality_v1` ajoute exactement sept entrées ;
- vérifier que deux observations au même pooling mais de cardinalités différentes produisent des
  scalaires différents ;
- vérifier les valeurs vides, les bornes et le clipping ;
- vérifier que les cardinalités adverses n'utilisent aucune zone cachée ;
- vérifier la désérialisation d'une observation historique et le forward complet ;
- vérifier des scores finis pour toutes les actions légales ;
- exécuter les tests neural et observation ciblés, puis la suite complète si le temps le permet.

## Rollout And Migration

L'implémentation ne modifie aucun profil actif ni checkpoint stable. Une future candidate doit
déclarer `observation_feature_set: zone_cardinality_v1` dans son profil, produire un checkpoint
mutable séparé dans le chemin canonique défini par le Makefile et être comparée à V004 avec les
mêmes dataset, split, seeds et panel.

Un résultat offline ou un screening court ne suffit pas à promouvoir la candidate.

## Files Expected To Change

- `shards_ai/ai/neural_model.py` : configuration, validation et encodage des scalaires ;
- `tests/ai/test_neural_model.py` : dimensions, cardinalités et compatibilité ;
- `tests/game/test_neural_observation.py` : uniquement si un scénario de masque supplémentaire est
  nécessaire ;
- `doc/TODO/001-cardinalites-observation-neural.md` : périmètre corrigé ;
- profil candidat neural ultérieur, après validation technique.
