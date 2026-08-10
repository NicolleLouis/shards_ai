# Ablation et pondération des voies identité et sémantique des cartes

## Objectif

Mesurer si la voie sémantique structurée de V4 apporte une politique utile
sans dépendre de l'identifiant exact de la carte, puis tester une pondération contrôlée des deux
voies. L'expérience est une campagne de recherche : elle ne modifie pas le moteur, ne remplace pas
`v002` et ne peut pas promouvoir un checkpoint sur un gain offline seul.

## État courant

`shards_ai/ai/structured_v004.py` calcule un embedding d'identité et un embedding sémantique, puis
les concatène avant `card_fusion`. La concaténation est le point d'injection de l'expérience. La
logique expérimentale sera portée par un scorer V5 dédié qui réutilise l'encodeur V4 ;
`StructuredSemanticV4Scorer`, ses poids et son comportement restent inchangés.

Le TODO associé est `doc/TODO/002-ablation-ponderation-identite-semantique.md`. Les changements
locaux déjà présents dans le dépôt sont hors périmètre de cette architecture et doivent être
préservés.

## Comportement cible

L'étape A compare uniquement :

| Candidate | Identité | Sémantique | But |
| --- | ---: | ---: | --- |
| contrôle | active | active | référence complète |
| sans identité | zéro | active | tester la sémantique seule |

L'étape B compare ensuite chaque pondération à un contrôle `1,0 / 1,0`, une variation à la fois :

- préférence sémantique : `id_scale=0,5`, `semantic_scale=1,5` ;
- préférence identité : `id_scale=1,5`, `semantic_scale=0,5`.

La candidate « sans sémantique » n'appartient pas à cette campagne. Elle pourra être ajoutée plus
tard comme diagnostic distinct, sans modifier l'interprétation de cette expérience.

Le contrôle expérimental V5 est distinct de la référence V4 brute : il utilise le scorer dédié avec
normalisation L2 et des scales `1,0 / 1,0`. La référence V4 brute conserve la fusion actuelle sans
normalisation et sert à mesurer le coût de ce changement de protocole, pas à remplacer le contrôle
de l'ablation.

## Hors périmètre

- Modifier les règles, le catalogue, l'observation partielle ou le masque d'information.
- Modifier V001, V002, le profil actif ou les checkpoints stables.
- Mélanger les scales avec un changement de dataset, de loss, de features ou de contexte d'actions.
- Convertir un checkpoint complet en ablation par masquage uniquement à l'inférence.
- Promouvoir un candidat avant la gate longue existante.

## Décisions clés

1. L'ablation est implémentée dans un scorer V5 dédié nommé
   `StructuredSemanticV5FusionScorer`, à l'entrée de `card_fusion`. Il réutilise
   `StructuredSemanticCardEncoder`, l'encodage des actions et les contrats d'observation de V4,
   sans modifier `StructuredSemanticV4Scorer`.
2. Les deux vecteurs sont normalisés séparément par norme L2 avec un epsilon explicite, puis
   multipliés par leurs scales. Le choix exact d'epsilon et de la dimension de normalisation est
   enregistré dans le profil et les métadonnées du checkpoint.
3. Pour l'étape A, la voie identité est remplacée par un tenseur nul de même forme ; la voie
   sémantique, les vocabulaires et le catalogue restent inchangés.
4. Chaque candidate est entraînée proprement avec la même initialisation contrôlée, dataset,
   hash, split `game_id`, seed, teacher, schedule, optimiser et nombre d'epochs que son contrôle.
5. Le contrôle et les candidates de l'expérience utilisent l'architecture explicite
   `structured_semantic_v5_fusion_experiment`. La référence `structured_semantic_v4` continue de
   suivre son chemin de fusion historique.
6. Les scales sont des paramètres de configuration validés positifs. Ils ne sont jamais déduits du
   nom de fichier et ne s'appliquent qu'au scorer expérimental.
7. Les checkpoints portent l'architecture, les scales, le mode d'ablation, la normalisation et
   l'empreinte de protocole. Un chargeur qui manque ces métadonnées refuse la campagne plutôt que
   d'inventer des valeurs.
8. L'étape A attribue la valeur pratique de la sémantique seule ; elle ne prétend pas mesurer la
   contribution causale complète de l'identité.

## Architecture proposée

Le chemin expérimental devient conceptuellement :

```text
card_id -> id_embedding -> normalise -> id_scale --------┐
                                                         ├─ concat -> card_fusion -> card_embedding
structure -> semantic_encoder -> normalise -> semantic_scale ┘
```

`StructuredSemanticV5FusionScorer` reçoit les mêmes représentations que V4 et applique sa
propre fusion. Le mode `without_identity` produit un zéro différentiable de la forme du vecteur
d'identité ; il ne supprime ni l'index `card_id`, ni le vocabulaire, ni la représentation
structurée. Le chemin d'entraînement reste différentiable et le chemin d'inférence conserve son
cache compatible avec les poids courants.

`StructuredSemanticV4Scorer` ne passe jamais par la fusion expérimentale. Les architectures
historiques ne passent jamais par l'un ou l'autre chemin V4. `build_neural_scorer` et le loader
sélectionnent explicitement `structured_semantic_v4` ou
`structured_semantic_v5_fusion_experiment` selon le checkpoint.

## Configuration et artefacts

Le profil candidat doit déclarer notamment :

```yaml
architecture: structured_semantic_v5_fusion_experiment
card_fusion_id_scale: 1.0
card_fusion_semantic_scale: 1.0
card_fusion_ablation: none
card_fusion_normalization: l2
card_fusion_normalization_epsilon: 1.0e-8
```

Les variantes d'étape A et B utilisent des profils candidats séparés et le checkpoint de travail
canonique défini par le `Makefile`. Aucun checkpoint nommé par expérience ne doit être ajouté dans
`artifacts/`. Les rapports détaillés restent dans `doc/Experiments/`, tandis que cette architecture
reste historique une fois la conception terminée.

## Tests et validation

- vérifier que `structured_semantic_v4` conserve exactement son chemin de fusion, ses scores finis
  et l'équivalence à l'ordre des actions ;
- vérifier que `structured_semantic_v5_fusion_experiment` charge un scorer distinct avec des
  métadonnées explicites ;
- vérifier que `without_identity` produit bien un zéro à l'entrée de `card_fusion` et conserve la
  dimension, les gradients de la voie sémantique et les vocabulaires ;
- vérifier les scales, l'epsilon et les modes invalides de configuration ;
- vérifier que V001 et V002 chargent et scorent comme avant ;
- vérifier la sauvegarde/restauration des métadonnées de campagne ;
- effectuer un smoke test sur un enregistrement avant toute campagne longue ;
- comparer offline puis en parties avec les mêmes seeds, le même split et le même panel Random,
  v007, v008 et profils neural requis.

Les mesures minimales sont top-1/top-3, regret teacher, divergence d'argmax, accord au contrôle,
résultats par phase et type d'action, puis victoires/défaites/nuls, deck final, nombre de décisions,
tours et latence. Un résultat offline ou contre un seul adversaire ne suffit pas.

## Critère de décision et rollout

L'étape A est diagnostique et ne promeut rien. L'étape B ne peut être candidate à la promotion que
si elle passe la gate longue en vigueur, sans régression contre v008 et avec une amélioration
agrégée robuste. `configs/neural_profiles/active.yaml` reste inchangé pendant toute la campagne.

## Questions ouvertes

- Non bloquant avant implémentation : confirmer les noms définitifs des champs de profil avec le
  schéma actuel de `NeuralModelConfig`.
- Non bloquant : décider si les rapports doivent comparer une ou plusieurs seeds avant le smoke
  test ; une seule seed ne pourra jamais constituer une preuve de promotion.

## Fichiers attendus

- `shards_ai/ai/neural_model.py` ou le module V4 actuel : configuration et factory ;
- `shards_ai/ai/structured_v004.py` : encodeur partagé et scorer V4 inchangé ;
- nouveau module V5 expérimental : `StructuredSemanticV5FusionScorer` et sa fusion ;
- `configs/neural_training_profiles/candidates/` : profils isolés des candidates ;
- tests IA V4 et tests de restauration de checkpoint ;
- scripts de rapport/benchmark si les métadonnées ne sont pas déjà propagées.
