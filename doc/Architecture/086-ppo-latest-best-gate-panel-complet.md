# PPO macro avec états latest/best et panel de promotion complet

## Objectif

Permettre au training PPO macro de continuer à explorer après une évaluation défavorable, tout en conservant le meilleur état connu selon le gate de promotion pondéré. L’évaluation doit utiliser le même panel et les mêmes poids que le gate qualité.

## Décisions

- Le fichier physique mutable reste unique : `artifacts/neural_training/checkpoint.pt`.
- Ce fichier contient deux états logiques : `latest` (modèle/optimizer courant) et `best` (modèle/optimizer du meilleur score).
- Une évaluation négative ne restaure plus le modèle courant.
- `best` est remplacé uniquement si le taux de victoire pondéré strictement progresse.
- Il n’existe plus de contrainte de non-régression par adversaire dans le gate PPO.
- Le panel d’évaluation et d’entraînement est : V7, V8, Neural V1 à V5. Random est exclu du gate et du mélange PPO macro.
- Les poids sont ceux du gate : V7 1,5 ; V8 2 ; V1/V2 0,5 ; V3/V4 0,25 ; V5 1, soit 6 au total.
- L’évaluation utilise 100 parties par adversaire, donc 700 parties par validation. L’intervalle de validation est aligné sur le prochain bloc PPO complet, soit 768 parties avec `games_per_update=128`.

## Architecture

Le checkpoint latest conserve les champs historiques de l’actor courant ainsi que des champs `best_*` sérialisés. Une reprise standard charge `latest`; une reprise explicite peut sélectionner `best`. Les compteurs et l’optimizer associés au meilleur état sont conservés pour permettre une reprise cohérente.

Le gate interne appelle le score pondéré commun aux sept adversaires. Le score candidat doit être strictement supérieur au score best; les taux individuels peuvent régresser. Cette décision est un mécanisme de sélection de checkpoint, pas une promotion stable : la promotion officielle reste réalisée par le validator dédié avec son panel complet.

## Risques et limites

700 parties rendent chaque validation coûteuse mais réduisent le bruit d’une évaluation à 10 parties. Le training peut poursuivre une mauvaise trajectoire entre deux validations; `latest` est donc diagnostique et `best` est l’état à retenir pour une validation officielle. Les parties d’évaluation restent un échantillon et ne remplacent pas plusieurs seeds indépendants.

## Validation

Tester le chargement latest/best, la conservation de `best` après une validation négative, l’acceptation d’une progression pondérée malgré une régression individuelle, le panel de sept adversaires, la normalisation des poids et l’absence de rollback du modèle courant.
