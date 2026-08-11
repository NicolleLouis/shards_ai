# TODO — Expérience 4 : regroupement conditionnel des nouvelles features

## Statut

Terminée côté intégration technique et tests. Le regroupement cardinalités + composition +
activation tactique est présent dans le pipeline V4 (`deck_state_v1` et encodeur tactique). Les
comparaisons d'ablation et la validation de qualité restent différées conformément à
l'architecture 081.

## Objectif

Déterminer si plusieurs familles de features apportent une information complémentaire, ou si leur
combinaison augmente seulement la dimension, le coût et le risque de surapprentissage.

## Conditions d'entrée

Une expérience précédente ne peut entrer dans le regroupement que si elle possède :

- un résultat holdout indépendant ;
- une slice de gain ou de non-régression identifiable ;
- des tests de masque et de sérialisation passants ;
- une mesure de dérive d'argmax ;
- un screening en partie sans régression manifeste.

Une famille rejetée ou inconclusive ne doit pas être incluse par défaut dans le regroupement.

## Variantes autorisées

Tester une combinaison à la fois, dans cet ordre :

1. cardinalité + composition factionnelle ;
2. cardinalité + activation tactique ;
3. composition factionnelle + activation tactique ;
4. combinaison complète uniquement si les trois précédentes restent justifiées.

Chaque variante conserve un ablateur sans la famille supposée complémentaire. Le contrôle est le
meilleur candidat individuel comparable, pas une nouvelle recette choisie après observation des
résultats.

## Critères de challenge

Le regroupement est refusé si :

- le gain individuel ne se retrouve pas dans la combinaison ;
- une feature devient redondante sans gain mesurable ;
- la latence augmente matériellement sans gain en partie ;
- la dérive hors slice dépasse le budget ;
- le résultat dépend d'une seule seed, d'un seul adversaire ou d'un screening court.

Une combinaison peut être conservée comme candidate de recherche même sans promotion, mais son
rapport doit expliquer quelle complémentarité était attendue et pourquoi elle n'a pas été démontrée.

## Protocole final

Utiliser le même protocole de données, splits, seeds et teacher que les expériences parentes.
Comparer sur holdout puis sur le panel complet, avec mesures par phase, action, cardinalité légale,
taille de deck, composition factionnelle et présence de synergies conditionnelles.

La promotion suit exclusivement la gate panel courante et son agrégation pondérée. Le profil actif,
les références historiques et le checkpoint stable restent inchangés jusqu'à la décision finale.

## Architecture obligatoire

Une nouvelle architecture numérotée doit être créée avant toute implémentation d'une combinaison.
Elle doit récapituler les feature sets retenus, leur ordre, leurs normalisations, les métadonnées de
checkpoint et la stratégie de migration ou de réentraînement.
