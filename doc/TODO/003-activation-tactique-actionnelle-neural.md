# TODO — Expérience 3 : activation tactique des synergies par action

## Statut

À implémenter séparément du calcul global des cardinalités et de la composition. Cette expérience est prioritaire
si l'objectif est d'améliorer les erreurs de choix pendant `PLAY`.

## Question

Le réseau sait-il qu'une carte candidate active réellement Union, Echo ou Domination dans l'état
courant, plutôt que de connaître uniquement les prérequis déclarés par la carte ?

## Hypothèse falsifiable

Des features calculées pour chaque couple `(observation, action)` améliorent le classement des cartes
jouables conditionnelles, sans propager la dérive vers les décisions non concernées.

## Décision de conception principale

Ces informations ne doivent pas être ajoutées comme scalaires d'état communs à toutes les actions.
Elles doivent être ajoutées à l'encodage de l'action, car leur valeur dépend de la carte candidate.

Le modèle peut dériver, sans modifier `Game.legal_actions()` ni `Game.apply()` :

- `union_active_for_candidate` ;
- `echo_active` ;
- `domination_active_for_candidate` ;
- éventuellement le nombre de factions manquantes pour Domination.

Les règles de calcul doivent reproduire les zones utilisées par le moteur. La carte candidate ne
doit jamais compter comme son propre allié. Les champions joués ce tour sont pris via
`played_champion_faction_mask`; les champions seulement activés sont exclus.

## Réutilisation de l'existant

La représentation sémantique contient déjà les propriétés `requires_union`, `requires_echo` et
`requires_domination`. L'expérience ajoute l'état d'activation réelle ; elle ne duplique pas ces
propriétés ni ne prétend les remplacer.

## Décisions à prendre dans l'architecture

Avant le code, documenter :

1. les zones exactes utilisées pour chaque condition ;
2. le traitement des cartes neutres et des champions ;
3. le calcul pour les actions qui ne sont pas `play_card` ;
4. le format de la feature actionnelle et sa compatibilité dataset ;
5. la nouvelle version d'architecture/checkpoint ;
6. le budget de dérive d'argmax hors des actions conditionnelles.

## Tests techniques

- Union ne compte pas la carte candidate elle-même.
- Echo ne dépend que de la défausse prévue par la règle.
- Domination distingue précisément les trois factions requises.
- Un champion activé ce tour est traité comme le moteur le traite.
- Deux actions candidates différentes peuvent produire des features différentes dans le même état.
- Les actions non concernées reçoivent des valeurs neutres et finies.
- Le forward reste valide pour toutes les actions légales sérialisées.

## Évaluation

Évaluer en priorité les `play_card` avec opérations conditionnelles, puis contrôler les slices
protégées : achats, bannissements, recrutement et passages de phase. Mesurer accord teacher,
regret, dérive d'argmax par slice, calibration et résultats en partie appariés sur les mêmes seeds.

Une amélioration offline sur les cartes conditionnelles ne suffit pas : il faut une amélioration ou
une non-régression du panel complet avec une dérive hors slice sous le budget annoncé.
