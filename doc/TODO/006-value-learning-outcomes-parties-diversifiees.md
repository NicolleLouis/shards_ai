# TODO — Value learning sur des parties diversifiées

## Statut

Idée à évaluer. Aucune implémentation ni collecte dédiée n'est engagée.

Cette note décrit une évolution de l'entraînement. Si elle implique une nouvelle tête de valeur,
une nouvelle loss ou une modification du contrat macro/atomique, écrire une architecture dédiée
avant l'implémentation.

## Hypothèse

Apprendre une valeur de position `V(état)` à partir du résultat final de nombreuses parties peut
fournir un signal plus général que l'imitation directe d'un seul teacher (`v008`). Une extension
`Q(état, action)` pourrait ensuite aider à choisir entre les candidats macro et atomiques, mais
nécessiterait d'observer des actions alternatives ou d'utiliser une recherche/du bootstrapping.

## Données nécessaires

Collecter des trajectoires de parties complètes provenant de politiques diversifiées :

- `v008`, `v007` et le neural actuel ;
- politiques exploratoires ou aléatoires pour élargir la couverture des états ;
- plusieurs adversaires, matchups, sièges et seeds ;
- provenance du joueur, observation masquée, action légale, phase, tour et outcome final.

Le résultat doit être exprimé du point de vue du joueur actif ou du joueur évalué. L'observation
ne doit jamais contenir le résultat futur, le profil adverse ou une information cachée.

## Première variante recommandée : `V(état)`

Commencer par une tête qui estime la valeur d'un état visité :

```text
victoire = 1.0
nul      = 0.5
défaite  = 0.0
```

Réutiliser l'encodeur d'observation macro/atomique unifié, sans prétendre obtenir de valeur
contrefactuelle pour les actions non jouées. La sortie devra être évaluée comme une estimation
calibrée, pas seulement comme un score de classement.

## Variante ultérieure : `Q(état, action)`

Associer une valeur à chaque couple état/action. Ne pas considérer le résultat de la seule action
choisie comme une preuve de la valeur des autres actions légales : les contrefactuels sont absents.
Une expérimentation `Q` devra donc spécifier une stratégie d'exploration, de recherche ou de
bootstrapping et conserver séparément les actions choisies et les actions alternatives.

## Protocole

1. Conserver l'imitation V4 comme contrôle.
2. Générer un dataset de ligue avec plusieurs politiques et des splits stricts par `game_id`.
3. Garder un holdout indépendant, non utilisé pour choisir les epochs ou les hyperparamètres.
4. Entraîner d'abord `V(état)` offline avec plusieurs seeds.
5. Mesurer Brier score, ECE, calibration par phase/action/matchup et win-rate en parties complètes.
6. Comparer à l'imitation V4 et à `v004` sur les mêmes seeds et le panel complet.

Les résultats du benchmark de promotion ne doivent pas être réinjectés directement dans le
dataset d'entraînement.

## Risques et critères d'arrêt

Risques principaux : biais de distribution vers un joueur, récompense terminale bruitée, fuite
entre parties, états rares sous-représentés et confusion entre corrélation et causalité d'une
action.

Arrêter ou revoir le protocole si une politique domine le volume, si une slice importante disparaît,
si la calibration se dégrade fortement ou si le gain offline n'est pas confirmé en parties
complètes. Une amélioration de Brier/ECE seule ne justifie pas une promotion.
