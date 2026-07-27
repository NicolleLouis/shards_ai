# Insights des décisions du NeuralPlayer

## Objective

Compléter le rapport de campagne du NeuralPlayer avec des statistiques explicatives sur deux
décisions importantes : le choix entre recruter immédiatement un mercenaire et l'acheter pour le
deck durable, ainsi que l'activation de `GainMastery` (1 Gem pour 1 maîtrise).

## Key Decisions

- `RecruitMercenary` est compté comme utilisation immédiate ; le moteur le joue puis le remet dans
  le deck central au cleanup.
- `BuyCard` d'une carte mercenaire est compté comme achat long terme ; la carte rejoint les zones
  possédées du joueur.
- Les statistiques mercenaires sont groupées par `card_definition_id`.
- Pour `GainMastery`, le benchmark enregistre toutes les opportunités où l'action est légale, puis
  distingue les activations effectives. Le taux est donc calculé par tour et par maîtrise courante,
  et non uniquement comme un nombre brut d'activations.
- Ces événements sont capturés uniquement lorsque le joueur actif est le NeuralPlayer.

## Metrics

Par mercenaire : utilisations immédiates, achats durables, total et proportion immédiate/durable.

Pour la maîtrise : opportunités, activations, taux d'activation, ventilés par `turn_number` et
`mastery` avant décision. Le rapport HTML affiche les deux ventilations et le JSON conserve les
détails agrégés.

## Data Flow

Le `decision_observer` du `GameRunner` reçoit l'observation, la liste d'actions légales et l'action
choisie avant application. Il ne capture que les décisions du joueur neural. Le mercenaire est
résolu depuis la rivière publique ; la maîtrise est lue dans `NeuralObservation.active_player`.

## Edge Cases

- une carte mercenaire achetée avec `BuyCard` est classée long terme ;
- une action `BuyCard` non mercenaire n'est pas incluse dans les statistiques mercenaires ;
- une décision sans `GainMastery` légal n'est pas une opportunité et n'entre pas au dénominateur ;
- une partie interrompue conserve ses événements déjà observés ;
- une absence d'événement produit des tableaux vides explicites.

## Testing Strategy

- test de classification `RecruitMercenary`/`BuyCard` par carte ;
- test d'enregistrement d'une opportunité et d'une activation de maîtrise ;
- test d'agrégation par mercenaire, tour et maîtrise ;
- test de présence des tableaux dans le JSON et le HTML ;
- suite complète du moteur et du joueur neural.
