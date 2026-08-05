# Apprentissage ciblé des choix de mercenaires

## Objective

Donner davantage de signal aux décisions où le même mercenaire peut être recruté immédiatement ou
acheté pour le long terme, sans surpondérer tous les achats ni utiliser le dataset pruné comme seule
distribution d'entraînement.

## Target Behavior

Un enregistrement est ciblé si ses actions légales contiennent à la fois `BuyCard` et
`RecruitMercenary` pour le même `card_definition_id`. La loss de cet enregistrement reçoit un poids
configurable ; les autres décisions gardent le poids naturel.

## Key Decisions

- L'expérience repart de `configs/neural_profiles/v001.pt` pour isoler l'effet du ciblage.
- Le dataset naturel 1M est utilisé pour l'entraînement.
- La validation naturelle est fournie séparément.
- Le poids ciblé par défaut reste `1.0`; l'expérience initiale utilise `3.0`.
- Le ciblage apprend les préférences de v008 dans ces situations ; il ne constitue pas encore une
  exploration contrefactuelle supérieure au teacher.

## Non-Goals

- forcer le recrutement immédiat ;
- supprimer les achats de mercenaires ;
- modifier les actions légales ou le checkpoint stable ;
- conclure à partir des seules métriques offline.

## Testing Strategy

- vérifier la détection d'une paire immédiat/long terme sur le même mercenaire ;
- vérifier qu'une décision sans paire n'est pas ciblée ;
- vérifier que le poids est enregistré dans la configuration effective ;
- benchmarker le candidat contre Random, v007 et v008 après l'epoch ciblée.
