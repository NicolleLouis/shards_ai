# Gate de promotion Neural — panel V006

## Objectif

Réduire le gate de promotion au panel de référence jugé pertinent après la promotion de Neural V006.
La décision reste fondée sur une moyenne pondérée strictement positive des deltas candidat contre
la référence active.

## État courant

Le gate précédent incluait V007, V008, Neural V001 à V006 et excluait déjà Random. Neural V003
était encore une référence obligatoire, malgré son poids faible de `0,25`. Le code centralisant ce
contrat est `scripts/validate_neural_profile.py`, réutilisé par la validation batch et les tests de
profils.

## Décision cible

Le panel de qualité devient :

| Adversaire | Poids | Ratio normalisé |
|---|---:|---:|
| V008 | 2,0 | 30,77 % |
| V007 | 1,0 | 15,38 % |
| Neural V005 | 1,0 | 15,38 % |
| Neural V006 | 1,0 | 15,38 % |
| Neural V001 | 0,5 | 7,69 % |
| Neural V002 | 0,5 | 7,69 % |
| Neural V004 | 0,5 | 7,69 % |

Le total des poids est `6,5`. Neural V003 et Random ne sont pas des adversaires du gate. Ils
restent utilisables dans les benchmarks ou expériences explicitement diagnostiques.

## Compatibilité et non-objectifs

- Les checkpoints V001 à V006 ne sont pas supprimés.
- Les rapports de validation historiques ne sont pas réécrits.
- Les profils d'entraînement historiques ne sont pas mutés pour changer rétroactivement leur panel.
- Le benchmark neural général peut continuer à inclure Random et Neural V003.
- La règle de décision reste une moyenne pondérée positive ; aucune contrainte dure par adversaire
  n'est ajoutée.

## Impact d'implémentation

`QUALITY_OPPONENT_WEIGHTS` et `NEURAL_REFERENCE_PROFILE_IDS` deviennent la source unique du panel
de validation. Les tests doivent vérifier l'absence de V003 dans le panel, la présence de V001,
V002, V004, V005 et V006, les poids exacts et le ratio pondéré.

La documentation current state et le README doivent décrire ce panel. Aucun changement moteur,
joueur, checkpoint ou runtime n'est requis.

## Validation

Exécuter les tests de profils et de validation PPO, puis `git diff --check`. Une prochaine candidate
devra être évaluée avec les mêmes adversaires, poids et seeds pour rester comparable.
