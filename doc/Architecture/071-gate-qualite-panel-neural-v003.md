# Gate de qualité avec panel neural v003

## Objective

Faire évaluer chaque nouveau profil neural contre les trois références neural les plus récentes,
dont `v003` dans le panel courant, tout en conservant un poids total de `1` pour cet ensemble.
Chaque profil neural contribue donc pour `1/3` à la moyenne qualité.

## Current State

La gate est centralisée dans `scripts/validate_neural_profile.py` et réutilisée par la validation
batchée et `scripts/meta_improve.py`. Le panel dynamique sélectionne actuellement au plus deux
checkpoints neural promus. Les deltas neural sont regroupés dans `neural_group` avec un poids de
`0,5`, tandis que Random, v007 et v008 ont respectivement les poids `0,5`, `1` et `2`.

## Target Behavior

- Le panel sélectionne trois profils neural promus, hors candidat, par ordre de version décroissante.
- Dans l'état courant, cela donne `v003`, `v002` et `v001` pour une nouvelle candidate.
- `v008` reste une garde dure de non-régression.
- Les poids appliqués sont explicites : `neural:v001 = 1/3`, `neural:v002 = 1/3`,
  `neural:v003 = 1/3`.
- Le poids total du groupe neural reste exactement `1`; les poids des autres adversaires ne changent pas.
- La validation batchée et `meta-improve` utilisent la même fonction de décision.

## Non-Goals

- Ne pas modifier les checkpoints, les profils actifs, le moteur, les heuristiques ou le masque d'information.
- Ne pas changer la garde v008, les poids Random/v007/v008 ou la règle de moyenne strictement positive.
- Ne pas agréger les victoires entre profils avant le calcul des deltas par adversaire.

## Key Decisions

- Remplacer le plafond de deux références neural par trois.
- Exposer les poids individuels dans `opponent_weights` afin de rendre la pondération vérifiable dans les rapports.
- Refuser implicitement un panel neural incomplet dans les tests de gate ; la validation d'un dépôt sans trois checkpoints reste diagnostique et doit signaler les profils absents plutôt que renormaliser silencieusement.

## Open Questions

- Non bloquant : après plusieurs promotions, faut-il conserver les trois derniers profils ou figer explicitement
  `v001`, `v002`, `v003` ? Cette modification conserve le comportement dynamique des trois derniers profils.

## Proposed Architecture

`_panel()` charge les trois derniers profils versionnés disponibles hors candidat. `acceptance_metrics()`
traite chaque clé `neural:<profile_id>` comme un adversaire indépendant, applique `1/3`, puis calcule la
moyenne pondérée globale avec les poids existants. La validation normale et batchée restent des clients
de cette même fonction.

## Observability And Operations

Les rapports doivent conserver la liste des adversaires et `opponent_weights`. Une promotion est refusée
si le panel requis n'est pas disponible ; les fichiers de progression batchée incluent la configuration
du panel pour empêcher une reprise avec une pondération différente.

## Testing Strategy

- Tester la sélection de trois profils neural dans le panel.
- Tester les poids individuels `1/3` et leur somme `1`.
- Tester que le résultat change correctement lorsqu'un des trois deltas neural régresse.
- Exécuter les tests des validateurs et de `meta-improve` concernés.

## Files Expected To Change

- `scripts/validate_neural_profile.py`
- `scripts/validate_neural_profile_batched.py` via la fonction partagée et le panel
- `tests/ai/test_neural_training_profiles.py`
- `doc/Current state/Neural player.md`
- `README.md`
