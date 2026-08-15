# Gate de promotion Neural — panel Hybrid V003

## Objective

Faire évoluer le panel de promotion neural pour mesurer les candidates contre les profils
heuristiques, neural et hybrides actuellement jugés représentatifs.

## Current State

`scripts/validate_neural_profile.py` compare une candidate au profil neural actif et exécute un
panel fixe composé uniquement de `HeuristicPlayer` et `NeuralPlayer`. Les profils hybrides
versionnés existent dans `configs/hybrid_profiles/`, mais ne sont pas utilisables par ce validator.

## Target Behavior

Le panel qualité devient :

| Adversaire | Poids |
|---|---:|
| `v008` | 1,5 |
| `hybrid:v003` | 1,5 |
| `v007` | 1,0 |
| `neural:v005` | 1,0 |
| `neural:v006` | 1,0 |
| `hybrid:v002` | 0,5 |
| `neural:v002` | 0,5 |

Le total est `7,0`. La décision reste l'acceptation d'une moyenne pondérée strictement positive
des deltas de taux de victoire candidat moins référence active. Aucun adversaire n'est une garde
individuelle de non-régression. `Random`, Neural V1/V3/V4 et les autres hybrides restent hors gate.

## Key Decisions

- Les identifiants de panel sont `hybrid:v002` et `hybrid:v003`.
- Le validator charge chaque profil hybride exactement via `load_hybrid_profile()` et construit le
  joueur via `build_hybrid_player()`.
- Candidate et référence utilisent les mêmes seeds par adversaire, comme pour le panel existant.
- Le panel et ses poids sont définis dans le code central du validator ; la validation batch le
  réutilise.
- Les profils hybrides et checkpoints référencés restent immuables pendant une campagne.

## Open Questions

Aucune question bloquante pour cette évolution. La taille de campagne reste celle du protocole
existant (200 parties par adversaire dans `meta_improve.py`).

## Proposed Architecture

`_panel()` retourne les profils heuristiques, les profils neural et les profils hybrides. `_play()`
sélectionne l'implémentation d'adversaire à partir du préfixe d'identifiant. Le calcul des deltas et
la promotion restent inchangés.

## Testing Strategy

- Vérifier la composition exacte du panel et les poids.
- Vérifier que les deux profils hybrides sont chargés et jouables dans une partie déterministe.
- Vérifier le calcul pondéré et le rejet d'un panel incomplet.
- Exécuter les tests ciblés de validation et le formatage de diff.

## Rollout And Migration

Mettre à jour le validator, le validator batch, les paramètres de campagne et les tests. Aucun
checkpoint, profil historique ou moteur de jeu n'est modifié.

## Files Expected To Change

- `scripts/validate_neural_profile.py`
- `scripts/validate_macro_neural_profile.py`
- `configs/meta_improvement.yaml`
- `scripts/meta_improve.py`
- `tests/ai/test_neural_training_profiles.py`
