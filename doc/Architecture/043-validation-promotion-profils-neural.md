# Validation et promotion des profils neural

## Objective

Comparer un checkpoint neural candidat au profil neural actif sur un panel reproductible
d'adversaires, publier le candidat uniquement s'il ne régresse contre aucun adversaire et progresse
contre au moins un, puis le rendre explicitement actif pour les futurs `NeuralPlayer` sans
checkpoint fourni.

## Current State

Les profils d'entraînement sont versionnés sous `configs/neural_training_profiles/`, mais il n'existe
pas encore de promotion automatique. `NeuralPlayer` exige un chemin de checkpoint et le benchmark
neural mixte ne compare pas deux checkpoints sur les mêmes seeds.

Les adversaires disponibles sont `RandomPlayer`, les profils heuristiques `v007` et `v008`, et les
checkpoints neural produits par les profils existants. Les artefacts sont volumineux et restent hors
Git.

## Target Behavior

`scripts/validate_neural_profile.py` reçoit un profil candidat et son checkpoint, charge le profil
neural actif comme référence, découvre au plus les deux derniers profils neural antérieurs dont le
checkpoint existe, puis joue chaque adversaire sur les mêmes seeds pour le candidat et la référence.

Le rapport de sortie détaille, par adversaire, le nombre de parties, victoires, défaites, nuls et
taux de victoire pour les deux modèles, ainsi que le delta et la décision. La validation réussit si
le taux candidat est supérieur ou égal à la référence pour chaque adversaire et strictement
supérieur pour au moins un adversaire. En cas de succès, le profil reçoit le prochain numéro libre,
est copié dans `configs/neural_training_profiles/` et `active.yaml` est mis à jour. En cas d'échec,
aucun fichier de profil n'est ajouté ni modifié.

## Non-Goals

- sélectionner un candidat sur une métrique autre que le taux de victoire ;
- promouvoir automatiquement un checkpoint sans rapport complet ;
- supprimer ou réécrire un profil historique ;
- versionner les checkpoints ou résultats JSON ;
- introduire une décision statistique différente de la règle explicite de non-régression demandée.

## Key Decisions

- Les comparaisons utilisent les mêmes seeds, le même nombre de parties et les mêmes limites de
  moteur pour le candidat et la référence.
- Le profil actif est un petit pointeur committable `active.yaml` contenant `active_profile_id`.
  Il pointe vers un fichier versionné `vNNN.yaml` et ne duplique pas sa configuration.
- La découverte des adversaires neural exclut le candidat et ne retient que les profils dont le
  checkpoint référencé existe. Les profils sont triés par version décroissante et limités à deux.
- Les adversaires heuristiques v007 et v008 doivent exister ; leur absence est une erreur de
  configuration, car ils font partie du panel contractuel.
- La promotion est atomique au niveau des fichiers : le profil versionné est écrit avant le pointeur
  actif. Le pointeur n'est mis à jour que si l'écriture du profil réussit.
- `NeuralPlayer(checkpoint_path=None)` charge le profil actif et son checkpoint. Un checkpoint absent
  provoque une erreur explicite plutôt qu'un fallback silencieux.
- Le script retourne un code non nul en cas de rejet ou d'erreur, et un code nul uniquement après
  validation et promotion réussies.

## Open Questions

- Non-blocking: ajouter plus tard des intervalles de confiance ou un test apparié ; la première
  version applique volontairement la règle de taux observé demandée.
- Non-blocking: le panel pourra être déclaré dans un profil de validation RL lorsque PPO sera
  implémenté.

## Proposed Architecture

Ajouter dans `neural_training_profiles.py` le chargement du pointeur actif, la découverte des profils
versionnés et l'attribution du prochain identifiant. Ajouter un module de campagne de validation qui
construit les joueurs avec un scorer neural partagé par campagne et exécute les deux checkpoints sur
les mêmes seeds.

Le script sérialise un rapport JSON optionnel et imprime toujours un résumé lisible. Il ne modifie le
profil candidat fourni hors du dossier canonique ; à la promotion, il crée une copie normalisée avec
le nouvel identifiant et le parent actif.

## Data Model

`configs/neural_training_profiles/active.yaml` :

```yaml
schema_version: 1
active_profile_id: v001
```

Le profil promu reprend la recette candidate, avec `profile_id` et `parent_profile_id` normalisés.
Son champ `output` référence le checkpoint évalué afin que le chargement par défaut soit
reproductible.

## Observability And Operations

Le rapport conserve la référence, le candidat, les seeds, les adversaires sélectionnés, les chemins
de checkpoints, les scores par catégorie et la décision. Il reste sous `artifacts/` et est ignoré
par Git. La console affiche une ligne par adversaire puis la décision finale.

## Edge Cases

- candidat identique à la référence : rejet, aucune progression stricte ;
- candidat inférieur sur une seule catégorie : rejet global ;
- égalité sur toutes les catégories : rejet ;
- profil neural récent sans checkpoint : ignoré dans la liste des adversaires ;
- aucun profil neural antérieur disponible : validation avec zéro ou un adversaire neural selon les
  checkpoints présents ;
- checkpoint incompatible avec le vocabulaire ou le modèle : erreur avant promotion ;
- numéro de profil déjà utilisé : calcul du prochain numéro à partir de tous les YAML présents.

## Testing Strategy

Tester la sélection des profils neural, la règle de décision, la génération du prochain identifiant,
le chargement du pointeur actif et le refus de promotion. Tester `NeuralPlayer` avec `None` sur un
profil actif temporaire. Les campagnes réelles restent des validations d'intégration contrôlées par
le nombre de parties.

## Rollout And Migration

Créer `active.yaml` pointant vers `v001`. Les appels existants qui fournissent un checkpoint restent
compatibles. Le premier candidat validé crée `v002.yaml` et met à jour `active.yaml` ; un candidat
rejeté ne laisse aucun nouveau profil.

## Files Expected To Change

- `configs/neural_training_profiles/active.yaml`
- `shards_ai/ai/neural_training_profiles.py`
- `shards_ai/ai/neural_player.py`
- `scripts/validate_neural_profile.py`
- `tests/ai/test_neural_training_profiles.py`
- `tests/ai/test_neural_player.py`
- `tests/ai/test_neural_profile_validation.py`
- `doc/Current state/Neural player.md`
- `README.md`
