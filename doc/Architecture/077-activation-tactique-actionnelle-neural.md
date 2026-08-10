# Architecture — Activation tactique actionnelle neural

## Décision

L'expérience tactique ajoute sept valeurs à l'encodage de chaque action :

```text
requires_union, union_active,
requires_echo, echo_active,
requires_domination, domination_active,
domination_missing_count
```

Ces valeurs sont calculées à partir du couple `(NeuralObservation, ActionRepresentation)`. Elles
ne deviennent pas des scalaires d'état partagés entre toutes les actions.

## Périmètre

Les conditions sont calculées pour les actions `play_card`. Les autres actions reçoivent un vecteur
nul : leur encodage reste fini et leur comportement ne dépend pas accidentellement des synergies
d'une carte qu'elles ne jouent pas.

Les prérequis sont lus dans la représentation sémantique de la carte et limités à la branche
d'effet active au niveau de maîtrise et de santé observé. Les cartes candidates sont exclues des
zones lors du calcul.

## Règles de calcul

- Union : autre carte de même faction dans la main ou la `play_zone` ; les champions sont exclus.
- Echo : au moins une carte Spectra dans la défausse active.
- Domination : factions Maquis, Spectra et Homodeus présentes dans la main ou la `play_zone`, plus
  les factions de `played_champion_faction_mask`. Les champions seulement activés ne comptent pas.
- La composition peut mélanger les zones ; `domination_missing_count` vaut le nombre de ces trois
  factions absentes.
- Les cartes neutres ne complètent aucune condition.

Les features d'activation sont gated par leur prérequis : une carte sans `requires_union` reçoit
`union_active=0`, même si le joueur possède une carte alliée.

## Checkpoints et compatibilité

Le scorer est versionné `structured_semantic_v6_tactical_action_v1` et dérive du scorer V005 avec
`deck_state_v1`. La migration V005 insère sept colonnes nulles dans la première couche de
`action_encoder`. Les anciennes architectures restent inchangées et chargeables.

Le moteur, les actions légales et le masque d'information ne sont pas modifiés par cette
expérience.

## Validation

Les tests couvrent l'exclusion de la candidate, les mélanges de zones, les champions joués contre
les champions seulement activés, Echo, les actions non concernées, les actions candidates
différentes dans un même état et le chargement d'un checkpoint migré.
