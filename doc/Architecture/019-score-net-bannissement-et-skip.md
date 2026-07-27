# Score net de bannissement et choix de `SkipBanish` — Architecture

## Objective

Permettre au joueur heuristique de comparer le bénéfice de retirer une carte du deck avec la valeur
de cette carte. Supprimer une carte forte doit pouvoir être moins intéressant que `SkipBanish`.
`banish_threshold` contrôlera l’agressivité du thinning et pourra être calibré par entraînement.

## Current State

`Game.legal_actions()` expose les `BanishCard` de la main et de la défausse ainsi que
`SkipBanish` lorsque `pending_banishes > 0`. Le moteur accepte les deux actions ; `SkipBanish`
annule la séquence de bannissements encore en attente.

L’extracteur calcule actuellement :

```text
max(0, 3 - card_acquisition_value)
```

Le plancher à zéro empêche un bannissement d’être moins attractif que `SkipBanish`, dont le score
est négatif via `action_penalty`. Il transforme également toutes les cartes de valeur supérieure à
3 en égalités départagées artificiellement.

La valeur durable d’un champion inclut sa valeur intrinsèque et son effet de pose, mais pas
systématiquement son pouvoir activable. Evokatus inclut sa pioche à la pose, mais son pouvoir
`+1 Power par champion Homodeus` doit également participer à sa valeur durable.

## Target Behavior

Pour une `BanishCard`, calculer une valeur nette signée :

```text
net_banish_value = banish_threshold - card_acquisition_value
```

Cette valeur devient la feature `deck_thinning` sans être bornée à zéro.

- Une carte faible produit un score positif et favorise le thinning.
- Une carte forte produit un score négatif et peut perdre contre `SkipBanish`.
- Une carte proche du seuil est comparée directement au coût du skip.
- Le tie-break entre cartes ne s’applique qu’après cette comparaison principale.

`SkipBanish` conserve sa pénalité actuelle. Le seuil décrit la valeur de référence des cartes ; la
pénalité de skip reste distincte et fixe dans un premier temps.

## Non-Goals

- modifier les actions légales ou la règle moteur du bannissement ;
- rendre le skip partiel : il annule toujours la séquence restante ;
- simuler la pioche ou la composition future du deck ;
- entraîner simultanément `banish_threshold` et la pénalité globale `action_penalty` ;
- modifier le tie-break déterministe quand le score principal distingue déjà les actions ;
- modifier la règle d’Evokatus : pioche à la pose, pouvoir activable à +1 Power par champion
  Homodeus.

## Key Decisions

1. `deck_thinning` représente désormais le gain net du bannissement et peut être négatif.
2. `CardAcquisitionWeights` reçoit `banish_threshold: float = 3.0` pour conserver l’échelle actuelle
   des profils historiques avant calibration.
3. `banish_threshold` règle l’agressivité contre la valeur de la carte ; `action_penalty` conserve
   le coût général du skip.
4. La première campagne ne fait varier que `banish_threshold`, afin d’éviter que deux paramètres
   apprennent la même préférence avec des effets indiscernables.
5. `_card_acquisition_value()` inclut la valeur observable du pouvoir activable des champions, en
   plus de leur valeur intrinsèque et de leur effet de pose.
6. Les profils YAML sans champ utilisent `banish_threshold=3.0`.

## Open Questions

- **Non bloquante — calibration :** `3.0` est une valeur de compatibilité, à calibrer avec les autres
  poids gelés.
- **Non bloquante — pénalité dédiée :** si le seuil seul ne suffit pas, une future évolution pourra
  ajouter `skip_banish_penalty`, sans réutiliser `action_penalty`, qui est global.
- **Non bloquante — bannissements multiples :** le comportement actuel du skip global reste hors
  périmètre.

## Proposed Architecture

### Valeur durable

`_card_acquisition_value()` reste le point central. Pour un champion, il agrège :

1. valeur intrinsèque ;
2. effet de pose, notamment la pioche d’Evokatus ;
3. pouvoir activable estimé dans l’état observable, notamment le nombre de champions Homodeus.

Cette estimation ne mute jamais l’observation.

### `BanishCard`

`features_for_action()` récupère `banish_threshold`, calcule la valeur durable complète, puis
retourne :

```python
ActionFeatures(
    deck_thinning=banish_threshold - card_acquisition_value,
)
```

Le moteur ne connaît ni les poids ni la valeur des cartes.

### `SkipBanish`

La branche `SkipBanish` reste inchangée et produit son `action_penalty`. Le classement compare donc
le score net du bannissement au score du skip. Une carte suffisamment précieuse pourra naturellement
faire préférer le skip.

### Tie-break

L’ordre de l’architecture 018 reste applicable uniquement entre bannissements dont le score
principal est égal : valeur immédiate de jouer, coût imprimé, identifiant stable, index courant.
Il ne peut pas faire battre `SkipBanish` à une carte dont le score net est inférieur.

## Data Model

Aucune table ni migration. Le modèle Python ajoute :

```python
banish_threshold: float = 3.0
```

Le champ est sérialisé sous `card_acquisition_weights`. Le chargeur reste compatible avec les
profils historiques.

## Backend Flow

1. Charger le profil et `banish_threshold`.
2. Générer `BanishCard` et `SkipBanish`.
3. Évaluer la valeur durable de chaque carte.
4. Calculer `banish_threshold - card_acquisition_value`.
5. Évaluer le skip avec sa pénalité existante.
6. Sélectionner le meilleur score complet et appliquer l’action.

Les cartes conditionnelles de victoire restent protégées avant le classement.

## Frontend Flow

Sans objet. Les rapports HTML pourront ultérieurement afficher la fréquence des skips, la valeur des
cartes ignorées et les taux conditionnels par carte.

## Authorization And Feature Gates

Sans objet. La compatibilité est assurée par la valeur par défaut.

## Observability And Operations

Les analyses doivent distinguer bannissements, `SkipBanish`, valeur d’acquisition, score net et
résultat de partie. Les comparaisons utiliseront le taux conditionnel
`bannis / cartes candidates`, pas seulement les nombres bruts.

## Edge Cases

- aucune carte bannissable : seul `SkipBanish` est disponible ;
- valeur exactement égale au seuil : score net nul ;
- valeur très élevée : score net négatif, sans plancher ;
- pouvoir sans condition observable : valeur déclarative de base uniquement ;
- profil historique sans champ : seuil `3.0` ;
- égalité entre cartes : tie-break 018 ;
- branche de victoire : protection existante conservée.

## Testing Strategy

- vérifier que `SkipBanish` est légal avec des cibles ;
- vérifier qu’une carte au-dessus du seuil peut perdre contre le skip ;
- vérifier qu’une carte faible est bannie plutôt que skipée ;
- vérifier le cas exactement au seuil ;
- vérifier le round-trip YAML avec et sans `banish_threshold` ;
- vérifier qu’Evokatus inclut sa pioche et son pouvoir activable ;
- vérifier que le tie-break 018 intervient seulement après égalité du score principal ;
- exécuter la suite complète et une campagne reproductible avant/après.

## Rollout And Migration

Le code utilisera `3.0` pour les profils existants. `v005` recevra explicitement le champ. Une
campagne indépendante fera varier uniquement `banish_threshold` avant toute publication.

## Files Expected To Change

- `shards_ai/ai/heuristic_evaluator.py` : nouveau champ et défaut ;
- `shards_ai/ai/heuristic_features.py` : valeur signée et pouvoirs de champions ;
- `configs/heuristic_profiles/v005.yaml` : valeur explicite ;
- optimisation et tests de profils : champ calibrable et round-trip ;
- tests heuristiques : décisions bannissement/skip ;
- `doc/Current state/Heuristic player.md` : comportement livré ;
- rapport HTML : métriques skip et taux conditionnels, si incluses.
