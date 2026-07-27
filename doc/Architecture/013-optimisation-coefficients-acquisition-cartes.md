# Optimisation des coefficients d’acquisition des cartes

## Objective

Permettre une première campagne d’optimisation dédiée aux coefficients internes de
`card_acquisition_value`, afin d’améliorer l’estimation de l’efficacité des cartes sans modifier
simultanément la politique heuristique complète.

La phase 1 doit produire un profil versionné, reproductible et comparable au profil de départ. Elle
doit pouvoir être exécutée seule, puis servir de base à une phase 2 combinant ces coefficients avec
les poids de `HeuristicWeights`.

## Current State

`_card_acquisition_value()` dans `shards_ai/ai/heuristic_features.py` utilise des constantes
internes : Gems, Power, maîtrise, santé, pioche, deck thinning et target denial. Ces constantes ne
sont pas présentes dans les profils YAML et ne sont pas optimisées.

Le coefficient global `HeuristicWeights.card_acquisition_value` existe, mais il est resté gelé dans
les campagnes `v002` → `v003`. Le script `scripts/optimize_heuristic.py` sait déjà limiter les
champs actifs via `--active-fields` et publier un profil YAML validé.

## Target Behavior

Un profil peut contenir une section `card_acquisition_weights`. Les cartes sont évaluées avec ces
coefficients au lieu de constantes codées en dur. Une campagne phase 1 active uniquement ces
coefficients ; les poids d’action de `HeuristicWeights` restent inchangés.

Le résultat conserve les coefficients internes dans le YAML publié et dans l’historique JSON. Un
profil ancien sans section dédiée continue à fonctionner avec les valeurs V1 par défaut.

## Non-Goals

- optimiser simultanément `HeuristicWeights` pendant la phase 1 ;
- ajouter une simulation future du deck ou de la qualité des cartes piochées ;
- optimiser séparément les champions ou les effets `on_play` dans cette première phase ;
- modifier les règles du moteur ou les actions légales ;
- garantir un optimum global.

## Key Decisions

1. Les coefficients internes sont un objet typé `CardAcquisitionWeights`, sérialisé sous une clé
   dédiée du profil YAML.
2. Les valeurs par défaut reproduisent exactement la formule actuelle :
   `gems=1.5`, `power=2.0`, `mastery=1.5`, `health=0.5`, `card_draw=2.5`,
   `deck_thinning=1.0`, `target_denial=1.0`.
3. La phase 1 utilise une recherche coordonnée existante avec un espace actif distinct, des bornes
   positives et des pas explicites.
4. Les coefficients sont injectés dans l’extracteur via le profil, sans état global mutable.
5. Le profil initial reste la référence figée de la campagne mixte ; seule la copie candidate évolue.
6. La validation terminale reste obligatoire contre `RandomPlayer` et le profil précédent.
7. La compatibilité ascendante est obligatoire : l’absence de section YAML signifie les valeurs par
   défaut et l’absence de changement comportemental.

## Open Questions

- **Non bloquante :** faut-il optimiser plus tard les bonus fixes de champion, d’effet `on_play` et
  de bouclier ? Ils restent hors phase 1 pour éviter d’élargir l’espace de recherche.
- **Non bloquante :** faut-il exposer le coefficient global `card_acquisition_value` dans la phase 2 ?
  Oui, mais après calibration des coefficients internes afin de limiter les compensations entre
  niveaux.

## Proposed Architecture

### Profil

`HeuristicProfile` contient `weights` et `card_acquisition_weights`. Le loader valide les deux
structures et complète une section absente avec `CardAcquisitionWeights()`.

### Extraction

`HeuristicPlayer` reçoit un profil ou les deux objets de poids. Pour préserver l’API existante,
`HeuristicPlayer(weights=...)` continue à fonctionner avec les valeurs d’acquisition par défaut.
Les chemins d’achat, de recrutement, de bannissement et de décision en attente transmettent les
poids internes à `_card_acquisition_value()`.

### Optimisation

Le module `shards_ai/optimization/heuristic.py` maintient deux espaces :

- `HeuristicWeights` pour les champs actuellement optimisables ;
- `CardAcquisitionWeights` pour la phase 1.

La campagne phase 1 fait varier uniquement le second espace. Le candidat et la référence portent
les deux objets, et les résultats enregistrent les deux mappings. Une phase future pourra activer
les deux espaces explicitement.

### CLI

Le script ajoute un mode `--acquisition-only` et une sélection
`--active-acquisition-fields`. En mode acquisition-only, les champs `--active-fields` sont gelés et
la publication conserve les poids heuristiques initiaux.

Exemple de campagne courte :

```bash
PYTHONPATH=. nice -n 10 poetry run python scripts/optimize_heuristic.py \
  --profile configs/heuristic_profiles/v003.yaml \
  --start-mixed \
  --acquisition-only \
  --duration-seconds 3600 \
  --initial-games 200 \
  --racing-games 500 \
  --validation-games 1000 \
  --test-games 3000 \
  --seed 46 \
  --publish-profile configs/heuristic_profiles/v004.yaml
```

## Data Model

```yaml
card_acquisition_weights:
  gems_produced: 1.5
  power_produced: 2.0
  mastery_gained: 1.5
  health_gained: 0.5
  card_draw: 2.5
  deck_thinning: 1.0
  target_denial: 1.0
```

Les champs sont finis, positifs ou nuls et validés comme nombres finis. Les profils historiques
restent lisibles sans migration.

## Observability And Operations

Les résultats JSON enregistrent les coefficients internes du candidat accepté, le profil parent,
les champs actifs, les bornes, la seed et la validation. Le YAML publié est écrit uniquement après
validation réussie. Les artefacts restent hors de `doc/` sous `artifacts/heuristic_optimization/`.

## Edge Cases

- section YAML absente : valeurs par défaut V1 ;
- champ inconnu : erreur explicite au chargement ou à la configuration ;
- coefficient négatif ou non fini : erreur explicite ;
- campagne interrompue : le candidat partiel ne peut pas être promu ;
- phase 1 sans champ actif : erreur de configuration plutôt qu’une campagne vide.

## Testing Strategy

- round-trip YAML avec et sans `card_acquisition_weights` ;
- égalité numérique entre les valeurs par défaut et l’ancienne formule ;
- injection de coefficients dans achat, mercenaire, bannissement et décision en attente ;
- vérification que `--acquisition-only` ne modifie pas `HeuristicWeights` ;
- optimisation courte déterministe avec un seul coefficient actif ;
- publication et validation d’un profil contenant les deux sections ;
- suite complète existante du moteur et de l’IA.

## Rollout And Migration

Les profils `v001` à `v003` ne sont pas modifiés. La phase 1 démarre avec `v003` et publie vers un
nouveau fichier, recommandé `v004.yaml`. La phase 2 pourra charger `v004` et activer explicitement
les deux familles de coefficients.

## Files Expected To Change

- `shards_ai/ai/heuristic_evaluator.py` — nouveau type de coefficients internes ;
- `shards_ai/ai/heuristic_profiles.py` — chargement et sauvegarde de la section dédiée ;
- `shards_ai/ai/heuristic_features.py` — injection dans l’estimation d’acquisition ;
- `shards_ai/ai/heuristic_player.py` — propagation des poids de profil ;
- `shards_ai/optimization/heuristic.py` — candidats et résultats avec deux espaces ;
- `scripts/optimize_heuristic.py` — mode acquisition-only et options CLI ;
- `tests/` — compatibilité, extraction et campagne ;
- `doc/Current state/Heuristic player.md` — comportement effectivement disponible.
