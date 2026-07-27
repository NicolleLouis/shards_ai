# Poids des contraintes des effets conditionnels

## Objective

Rendre la pénalité des effets conditionnels plus cohérente en distinguant la difficulté des
contraintes. Une domination est plus difficile à atteindre qu’une union, elle-même plus difficile
qu’un Echo ou qu’un champion déjà en jeu. La valeur d’un effet reste positive ou nulle après
application de sa pénalité.

## Current State

`_effect_constraint_penalty()` applique actuellement une pénalité uniforme de `1.0` pour chaque
condition booléenne absente. Les seuils de maîtrise et de santé utilisent des formules graduelles,
mais sans poids propre. La pénalité est ensuite multipliée par `HeuristicWeights.constraint_penalty`.

Les contraintes concernées sont `requires_domination`, `requires_union`, `requires_echo`,
`requires_inspiration`, `mastery_at_least` et `health_at_least`. Les profils YAML ne portent pas
encore de coefficients dédiés.

## Target Behavior

Chaque profil peut contenir une section `constraint_weights`. Les effets évalués calculent une valeur
propre, puis soustraient uniquement les pénalités de leurs contraintes. Chaque contribution est
bornée :

```text
score_effet = max(0, valeur_effet - penalite_contraintes)
```

Les effets indépendants d’une même carte sont additionnés. Une contrainte difficile ne peut donc pas
rendre négatif un effet plat non contraint.

## Non-Goals

- optimiser ces poids dans cette première implémentation ;
- apprendre la probabilité réelle d’atteindre une condition ;
- modifier la légalité ou la résolution des cartes ;
- traiter les effets futurs comme une simulation complète de deck ;
- recalibrer les bonus fixes de champions, boucliers ou effets `on_play`.

## Key Decisions

1. Les poids sont regroupés dans un type `CardConstraintWeights` et sérialisés dans les profils.
2. Les valeurs initiales actives sont :
   - `domination=1.5` ;
   - `union=1.0` ;
   - `echo=0.75` ;
   - `inspiration=0.5` ;
   - `mastery=1.0` ;
   - `health=0.75`.
3. La maîtrise reste graduelle : le poids multiplie la base du seuil et l’écart restant.
4. La santé reste graduelle : le poids multiplie l’écart à `health_at_least`.
5. Une condition booléenne non satisfaite ajoute son poids une fois par opération concernée.
6. Un profil historique sans `constraint_weights` utilise des valeurs legacy toutes égales à `1.0`,
   afin de ne pas changer silencieusement le comportement de `v001` à `v003`.
7. Le profil actif `v004` porte explicitement les nouvelles valeurs ordonnées.

## Open Questions

- **Non bloquante :** les poids pourront être activés dans une future campagne après confirmation
  indépendante de `v004`.
- **Non bloquante :** les effets conditionnels futurs pourront recevoir un facteur de potentiel
  séparé ; cette architecture ne l’introduit pas encore.

## Proposed Architecture

`CardConstraintWeights` est défini à côté de `HeuristicWeights`. `HeuristicProfile` contient les
deux objets : poids de décision et poids de contraintes. `HeuristicPlayer` les transmet à
`features_for_action()`.

L’extracteur transmet les poids aux calculs de pénalité de carte, d’effet et d’opération. Les
contraintes sont évaluées localement à l’effet concerné. Les chemins `BuyCard`, `RecruitMercenary`,
`RecruitFreeCard`, bannissement et décision en attente utilisent le même profil.

## Data Model

```yaml
constraint_weights:
  mastery: 1.0
  health: 0.75
  inspiration: 0.5
  echo: 0.75
  union: 1.0
  domination: 1.5
```

Les champs doivent être finis et non négatifs. La section est optionnelle à la lecture pour préserver
les profils historiques.

## Edge Cases

- une condition satisfaite ne produit aucune pénalité ;
- plusieurs conditions absentes sur une opération se cumulent ;
- une opération conditionnelle inactive ne contribue pas à la valeur actuelle de l’effet ;
- une pénalité supérieure à la valeur de l’effet donne une contribution nulle, jamais négative ;
- les thresholds présents dans des branches d’effet restent interprétés selon les règles actuelles
  de sélection de branche ; l’estimation d’un potentiel futur est hors périmètre.

## Testing Strategy

- round-trip YAML et lecture d’un profil sans section ;
- vérification des valeurs legacy et `v004` ;
- tests séparés pour domination, union, echo et inspiration ;
- tests graduels de maîtrise et de santé ;
- test de borne à zéro d’un effet surpénalisé ;
- test que les effets indépendants sont additionnés sans pénalité croisée ;
- suite complète moteur, IA et optimisation.

## Rollout And Migration

Les profils historiques ne sont pas réécrits. `v004.yaml` reçoit explicitement la section de
contraintes. Les nouvelles valeurs deviennent les défauts actifs de `HeuristicPlayer()` ; le loader
continue d’utiliser les valeurs legacy lorsqu’un ancien YAML ne contient pas la section.

## Files Expected To Change

- `shards_ai/ai/heuristic_evaluator.py` — type `CardConstraintWeights` ;
- `shards_ai/ai/heuristic_profiles.py` — lecture et écriture YAML ;
- `shards_ai/ai/heuristic_player.py` — injection des poids ;
- `shards_ai/ai/heuristic_features.py` — pénalités pondérées et borne positive ;
- `configs/heuristic_profiles/v004.yaml` — valeurs actives explicites ;
- `tests/` — contraintes et compatibilité ;
- `doc/Current state/Heuristic player.md` — comportement courant.
