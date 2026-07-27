# Optimisation robuste des coefficients du HeuristicPlayer — Architecture

**Statut : implémenté** — Architecture de référence de la recherche hybride et de sa validation.

## Objective

Rendre les campagnes d’optimisation des coefficients plus fiables après la campagne `v003`, qui
n’a pas démontré de gain contre `v002`. L’objectif est de sélectionner un profil qui bat réellement
la référence heuristique sur des seeds indépendantes, plutôt qu’un candidat gagnant sur un petit
échantillon bruité.

## Current State

- `HeuristicPlayer` reçoit un mapping immuable de `HeuristicWeights` et choisit l’action de score
  maximal.
- `scripts/optimize_heuristic.py` charge un profil initial, exécute une descente coordonnée et
  écrit un historique JSON ainsi qu’un profil YAML.
- Les candidats d’un même batch partagent déjà des seeds dérivées, mais chaque candidat reçoit
  souvent seulement 100 parties avant décision.
- Le shaping est calculé avec `alpha=0.10` au début puis décroît jusqu’à zéro. Il sert de signal
  secondaire et non de récompense directement utilisée par le joueur.
- La phase mixte 50/50 avec le profil précédent existe, mais le seuil de 90 % contre Random peut
  empêcher son démarrage.
- La campagne `v003` a changé seulement `gems_produced` et `power_produced`, sans gain confirmé
  contre `v002`.

## Target Behavior

Une campagne doit pouvoir :

1. démarrer explicitement depuis `v002` ou tout autre profil fourni ;
2. évaluer directement un candidat contre 50 % `RandomPlayer` et 50 % une référence heuristique
   figée ;
3. explorer des mutations simples, conjointes et aléatoires sur un sous-ensemble de coefficients ;
4. éliminer progressivement les candidats faibles avec des lots de tailles croissantes ;
5. utiliser des seeds appariées entre candidats d’un même round ;
6. séparer les seeds de recherche, validation et test ;
7. publier uniquement un candidat dont l’amélioration est confirmée statistiquement ;
8. conserver les résultats détaillés permettant d’expliquer chaque promotion ou élimination.

## Non-Goals

- Modifier les règles du moteur ou la légalité des actions.
- Ajouter du look-ahead, du multiprocessing ou des threads dans cette évolution.
- Optimiser `terminal_win` ou `lethal`.
- Faire du shaping le critère principal de publication.
- Remplacer silencieusement le profil actif du `HeuristicPlayer`.

## Key Decisions

1. **Recherche hybride.** La campagne combine mutations par coordonnée, mutations conjointes et
   mutations aléatoires, puis applique un racing progressif. La descente coordonnée reste
   disponible comme stratégie de référence.
2. **Coefficients actifs limités.** La première passe optimise `gems_produced`, `power_produced`,
   `damage_value`, `card_draw`, `health_gained`, `mastery_gained` et `champion_value`. Les autres
   coefficients sont gelés jusqu’à ce qu’une amélioration soit confirmée.
3. **Racing déterministe.** Tous les candidats d’un round utilisent les mêmes seeds, le même
   adversaire et la même alternance de position. Les niveaux par défaut sont 200, 500 puis 2 000
   parties pour les candidats conservés.
4. **Adversaires.** Une campagne visant à battre une référence utilise directement 50 % Random et
   50 % de la référence figée. Le seuil de 90 % contre Random devient une métrique, pas un verrou.
5. **Objectif.** L’utilité terminale vaut victoire 1, nulle 0,5 et défaite 0. Le shaping ne départage
   que les candidats dont les utilités sont statistiquement équivalentes. La validation finale
   utilise `alpha=0`.
6. **Publication statistique.** La validation utilise au moins 1 000 parties par adversaire et un
   intervalle de confiance à 95 % sur la différence appariée. La borne basse du gain contre la
   référence doit dépasser +0,01 ; aucune baisse de plus de 0,01 n’est acceptée contre Random.
7. **Versionnement.** `--publish-profile configs/heuristic_profiles/v003.yaml` écrit également
   `profile_id: heuristic-v003` et `parent_profile_id: heuristic-v002`.
8. **Ressources.** La campagne reste mono-processus et mono-cœur par défaut. Le budget global et
   les tailles de lots sont explicites afin d’éviter une consommation incontrôlée.

## Open Questions

- **Non bloquante :** une recherche évolutionnaire complète pourra être comparée au racing hybride
  après mesure sur un budget identique.
- **Non bloquante :** `champion_threat_scale` reste fixe jusqu’à une campagne dédiée de calibration.

## Proposed Architecture

### Génération des candidats

Le générateur part du meilleur profil courant et produit :

- une variation positive et négative de chaque coefficient actif ;
- des variations conjointes sur deux coefficients actifs ;
- un petit nombre de profils aléatoires bornés autour du profil courant.

Les doublons et les candidats identiques au profil courant sont éliminés avant simulation.

### Racing

Les candidats sont d’abord évalués sur 200 parties. Les cinq meilleurs passent à 500 parties et les
deux meilleurs à 2 000 parties. Le meilleur candidat du dernier niveau devient le profil courant
uniquement s’il bat la référence sur l’objectif terminal ou si son shaping départage une égalité
statistique.

Les validations ne sont jamais utilisées pour générer une mutation. Elles sont réservées à la
décision de publication.

### Statistiques

Chaque partie produit une utilité individuelle. Lorsque les seeds sont communes, la différence
entre candidat et référence est conservée par partie. L’intervalle de confiance à 95 % est calculé
sur ces différences appariées ; il est utilisé pour éviter qu’une fluctuation de quelques victoires
déclenche une publication.

### Shaping

Le potentiel existant reste inchangé : santé, maîtrise et menace de champions sont normalisées et
agrégées par transition. `alpha` reste fixé pour tous les candidats d’un round et décroît avec la
progression de la campagne. Il ne peut pas transformer une défaite statistiquement significative en
victoire de recherche.

## Data Model

Les résultats JSON ajoutent ou conservent :

- niveau de racing ;
- type de mutation et coefficients modifiés ;
- seeds et nombre de parties ;
- résultats par adversaire ;
- utilité, shaping et intervalle de confiance ;
- décision de promotion, rejet ou publication.

Les artefacts restent sous `artifacts/heuristic_optimization/<run-id>/`. Le YAML publié contient
l’identifiant, le parent, les coefficients et les métadonnées de campagne.

## CLI

Le script accepte les paramètres suivants, avec des valeurs raisonnables pour une campagne d’une
heure :

```text
--profile
--start-mixed
--initial-games
--racing-games
--validation-games
--test-games
--active-fields
--confidence-level
--minimum-gain
--publish-profile
```

Exemple :

```bash
PYTHONPATH=. nice -n 10 poetry run python scripts/optimize_heuristic.py \
  --profile configs/heuristic_profiles/v002.yaml \
  --start-mixed \
  --duration-seconds 3600 \
  --initial-games 200 \
  --racing-games 500 \
  --validation-games 1000 \
  --test-games 3000 \
  --seed 44 \
  --publish-profile configs/heuristic_profiles/v003.yaml
```

## Testing Strategy

- tester les mutations, bornes, déduplications et coefficients gelés ;
- vérifier que les candidats d’un round utilisent les mêmes seeds ;
- vérifier que la référence mixte reste figée ;
- tester le calcul de différence appariée et les décisions selon l’intervalle de confiance ;
- refuser un profil gagnant sur un petit échantillon mais non confirmé en validation ;
- vérifier la cohérence entre le nom de fichier publié et `profile_id` ;
- exécuter une campagne smoke de 60 secondes sans erreur et une campagne déterministe courte.

## Rollout

La première campagne de cette architecture publie vers un nouveau fichier, sans modifier le profil
par défaut intégré à `HeuristicPlayer`. Le profil actif ne sera changé qu’après comparaison du
rapport de validation et revue des coefficients. Les anciennes campagnes restent lisibles grâce à
la conservation du format JSON existant.

## Files Expected To Change

- `shards_ai/optimization/heuristic.py` : stratégie hybride, racing et statistiques ;
- `scripts/optimize_heuristic.py` : options CLI et publication versionnée ;
- `tests/optimization/test_heuristic.py` : tests de recherche, seeds et validation ;
- `doc/Current state/Heuristic player.md` : comportement effectivement livré après implémentation.
