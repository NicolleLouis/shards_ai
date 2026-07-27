# Campagne d’optimisation combinée — Architecture

## Objective

Ajouter un mode de campagne capable d’optimiser dans une même recherche :

- les `HeuristicWeights` de décision ;
- les `CardAcquisitionWeights` ;
- les `CardConstraintWeights`.

Le mode doit permettre à une évolution d’un bloc d’être évaluée avec les valeurs courantes des deux
autres blocs, tout en conservant les modes historiques spécialisés.

## Current State

`optimize_heuristic()` optimise uniquement `HeuristicWeights` lorsque les modes spécialisés sont
désactivés. `--acquisition-only` délègue à `optimize_acquisition_weights()` et
`--constraints-only` délègue à `optimize_constraint_weights()`.

Les trois optimiseurs utilisent déjà `_evaluate_candidate()`, qui sait évaluer simultanément les
trois familles de poids, mais leur état courant et leurs voisinages ne sont pas combinés. Il est donc
impossible de faire évoluer les trois familles dans une même campagne depuis le CLI.

## Target Behavior

Ajouter un mode explicite `--combined` qui :

1. charge un profil complet ;
2. représente un candidat comme un triplet `(action, acquisition, constraints)` ;
3. génère des voisins dans chacune des trois familles ;
4. évalue chaque voisin avec le triplet complet ;
5. conserve le meilleur candidat complet par racing et finalistes ;
6. valide le triplet final contre RandomPlayer et le profil de référence ;
7. ne publie le profil que si la validation complète passe.

Sans `--combined`, les comportements actuels restent inchangés. `--combined` est incompatible avec
`--acquisition-only` et `--constraints-only`.

## Non-Goals

- remplacer immédiatement la recherche par une méthode évolutionnaire, bayésienne ou exhaustive ;
- garantir l’exploration de toutes les combinaisons possibles ;
- modifier les bornes ou pas de recherche existants ;
- changer les règles du jeu ou la politique heuristique hors poids ;
- ajouter parallélisme, cache ou multiprocessing ;
- publier automatiquement un profil malgré une validation négative.

## Key Decisions

1. **État atomique complet.** Un candidat et le profil courant contiennent toujours les trois
   familles de poids afin qu’aucune évaluation ne mélange un candidat partiel avec une référence
   implicite.
2. **Voisinage par blocs.** La première version génère les voisins unitaires et les paires conjointes
   déjà définies à l’intérieur de chaque famille. Les modifications des familles interagissent par
   leur évaluation commune et par l’état courant accepté à chaque batch.
3. **Pas d’explosion combinatoire.** Les paires croisées entre les 36 paramètres ne sont pas toutes
   générées : cela multiplierait le coût de chaque batch sans garantie de meilleure information. La
   prochaine évolution de l’algorithme de recherche pourra ajouter une exploration croisée contrôlée.
4. **Validation commune.** Le candidat combiné doit battre le profil de référence sur l’adversaire
   heuristique avec le gain minimal configuré et ne pas régresser significativement contre Random.
5. **Compatibilité CLI.** Le mode combiné est opt-in ; les anciennes commandes et profils restent
   valides.
6. **Traçabilité.** Chaque entrée d’historique sérialise les trois dictionnaires de poids, même si
   une seule famille a changé dans le voisin testé.

## Open Questions

- **Non bloquante — recherche croisée :** après une première campagne combinée, décider si le
  voisinage doit ajouter des paires inter-familles ciblées ou une recherche différente.
- **Non bloquante — durée :** le nombre de dimensions augmente fortement ; la durée et les tailles de
  lots devront être calibrées après un premier run.
- **Non bloquante — sélection :** un futur algorithme pourra réserver une part du budget aux mutations
  conjointes plutôt qu’aux seuls meilleurs voisins locaux.

## Proposed Architecture

### Paramètres du mode

Ajouter `combined: bool` à `OptimizationConfig` et `--combined` au script CLI. La validation de la
configuration interdit les combinaisons ambiguës avec les modes spécialisés.

Le mode utilise :

- `active_fields` et `DEFAULT_BOUNDS` pour les actions ;
- `active_acquisition_fields` et `ACQUISITION_BOUNDS` pour l’acquisition ;
- `active_constraint_fields` et `CONSTRAINT_BOUNDS` pour les contraintes.

Pour le mode combiné, les contraintes actives par défaut deviennent toutes les contraintes disponibles
si le CLI ne reçoit pas de liste explicite.

### Génération des candidats

Une fonction dédiée génère un ensemble dédoublonné de triplets :

```text
(action courant ou voisin,
 acquisition courante ou voisin,
 contraintes courantes ou voisine)
```

Chaque voisin d’un bloc est combiné avec les valeurs courantes des deux autres blocs. Les paires
conjointes existantes sont conservées au sein de leur bloc. Le classement reste fondé sur
`aggregate_objective` et ne change pas les règles de promotion.

### Évaluation et progression

`_evaluate_candidate()` reçoit toujours les trois blocs du triplet. Après chaque batch, le meilleur
triplet complet remplace l’état courant uniquement si son objectif est strictement supérieur. En cas
d’échec, les échelles de pas diminuent comme dans les optimiseurs spécialisés.

Les phases `initial`, `racing`, `finalist` et `validation-sample` suivent le même budget global et
les mêmes seeds déterministes que le mode historique.

## Data Model

Aucune migration ni nouvelle donnée persistante. `OptimizationConfig` reçoit `combined`. Les profils
YAML restent inchangés ; les trois sections existantes sont utilisées.

## Backend Flow

1. Parser `--combined` et les listes de champs actives.
2. Charger le profil initial et la référence complète.
3. Construire l’état courant triplet.
4. Générer les voisins de chaque bloc et les combiner avec l’état courant.
5. Évaluer les candidats contre Random et le profil précédent.
6. Effectuer racing et validation des finalistes.
7. Accepter le meilleur triplet strictement améliorant.
8. Valider le triplet final ; publier uniquement en cas de succès.

Les erreurs ou évaluations incomplètes restent exclues de la promotion, comme actuellement.

## Frontend Flow

Sans objet.

## Authorization And Feature Gates

Sans objet. Le mode est contrôlé par un flag CLI local.

## Observability And Operations

Les métadonnées du profil et `results.json` enregistrent :

- `combined=true` ;
- les trois listes de champs actives ;
- les trois blocs du candidat accepté ;
- le nombre de batches ;
- les validations par adversaire.

Le résumé CLI indique explicitement le mode combiné pour éviter de confondre une campagne action-only
avec une campagne complète.

## Edge Cases

- aucune famille active : le candidat courant est simplement réévalué ;
- liste de contraintes vide en mode combiné : toutes les contraintes disponibles sont utilisées par
  défaut ;
- champ inconnu : erreur de configuration avant toute partie ;
- options `--combined` et `--acquisition-only` ou `--constraints-only` : erreur CLI ;
- deadline pendant un candidat : le candidat incomplet ne peut pas être promu ;
- profil historique : toutes les sections manquantes utilisent les défauts existants.

## Testing Strategy

- tester la validation des options CLI et les incompatibilités ;
- tester la génération de voisins dans chaque bloc et le dédoublonnage des triplets ;
- vérifier qu’un candidat d’action conserve les acquisitions/contraintes courantes ;
- vérifier qu’un candidat d’acquisition conserve les actions/contraintes courantes ;
- vérifier le même comportement pour les contraintes ;
- tester l’acceptation et la sérialisation d’un triplet complet ;
- tester qu’une évaluation incomplète ne peut pas être publiée ;
- exécuter la suite complète et un mini-run combiné déterministe.

## Rollout And Migration

Le mode historique reste le défaut. Le nouveau mode sera utilisé explicitement avec `--combined` et
un profil de publication distinct. Une campagne courte servira à vérifier le coût et le nombre de
candidats avant une campagne longue.

## Files Expected To Change

- `shards_ai/optimization/heuristic.py` : configuration, voisinage et boucle combinée ;
- `scripts/optimize_heuristic.py` : option `--combined` et résumé ;
- `tests/optimization/test_heuristic.py` : tests du triplet et du voisinage ;
- `doc/Current state/Heuristic player.md` ou une page d’état dédiée : comportement de l’optimiseur.
