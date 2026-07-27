# Évaluation inter-phase de `GainMastery` — Architecture

## Objective

Corriger la décision du joueur heuristique lorsqu'il peut dépenser une gemme pour gagner un point
de maîtrise. Aujourd'hui, `GainMastery` est comparé aux cartes jouables pendant `PLAY`, alors que
la gemme consommée modifie directement les achats possibles pendant `BUY`.

L'objectif est que l'heuristique puisse arbitrer entre :

- le gain immédiat de maîtrise ;
- les effets de seuil rendus accessibles par ce point de maîtrise ;
- la valeur de l'achat auquel la gemme renoncera éventuellement.

La décision doit rester déterministe, explicable, compatible avec les observations publiques et
suffisamment légère pour être exécutée dans chaque décision de jeu.

## Current State

Le moteur expose `GainMastery()` uniquement pendant la phase `PLAY`, lorsque le joueur possède au
moins une gemme, n'a pas atteint 30 de maîtrise et n'a pas déjà utilisé cette action pendant le
tour. L'action coûte une gemme et augmente la maîtrise de un ; le moteur réinitialise le verrou au
cleanup.

`HeuristicPlayer.choose_action()` compare les actions légales de la phase courante. En phase
`PLAY`, ces actions sont les cartes de la main, les activations de champions, `GainMastery()` et
`PassPlayPhase()`. Les `BuyCard` et `RecruitMercenary` ne sont donc pas présents dans la même
comparaison.

Dans `shards_ai/ai/heuristic_features.py`, `GainMastery` expose actuellement :

- `cost_paid = 1` ;
- `mastery_gained = 1` ;
- `mastery_advantage_delta = 1 / 30`.

Le profil `v004` donne une valeur positive au gain de maîtrise, mais `cost_paid` vaut `0.0` et
aucune valeur d'achat futur n'est calculée. La politique peut donc dépenser une gemme sans tenir
compte de la carte achetable qu'elle rend inaccessible.

Les achats sont évalués séparément en phase `BUY` avec `_purchase_features()` et
`_card_acquisition_value()`. La rivière contient au plus six cartes et les mercenaires sont des
actions d'achat distinctes.

## Target Behavior

Lorsqu'elle évalue `GainMastery`, l'heuristique ajoute deux signaux projetés :

1. une valeur de seuil, correspondant à l'amélioration observable de toutes les cartes de la main
   jouables après le passage de `mastery` à `mastery + 1` ;
2. un delta d'opportunité, correspondant à la différence entre la meilleure acquisition accessible
   avec les gemmes et la maîtrise actuelles et la meilleure acquisition accessible après avoir
   dépensé une gemme et gagné un point de maîtrise.

Conceptuellement :

```text
score(GainMastery)
  = gain de maîtrise actuel
  + valeur des seuils débloqués
  - delta d'opportunité d'achat
```

Le calcul doit tenir compte de `BuyCard` et `RecruitMercenary`, de la rivière observée, des coûts
réels et des poids d'acquisition actifs. La valeur de l'achat postérieur sera évaluée avec la
maîtrise projetée (`mastery + 1`) afin de représenter les seuils qui améliorent une carte achetée.
Si aucun achat n'est disponible avant ou après la dépense, la perte correspondante vaut zéro.

Le gain de seuil couvre les cartes de la main qui deviennent plus fortes immédiatement après
`GainMastery`, sans supposer que le joueur connaît des cartes cachées ou l'ordre futur de la pioche.
Les effets de seuil des cartes de la rivière sont couverts par la comparaison de valeur d'achat et
ne doivent pas être ajoutés une seconde fois à `mastery_threshold_value`.

## Non-Goals

- modifier les règles moteur de la maîtrise, des gemmes ou des achats ;
- appliquer réellement `GainMastery`, cloner une partie ou simuler un tour complet pour scorer une
  action ;
- effectuer une recherche exhaustive sur l'ordre des cartes, les achats futurs ou les synergies de
  plusieurs tours ;
- rendre l'heuristique consciente de la main ou du deck adverse ;
- optimiser les nouveaux coefficients dans la même évolution que leur introduction ;
- remplacer l'évaluation complète des cartes par une stratégie d'achat indépendante.

## Key Decisions

1. **Projection pure et locale.** Le calcul sera effectué sur l'observation détachée et ne mutera
   ni `GameState`, ni `PlayerState`, ni la source aléatoire. Aucune copie complète de `Game` ne sera
   créée.
2. **Deux signaux explicites.** La valeur du seuil et le coût d'opportunité auront des features
   distinctes dans `ActionFeatures`. Le champ générique `cost_paid` ne sera pas réutilisé pour
   représenter la valeur d'un achat perdu.
3. **Meilleur achat accessible.** Le coût d'opportunité comparera la meilleure valeur parmi les
   `BuyCard` et `RecruitMercenary` légaux avec `gems` et la maîtrise actuelle, puis avec `gems - 1`
   et la maîtrise projetée. Il ne valorisera pas une carte simplement parce qu'elle existe dans la
   rivière si elle n'est pas achetable. Les mercenaires sont explicitement inclus avec leur valeur
   d'effet immédiat.
4. **Valeur cohérente avec la politique actuelle.** La projection réutilisera les fonctions
   d'évaluation existantes et les poids injectés, afin que le calcul ne crée pas une seconde
   définition concurrente de la valeur d'une carte.
5. **Projection bornée du seuil.** La première version additionnera l'amélioration observable de
   toutes les cartes actuellement présentes dans la main qui peuvent être jouées après
   `GainMastery`. Le joueur joue normalement presque toute sa main ; cette hypothèse est donc plus
   représentative que la seule meilleure carte. La valeur des cartes de la rivière reste dans la
   projection d'achat et n'est pas ajoutée à ce signal. La projection ne cherchera toutefois pas le
   meilleur ordre ni les synergies entre plusieurs cartes jouées dans le même tour.
6. **Compatibilité des profils.** Les profils YAML existants qui ne contiennent pas les nouveaux
   poids continueront à se charger avec des valeurs par défaut explicites. Les nouveaux poids seront
   optimisables séparément après validation comportementale.
7. **Calibration post-implémentation.** Les valeurs initiales des nouveaux poids ne seront pas
   déduites artificiellement des échelles existantes. Après l'implémentation et la validation des
   tests comportementaux, une campagne dédiée calibrera séparément la valeur des seuils et le coût
   d'opportunité d'achat.
8. **Égalités de valeur.** Lorsque plusieurs achats ont la même valeur projetée, le coût
   d'opportunité dépendra uniquement de cette valeur numérique et jamais du choix arbitraire d'une
   carte parmi les options équivalentes.
9. **Delta signé.** Le signal d'opportunité sera calculé comme `valeur_avant - valeur_après`. Il
   peut donc être négatif lorsque le point de maîtrise augmente suffisamment la valeur de l'achat
   postérieur ; avec un poids négatif, ce cas récompense correctement `GainMastery`.
10. **Priorités inchangées.** Une victoire terminale et une action létale resteront prioritaires sur
   le score pondéré. La nouvelle projection ne modifiera pas la validation moteur des actions.

## Open Questions

La décision fonctionnelle est complète. Les valeurs initiales des nouveaux poids restent
volontairement à calibrer après l'implémentation et les tests.

## Proposed Architecture

### Features

Étendre `ActionFeatures` avec des champs dédiés, par exemple :

- `purchase_opportunity_cost` : delta signé entre la meilleure valeur d'achat avant et après
  `GainMastery` ;
- `mastery_threshold_value` : valeur de l'amélioration des actions rendues accessibles par
  `mastery + 1`.

Étendre `HeuristicWeights.score()` avec les coefficients correspondants. Le coefficient du coût
d'opportunité sera négatif ; celui du seuil sera positif. Les noms définitifs devront rester
explicites dans les profils et les rapports d'optimisation.

### Projection des achats

Ajouter un helper pur, proche de l'évaluation des features, qui :

1. énumère les cartes de la rivière achetables avec un budget donné ;
2. construit les actions `BuyCard` et `RecruitMercenary` correspondantes ;
3. évalue chaque action avec les poids d'acquisition et de contraintes existants, en permettant de
   fournir une maîtrise projetée ;
4. conserve la meilleure valeur, avec zéro si aucune action n'est disponible ;
5. renvoie le delta signé `best_value(gems, mastery) - best_value(gems - 1, mastery + 1)` pour
   `GainMastery`.

Le helper ne doit pas appeler `Game.legal_actions()` sur une copie, car cela introduirait une
dépendance à une transition de moteur inutile et plus coûteuse. Il peut réutiliser une fonction
partagée d'énumération de la rivière si celle-ci reste pure.

### Projection des seuils

Pour `GainMastery`, l'extracteur évaluera les effets conditionnels avec la maîtrise courante puis
avec la maîtrise projetée, dans une vue détachée ou via un paramètre explicite de maîtrise. Il
comparera notamment :

- les `PlayCard` des cartes actuellement présentes dans la main ;
Pour chaque carte de la main jouable après `GainMastery`, l'extracteur évaluera la différence entre
son effet à la maîtrise courante et son effet à la maîtrise projetée. Les différences positives
seront additionnées pour constituer `mastery_threshold_value`. Les effets déjà actifs à la maîtrise
courante ne seront pas comptés comme un nouveau bonus.

Cette addition représente une valeur de tour : elle suppose que les cartes actuellement jouables
seront effectivement jouées, mais ne suppose pas que des cartes inconnues de la pioche seront
piochées ni qu'une séquence optimale de synergies sera trouvée. Les cartes déjà jouées ou bannies
ne sont pas incluses.

### Intégration dans `features_for_action`

Le cas `GainMastery` appellera ces projections en plus de ses signaux actuels. Les autres actions
continueront à être évaluées comme aujourd'hui. La projection ne sera donc exécutée que lorsque
`GainMastery` figure parmi les actions examinées.

`HeuristicPlayer.choose_action()` conservera son classement lexicographique actuel :

1. victoire terminale ;
2. action létale ;
3. score pondéré incluant les nouveaux signaux ;
4. priorité de fin de phase ;
5. ordre stable des actions légales.

## Data Model

Aucune table, migration ou donnée persistante n'est nécessaire.

Les changements de données sont limités à :

- deux champs optionnels de `ActionFeatures` avec valeurs par défaut nulles ;
- deux coefficients sérialisés dans `HeuristicWeights` et les profils YAML ;
- éventuellement deux colonnes de diagnostic dans les journaux de décision futurs.

Les observations de jeu, les actions publiques et le format des snapshots de partie restent
compatibles.

## Backend Flow

1. Le moteur produit la liste légale habituelle. `GainMastery` n'est ajouté que si ses préconditions
   sont satisfaites.
2. `HeuristicPlayer` demande les features de chaque action.
3. Pour `GainMastery`, l'extracteur calcule sans mutation la meilleure valeur d'achat avec `gems`
   puis `gems - 1`, ainsi que la valeur du seuil avec `mastery` puis `mastery + 1`.
4. Le score pondéré compare le résultat aux cartes de la main, aux champions et à
   `PassPlayPhase`.
5. Le joueur renvoie l'action sélectionnée.
6. Le moteur valide et applique l'action ; il reste l'unique source de vérité pour la consommation
   de gemme, le gain de maîtrise et le verrou par tour.

Le calcul ne doit pas produire d'effet de bord ni consommer de hasard. Une valeur de projection
manquante ou non supportée doit être traitée comme zéro avec un signal de diagnostic local, plutôt
que de rendre une action illégale.

## Frontend Flow

Sans objet pour cette évolution. Aucun frontend n'est modifié.

## Authorization And Feature Gates

Sans objet. Le joueur heuristique et le moteur tournent localement ; aucune permission ni feature
flag applicatif n'est nécessaire.

Un drapeau de profil ou un constructeur pourra être utilisé temporairement pendant la calibration,
mais la règle moteur ne doit pas dépendre de ce drapeau.

## Observability And Operations

La première implémentation ne journalisera pas chaque projection par défaut, afin de préserver le
débit des campagnes. Un mode de diagnostic optionnel devra pouvoir exposer pour un état donné :

- score de `GainMastery` avant et après projection ;
- meilleure valeur d'achat avec `gems` ;
- meilleure valeur d'achat avec `gems - 1` ;
- valeur de seuil ;
- carte ou action ayant fourni chaque maximum.

Les campagnes d'optimisation devront conserver les nouveaux poids dans leur JSON et leur YAML,
comme les coefficients actuels. Les erreurs de projection devront être visibles dans les tests et
les outils de diagnostic, sans être silencieusement confondues avec une absence d'achat.

## Edge Cases

- zéro gemme : `GainMastery` n'est pas légal ; aucun coût d'opportunité n'est calculé ;
- maîtrise à 29 : le gain est légal et peut débloquer le seuil 30, sans dépasser le plafond ;
- maîtrise à 30 : `GainMastery` n'est pas légal ;
- pouvoir déjà utilisé : l'action n'est pas légale ;
- aucune carte achetable avec les deux budgets : coût d'opportunité nul ;
- achat disponible avec `gems` mais aucun achat avec `gems - 1` : toute la valeur du meilleur achat
  accessible est perdue ;
- plusieurs cartes de même valeur : le résultat ne dépend pas de leur ordre ;
- carte de la main dont l'effet est déjà actif : aucune valeur de seuil artificielle ;
- plusieurs cartes franchissant le même seuil : leurs gains positifs sont additionnés une seule fois
  par carte, sans multiplier la valeur par le nombre de branches de l'effet ;
- seuil débloquant une branche conditionnelle mais non observable ou non supportée : valeur nulle et
  diagnostic, sans inventer de règle ;
- mercenaire achetable uniquement avant la dépense : sa valeur immédiate est incluse dans la perte ;
- rivière vide ou carte `None` : aucune action d'achat projetée ;
- observation détachée : le calcul ne doit jamais modifier l'état réel ni l'état observé ;
- égalité entre `GainMastery` et une autre action : le classement stable existant reste applicable.

## Testing Strategy

Ajouter des tests unitaires déterministes couvrant :

1. la valeur nulle lorsque la gemme ne change aucun achat accessible ;
2. la perte d'une carte chère accessible avec une gemme mais plus avec une gemme de moins ;
3. la comparaison entre `BuyCard` et `RecruitMercenary`, y compris lorsqu'un seuil de maîtrise
   améliore la valeur postérieure d'une carte de la rivière ;
4. le franchissement d'un seuil qui améliore plusieurs cartes de la main, avec addition des gains ;
5. le plafond de maîtrise et l'absence de seuil au-delà de 30 ;
6. l'absence de mutation de l'observation et du joueur pendant la projection ;
7. la sélection de `GainMastery` lorsque le gain net dépasse les cartes jouables ;
8. la conservation d'une carte jouable ou d'un achat lorsque le coût d'opportunité domine ;
9. la compatibilité de chargement des anciens profils sans les nouveaux coefficients ;
10. la reproductibilité des scores sur état et actions identiques.

Prévoir ensuite une petite campagne contrôlée avec des seeds fixes et trois scénarios :

- gemme nécessaire à une carte forte ;
- gemme dépensée pour débloquer un seuil décisif ;
- gemme sans impact sur la rivière.

La suite complète devra être exécutée avant toute optimisation des nouveaux coefficients.

## Rollout And Migration

1. Ajouter les features, les projections pures et les tests unitaires.
2. Ajouter les coefficients avec des valeurs par défaut compatibles, sans changer de profil publié
   automatiquement.
3. Vérifier les scénarios ciblés et comparer le comportement de `v004` avant/après sur des seeds
   appariées.
4. Lancer une campagne de calibration dédiée aux nouveaux coefficients, séparément de
   `durable_replay_factor` et des autres acquisitions.
5. Publier un nouveau profil uniquement après validation contre Random et contre la référence
   heuristique.

Le rollback consiste à recharger `v004` ou un profil antérieur. Les anciennes parties et les
anciens résultats restent lisibles car aucune structure moteur persistante n'est modifiée.

## Files Expected To Change

- `shards_ai/ai/heuristic_evaluator.py` — nouvelles features et coefficients ;
- `shards_ai/ai/heuristic_features.py` — projections d'achat et de seuil pour `GainMastery` ;
- `shards_ai/ai/heuristic_profiles.py` — compatibilité de chargement/sauvegarde si nécessaire ;
- `configs/heuristic_profiles/*.yaml` — uniquement après calibration et validation ;
- `tests/game/test_heuristic_player.py` — sélection comportementale ;
- `tests/game/test_heuristic_state_features.py` — signaux et absence de mutation ;
- `tests/optimization/test_heuristic.py` — profils, coefficients et campagnes ciblées ;
- `doc/Current state/Heuristic player.md` — comportement final après implémentation.

Le fichier d'architecture est historique et ne devra pas être réécrit pour suivre les détails de
l'implémentation finale ; l'état courant sera mis à jour après livraison.
