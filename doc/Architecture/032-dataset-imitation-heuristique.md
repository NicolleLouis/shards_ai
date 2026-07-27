# Dataset d'imitation de l'heuristique

## Objectif

Définir le pipeline de génération du dataset initial d'imitation pour le réseau neural. Chaque
exemple doit décrire une décision atomique d'un joueur heuristique, avec l'observation partiale
accessible au joueur, toutes les actions légales, leurs représentations et leurs scores heuristiques,
puis le résultat final de la partie.

Le dataset doit être rejouable, versionné, mélangeable entre plusieurs profils et exploitable en
streaming sans transmettre d'information cachée au futur modèle.

## Cible de volume

La métrique principale est le nombre de points de décision heuristiques, et non le nombre de
parties. Une mesure rapide sur 20 parties `v007` contre RandomPlayer donne environ 340 décisions
totales par partie, soit environ 170 décisions heuristiques par partie lorsque les rôles sont
alternés.

Les cibles de la première campagne sont :

- validation technique : 5 000 décisions heuristiques ;
- premier dataset d'entraînement sérieux : 1 000 000 de décisions heuristiques ;
- extension ultérieure possible : 5 000 000 si la diversité ou la validation montrent que le
  million est insuffisant.

Avec le mélange actuel, le million représente approximativement 3 000 à 6 000 parties selon la
proportion de rencontres heuristique/heuristique et heuristique/RandomPlayer. Le générateur doit
donc pouvoir s'arrêter sur `target_decisions` en plus d'une limite de parties.

Le manifest enregistrera la cible demandée, le nombre réellement produit et la distribution par
profil et par type de rencontre. Une campagne ne sera considérée prête pour l'entraînement qu'après
la validation technique et l'inspection d'un échantillon ; la cible d'un million est un objectif de
volume, pas une garantie de qualité.

## État actuel

`GameRunner.run()` expose déjà un `decision_observer` appelé avant chaque transition. Il reçoit :

- une observation complète détachée utilisée par les joueurs actuels ;
- la liste des actions légales ;
- l'action choisie ;
- le joueur actif.

`HeuristicPlayer` dispose de `score_action()` et `features_for_action()`, mais sa décision complète
peut aussi dépendre de priorités ou filtres complémentaires : victoire terminale, action létale,
seuil d'achat, protection de bannissement, priorité de phase et départage.

Les profils sont chargés avec `load_profile()` depuis `configs/heuristic_profiles/*.yaml`. Les
résultats et artefacts d'optimisation existants sont stockés sous `artifacts/` et les analyses sous
`analysis_output/`. Aucun de ces répertoires ne doit être utilisé pour stocker de la documentation.

## Comportement cible

Le générateur exécute des parties avec des profils heuristiques explicitement fournis. À chaque
décision où le joueur actif est heuristique, il capture un exemple avant `Game.apply()`.

Les décisions d'un `RandomPlayer` ne sont pas utilisées comme labels d'imitation, mais ses parties
peuvent fournir le contexte d'entraînement lorsqu'il affronte l'heuristique.

Chaque exemple contient au minimum :

```text
dataset_schema_version
game_id
game_seed
decision_index
turn_number
acting_player_role
heuristic_profile_id
heuristic_profile_path
opponent_type
opponent_profile_id éventuel
observation_masked
legal_actions
action_representations
heuristic_scores
heuristic_ranks éventuels
chosen_action_index
chosen_action
final_outcome_from_acting_player_view
```

L'observation stockée est exclusivement `Game.neural_observation_for(active_player)`. L'état complet
utilisé par l'heuristique et le moteur ne doit jamais être sérialisé dans le dataset.

## Mélange des parties

La configuration de campagne doit accepter une liste de profils validés et des proportions de
rencontres, par exemple :

- profil heuristique contre `RandomPlayer` ;
- profil heuristique contre un autre profil heuristique ;
- versions heuristiques différentes dans les deux rôles ;
- alternance du joueur actif et des positions de départ.

Les profils candidats non validés ne doivent pas être inclus par défaut. La version active du
`Makefile` ne doit pas être modifiée automatiquement par le générateur.

Les seeds de parties doivent être dérivées de manière déterministe à partir d'une seed racine et de
l'index de partie, comme dans les benchmarks existants. Une même configuration relancée avec la
même seed doit produire le même dataset, sous réserve d'un ordre d'écriture identique.

## Score et classement heuristique

Pour chaque action légale, conserver le score brut retourné par l'évaluation heuristique, sans
normalisation destructive. Les normalisations et fonctions de perte seront expérimentées au step
d'entraînement.

Le score brut n'est pas nécessairement suffisant pour reproduire l'action choisie :

- une victoire terminale ou une action létale peut être prioritaire ;
- certains achats sont filtrés par `buy_threshold` ;
- certaines cartes ne doivent pas être bannies ;
- la phase et les départages interviennent dans le choix final.

Le dataset doit donc conserver séparément :

- `heuristic_score` pour chaque action légale ;
- l'action choisie par la politique complète ;
- le rang brut si calculé ;
- le rang ou les composantes de sélection complètes si l'API de diagnostic les expose.

Le générateur ne doit pas recalculer une politique simplifiée qui divergerait de
`HeuristicPlayer.choose_action()`. Si le classement complet n'est pas encore exposé proprement, le
dataset conserve l'action choisie et les scores bruts, et marque le rang complet comme indisponible.

## Format de stockage

Le format recommandé est un JSONL, un enregistrement par décision, afin de permettre :

- la lecture streaming pendant l'entraînement ;
- l'inspection d'un exemple isolé ;
- l'ajout de statistiques sans charger toutes les parties en mémoire ;
- la reprise ou la suppression d'une partie complète via `game_id`.

Les observations, actions et représentations utiliseront les sérialisations versionnées des steps
8.b et 8.d. Les identités de cartes utilisées par les actions sont complétées avec le descripteur
de carte ou sa référence de catalogue selon le contrat retenu par l'encodeur ; le dataset doit
conserver suffisamment d'information pour régénérer le card embedding du step 8.c.

Une partie est d'abord bufferisée jusqu'à connaître son résultat terminal. Les enregistrements ne
sont ajoutés au dataset final qu'après la fin de la partie, afin qu'aucune ligne ne manque son label
de victoire, défaite ou partie nulle. Une interruption ne doit pas produire de lignes prétendument
valides sans résultat final.

Les sorties de dataset doivent rester hors de `doc/`, par exemple sous `artifacts/imitation_dataset/`.
Les gros datasets peuvent être compressés après génération, mais le format logique reste JSONL.

## Label de résultat

Le résultat final est exprimé du point de vue du joueur qui a pris la décision :

```text
win
loss
draw
```

Une partie terminée par `GameStatus.DRAW` produit `draw`. Une erreur de simulation ou une partie
incomplète est exclue du dataset final et enregistrée séparément dans les statistiques de campagne.

Le résultat final ne remplace pas les scores heuristiques dans cette première phase ; il prépare
l'évaluation future et l'évolution vers le reinforcement learning.

## Architecture proposée

Créer un pipeline séparé du moteur et des joueurs :

```text
CampaignConfig
    ↓
seed planner
    ↓
Game + joueurs heuristiques/random
    ↓ decision_observer avant apply
masked observation + legal actions
    ↓
heuristic score collector + action representations
    ↓
per-game buffer
    ↓ résultat final
JSONL writer + manifest + statistiques
```

Le collector doit :

1. vérifier que l'action choisie appartient à `legal_actions` ;
2. construire l'observation masquée à partir de l'état courant ;
3. calculer le score de chaque action légale avec le profil de l'acteur ;
4. construire les représentations d'action dans le même ordre ;
5. enregistrer l'index de l'action choisie ;
6. ne conserver comme exemple que les décisions de joueurs heuristiques ;
7. finaliser les exemples avec le résultat relatif du joueur après la partie.

Le manifest du dataset contiendra :

- seed racine et méthode de dérivation des seeds ;
- profils utilisés, chemins et identifiants ;
- versions des schémas observation, carte et action ;
- version du catalogue de cartes ou empreinte du catalogue ;
- configuration du mélange d'adversaires ;
- nombre de parties tentées, terminées, exclues et décisions produites ;
- compteurs win/loss/draw et erreurs.

## Fuite d'information et reproductibilité

Le collector peut recevoir un `GameState` complet parce qu'il doit appeler le score heuristique, mais
il ne doit sérialiser que l'observation neural et les champs d'action autorisés. Des tests devront
vérifier qu'aucune représentation JSON ne contient les cartes de la main ou de la pioche adverse.

Les partitions train/validation/test seront faites par partie ou par seed, jamais par décision
individuelle. Sinon des décisions successives d'une même partie pourraient se retrouver dans des
partitions différentes et créer une fuite temporelle.

Les profils et versions devront être conservés dans le manifest. Un dataset ne doit pas être
reconstruit avec un profil dont le fichier a été modifié sans que son identité ou son empreinte
change.

## Performance et volume

La génération peut produire plusieurs dizaines de décisions par partie, et chaque décision contient
plusieurs actions légales. Le pipeline doit donc :

- utiliser les observations et représentations sérialisables plutôt que des copies arbitraires du
  moteur ;
- éviter de recalculer les représentations de cartes immuables ;
- écrire en streaming après finalisation de chaque partie ;
- proposer un mode de validation court avant les campagnes longues ;
- séparer le nombre de parties, le nombre de décisions et la taille finale du fichier dans les
  statistiques.

Le générateur ne doit pas lancer automatiquement une campagne overnight ni publier de profil
heuristique. Sa responsabilité s'arrête à la production d'un artefact explicitement demandé.

## Cas particuliers

- partie terminée pendant l'action précédente : la dernière décision reçoit le résultat final ;
- partie nulle par limite de tours : label `draw` si la partie est correctement finalisée ;
- erreur moteur ou action invalide : partie exclue, erreur conservée dans le manifest ;
- aucun joueur heuristique dans une partie : partie valide mais zéro exemple d'imitation ;
- plusieurs joueurs heuristiques avec des profils différents : chaque décision porte le profil de
  son acteur ;
- égalité heuristique : conserver l'action réellement choisie et le mécanisme de départage dans les
  métadonnées si disponible ;
- évolution du catalogue : incompatible si une carte utilisée ne peut plus être résolue ; le
  générateur doit échouer explicitement plutôt que produire une représentation vide.

## Stratégie de test

Ajouter des tests qui vérifient :

- un exemple par décision heuristique et aucun exemple pour les décisions RandomPlayer ;
- présence de toutes les actions légales et de leurs scores ;
- correspondance positionnelle actions/représentations/scores ;
- action choisie et index cohérents ;
- résultat relatif win/loss/draw ;
- absence de données adverses cachées dans le JSONL ;
- reproductibilité d'une campagne courte avec une seed fixe ;
- séparation par partie pour les partitions ;
- exclusion des erreurs et parties incomplètes ;
- manifest cohérent avec le nombre de parties et de décisions ;
- impossibilité de publier ou modifier automatiquement un profil heuristique.

## Questions ouvertes

Aucune question bloquante pour l'architecture du générateur. La taille de campagne initiale, la
compression et la stratégie exacte de normalisation des scores seront réglées expérimentalement
après production d'un premier dataset court.

## Fichiers attendus

- `shards_ai/ai/imitation_dataset.py` — configuration, collector, writer et manifest ;
- `scripts/generate_imitation_dataset.py` — interface CLI ;
- `tests/ai/test_imitation_dataset.py` — tests du pipeline et de la reproductibilité ;
- `artifacts/imitation_dataset/` — sorties générées, jamais versionnées dans `doc/` ;
- `doc/Roadmap.md` — passage du step 8.e à `DONE` après validation.

## Validation attendue

Générer un petit dataset de validation avec plusieurs profils et adversaires, inspecter quelques
exemples JSONL, vérifier l'absence de fuite et comparer deux générations à seed identique. Exécuter
ensuite la suite complète avant toute campagne plus longue.
