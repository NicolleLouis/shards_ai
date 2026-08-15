# État courant — Profils de joueurs

Cette page sert de guide rapide aux joueurs et profils versionnés présents dans le dépôt. Elle
sépare les profils heuristiques, les générations de modèles neural et le statut de chaque version.
Les numéros ne désignent pas une échelle de force : ils identifient des recettes ou des architectures
historiques.

## Vue d’ensemble

| Profil | Famille | Statut | Particularité principale |
|---|---|---|---|
| `v007` | heuristique | référence historique | Poids heuristiques optimisés, avec pondération séparée des achats et contraintes. |
| `v008` | heuristique | profil heuristique par défaut | Optimisation combinée des poids d’action, d’acquisition et de contraintes. |
| `v001` | neural | référence historique | Première imitation supervisée de `Heuristic v008`, actions encodées indépendamment. |
| `v002` | neural | référence historique | Dataset issu de cycles DAgger et contexte global des actions candidates. |
| `v003` | neural | référence historique | Encodeur sémantique structuré et chemin `semantic_identity_v3`. |
| `v004` | neural | contrôle atomique / rollback | Fusion normalisée entre identité et sémantique des cartes. |
| `v005` | neural | référence neural historique | Joueur macro/atomique unifié issu de l'imitation. |
| `v006` | neural | profil neural actif | Fine-tuning PPO de V005, même contrat macro/atomique et récompense terminale. |

Le joueur réellement actif pour le neural est déterminé par les deux pointeurs
`configs/neural_profiles/active.yaml` et `configs/neural_training_profiles/active.yaml`. Ils
pointent actuellement vers `v006`. Le profil heuristique par défaut du constructeur est `v008`.

Lorsqu'un profil historique est exécuté via `GameRunner`, il passe par
`LegacyActionMiddleware`. Cette couche conserve la compatibilité de la vue `PLAY/BUY` pour Random,
Heuristic, Neural et MacroNeural, sans modifier les checkpoints ; les nouveaux joueurs doivent
déclarer explicitement `full_main_v1` pour utiliser les décisions intercalables.

## Profils heuristiques

Tous les profils heuristiques utilisent la même implémentation `HeuristicPlayer`. Une action légale
est transformée en features puis évaluée par un produit scalaire avec des poids configurables. Le
moteur reste responsable de produire et de valider les actions légales ; le profil ne change donc
pas l’interface du moteur.

Les profils peuvent différer sur trois familles de paramètres :

- les poids d’action : ressources, dégâts, santé, pioche, champions, bannissement, progression et
  pénalités ;
- `card_acquisition_weights` : valeur durable d’une carte achetée, incluant notamment la pioche,
  l’épuration du deck et le potentiel de rejouabilité ;
- `constraint_weights` : risque ou valeur des contraintes d’effets comme `mastery`, `health`,
  `inspiration`, `echo`, `union` et `domination`.

### Heuristic V007

`v007` est une référence heuristique historique et le parent direct de `v008`. Il utilise les
features heuristiques complètes, mais son optimisation publiée a principalement retenu le
`buy_threshold` comme champ actif final. Son profil est conservé dans
`configs/heuristic_profiles/v007.yaml` pour les benchmarks et les comparaisons appariées.

### Heuristic V008

`v008` est le profil heuristique par défaut. Sa campagne a optimisé conjointement les poids
d’action, d’acquisition et de contraintes (`hybrid_racing_combined`). Parmi ses différences
effectives avec `v007`, le profil donne davantage de poids au `card_draw`, aux dégâts et au coût
d’opportunité d’achat, tout en conservant un seuil d’achat durable de `0.625`.

Les valeurs exactes restent celles du YAML :
`configs/heuristic_profiles/v008.yaml`. Il ne faut pas interpréter V008 comme une nouvelle classe
de joueur : c’est une nouvelle calibration de la même politique heuristique.

## Profils neural historiques

Les joueurs neural observent uniquement `NeuralObservation` et ne génèrent pas eux-mêmes les actions.
Ils scorent les actions fournies par le moteur. Les profils historiques restent chargeables afin de
servir de références expérimentales et de panel de benchmark.

| Version | Architecture / contexte | Données ou objectif | Rôle actuel |
|---|---|---|---|
| `v001` | `independent_action` : chaque action est encodée indépendamment | Imitation supervisée de `Heuristic v008` | Baseline neural historique. |
| `v002` | `global_candidate_context` : chaque action reçoit un résumé du groupe de candidats | DAgger et itérations d’imitation | Référence historique ; conserve la compatibilité avec son architecture propre. |
| `v003` | `semantic_identity_v3` | Encodeur sémantique/identité des cartes, split par `game_id` | Référence historique d’une représentation structurée. |
| `v004` | `structured_semantic_v4` puis fusion V5 expérimentale | Fusion L2 identité/sémantique, `id=0.5`, `semantic=1.5` | Contrôle atomique et rollback explicite. |
| `v005` | `structured_semantic_v5_macro_tactical_action_v1` | Imitation macro/atomique depuis V004 | Référence parent historique de V006. |
| `v006` | `structured_semantic_v5_macro_tactical_action_v1` | Fine-tuning PPO depuis V005 | Profil neural actif. |

### Neural V001

V001 est la première génération neural : elle apprend à reproduire le choix du teacher heuristique,
principalement V008, sur les décisions enregistrées. Son contexte d’action est
`independent_action`, ce qui signifie qu’une action est représentée sans résumé explicite de ses
concurrentes.

### Neural V002

V002 conserve la logique générale d’imitation mais ajoute un contexte global des candidats. Son
dataset inclut des cycles DAgger, destinés à exposer le modèle à des états rencontrés par la
politique neural plutôt qu’aux seuls états du teacher initial. Cette version est historique et ne
doit pas être confondue avec le PPO macro actuellement disponible.

### Neural V003

V003 introduit le chemin `semantic_identity_v3`. Les cartes sont décrites par leur identité de
définition et par des caractéristiques sémantiques structurées, plutôt que par une représentation
entièrement plate. Cette version reste une politique atomique historique.

### Neural V004

V004 utilise l’encodeur structuré et la fusion séparée des voies identité et sémantique. La
configuration publiée applique une normalisation L2 avec les échelles `0.5` pour l’identité et
`1.5` pour la sémantique. V004 constitue le contrôle atomique à partir duquel V005 a été migré ; il
n’est pas le profil neural actif.

## Neural V005 : référence parent

V005 est une évolution de représentation et de granularité, pas seulement une nouvelle calibration
des poids. Il utilise `PlayTurnSolver` pour regrouper certains segments de la phase PLAY en
candidats macro, puis évalue ces candidats avec un scoreur action-conditionnel.

Ses caractéristiques principales sont :

- représentation de l’action racine ;
- conséquences connues de la branche, comme les deltas de ressources et de zones ;
- features tactiques action-conditionnées liées notamment à Union, Echo et Domination ;
- représentation commune des décisions `macro_play` et `atomic` ;
- suppression des identifiants d’instance et des cartes inconnues révélées par une pioche future ;
- regroupement de variantes physiques équivalentes avant le scoring ;
- validation finale des actions par `Game.apply()`.

Lorsqu’une résolution ne comporte qu’une seule candidate, elle est rejouée automatiquement sans
appel neural. Les actions automatiquement rejouées ne constituent pas des décisions
d’apprentissage. Si l’abstraction macro atteint ses budgets, le joueur utilise les candidats
atomiques unifiés restants.

Les limites nominales du solveur sont de 256 expansions, 128 états mémoïsés, 16 candidats macro et
32 actions atomiques par segment. Le profil et la recette correspondants sont
`configs/neural_profiles/v005.pt` et `configs/neural_training_profiles/v005.yaml`.

## Neural V006 : profil actif

V006 est le profil neural actif. Il conserve l'architecture
`structured_semantic_v5_macro_tactical_action_v1` et le contrat de décisions macro/atomiques de
V005, mais ajoute un fine-tuning PPO depuis V005. Le profil utilise `learning_rate=0.0005`,
`gamma=1`, `gae_lambda=1` et une récompense strictement terminale victoire/défaite.

Le checkpoint stable est `configs/neural_profiles/v006.pt` et sa recette est
`configs/neural_training_profiles/v006.yaml`. Le checkpoint mutable d'entraînement reste
`artifacts/neural_training/checkpoint.pt` et ne doit pas être entraîné directement dans le dossier
stable.

## Ce que les versions ne signifient pas

- `Heuristic V007` et `Heuristic V008` sont deux calibrations de la même politique heuristique.
- `Neural V001` à `V004` sont des générations de représentation et d’entraînement ; elles ne sont
  pas nécessairement strictement ordonnées par force de jeu.
- `Neural V005` change aussi la granularité des décisions avec le solveur macro. Une comparaison
  directe avec une ancienne version doit donc tenir compte du nombre et du type de décisions.
- Une meilleure loss ou un meilleur top-1 offline ne suffit pas à promouvoir un profil : la preuve
  de qualité repose sur des parties complètes, reproductibles et comparées au panel prévu.

## Références

- [Joueur heuristique](Heuristic%20player.md) — comportement détaillé et optimisation des poids.
- [Joueur neural](Neural%20player.md) — pipeline, runtime, benchmark et contrat macro.
- [Encodeur sémantique structuré](../Architecture/070-encodeur-semantique-structure-cartes.md) —
  tableau historique des architectures V001 à V004.
- [Représentation et apprentissage macro](../Architecture/081-representation-et-apprentissage-macro-neural-v2.md)
  — évolution vers la représentation macro/action-conditionnée.
- [PPO macro](../Architecture/085-ppo-joueur-macro-play-turn-solver.md) — fine-tuning PPO de la
  voie macro, distinct de la définition du profil V005.
