# Neural player

## État

Le socle d'imitation supervisée et son fine-tuning PPO sont disponibles. Ils utilisent PyTorch et
reçoivent uniquement `NeuralObservation`, l'observation masquée construite par le moteur, ainsi
que les représentations des actions légales produites par `representation_for_neural_action`.

Les entraînements sont décrits par des profils YAML versionnés sous
`configs/neural_training_profiles/`. `v001.yaml` indique le dataset, les seeds, les hyperparamètres,
le split et la configuration du modèle. `scripts/train_neural_imitation.py --profile <profile.yaml>`
charge cette recette ; les options CLI peuvent la surcharger pour une expérience ponctuelle. Le
seul checkpoint mutable est `artifacts/neural_training/checkpoint.pt`, partagé par l'entraînement,
la reprise, le reporting, le benchmark et la validation. Le checkpoint et le fichier de métriques
enregistrent l'identifiant du profil, son fingerprint et celui de la configuration effective. Les
checkpoints promus sont versionnés sous `configs/neural_profiles/` et ne sont plus entraînés ; les
datasets et rapports restent sous `artifacts/` et ne doivent pas être versionnés dans Git.

Le profil stable actif est `v006`, issu d'un fine-tuning PPO macro
`structured_semantic_v5_macro_tactical_action_v1` depuis V005. Il est conservé dans
`configs/neural_profiles/v006.pt`; V005 reste la référence parent immédiate, V004 le contrôle
atomique historique, et V001 à V005 restent des références historiques protégées.

## Modèle

`NeuralActionScorer` encode les cartes avec une voie d'identité (`card_definition_id`) et une voie
sémantique structurée, agrège les zones de cartes comme ensembles/multiensembles, puis calcule un
score action-conditionnel pour chaque action. Il ne génère aucune action et ne remplace pas
`Game.legal_actions()`.

Les dimensions sont configurables via `NeuralModelConfig`. Le modèle fonctionne sur CPU et peut
être déplacé sur un autre device PyTorch. L'inférence de production devra utiliser `torch.no_grad()`
et mettre en cache les tenseurs préparés si les mesures de latence le justifient.

Un feature set expérimental opt-in `zone_cardinality_v1` est disponible dans `NeuralModelConfig`. Il
ajoute sept scalaires de cardinalité pour les zones de recyclage et les champions du joueur actif,
son total possédé et les zones publiques agrégées de l'adversaire. Les tailles de la main, de la
zone de jeu, de la rivière et du deck central ne sont pas ajoutées comme scalaires. La valeur par
défaut `baseline` conserve la dimension et le comportement des architectures historiques ; le
feature set expérimental nécessite un entraînement compatible et n'est pas utilisé par le profil
actif tant qu'il n'a pas passé la validation.

Une architecture expérimentale V5, `structured_semantic_v5_fusion_experiment`, est disponible via
`StructuredSemanticV5FusionScorer`. Elle réutilise l'encodeur sémantique structuré V4, mais isole
la fusion des voies identité et sémantique avec une normalisation L2 et des scales configurables.
L'ablation `without_identity` a été testée puis rejetée dans `exp-00101` ; son support a été retiré.
`StructuredSemanticV4Scorer` et ses checkpoints restent inchangés. Le profil actif V5 utilise une
normalisation L2 par voie avec `card_fusion_id_scale=0.5` et `card_fusion_semantic_scale=1.5`.

Le scoreur expérimental `structured_semantic_v5_macro_deck_state_v1` choisit entre les candidats
macro produits par `PlayTurnSolver`. Il réutilise l'encodeur structuré avec le feature set
`deck_state_v1`, donc les sept cardinalités de zones et les quatre comptes factionnels actifs sont
présents dans son état. Les candidats sont encodés par histogramme des types de trace, phase, état
terminal et conséquences numériques. Il est entraîné séparément du scoreur atomique avec
`scripts/train_macro_imitation.py` et le profil candidat
`configs/neural_training_profiles/candidates/exp00109-macro-v8-deck-state.yaml`.

## Fine-tuning PPO V006

V006 conserve le contrat macro/atomique et l'architecture de V005. Il a été initialisé depuis
`configs/neural_profiles/v005.pt`, puis entraîné avec un PPO à récompense terminale uniquement,
`gamma=1`, `gae_lambda=1` et `learning_rate=0.0005`. La validation paired de 100 parties par
adversaire a produit un gain pondéré de `+1,125 point` contre V005 ; le candidat a ensuite été
promu dans `configs/neural_profiles/v006.pt`.

Le panel de qualité comprend V007, V008 et Neural V001 à V006. Ses poids sont respectivement
`1,5`, `2`, `0,5`, `0,5`, `0,25`, `0,25`, `1`, `1`. Random reste diagnostique et hors gate.

## Entraînement

`neural_training.py` lit le JSONL en streaming. Les losses disponibles sont : préférence
paire-à-paire, imitation de l'action choisie et régression optionnelle des scores normalisés par
décision. `train_neural_imitation.py` produit un checkpoint hors de `doc/` et partitionne les
décisions par `game_id` de façon déterministe.

`macro_training.py` lit le dataset unifié en streaming. Le lecteur strict V4 consomme les décisions
`macro_play` et `atomic`, exclut les candidats uniques, et conserve un split groupé par `game_id`.
`train_macro_imitation.py` utilise des pondérations `decision_kind` déclarées dans le profil,
rapporte les métriques par type de décision, phase, action, matchup et collisions, initialise par
défaut les modules communs depuis V004 après migration vers `deck_state_v1`, puis écrit le
checkpoint mutable canonique et ses métriques. Le dataset unifié n'est pas compatible avec
`train_neural_imitation.py`.

Chaque epoch peut maintenant être suivi dans les fichiers `.metrics.json` et `.metrics.csv` ; un
fichier `.svg` est également généré par défaut. Les métriques de validation incluent la loss,
l'accord top-1, le rang moyen de l'action choisie, son rang normalisé et la précision paire-à-paire.
L'évaluation peut être plafonnée avec `--max-validation-records` pour contrôler la durée.

Un rapport `.html` autonome est également généré. Il contient les graphiques de progression,
un résumé du meilleur epoch et un tableau détaillé. Il peut être régénéré depuis un JSON existant
avec `scripts/generate_neural_training_report.py`.

Le checkpoint inclut aussi l'état de l'optimiseur et l'epoch atteint. Le training peut donc être
repris avec `--resume-from artifacts/neural_training/checkpoint.pt` ; les métriques précédentes
sont conservées et les nouveaux epochs sont ajoutés au JSON, au CSV et au rapport HTML.

Pendant la collecte PPO et chaque update, l'actor et le critic partagent l'encodage de l'observation
et des actions légales afin d'éviter un second forward identique ; cela ne change pas le contrat
d'inférence gloutonne de `NeuralPlayer`.

Le chemin CPU du modèle batch les embeddings des cartes distinctes par décision afin d'éviter de
recalculer plusieurs fois les mêmes cartes dans les zones et les actions. Les caractéristiques
sémantiques statiques du catalogue sont stockées dans un buffer de modèle non persisté et
sélectionnées par index à chaque décision, ce qui réduit les allocations sans modifier le
checkpoint ni le résultat observable. En évaluation sous `torch.inference_mode()`, les vecteurs
d'embedding fusionnés sont maintenant conservés par scorer pour éviter leur reconstruction entre
décisions ; ce cache n'est pas utilisé pendant l'entraînement. Le script d'entraînement
utilise par défaut un thread PyTorch (`--torch-threads`) adapté aux petits forwards successifs ;
ce paramètre reste configurable pour les autres machines et workloads.

En évaluation sous `torch.inference_mode()`, les représentations d'actions déjà encodées sont
également conservées par scorer et réutilisées lorsque la même représentation réapparaît. Ce cache
est désactivé pendant l'entraînement ; il ne modifie ni les poids, ni les sorties initiales, ni le
contrat des actions légales.

Le `NeuralPlayer` charge un checkpoint, reçoit uniquement `NeuralObservation` via `GameRunner`,
représente les actions légales depuis les zones publiques, puis choisit le meilleur score sous
`torch.inference_mode()`. Les égalités sont départagées par un `GameRandom` injectable. Le benchmark
`benchmarks/benchmark_neural_players.py` mesure les victoires, défaites, matchs nuls, actions,
latence d'inférence et temps total contre Random, un profil heuristique ou un autre checkpoint
neural. Il utilise par défaut un
thread PyTorch (`--torch-threads 1`) adapté à ce hot path de petites inférences successives.
Pour une campagne, les scorers du NeuralPlayer et de l'adversaire neural sont chargés une seule fois
et réutilisés entre les parties ; les seeds et les RNG de chaque partie restent indépendants.

`benchmarks/benchmark_neural_mix.py` lance une campagne mixte déterministe (20 % Random, 50 % v007,
30 % v008) et génère `neural_mix.json` ainsi qu'un rapport HTML. Ce rapport compare les adversaires
et détaille durée, nombre total de tours, tours moyens par joueur, actions, maîtrise, PV et
composition agrégée des decks finaux des deux joueurs. Le nombre de tours par joueur est calculé
comme le nombre total de tours divisé par le nombre de joueurs du runner. Il détaille aussi, pour le NeuralPlayer, les mercenaires recrutés immédiatement ou achetés
à long terme, ainsi que le taux d'activation de `GainMastery` par tour et par maîtrise disponible.
Le résumé compare également la taille finale moyenne du deck Neural et de celui de l'adversaire,
leur delta, et le taux de parties dans lesquelles le NeuralPlayer a passé la phase de jeu alors
qu'une action `PlayCard` restait légale dans sa main. Ce dernier indicateur mesure un comportement
observable, sans prétendre déterminer si jouer la carte était stratégiquement optimal.
Le rapport inclut également, pour chaque adversaire, neuf graphiques SVG par multiplicité centrale
(`×1`, `×2`, `×3`) : moyenne NeuralPlayer, moyenne adverse et delta des copies moyennes. Les cartes
de chaque groupe sont triées par moyenne décroissante, ou par valeur absolue du delta décroissante.
Le JSON et le HTML indiquent aussi le nombre moyen de décisions neural par partie ; avec un
checkpoint macro, ils séparent les décisions macro du PLAY et les décisions atomiques unifiées.

`benchmarks/benchmark_neural_panel.py`, appelé par `make neural-benchmark-panel`, applique la même
instrumentation à un panel fixe composé de Random, Heuristic v007, Heuristic v008 et des profils
neural v001, v002, v003, v004, v005 et v006. Il joue un nombre égal de parties contre chaque adversaire, conserve
chaque partie dans `artifacts/neural_benchmark/neural_panel.json` et produit le rapport autonome
`artifacts/neural_benchmark/neural_panel.html`. Le profil testé est v006 par défaut et peut être
remplacé par `NEURAL_PANEL_CHECKPOINT` pour analyser le checkpoint mutable ou une autre version.

Le self-play n'est pas encore implémenté. Les anciennes cibles Makefile PPO et DAgGER ont été
retirées car elles pointaient vers des recettes candidates historiques. Les nouveaux entraînements
sont orchestrés comme des expériences candidates avec `v003` comme parent explicite. Le checkpoint
mutable reste `artifacts/neural_training/checkpoint.pt`; les checkpoints stables sous
`configs/neural_profiles/` ne sont jamais entraînés en place. Les pointeurs actifs désignent
actuellement `v006` et son architecture `structured_semantic_v5_macro_tactical_action_v1`.

`scripts/validate_neural_profile.py` compare un candidat au dernier profil neural actif sur les mêmes seeds,
contre `v007`, `v008` et les profils neural v001, v002, v004, v005 et v006 dont les
checkpoints existent. Il imprime les résultats par adversaire avec deux décimales et les comptes
victoires/parties. La gate pondère v007 à `1`, v008 à `2`, v001/v002/v004 à `0,5` et v005/v006
à `1` ; Neural V003 et Random sont exclus de la gate.
Le défaut Makefile est de 100 parties par adversaire ; un panel d'au moins 200
parties est recommandé pour une promotion finale. Le script propose de ne promouvoir le candidat
que par rapport au profil neural actif courant (`v006` actuellement) ; `v008` reste l'adversaire
heuristique protégé et ne constitue pas la référence neural remplacée à chaque promotion.
La promotion est autorisée si la moyenne pondérée des deltas de tous les adversaires est strictement
positive, y compris lorsqu'elle contient une baisse contre v008. Random n'appartient pas à cette gate, mais reste
disponible dans les benchmarks diagnostiques. Chaque adversaire compte une fois, indépendamment
du nombre de parties jouées ; les résultats sont d'abord agrégés par adversaire. Les résultats de
catégories éventuelles sont conservés mais ne modifient pas cette gate. Cette validation
peut être exécutée hors de Codex ; le résultat doit alors être conservé ou fourni avant promotion.
Une promotion crée le prochain
profil versionné et met à jour `active.yaml` ; un rejet ne modifie aucun profil.

Pour les campagnes d'entraînement longues, le profil actif et son checkpoint stable constituent le
contrôle : il est inutile de réentraîner une copie du contrôle pour chaque seed candidate lorsque le
dataset et le protocole de comparaison restent inchangés. Un contrôle séparé n'est nécessaire que
si l'expérience change ces éléments au point de rendre le profil actif incomparable. Toute campagne
multi-seed doit également prévoir une durée cumulée inférieure à 15 heures avant son lancement ; le
nombre de seeds, d'epochs, de décisions d'entraînement ou de parties doit être réduit si cette
estimation est dépassée.

Le générateur d'imitation accepte aussi des matchups ciblés avec `--opponent-profile`. Les profils
passés par `--profile` sont alors les seuls teachers enregistrés ; les adversaires servent uniquement
à jouer les parties. `--target-decisions` compte donc les lignes du teacher, ce qui permet par
exemple de générer un dataset v008 contre Random et v007 sans introduire de labels v007.

La promotion recopie le checkpoint mutable unique dans `configs/neural_profiles/vNNN.pt`, synchronise
ses métadonnées de profil et crée `configs/neural_training_profiles/vNNN.yaml`. Un checkpoint PPO
conserve aussi l'état actor-critic et de l'optimiseur pour la reprise. Le benchmark
`benchmarks/benchmark_neural_players.py` accepte `--opponent neural` et
`--opponent-checkpoint`, ce qui permet de comparer deux versions historiques sans dépendre du
profil actif.

## Abstraction macro expérimentale des play turns

`MacroNeuralPlayer` utilise `PlayTurnSolver` pour canoniser les segments de `PLAY` dont les effets
fixes ou les conditions déjà actives ne créent pas de choix stratégique. Les effets conditionnels
restent des branches si leur seuil maximal, leur Union, leur Domination, leur santé ou leur présence
de champion n'est pas garantie. Echo/Spectra reste volatile lorsqu'une action disponible peut
piocher ou modifier la défausse. Les actions atomiques produites par le solveur sont toujours
rejouées et validées par `Game.apply()`.

Les actions physiques équivalentes sont regroupées avant le scoring : même définition et même zone
logique donnent un représentant déterministe, avec `physical_variant_count` conservé dans la
candidate et le payload de décision. Les `instance_id` restent dans la trace atomique rejouée mais
ne sont pas exposés au tenseur. Les cartes de la pioche, les slots de la rivière et les champions
dont l'état d'activation diffère restent distincts. Les cibles `BanishCard` identiques dans une même
zone suivent la même canonisation ; `SkipBanish` et les définitions différentes restent distincts.

Les budgets sont des constantes fixes du solveur : 256 expansions, 128 états mémoïsés, 16 candidates
macro et 32 actions atomiques par segment. Un dépassement utilise les candidats atomiques unifiés. Le hook
`GameRunner.macro_decision_observer` expose un payload uniquement lorsqu'une candidate macro est
choisie ; les actions automatiquement rejouées ne produisent pas de décisions d'apprentissage.
Une résolution ne contenant qu'une candidate est également rejouée automatiquement, sans appel au
scorer, sans compteur `macro_decisions` et sans payload.

Le scorer macro neural V4 et son entraînement dédié constituent désormais la voie macro active par
défaut via `configs/neural_profiles/v006.pt`. Le contrat structured_semantic_v5_macro_tactical_action_v1 conserve l'identité racine et
les conséquences connues V3, puis ajoute au candidat PlayCard les features tactiques V6
action-conditionnées Union, Echo et Domination.
Le contrat V3 conserve, pour chaque branche, les deltas bornés de ressources et de zones, les choix pending,
les masques de factions/champions, la victoire immédiate et les définitions de cartes résolues depuis
la vue pré-décision. Les identifiants d'instance et les cartes révélées par une pioche future ne
parviennent jamais au tenseur. Les schémas V1/V2 restent lisibles pour les artefacts historiques.
Le sélecteur par défaut est déterministe et sert uniquement à mesurer le coût de l'abstraction. Sur un match seedé
`104` contre Heuristic V8, trois répétitions donnent une médiane de 175 actions,
22 tours, 39 décisions exposées et 80.3 ms. L'optimisation de `Game.clone()` utilise la copie
manuelle de l'état et une copie explicite du flux RNG ; elle a réduit le solveur de 315.2 ms à 13.3 ms
sur le même workload avant l'arrêt de la campagne.

La génération du dataset macro utilise désormais le schéma candidat V4 et la représentation avec
conséquences connues et tactiques. Elle ajoute l'action racine masquée au résumé macro et retire
card_instance_id et choice_id avant encodage ; les représentations historiques V1/V2 restent
disponibles pour lecture. Le nouveau profil d'entraînement est
configs/neural_training_profiles/candidates/exp00112-macro-v4-tactical-action.yaml. Le profil stable
correspondant est `configs/neural_training_profiles/v006.yaml` et son pointeur actif est `v006`.

La couverture atomique unifiée est implémentée : hors PLAY et lorsque le solveur ne
peut pas abstraire une décision, les actions légales sont transformées en candidats de trace de
longueur 1 avec `decision_kind=atomic` et présentées au même scoreur V4. Les champs de conséquences
macro sont nuls pour ces candidats ; les features tactiques applicables restent action-conditionnées.

## Analyse offline de l'imitation

`scripts/analyze_neural_imitation.py` évalue un checkpoint sans entraînement sur les représentations
d’actions sérialisées dans le dataset d’imitation. Il score toutes les actions légales avec
`NeuralActionScorer` sous `torch.inference_mode()`, en streaming, puis écrit un rapport HTML autonome
et, sur demande, un JSON dans `artifacts/analysis/`. Par défaut, `--split non_train` exclut le split
train déterminé par le même hash de `game_id` que l’entraînement ; `--split validation`, `test` ou
`all` permet de sélectionner un autre périmètre.

Le rapport fournit l’accord top-1, l’accord top-3, le score heuristique moyen du choix teacher et le
regret moyen `H(s,a_H) - H(s,a_NN)`. Les résultats sont ventilés par phase, par familles Achat,
Attaque, Recrutement et Ciblage, ainsi que par `action_type` exact. Ciblage est un regroupement
non exclusif lorsqu’une action porte une cible. La cible `make neural-imitation-analysis` utilise
le dataset v008 contre Random/v007 et le checkpoint mutable `NEURAL_CHECKPOINT`.

## Analyse des états réellement visités

La cible `make neural-visited-state-analysis` joue le NeuralPlayer contre v008 et utilise le hook
`GameRunner.decision_observer` pour analyser uniquement les décisions effectivement rencontrées par
le NeuralPlayer. Sur chaque état, elle demande contrefactuellement à v008 son action sans appliquer
cette action, puis calcule l’accord top-1/top-3, le rang v008 dans les logits neural, le regret
heuristique et la première divergence de chaque partie. Les sorties JSON/HTML sont écrites sous
`artifacts/analysis/`. Cette mesure est conditionnelle à la trajectoire du NeuralPlayer et ne simule
pas une seconde branche de partie.

Le training d'imitation utilise l'implémentation CPU `foreach` d'Adam. Elle
réduit le coût de dispatch des mises à jour effectuées à chaque décision sans
modifier la taille des lots, l'ordre des records, la loss ou le nombre de
threads configuré.

## Joueur composé acquisition / play / banishment

`shards_ai.ai.HybridPlayer` route maintenant indépendamment les familles de
décisions. La première composition algorithmique construite par
`build_hybrid_player(profile="hybrid-v002")` est `neural_v006` pour les achats
et recrutements, `algorithmic_play_v001` pour PLAY et
`deterministic_blaster_crystal_v001` pour les bannissements. `hybrid-v001`
reste la composition historique avec PLAY heuristique V008.

Les versions sont décrites par des profils YAML immuables sous
`configs/hybrid_profiles/`. Les politiques indépendantes sont conservées sous
`configs/player_policies/{acquisition,play,banish}/`; `hybrid-v002` référence
explicitement `acquisition/v006`, `play/v001` et `banish/v001`.
`load_hybrid_profile()` charge une version exacte ; aucun pointeur actif n'est
consulté. La fingerprint du document permet de vérifier qu'une composition
rejouée n'a pas changé. Une nouvelle composition ou une nouvelle politique
doit créer un nouveau profil et ne pas réécrire une version déjà utilisée.

La politique de bannissement choisit un Blaster visible dans la main ou la
défausse ; à défaut elle choisit un Crystal de la défausse ; sinon elle choisit
`SkipBanish`. Les égalités sont départagées par `instance_id`. Le routeur donne
la priorité au bannissement avant l'acquisition, notamment lorsqu'un effet
pendant PLAY ou BUY demande un bannissement.

L'acquisition utilise le checkpoint macro V006 via `MacroNeuralPlayer`, mais
uniquement lorsque les actions légales sont `BuyCard`, `RecruitMercenary`,
`RecruitFreeCard` ou `StopBuying`. Le joueur composé conserve l'état public et
chaque politique renvoie une action qui est ensuite validée par le moteur.
`last_decision` expose l'identifiant de politique, la famille et le type de
l'action choisie. Le PPO multi-tête partagé reste une architecture future ; il
n'est pas activé par cette composition.

`AlgorithmicPlayPolicy` est la version `algorithmic_play_v001`. À chaque
décision PLAY, elle recalcule les contraintes visibles et classe les cartes dans
l'ordre suivant : pioche sans remélange, effet de bannissement, cartes Spectra
avec Echo, cartes sans contrainte ou déjà validées, pioches avec remélange,
contraintes nouvellement validées, pioches dont la contrainte reste invalide,
puis cartes restantes. Les contraintes suivies sont santé, maîtrise, Echo,
Domination, Union et Inspiration ; les actions `BanishCard` elles-mêmes restent
routées à la politique banishment.
