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

Le profil stable actif est `v004`, issu de l'expérience V5 `structured_semantic_v5_fusion_experiment`
et entraîné depuis `v003`. Il est conservé dans `configs/neural_profiles/v004.pt`; `v003` et `v002`
restent des références historiques protégées.

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

## Entraînement

`neural_training.py` lit le JSONL en streaming. Les losses disponibles sont : préférence
paire-à-paire, imitation de l'action choisie et régression optionnelle des scores normalisés par
décision. `train_neural_imitation.py` produit un checkpoint hors de `doc/` et partitionne les
décisions par `game_id` de façon déterministe.

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

`benchmarks/benchmark_neural_panel.py`, appelé par `make neural-benchmark-panel`, applique la même
instrumentation à un panel fixe composé de Random, Heuristic v007, Heuristic v008 et des profils
neural v001, v002, v003 et v004. Il joue un nombre égal de parties contre chaque adversaire, conserve
chaque partie dans `artifacts/neural_benchmark/neural_panel.json` et produit le rapport autonome
`artifacts/neural_benchmark/neural_panel.html`. Le profil testé est v004 par défaut et peut être
remplacé par `NEURAL_PANEL_CHECKPOINT` pour analyser le checkpoint mutable ou une autre version.

Le self-play n'est pas encore implémenté. Les anciennes cibles Makefile PPO et DAgGER ont été
retirées car elles pointaient vers des recettes candidates historiques. Les nouveaux entraînements
sont orchestrés comme des expériences candidates avec `v003` comme parent explicite. Le checkpoint
mutable reste `artifacts/neural_training/checkpoint.pt`; les checkpoints stables sous
`configs/neural_profiles/` ne sont jamais entraînés en place. Les pointeurs actifs désignent
actuellement `v004` et son architecture `structured_semantic_v5_fusion_experiment`.

`scripts/validate_neural_profile.py` compare un candidat au dernier profil neural actif sur les mêmes seeds,
contre `RandomPlayer`, `v007`, `v008` et les quatre derniers profils neural dont les
checkpoints existent. Il imprime les résultats par adversaire avec deux décimales et les comptes
victoires/parties. Chaque profil neural pèse `1/4`, soit un poids total de `1` pour le groupe neural.
Le défaut Makefile est de 100 parties par adversaire ; un panel d'au moins 200
parties est recommandé pour une promotion finale. Le script propose de ne promouvoir le candidat
que par rapport au profil neural actif courant (`v004` actuellement) ; `v008` reste l'adversaire
heuristique protégé et ne constitue pas la référence neural remplacée à chaque promotion.
La promotion est autorisée si la moyenne pondérée des deltas de tous les adversaires est strictement
positive, y compris lorsqu'elle contient une baisse contre v008. Chaque adversaire compte une fois, indépendamment
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

## Joueurs hybrides et ablation en partie

`shards_ai.ai.HybridPlayer` encapsule un `NeuralPlayer` et un `HeuristicPlayer`.
La politique `purchase_recruitment` délègue les achats, recrutements et sorties
de la phase d'achat à Heuristic V8; `play_phase` délègue toutes les décisions de
`Phase.PLAY`; `banish` délègue les décisions dont les actions légales contiennent
un `BanishCard`. Les autres décisions restent neuronales.

`benchmarks/benchmark_neural_hybrids.py`, appelé par `make neural-hybrid-benchmark`,
fait jouer le NeuralPlayer contre Neural, Heuristic V8 et ces trois hybrides. Le
nombre de parties doit être un multiple de cinq; l'index de partie répartit les
adversaires à parts égales. Le JSON conserve les résultats détaillés et le HTML
les taux de victoire par intervention. Cette campagne mesure une ablation sur
des trajectoires différentes, pas un effet causal isolé état par état.

Le training d'imitation utilise l'implémentation CPU `foreach` d'Adam. Elle
réduit le coût de dispatch des mises à jour effectuées à chaque décision sans
modifier la taille des lots, l'ordre des records, la loss ou le nombre de
threads configuré.
