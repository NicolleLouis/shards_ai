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

Le profil stable actif est `v002`, issu du fine-tuning d'imitation sur l'historique et les deux
cycles DAgGER avec l'architecture `global_candidate_context`. Il est conservé dans
`configs/neural_profiles/v002.pt`; les prochaines recherches ciblent des améliorations hors DAgGER,
car les cycles DAgGER n'ont pas débloqué la faiblesse principale contre v008.

## Modèle

`NeuralActionScorer` encode les cartes avec une voie d'identité (`card_definition_id`) et une voie
sémantique structurée, agrège les zones de cartes comme ensembles/multiensembles, puis calcule un
score action-conditionnel pour chaque action. Il ne génère aucune action et ne remplace pas
`Game.legal_actions()`.

Les dimensions sont configurables via `NeuralModelConfig`. Le modèle fonctionne sur CPU et peut
être déplacé sur un autre device PyTorch. L'inférence de production devra utiliser `torch.no_grad()`
et mettre en cache les tenseurs préparés si les mesures de latence le justifient.

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
checkpoint ni le résultat observable. Le script d'entraînement
utilise par défaut un thread PyTorch (`--torch-threads`) adapté aux petits forwards successifs ;
ce paramètre reste configurable pour les autres machines et workloads.

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

Le self-play n'est pas encore implémenté. Le `Makefile` expose
`make neural-rl-train` et `make neural-rl-train-resume`, qui utilisent le profil candidat indiqué
par `NEURAL_RL_PROFILE` et le checkpoint mutable unique `NEURAL_CHECKPOINT`. Le training RL est
budgété en nombre de parties (`total_games`) ; les `optimization_epochs` sont des passes sur les
transitions du rollout. Les checkpoints stables sous `configs/neural_profiles/` ne sont jamais
entraînés. `configs/neural_profiles/active.yaml` désigne le profil stable chargé par défaut
lorsqu'un `NeuralPlayer` est construit sans checkpoint explicite. La collecte des parties accepte
`--workers` via `NEURAL_RL_WORKERS`, avec une valeur par défaut de `1` ; seul le rollout est
parallélisé, tandis que l'update PPO et l'écriture du checkpoint restent séquentiels. Le profil PPO
v002 conserve une régularisation KL vers v001, une entropie réduite et un mélange `20 % Random /
30 % v007 / 50 % v008`. Une évaluation gloutonne périodique contre les trois adversaires permet de
restaurer le meilleur état dans le checkpoint mutable unique. Le score de sélection est pondéré par
ce mélange ; une baisse d'une victoire est tolérée contre Random et v007 pour éviter de rejeter une
progression contre v008, mais aucune baisse n'est tolérée contre v008. La validation finale reste
stricte avant promotion. Le résultat détaillé du meilleur état est conservé dans le checkpoint pour
rendre les reprises cohérentes. Chaque évaluation périodique utilise 64 parties par adversaire, soit
192 parties, tandis que la validation finale doit rester plus large.

Le profil expérimental `configs/neural_training_profiles/candidates/v003.yaml` active un reward
shaping de deckbuilding basé sur une table fixe de 48 valeurs dérivées de l'acquisition v008.
Le potentiel est la moyenne des valeurs des cartes propres connues dans la pioche, la main, la
défausse, la zone de jeu et les champions. Un delta borné et pondéré par `beta` est ajouté seulement
aux transitions d'achat, de recrutement, de maîtrise, de bannissement et de fin de phase ; la
récompense terminale `+1/-1` reste présente. Le réseau ne reçoit pas ces zones supplémentaires :
elles servent uniquement au calcul de la reward dans le collecteur PPO.

`scripts/validate_neural_profile.py` compare un candidat au dernier profil neural actif sur les mêmes seeds,
contre `RandomPlayer`, `v007`, `v008` et au plus les deux derniers profils neural dont les
checkpoints existent. Il imprime les résultats par adversaire avec deux décimales et les comptes
victoires/parties. Le défaut Makefile est de 100 parties par adversaire ; un panel d'au moins 200
parties est recommandé pour une promotion finale. Le script propose de ne promouvoir le candidat
que par rapport au profil neural actif courant (v002 actuellement) ; `v008` reste l'adversaire
heuristique protégé et ne constitue pas la référence neural remplacée à chaque promotion.
que si son taux de victoire contre V008 ne baisse pas, que les baisses contre Random ou v007 restent
dans une tolérance de 5 points, que le gain pondéré du panel atteint au moins 0,5 point et qu'un
gain individuel atteint 1 point. Cette validation
peut être exécutée hors de Codex ; le résultat doit alors être conservé ou fourni avant promotion.
Une promotion crée le prochain
profil versionné et met à jour `active.yaml` ; un rejet ne modifie aucun profil.

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

## Architecture expérimentale avec contexte global des actions

Une architecture `global_candidate_context` est disponible pour l'imitation. Elle réutilise les
encodeurs baseline, moyenne les embeddings des actions légales, injecte ce contexte global dans le
score de chaque action et reste indépendante de l'ordre de la liste. Elle est chargée automatiquement
par `NeuralPlayer` lorsque le checkpoint porte cette métadonnée. Le profil
`configs/neural_training_profiles/candidates/v004.yaml` prépare ce réentraînement ; il n'est pas
encore promu et le PPO n'utilise pas encore cette architecture.

L'architecture expérimentale `semantic_identity_v3` est également disponible. Elle conserve le
contrat action-conditionnel de la baseline mais configure séparément l'embedding d'identité de la
carte, la couche sémantique et la représentation finale. Le profil
`configs/neural_training_profiles/candidates/v003-embedding.yaml` utilise respectivement 24, 64 et
64 dimensions, et enregistre les alternatives 16/24 et 48/64 dans sa recette. La résolution passe
par `build_neural_scorer` dans tous les chemins d'inférence, d'imitation, de PPO et d'analyse ; les
checkpoints V001 et V002 continuent donc d'utiliser leurs architectures historiques.

Le support PPO contextualisé est préparé dans `NeuralActorCritic` et le profil candidat
`configs/neural_training_profiles/candidates/v005.yaml`. L’actor et la référence KL chargent la
même architecture depuis le checkpoint ; le critic reste dépendant de l’observation seule. Ce
profil ne doit être lancé qu’après l’analyse offline par phase/action et un benchmark en parties du
checkpoint v004. Il utilise toujours le checkpoint mutable canonique et ne crée pas de `best.pt`
séparé.

## Analyse offline de l’imitation

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

## Cycle DAgGER ciblé

`scripts/collect_dagger_dataset.py` fait jouer le checkpoint mutable actuel
contre une liste explicite d'adversaires v008, v007 et Neural. Pour chaque
décision du Neural évalué, il conserve toutes les actions légales, les scores et
le choix de v008, le choix Neural, le rang, le regret, la phase, la divergence
et l'issue de la partie. Les décisions hors `PLAY` restent dans le dataset.

Pour les phases `PLAY`, le collecteur rejoue v008 sur une copie de l'état de
début de phase et compare l'état de fin avec la trajectoire réelle. Une
différence d'action peut donc être marquée comme permutation équivalente ou
divergence stratégique. Les sorties brutes et leurs manifests sont écrites
sous `artifacts/imitation_dataset/`.

`scripts/sample_dagger_dataset.py` construit ensuite un dataset de fine-tuning
par réservoir pondéré et déterministe : 45 % d'ancien dataset, 35 % de nouveaux
exemples `PLAY` et 20 % d'autres décisions on-policy. Il écrit aussi un holdout
de validation historique et on-policy séparé. La cible `make neural-dagger-train`
reprend explicitement les poids du checkpoint courant avec un learning rate
réduit, une époque par défaut et un optimiseur réinitialisé.

Le cycle `make neural-dagger2-collect` utilise le préfixe `dagger_2-` pour éviter les collisions
de parties avec le premier cycle et refuse tout teacher différent de Heuristic V8. La cible
`make neural-dagger2-sample` échantillonne uniquement les états on-policy du cycle 2 avec les
pondérations par action `play_card=1.5`, `recruit_mercenary=3`, `assign_power=3` et
`choose_pending_decision=2`; le choix Neural est conservé séparément et `chosen_action` reste
le choix de V8. Enfin `make neural-dagger-merge` fusionne l'historique, DAgGER 1 et DAgGER 2,
ajoute `dataset_source`/`dagger_stage` et refuse les lignes DAgGER qui ne portent pas les scores
ou le label de Heuristic V8.
La fusion prend le raw du cycle 1 comme source DAgGER, car `dagger_cycle_1_train.jsonl` est déjà
un échantillon mélangé avec l'historique et ne doit pas être réutilisé comme un dataset on-policy
exclusif.

Le training d'imitation utilise l'implémentation CPU `foreach` d'Adam. Elle
réduit le coût de dispatch des mises à jour effectuées à chaque décision sans
modifier la taille des lots, l'ordre des records, la loss ou le nombre de
threads configuré.
