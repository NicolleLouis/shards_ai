# Neural player

## État

Le premier socle d'imitation supervisée est disponible. Il utilise PyTorch et reçoit uniquement
`NeuralObservation`, l'observation masquée construite par le moteur, ainsi que les représentations
des actions légales produites par `representation_for_action`.

Les entraînements sont décrits par des profils YAML versionnés sous
`configs/neural_training_profiles/`. `v001.yaml` reprend le baseline d'imitation et indique le
dataset, le checkpoint de sortie, les seeds, les hyperparamètres, le split et la configuration du
modèle. `scripts/train_neural_imitation.py --profile <profile.yaml>` charge cette recette ; les
options CLI peuvent la surcharger pour une expérience ponctuelle. Le checkpoint et le fichier de
métriques enregistrent l'identifiant du profil, son fingerprint et celui de la configuration
effective. Les checkpoints promus sont versionnés sous `configs/neural_profiles/`; les datasets,
checkpoints candidats et rapports restent sous `artifacts/` et ne doivent pas être versionnés dans
Git.

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
repris avec `--resume-from` ; les métriques précédentes sont conservées et les nouveaux epochs
sont ajoutés au JSON, au CSV et au rapport HTML.

Le chemin CPU du modèle batch les embeddings des cartes distinctes par décision afin d'éviter de
recalculer plusieurs fois les mêmes cartes dans les zones et les actions. Le script d'entraînement
utilise par défaut un thread PyTorch (`--torch-threads`) adapté aux petits forwards successifs ;
ce paramètre reste configurable pour les autres machines et workloads.

Le `NeuralPlayer` charge un checkpoint, reçoit uniquement `NeuralObservation` via `GameRunner`,
représente les actions légales depuis les zones publiques, puis choisit le meilleur score sous
`torch.inference_mode()`. Les égalités sont départagées par un `GameRandom` injectable. Le benchmark
`benchmarks/benchmark_neural_players.py` mesure les victoires, défaites, matchs nuls, actions,
latence d'inférence et temps total contre Random ou un profil heuristique. Il utilise par défaut un
thread PyTorch (`--torch-threads 1`) adapté à ce hot path de petites inférences successives.

`benchmarks/benchmark_neural_mix.py` lance une campagne mixte déterministe (20 % Random, 50 % v007,
30 % v008) et génère `neural_mix.json` ainsi qu'un rapport HTML. Ce rapport compare les adversaires
et détaille durée, nombre total de tours, tours moyens par joueur, actions, maîtrise, PV et
composition agrégée des decks finaux des deux joueurs. Le nombre de tours par joueur est calculé
comme le nombre total de tours divisé par le nombre de joueurs du runner. Il détaille aussi, pour le NeuralPlayer, les mercenaires recrutés immédiatement ou achetés
à long terme, ainsi que le taux d'activation de `GainMastery` par tour et par maîtrise disponible.
Le rapport inclut également, pour chaque adversaire, neuf graphiques SVG par multiplicité centrale
(`×1`, `×2`, `×3`) : moyenne NeuralPlayer, moyenne adverse et delta des copies moyennes. Les cartes
de chaque groupe sont triées par moyenne décroissante, ou par valeur absolue du delta décroissante.

Le reinforcement learning et le self-play ne sont pas encore implémentés. Le `Makefile` sélectionne
le profil actif via `NEURAL_VERSION`, avec `make neural-train` et `make neural-train-resume` comme
commandes opérationnelles versionnées. `configs/neural_profiles/active.yaml` désigne le profil
chargé par défaut lorsqu'un `NeuralPlayer` est construit sans checkpoint explicite.

`scripts/validate_neural_profile.py` compare un candidat à ce profil actif sur les mêmes seeds,
contre `RandomPlayer`, `v007`, `v008` et au plus les deux derniers profils neural dont les
checkpoints existent. Il imprime les résultats par adversaire et ne promeut le candidat que si son
taux de victoire ne baisse nulle part et progresse au moins une fois. Une promotion crée le prochain
profil versionné et met à jour `active.yaml` ; un rejet ne modifie aucun profil.

Le générateur d'imitation accepte aussi des matchups ciblés avec `--opponent-profile`. Les profils
passés par `--profile` sont alors les seuls teachers enregistrés ; les adversaires servent uniquement
à jouer les parties. `--target-decisions` compte donc les lignes du teacher, ce qui permet par
exemple de générer un dataset v008 contre Random et v007 sans introduire de labels v007.

La promotion recopie le checkpoint candidat dans `configs/neural_profiles/vNNN.pt`, synchronise ses
métadonnées de profil et crée `configs/neural_training_profiles/vNNN.yaml`. Le benchmark
`benchmarks/benchmark_neural_players.py` accepte `--opponent neural` et
`--opponent-checkpoint`, ce qui permet de comparer deux versions historiques sans dépendre du
profil actif.
