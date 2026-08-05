# Shards AI

Moteur déterministe et agents capables de jouer à Shards of Infinity.

Le projet est géré avec [Poetry](https://python-poetry.org/).

```bash
poetry install
poetry run pytest
```

## Organisation des scripts et sorties

La règle est de ne jamais écrire de sortie générée à côté du code :

- `scripts/` contient les commandes reproductibles qui préparent des données, entraînent un
  modèle, valident un profil ou génèrent un rapport ;
- `benchmarks/` contient uniquement les programmes de mesure de performance et de matchs de
  référence ;
- `shards_ai/analysis/` contient la logique Python réutilisable d'analyse, pas les rapports ;
- `configs/` contient les profils et configurations versionnés qui servent de sources de vérité ;
- `artifacts/` contient toutes les sorties locales générées, classées par type ;
- `doc/` contient uniquement la documentation Markdown, jamais une sortie d'expérience.

Les artefacts ne sont pas commités. `analysis_output/` et `scripts/analysis_output/` sont des
anciens chemins et ne doivent plus être utilisés. Une sortie temporaire sans valeur de
reproduction va dans `/tmp`.

## Analyse de parties

Pour lancer une campagne avec deux `RandomPlayer` symétriques et produire les statistiques locales :

```bash
poetry run python scripts/analyze_games.py
```

La campagne dure 60 secondes par défaut et génère une seed aléatoire affichée dans la console.
Les résultats JSON, CSV et HTML/SVG sont écrits dans
`artifacts/analysis/random_vs_random/`. Une seed et une durée peuvent être fournies pour
reproduire une campagne :

```bash
poetry run python scripts/analyze_games.py --duration-seconds 30 --seed 42
```

Le benchmark de `HeuristicPlayer` v008 contre un mélange équilibré de Random et Heuristic v007
écrit son rapport dans `artifacts/analysis/heuristic_v008_mix_1000/report.html` :

```bash
PYTHONPATH=. poetry run python scripts/benchmark_heuristic_report.py \
  --games 1000 \
  --seed 87000 \
  --profile configs/heuristic_profiles/v008.yaml \
  --opponent-profile configs/heuristic_profiles/v007.yaml \
  --output-dir artifacts/analysis/heuristic_v008_mix_1000
```

## Profils d'entraînement neural

Les recettes d'entraînement sont versionnées sous `configs/neural_training_profiles/` et les
checkpoints stables promus sous `configs/neural_profiles/`. Le profil PPO candidat est sélectionné
par `NEURAL_RL_PROFILE` dans le `Makefile`.

Il n'existe qu'un seul checkpoint de training mutable :
`artifacts/neural_training/checkpoint.pt`. La variable `NEURAL_CHECKPOINT` est utilisée par les
cibles neural du `Makefile` : entraînement, reprise, benchmark et validation. Les checkpoints sous
`configs/neural_profiles/` sont stables, ne sont plus entraînés et sont les seules versions
conservées durablement. Les artefacts générés restent hors Git.

Pour expérimenter une nouvelle recette, copier `configs/neural_training_profiles/v001.yaml` vers un
nouvel identifiant, modifier `parent_profile_id` et les paramètres, puis lancer directement :

```bash
PYTHONPATH=. poetry run python scripts/train_neural_imitation.py \
  --profile configs/neural_training_profiles/v002.yaml
```

Le checkpoint et les métriques enregistrent l'identifiant et l'empreinte de la configuration
effective. Les overrides CLI sont adaptés aux essais ponctuels ; la recette reproductible doit être
conservée dans un nouveau profil YAML. L'ancien pipeline d'imitation reste disponible via
`scripts/train_neural_imitation.py`, mais il n'est plus exposé par une cible Makefile dédiée.

Un checkpoint promu contient les poids, l'état de l'optimiseur, les métriques et les métadonnées du
profil. Il est chargé directement par un `NeuralPlayer` et ne sert plus de sortie d'entraînement.

Pour valider un candidat contre Random, v007, v008 et les derniers checkpoints neural disponibles :

```bash
PYTHONPATH=. poetry run python scripts/validate_neural_profile.py \
  --candidate-profile /tmp/neural_candidate.yaml \
  --candidate-checkpoint artifacts/neural_training/checkpoint.pt \
  --games 100
```

`NEURAL_VALIDATION_GAMES` permet de modifier ce nombre depuis le `Makefile`. Le défaut de 100
parties par adversaire sert au contrôle rapide ; utiliser au moins 200 parties par adversaire pour
une décision finale de promotion lorsque le coût d'exécution est acceptable. Cette validation peut
être exécutée par l'utilisateur hors de Codex ; ses résultats doivent alors être conservés dans
`artifacts/neural_validation/` ou fournis explicitement avant promotion. La sortie affiche les
taux avec deux décimales et conserve les comptes `victoires/parties`.

Le script propose une règle stricte : le candidat doit progresser contre au moins un adversaire et
ne baisser contre aucun. L'utilisateur peut toutefois effectuer cette validation et cette décision
hors de Codex lorsque le benchmark a déjà été exécuté ; la promotion manuelle doit alors conserver
la preuve et mettre à jour les deux pointeurs `active.yaml`. En cas de rejet explicite, les profils
ne sont pas modifiés.
Le mode `--no-promote` permet de produire uniquement le rapport.

## Entraînement RL du candidat v002

Le profil candidat `configs/neural_training_profiles/candidates/v002.yaml` configure un training
PPO en parties réelles contre Random, v007 et v008. La récompense est uniquement terminale : `+1`
pour une victoire, `-1` pour une défaite et `0` pour un nul. Aucun reward shaping n'est utilisé.

Pour initialiser un run depuis le checkpoint stable v001 :

```bash
make neural-rl-train
```

Pour reprendre le checkpoint mutable unique :

```bash
make neural-rl-train-resume
```

Le budget principal est `total_games`. `games_per_update` contrôle le nombre de parties collectées
avant un update PPO ; `optimization_epochs` contrôle le nombre de passes PPO sur ces transitions.
Ce sont donc deux unités différentes. La collecte peut utiliser plusieurs workers avec
`NEURAL_RL_WORKERS` (1 par défaut) ; l'update PPO et l'écriture du checkpoint restent séquentiels.
Le profil v002 utilise aussi une régularisation KL vers v001, une entropie réduite et un mélange
`20 % Random / 30 % v007 / 50 % v008`. Une évaluation gloutonne périodique conserve le meilleur
état dans le checkpoint mutable unique. Un état n'est retenu que s'il ne régresse contre aucun des
tandis qu'une tolérance d'une victoire est admise pendant la sélection périodique contre Random et
v007. Aucune tolérance n'est admise contre v008 ; le score pondéré 20/30/50 sert ensuite à favoriser
v008. La validation finale reste stricte avant promotion. Les paramètres peuvent être surchargés
pour un smoke test. Le profil utilise 64 parties par adversaire pour chaque
évaluation périodique, soit 192 parties par point de sélection ; la validation finale doit utiliser
un panel encore plus large :

```bash
make neural-rl-train \
  NEURAL_RL_TOTAL_GAMES=10 \
  NEURAL_RL_GAMES_PER_UPDATE=2 \
  NEURAL_RL_OPTIMIZATION_EPOCHS=1 \
  NEURAL_RL_WORKERS=2
```

Le seul checkpoint de travail reste `artifacts/neural_training/checkpoint.pt`. La validation et la
promotion v002 utilisent ensuite ce checkpoint vers `configs/neural_profiles/v002.pt`.

Pour générer environ 100 000 décisions v008 uniquement, avec des parties contre Random et v007 :

```bash
PYTHONPATH=. poetry run python scripts/generate_imitation_dataset.py \
  --profile configs/heuristic_profiles/v008.yaml \
  --opponent-profile configs/heuristic_profiles/v007.yaml \
  --target-decisions 100000 \
  --seed 88008 \
--output artifacts/imitation_dataset/v008_vs_random_v007_100k.jsonl
```

Pour comparer deux checkpoints versionnés :

```bash
PYTHONPATH=. poetry run python benchmarks/benchmark_neural_players.py \
  --checkpoint configs/neural_profiles/v002.pt \
  --opponent neural \
  --opponent-checkpoint configs/neural_profiles/v001.pt \
  --games 1000 \
  --output artifacts/neural_benchmark/v002_vs_v001.json
```

Une nouvelle version suit donc ce cycle : copier un profil, entraîner vers
`configs/neural_profiles/vNNN.pt`, valider le candidat, puis committer le profil et le checkpoint après
promotion. Les datasets et rapports sous `artifacts/` ne sont pas commités.

Dans ce mode, `--profile` désigne le teacher et `--opponent-profile` désigne uniquement un
adversaire ; les décisions v007 ne sont pas ajoutées aux labels.

## Optimisation des coefficients heuristiques

Le profil manuel initial est conservé dans `configs/heuristic_profiles/v001.yaml`. Le profil
heuristique actif par défaut est `configs/heuristic_profiles/v008.yaml`. Une campagne courte de
60 secondes peut être lancée ainsi :

```bash
PYTHONPATH=. poetry run python scripts/optimize_heuristic.py \
  --duration-seconds 60 \
  --seed 42
```

La commande écrit un historique JSON et un profil YAML candidat dans
`artifacts/heuristic_optimization/<run-id>/`. Pour publier explicitement le profil obtenu :

```bash
PYTHONPATH=. poetry run python scripts/optimize_heuristic.py \
  --duration-seconds 21600 \
  --seed 42 \
  --publish-profile configs/heuristic_profiles/v002.yaml
```

La campagne utilise une recherche hybride par mutations et racing. Pour l’objectif recommandé
consistant à battre `v002`, elle peut démarrer directement dans un mélange 50/50 entre
`RandomPlayer` et le profil heuristique précédent :

```bash
PYTHONPATH=. poetry run python scripts/optimize_heuristic.py \
  --profile configs/heuristic_profiles/v002.yaml \
  --duration-seconds 60 \
  --start-mixed \
  --seed 42
```

La publication n’écrit un nouveau profil que si la validation indépendante confirme le gain
statistique. Les paramètres `--initial-games`, `--racing-games`, `--validation-games` et
`--test-games` contrôlent les niveaux de racing et de validation.
