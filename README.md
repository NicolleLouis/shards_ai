# Shards AI

Moteur déterministe et agents capables de jouer à Shards of Infinity.

Le projet est géré avec [Poetry](https://python-poetry.org/).

```bash
poetry install
poetry run pytest
```

## Analyse de parties

Pour lancer une campagne avec deux `RandomPlayer` symétriques et produire les statistiques locales :

```bash
poetry run python scripts/analyze_games.py
```

La campagne dure 60 secondes par défaut et génère une seed aléatoire affichée dans la console.
Les résultats JSON, CSV et HTML/SVG sont écrits dans
`scripts/analysis_output/random_vs_random/`. Une seed et une durée peuvent être fournies pour
reproduire une campagne :

```bash
poetry run python scripts/analyze_games.py --duration-seconds 30 --seed 42
```

Le benchmark de `HeuristicPlayer` v008 contre un mélange équilibré de Random et Heuristic v007
écrit son rapport dans `analysis_output/heuristic_v008_mix_1000/report.html` :

```bash
PYTHONPATH=. poetry run python scripts/benchmark_heuristic_report.py \
  --games 1000 \
  --seed 87000 \
  --profile configs/heuristic_profiles/v008.yaml \
  --opponent-profile configs/heuristic_profiles/v007.yaml \
  --output-dir analysis_output/heuristic_v008_mix_1000
```

## Profils d'entraînement neural

Les recettes d'entraînement sont versionnées sous `configs/neural_training_profiles/` et les
checkpoints promus sous `configs/neural_profiles/`. Le profil actif est sélectionné par `NEURAL_VERSION`
dans le `Makefile`. Les datasets, checkpoints candidats et rapports générés restent dans
`artifacts/` et ne sont pas versionnés.

Pour entraîner ou reprendre le profil actif :

```bash
make neural-train
make neural-train-resume
```

Ces commandes écrivent par défaut un checkpoint candidat sous `artifacts/`, afin de ne pas
écraser accidentellement une version déjà promue. Pour préparer une nouvelle version depuis une
future session :

1. copier `v001.yaml` vers un fichier candidat, par exemple `/tmp/neural_v002.yaml`, et modifier son
   `profile_id`, son `parent_profile_id` et ses paramètres ;
2. entraîner le candidat :

```bash
make neural-train \
  NEURAL_PROFILE=/tmp/neural_v002.yaml \
  NEURAL_VERSION=v002 \
  NEURAL_MODEL=artifacts/neural_imitation/v002-candidate.pt
```

3. reprendre ce même fichier si nécessaire avec `make neural-train-resume` et les mêmes variables ;
   pour repartir d'un autre checkpoint, utiliser `NEURAL_RESUME_FROM` séparément de
   `NEURAL_MODEL` ;
4. valider et promouvoir :

```bash
PYTHONPATH=. poetry run python scripts/validate_neural_profile.py \
  --candidate-profile /tmp/neural_v002.yaml \
  --candidate-checkpoint artifacts/neural_imitation/v002-candidate.pt \
  --games 1000
```

La promotion crée alors `configs/neural_training_profiles/v002.yaml`,
`configs/neural_profiles/v002.pt` et met à jour `active.yaml`. Le fichier temporaire du candidat peut
rester hors du dépôt.

Pour expérimenter une nouvelle recette, copier `configs/neural_training_profiles/v001.yaml` vers un
nouvel identifiant, modifier `parent_profile_id` et les paramètres, puis lancer directement :

```bash
PYTHONPATH=. poetry run python scripts/train_neural_imitation.py \
  --profile configs/neural_training_profiles/v002.yaml
```

Le checkpoint et les métriques enregistrent l'identifiant et l'empreinte de la configuration
effective. Les overrides CLI sont adaptés aux essais ponctuels ; la recette reproductible doit être
conservée dans un nouveau profil YAML.

Un checkpoint promu contient les poids, l'état de l'optimiseur, les métriques et les métadonnées du
profil. Il peut être repris pour un entraînement ou chargé directement par un `NeuralPlayer`.

Pour valider un candidat contre Random, v007, v008 et les derniers checkpoints neural disponibles :

```bash
PYTHONPATH=. poetry run python scripts/validate_neural_profile.py \
  --candidate-profile /tmp/neural_candidate.yaml \
  --candidate-checkpoint artifacts/neural_imitation/candidate.pt \
  --games 1000
```

Le candidat est promu uniquement si son taux de victoire progresse contre au moins un adversaire et
ne baisse contre aucun. En cas d'acceptation, le script crée le prochain profil `vNNN.yaml` et met à
jour `configs/neural_training_profiles/active.yaml`. En cas de rejet, il ne modifie pas les profils.
Le mode `--no-promote` permet de produire uniquement le rapport.

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
