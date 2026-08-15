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

Le benchmark de `HeuristicPlayer` v008 contre Random
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
checkpoints stables promus sous `configs/neural_profiles/`. Les deux pointeurs `active.yaml`
désignent actuellement `v006` (`structured_semantic_v5_macro_tactical_action_v1`), promu après
fine-tuning PPO de V005.

Il n'existe qu'un seul checkpoint de training mutable :
`artifacts/neural_training/checkpoint.pt`. La variable `NEURAL_CHECKPOINT` est utilisée par les
cibles neural restantes du `Makefile` : benchmark et validation. Les checkpoints sous
`configs/neural_profiles/` sont stables, ne sont plus entraînés et sont les seules versions
conservées durablement. Les artefacts générés restent hors Git.

La gate de qualité compare chaque candidat à Hybrid V006/V004/V005 et Heuristic V008 avec un poids `1`,
puis Hybrid V001/V003 avec un poids `0,75`. Neural V001 à V009, Heuristic V007, Hybrid V002,
les autres hybrides et Random restent diagnostiques et sont exclus de la gate ; la décision repose
uniquement sur une moyenne pondérée strictement positive, sans garde dure de non-régression v008.

Pour expérimenter une nouvelle recette, créer un profil candidat temporaire avec
`parent_profile_id: v003`, puis l'exécuter vers le checkpoint mutable :

```bash
PYTHONPATH=. poetry run python scripts/train_neural_imitation.py \
  --profile /tmp/neural_candidate.yaml \
  --output artifacts/neural_training/checkpoint.pt
```

Le checkpoint et les métriques enregistrent l'identifiant et l'empreinte de la configuration
effective. Les overrides CLI sont adaptés aux essais ponctuels ; la recette reproductible doit être
conservée dans un nouveau profil YAML. L'ancien pipeline d'imitation reste disponible via
`scripts/train_neural_imitation.py`, mais il n'est plus exposé par une cible Makefile dédiée.

Un checkpoint promu contient les poids, l'état de l'optimiseur, les métriques et les métadonnées du
profil. Il est chargé directement par un `NeuralPlayer` et ne sert plus de sortie d'entraînement.

Pour valider un candidat contre v008 et Hybrid V003 :

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

La décision repose sur une moyenne pondérée strictement positive contre le panel ; aucun adversaire
isolé ne constitue une garde dure. L'utilisateur peut toutefois effectuer cette validation et cette décision
hors de Codex lorsque le benchmark a déjà été exécuté ; la promotion manuelle doit alors conserver
la preuve et mettre à jour les deux pointeurs `active.yaml`. En cas de rejet explicite, les profils
ne sont pas modifiés.
Le mode `--no-promote` permet de produire uniquement le rapport.

Pour comparer un profil à un panel complet, `make neural-benchmark-panel` joue contre Random,
Heuristic v008. Le défaut est de 200 parties par adversaire, soit 400 parties, avec
`configs/neural_profiles/v006.pt` comme profil
testé. Le JSON détaillé et le rapport HTML sont écrits dans `artifacts/neural_benchmark/`.
Les variables `NEURAL_PANEL_CHECKPOINT`, `NEURAL_PANEL_GAMES`, `NEURAL_PANEL_SEED`,
`NEURAL_PANEL_OUTPUT` et `NEURAL_PANEL_HTML_OUTPUT` permettent de modifier la campagne.

## Entraînement neural

Les anciennes cibles Makefile PPO et DAgGER ont été retirées : elles pointaient vers des recettes
v001/v002/v003 obsolètes et pouvaient contourner le profil actif. Les entraînements doivent être
orchestrés comme des expériences candidates avec `v003` comme parent explicite, le checkpoint
mutable `artifacts/neural_training/checkpoint.pt`, puis une validation avant promotion. Les cibles
Makefile restantes servent au benchmark, à l'analyse, à la validation et à `meta-improve`.

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

Le profil heuristique actif par défaut est `configs/heuristic_profiles/v008.yaml`. Une campagne
courte de 60 secondes peut être lancée ainsi :

```bash
PYTHONPATH=. poetry run python scripts/optimize_heuristic.py \
  --duration-seconds 60 \
  --seed 42
```

La commande écrit un historique JSON et un profil YAML candidat dans
`artifacts/heuristic_optimization/<run-id>/`. Pour publier explicitement le profil obtenu depuis
la référence v007 :

```bash
PYTHONPATH=. poetry run python scripts/optimize_heuristic.py \
  --duration-seconds 21600 \
  --seed 42 \
  --profile configs/heuristic_profiles/v007.yaml \
  --publish-profile /tmp/heuristic_candidate.yaml
```

La campagne utilise une recherche hybride par mutations et racing. Pour démarrer dans un mélange
50/50 entre `RandomPlayer` et la référence heuristique précédente :

```bash
PYTHONPATH=. poetry run python scripts/optimize_heuristic.py \
  --profile configs/heuristic_profiles/v007.yaml \
  --duration-seconds 60 \
  --start-mixed \
  --seed 42
```

La publication n’écrit un nouveau profil que si la validation indépendante confirme le gain
statistique. Les paramètres `--initial-games`, `--racing-games`, `--validation-games` et
`--test-games` contrôlent les niveaux de racing et de validation.
