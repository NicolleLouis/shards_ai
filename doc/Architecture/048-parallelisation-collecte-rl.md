# Parallélisation de la collecte RL

## Contexte

La collecte PPO v002 joue les parties séquentiellement avant chaque update. Le réseau est utilisé
en lecture seule pendant cette phase ; les parties indépendantes peuvent donc être collectées par
plusieurs workers sans paralléliser l'optimisation ni l'écriture du checkpoint.

## Décision

Ajouter une option `--workers` au script `scripts/train_neural_rl.py`, avec une valeur par défaut de
`1`. Le `Makefile` expose cette valeur via `NEURAL_RL_WORKERS`.

Avec plusieurs workers, chaque worker collecte des épisodes complets dans un
`ThreadPoolExecutor`. Les résultats sont réassemblés dans l'ordre des indices de parties, puis un
seul thread exécute `ppo_update` et écrit le checkpoint mutable.

La valeur `1` conserve le chemin séquentiel et reste le mode reproductible de référence. Les
workers ne partagent pas de décision ou de RNG de partie : chaque épisode dérive un générateur
Torch de son `game_seed`. Le choix stochastique de la politique reste donc identique entre les
modes séquentiel et parallèle pour un même profil et une même plage d'indices.

## Limites

- le parallélisme concerne uniquement la collecte des jeux ; l'update PPO reste séquentiel ;
- le modèle actor-critic est partagé en lecture seule pendant la collecte ;
- le plafond `max_transitions_per_update` peut faire terminer des épisodes supplémentaires dans un
  batch parallèle, mais seuls les épisodes retenus dans l'ordre sont transmis à PPO ;
- la valeur de `--workers` doit rester adaptée au CPU disponible et à `--torch-threads` pour éviter
  la sur-souscription ;
- les checkpoints stables sous `configs/neural_profiles/` ne sont jamais écrits par cette option.

## Validation

Les tests comparent un rollout avec `workers=1` et `workers=2` sur les mêmes seeds, transitions,
adversaires, résultats et actions choisies. Un smoke test du script vérifie également que les deux
modes écrivent un checkpoint valide.
