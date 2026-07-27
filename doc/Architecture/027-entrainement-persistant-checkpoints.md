# Entraînement heuristique persistant et reprenable — Architecture

## Objective

Permettre à une campagne d’optimisation heuristique de s’exécuter sur plusieurs sessions, avec des
interruptions volontaires pendant les périodes d’utilisation de l’ordinateur, sans perdre les
progrès déjà validés.

La durée demandée doit représenter un budget de calcul réellement consommé, et non une durée
calendaire continue. Une campagne de 15 heures pourra ainsi être exécutée en plusieurs créneaux de
pause, avec reprise déterministe depuis le dernier batch terminé.

## Current State

`scripts/optimize_heuristic.py` construit un `OptimizationConfig`, appelle l’optimiseur et écrit
`results.json` ainsi que le profil accepté uniquement lorsque l’appel se termine. Les quatre modes
de recherche, dont `--combined`, conservent leur état dans des variables locales : candidat courant,
échelles de mutation, numéro de batch, mode mixte et historique.

Les boucles utilisent actuellement `time.monotonic()` avec une deadline calculée au démarrage. Une
interruption, un arrêt du processus ou une mise en veille peut donc empêcher la sauvegarde du travail
réalisé. Le résultat final ne permet pas de reprendre directement une campagne.

Les seeds des parties sont déjà dérivées de la seed racine, du batch, de l’index et de l’adversaire.
Cette propriété permet de rejouer un batch interrompu sans modifier les résultats, à condition de
reprendre au début du batch concerné.

## Target Behavior

Une campagne peut être lancée avec un chemin de checkpoint et interrompue proprement :

```bash
PYTHONPATH=. poetry run python scripts/optimize_heuristic.py \
  --profile configs/heuristic_profiles/v007.yaml \
  --reference-profile configs/heuristic_profiles/v007.yaml \
  --combined \
  --compute-seconds 54000 \
  --checkpoint artifacts/heuristic_optimization/v008/checkpoint.json \
  --seed 88
```

Une session suivante reprend le même run :

```bash
PYTHONPATH=. poetry run python scripts/optimize_heuristic.py \
  --resume artifacts/heuristic_optimization/v008/checkpoint.json
```

Le compteur `compute-seconds` est cumulatif entre les sessions. Une interruption pendant une
évaluation abandonne au plus le batch en cours ; le dernier batch terminé et accepté reste acquis.

## Non-Goals

- Reprendre au milieu d’une partie individuelle ou d’une évaluation partiellement exécutée.
- Persister chaque transition de jeu ou chaque observation.
- Permettre de modifier silencieusement la seed, le profil de référence ou les bornes pendant une
  reprise.
- Remplacer `tmux`, `nohup` ou un gestionnaire de processus : ces outils restent utiles pour garder
  une session ouverte, mais le checkpoint fournit la persistance de l’optimisation elle-même.
- Publier automatiquement un profil lors d’une interruption ou à partir d’un checkpoint incomplet.

## Key Decisions

1. **Checkpoint à la frontière des batches.** L’état est sauvegardé après la décision d’acceptation
   du batch et après la mise à jour de l’échantillon de validation. Aucun état partiellement évalué
   n’est présenté comme acquis.
2. **Interruption coopérative.** `SIGINT`/`Ctrl+C` demande un arrêt après l’évaluation courante et
   force une sauvegarde du dernier état cohérent. Une interruption forcée peut perdre le batch en
   cours, mais jamais les checkpoints précédents.
3. **Budget en temps CPU de processus.** Utiliser `time.process_time()` ou son équivalent monotone
   de précision suffisante pour ne compter que le temps CPU consommé par l’optimiseur. Le temps passé
   en veille, suspendu ou privé de CPU n’épuise pas le budget.
4. **Reprise au batch suivant.** Le checkpoint stocke `next_batch`, et non une continuation au milieu
   d’une évaluation. Cela simplifie la cohérence et réutilise les seeds déterministes existantes.
5. **Configuration immuable par run.** Au resume, le profil initial, le profil de référence, le mode,
   les champs actifs, les bornes, la seed et les tailles de lots doivent correspondre au checkpoint.
   Une empreinte de configuration est vérifiée avant reprise.
6. **Écriture atomique.** Écrire dans un fichier temporaire du même dossier, le synchroniser si
   possible, puis le remplacer par `os.replace()`. Une coupure pendant l’écriture conserve le dernier
   checkpoint valide.
7. **Validation finale rejouable.** Si l’interruption survient pendant la validation finale, le
   checkpoint conserve le candidat courant et l’état `validation_pending`. La validation est reprise
   depuis le début au prochain lancement ; elle ne modifie pas la recherche déjà acquise.
8. **Un seul processus écrivain.** Le checkpoint n’est pas conçu pour deux campagnes concurrentes.
   Un verrou de fichier ou une détection de PID/run actif doit empêcher deux processus de reprendre le
   même fichier simultanément.

## Proposed Architecture

### Checkpoint state

Introduire une structure sérialisable dédiée, distincte de `OptimizationResult`, contenant :

- `schema_version` ;
- `run_id` et timestamps de création/mise à jour ;
- profil initial et profil de référence, ou leurs identifiants et empreintes de contenu ;
- configuration normalisée et empreinte ;
- seed racine et mode d’optimisation ;
- `next_batch` ;
- candidat courant complet : poids d’action, d’acquisition et de contraintes ;
- échelles de mutation ;
- état `mixed` ;
- `compute_seconds_consumed` ;
- phase (`search`, `validation_pending`, `completed`) ;
- historique utile à l’analyse et aux résultats finaux.

Le checkpoint doit être lisible indépendamment du processus afin de diagnostiquer un run interrompu.

### Session budget

Créer un budget de session séparant :

- le compteur cumulatif restauré depuis le checkpoint ;
- le compteur CPU consommé dans la session courante ;
- le budget total demandé.

La boucle de recherche teste ce budget entre les évaluations. Une évaluation individuelle reste
atomique du point de vue de la reprise ; elle peut dépasser légèrement le seuil de session si elle a
déjà commencé.

### CLI

Ajouter :

- `--compute-seconds` pour un budget cumulatif de calcul ;
- `--checkpoint PATH` pour activer ou choisir le fichier ;
- `--resume PATH` pour restaurer une campagne ;
- éventuellement `--checkpoint-interval batches` si un jour le coût d’écriture de l’historique
  devient significatif.

`--duration-seconds` reste accepté pour compatibilité, mais son comportement doit être documenté
comme budget de session ou déprécié au profit de `--compute-seconds`. Une reprise ne doit pas
additionner implicitement deux budgets contradictoires.

### Final result

À la fin, `write_optimization_result()` continue d’écrire le résultat complet et le profil accepté.
Le checkpoint est marqué `completed` et conservé comme trace de reprise ; il n’est pas supprimé
automatiquement afin de permettre l’audit.

## Data Model

Le format JSON du checkpoint est versionné séparément du `schema_version` de `OptimizationResult`.
Les dataclasses existantes `HeuristicWeights`, `CardAcquisitionWeights` et
`CardConstraintWeights` restent la source de validation des poids restaurés.

Exemple minimal :

```json
{
  "schema_version": 1,
  "run_id": "v008-seed88",
  "next_batch": 7,
  "compute_seconds_consumed": 24831.4,
  "compute_seconds_target": 54000,
  "phase": "search",
  "mixed": true,
  "seed": 88,
  "current": {
    "weights": {},
    "card_acquisition_weights": {},
    "constraint_weights": {}
  },
  "step_scales": {},
  "configuration_fingerprint": "..."
}
```

Les anciens checkpoints ou les checkpoints dont le profil de référence a changé sont refusés avec
un message explicite ; ils ne sont jamais migrés silencieusement.

## Backend Flow

1. Le CLI crée un run ou charge et valide un checkpoint.
2. L’optimiseur restaure le candidat, les échelles et `next_batch`.
3. Il exécute les candidats du batch courant avec les seeds déjà déterministes.
4. Il accepte ou refuse le meilleur candidat.
5. Il met à jour le compteur CPU et écrit atomiquement le checkpoint.
6. Il recommence tant que le budget cumulatif n’est pas épuisé.
7. Il passe à `validation_pending`, sauvegarde, puis exécute la validation indépendante.
8. Il marque le checkpoint `completed` uniquement après validation et écrit le résultat final.

En cas d’exception, le dernier checkpoint reste disponible et l’erreur doit indiquer le chemin du
checkpoint récupérable. Une interruption durant une évaluation ne doit pas publier le candidat
partiel.

## Performance And Operations

Le checkpoint est écrit une fois par batch, pas par partie. L’historique complet peut devenir le
principal coût d’écriture si une campagne contient beaucoup de candidats ; la structure doit donc
permettre de conserver un historique borné ou séparé dans un fichier append-only ultérieur.

Le rapport final doit exposer au minimum le budget CPU consommé, le nombre de reprises, le dernier
batch terminé et la phase d’arrêt. Un log de session doit distinguer : `checkpoint_saved`,
`resume_loaded`, `budget_exhausted`, `interrupted` et `validation_completed`.

## Edge Cases

- `Ctrl+C` pendant le premier batch : le checkpoint contient le profil initial et `next_batch=0`.
- arrêt brutal pendant l’écriture : le fichier précédent reste valide grâce à l’écriture atomique ;
  le fichier temporaire peut être ignoré ou nettoyé au prochain démarrage.
- checkpoint terminé repris par erreur : refuser par défaut ou demander une option explicite de fork.
- budget déjà épuisé au resume : lancer uniquement la finalisation/validation si elle est en attente,
  sinon terminer sans nouvelle recherche.
- changement de code ou de profil : l’empreinte de configuration invalide le resume et force un
  nouveau run ou un fork explicite.
- plusieurs processus sur le même checkpoint : verrouillage ou refus avant toute mutation.
- reprise après une validation interrompue : rejouer la validation complète avec les mêmes seeds.

## Testing Strategy

- sauvegarde et restauration d’un checkpoint après un batch synthétique ;
- reprise avec le même résultat qu’une exécution continue sur une seed fixe ;
- interruption pendant une évaluation : dernier batch cohérent conservé ;
- compteur CPU qui n’avance pas pendant une attente simulée ;
- refus d’un checkpoint dont la configuration, le profil ou la seed diffère ;
- écriture atomique et récupération après fichier temporaire incomplet ;
- reprise de la validation finale ;
- absence de publication avant une validation réussie ;
- tests CLI des options `--checkpoint`, `--resume` et `--compute-seconds` ;
- suite complète du dépôt et benchmark court de non-régression du débit.

## Rollout And Migration

L’implémentation est rétrocompatible pour les campagnes sans checkpoint. Les campagnes existantes
continuent d’utiliser `--duration-seconds` jusqu’à migration explicite. Une nouvelle campagne
persistent devra commencer avec un nouveau fichier de checkpoint et une seed documentée.

La première version peut limiter la reprise aux quatre modes actuels sans modifier la sérialisation
des profils publiés. Les anciens `results.json` restent consultables mais ne deviennent pas des
checkpoints automatiquement, car ils ne contiennent pas nécessairement les échelles de mutation ni
la position exacte dans la boucle.

## Authorization And Feature Gates

Aucun contrôle d’accès ou feature flag applicatif n’est nécessaire. Le checkpoint est un artefact
local d’expérimentation ; ses permissions doivent rester celles de l’utilisateur qui lance Poetry.

## Files Expected To Change

- `shards_ai/optimization/heuristic.py` : état reprenable, budget CPU, sauvegarde par batch et
  restauration ;
- `scripts/optimize_heuristic.py` : options CLI, gestion de `SIGINT`, reprise et messages d’état ;
- tests d’optimisation et de CLI : checkpoint, resume, interruption, compatibilité ;
- `doc/Current state/Heuristic player.md` ou une page d’état dédiée : commandes et comportement
  persistant.
