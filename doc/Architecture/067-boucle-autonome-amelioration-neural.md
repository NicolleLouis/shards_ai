# Boucle autonome d'amélioration du joueur neural

## Objectif

Permettre de lancer une campagne de `N` expériences d'une durée nominale d'une heure. Pour
chaque expérience, un agent propose une hypothèse, l'implémente, entraîne ou évalue un candidat,
mesure le résultat et produit un rapport Markdown humainement analysable.

Chaque expérience terminée est persistée sur la branche présente au lancement :

- une expérience rejetée conserve son rapport et ses métriques, mais pas son code candidat ;
- une expérience acceptée conserve son code, sa recette, son rapport et son checkpoint promu ;
- aucun `git push` n'est effectué automatiquement ;
- une interruption ne peut perdre que l'expérience en cours, jamais les expériences déjà
  commitées.

Le système est un outil de recherche expérimental. Il ne garantit pas que l'agent trouve une bonne
idée ; il garantit que les idées essayées restent comparables, traçables et réversibles.

## État courant

Le dépôt sépare déjà le moteur de jeu, les joueurs heuristiques et les joueurs neuronaux. Les
points d'intégration existants sont :

- `NeuralPlayer` et les factories de modèles dans `shards_ai/ai/` ;
- les recettes sous `configs/neural_training_profiles/` ;
- le checkpoint mutable unique `artifacts/neural_training/checkpoint.pt` ;
- les checkpoints stables sous `configs/neural_profiles/vNNN.pt` ;
- les entraînements imitation, DAgGER et PPO exposés par le `Makefile` ;
- `benchmarks/benchmark_neural_mix.py` et les benchmarks neural spécialisés ;
- `scripts/validate_neural_profile.py` pour la comparaison et la promotion ;
- les rapports et sorties générés sous `artifacts/`.

Le checkout de départ peut contenir des changements non commités. L'orchestrateur ne doit ni les
réinitialiser ni les inclure silencieusement dans une campagne.

## Comportement cible

La commande prend une durée ou un nombre d'expériences, puis mémorise la branche, le commit et
l'état du checkout au démarrage. Elle exécute séquentiellement :

1. lire les expériences et résultats précédents ;
2. choisir une hypothèse unique et falsifiable ;
3. créer un worktree expérimental depuis le dernier commit de la branche de campagne ;
4. modifier uniquement le périmètre autorisé ;
5. exécuter les tests, un smoke test, puis l'entraînement et le screening dans le budget ;
6. exécuter la validation finale si le screening est prometteur ;
7. générer le rapport Markdown de l'expérience ;
8. commiter le résultat dans le worktree expérimental ;
9. intégrer ce commit sur la branche de campagne ;
10. supprimer le worktree temporaire et commencer l'expérience suivante.

Une expérience qui atteint son timeout ou échoue techniquement est `inconclusive` ou `failed`,
selon la cause, et reçoit elle aussi un rapport commitable. Elle n'est pas présentée comme un
rejet compétitif.

## Hors périmètre et protections

L'automate ne doit jamais modifier :

- `shards_ai/game/` et les règles, transitions, actions ou observations du moteur ;
- les implémentations de joueurs heuristiques ;
- `configs/heuristic_profiles/` ;
- `configs/neural_profiles/`, les profils d'entraînement stables et les pointeurs `active.yaml` ;
- les seeds, adversaires, datasets de référence ou seuils de promotion de la campagne ;
- un checkpoint stable déjà promu.

Le joueur neural peut évoluer librement dans son périmètre : architecture, dimensions, encodeurs,
losses, entraînement imitation, DAgGER, PPO, recherche Monte Carlo, données, sampling, reward
shaping, orchestration interne et analyses. Une architecture peut être ajoutée, remplacée ou
supprimée dans une expérience, sous réserve de conserver les checkpoints promus nécessaires à
leur rechargement historique.

L'absence de triche informationnelle est une propriété testée, pas seulement une instruction de
prompt. L'expérience doit utiliser l'observation fournie au joueur neural et ne doit pas accéder à
l'état complet du moteur pour décider. Les tests de non-régression vérifient notamment l'absence de
main adverse, pioche adverse et cartes futures de rivière.

## Décisions clés

1. **Un commit par expérience terminée.** Le commit est créé avant de passer à l'expérience
   suivante. Les résultats sont donc persistants pendant toute la campagne.
2. **Worktree isolé.** L'agent ne travaille jamais dans le checkout de l'utilisateur. Le worktree
   est créé depuis le dernier commit de la branche de campagne et supprimé après intégration.
3. **Préflight strict.** La commande refuse de démarrer si le checkout de départ est sale, si la
   branche change pendant l'exécution, ou si une autre campagne utilise le même répertoire.
4. **Intégration sans push.** Après le commit expérimental, l'orchestrateur l'intègre sur la
   branche mémorisée par `cherry-pick` ou mécanisme équivalent. Aucun push distant n'est réalisé.
5. **Interruption transactionnelle.** `SIGINT`, timeout du processus ou arrêt propre n'efface
   aucun commit déjà intégré. Le worktree courant peut être abandonné et nettoyé ; son expérience
   est la seule perte autorisée.
6. **Une hypothèse majeure à la fois.** L'agent peut explorer toute la surface autorisée, mais le
   rapport doit identifier le changement principal et ses variables. Les modifications multiples
   doivent être justifiées comme une seule hypothèse architecturale testable.
7. **Baseline et panel immuables.** Le candidat et la référence sont évalués avec le même
   protocole, les mêmes seeds et les mêmes adversaires. Le screening rapide ne peut pas modifier
   les critères de validation finale.
8. **Promotion indépendante.** L'agent propose un candidat ; un gate déterministe décide de la
   promotion. Une amélioration contre V008 ne peut pas être fabriquée en modifiant le benchmark.
9. **Rapports d'échec versionnés.** Les rapports Markdown vivent sous `doc/Experiments/`. Les
   checkpoints, logs volumineux et JSON intermédiaires restent sous `artifacts/` et ne sont pas
   nécessaires au commit d'un rejet.
10. **Provenance explicite.** Chaque candidat et chaque rapport enregistrent parent, commit,
    recette, dataset, seed, fingerprint d'architecture, hash du checkpoint et commandes exactes.
11. **Budget global borné.** Une expérience dispose par défaut de 3 600 secondes au total : 2 400
    secondes d'entraînement ou de collecte, 750 secondes de screening et 450 secondes de marge
    pour l'idée, les tests, l'analyse et l'arrêt propre. Ces valeurs sont configurables, mais leur
    somme ne peut pas dépasser le budget global.
12. **Budget dépendant de l'expérience.** L'agent peut choisir une classe légère, normale ou lourde
    et proposer une répartition différente, mais l'orchestrateur impose le plafond global et
    enregistre la répartition réellement demandée. Une architecture lente n'obtient pas un budget
    implicite supérieur.
13. **Performance mesurée à chaque expérience.** Le résultat doit fournir une mesure comparable
    avant/après. Une régression supérieure à 5 % sur le temps ou le débit rejette par défaut un
    candidat accepté, sans modifier le workload pour le sauver.
14. **Optimisation de performance séparée.** Les optimisations sont lancées avec
    `--experiment-kind performance` et le workflow `optimize-game-performance`. Elles ont leurs
    propres hypothèses, profils, benchmarks et rapports ; elles ne sont pas injectées dans chaque
    expérience de qualité.
15. **Backlog évolutif.** `doc/Ideas.md` est le catalogue initial et la mémoire courte des pistes
    à essayer. Codex peut supprimer une idée devenue incohérente, marquer `done` une idée testée
    avec ou sans succès, et ajouter une piste nouvelle. Toute suppression ou clôture doit être
    justifiée dans le rapport et reste visible dans le commit de l'expérience.
16. **Deux fenêtres d'écriture des idées.** Au début, Codex lit le catalogue, choisit l'hypothèse
    courante et peut ajouter les pistes apparues pendant cette sélection pour les expériences
    futures. À la fin, il analyse le résultat et peut ajouter des corrections, variantes ou next
    steps. Ces deux phases sont distinguées dans le rapport.

## Architecture proposée

### Orchestrateur

`scripts/meta_improve.py` sera responsable de la campagne, des budgets, de l'état persistant, des
worktrees et de l'intégration Git. Il ne délègue pas à l'agent les commandes de promotion, les
seeds ou les chemins critiques.

### Agent de recherche

L'agent reçoit l'état courant de l'IA, les rapports précédents et les métriques disponibles. Il
retourne une hypothèse structurée, une liste de fichiers autorisés à modifier, une recette et des
critères attendus. Il peut proposer une nouvelle architecture ou un nouveau concept, mais il doit
faire échouer l'expérience si sa modification touche une zone interdite.

La première implémentation utilise Codex CLI. L'orchestrateur ne dépend toutefois pas du binaire
Codex : il lance une commande d'agent configurable, actuellement fournie par Codex CLI, dans le
worktree expérimental. Le prompt, le chemin du manifeste, le budget, le commit parent et le type
d'expérience sont transmis par variables d'environnement et fichiers locaux. Codex doit modifier
le worktree, exécuter les validations autorisées et écrire `result.json` dans le répertoire de
l'expérience.

Le catalogue initial d'hypothèses est `doc/Ideas.md`. Codex le lit avant de choisir une expérience
et privilégie une idée existante lorsqu'elle correspond aux métriques observées. Pendant cette
phase de préparation, il peut ajouter des hypothèses non sélectionnées pour les campagnes futures,
ou retirer une piste devenue sans objet. Après l'entraînement et l'évaluation, il dispose d'une
seconde phase d'analyse pendant laquelle il peut marquer l'idée comme réussie ou échouée et ajouter
des corrections ou next steps. Le rapport conserve le diff exact du catalogue et sépare les
changements de préparation de ceux issus de l'analyse finale.

Un appel API pourra remplacer cette commande ultérieurement sans modifier les règles Git, les
budgets, les gates ou le format des rapports. Le push distant et les décisions de promotion restent
hors du périmètre de Codex ; ils appartiennent à l'orchestrateur.

### Runner

Le runner exécute les commandes allowlistées dans le worktree : tests ciblés, smoke test,
entraînement, analyses offline, benchmarks et validation. Chaque étape utilise un timeout et écrit
une sortie persistante dans le répertoire de l'expérience.

### Gate de comparaison

Le screening sert uniquement à éliminer rapidement les candidats manifestement mauvais. La
validation finale compare le candidat et la référence contre Random, V007, V008 et les profils
neural disponibles selon le protocole existant. Elle utilise un score pondéré : V008 pèse 50 %,
V007 20 %, Random 10 % et les profils neural disponibles 20 % au total. Une petite baisse contre
Random ou V007 est donc tolérée si le gain global est suffisant.

V008 reste un garde-fou stratégique : son delta ne peut pas être négatif. Les adversaires
secondaires peuvent baisser jusqu'à 5 points de taux de victoire, le gain pondéré doit être d'au
moins 0,5 point et au moins un delta doit atteindre 1 point. Les poids, seuils, deltas par
adversaire et comptes exacts sont conservés dans le rapport. Le résultat peut aussi fournir des
`categories` avec un `delta` ou `delta_score` et un `weight` ; leur score moyen contribue alors à
30 % du gain global, les résultats du panel conservant 70 %. Une amélioration forte sur plusieurs
catégories peut donc compenser une petite perte secondaire, sans pouvoir contourner le garde-fou
V008.

Le gate vérifie aussi la performance d'exécution avec le même workload. Le résultat agent doit
contenir `performance.baseline` et `performance.candidate`, avec `elapsed_seconds` ou
`throughput`. Cette mesure constitue un garde-fou ; elle ne remplace pas la validation compétitive.

### Maintenance de performance

Une campagne de qualité peut être suivie d'une campagne de maintenance toutes les 8 à 12
expériences, ou plus tôt après une régression persistante. Cette campagne profile et optimise les
chemins neural, l'entraînement, la collecte et l'orchestration dans le périmètre autorisé. Elle
n'altère jamais le moteur, les joueurs heuristiques ou le masque d'information. Une optimisation
n'est conservée qu'avec un benchmark identique, des tests passants et un gain robuste d'au moins
2 % ; sinon elle produit un rapport de rejet.

### Registre des expériences

Le registre lisible par machine est un manifeste par expérience sous `artifacts/experiments/`.
Le résumé humain commité est sous `doc/Experiments/`. Les deux portent le même identifiant.

## Modèle de données

Chaque manifeste contient au minimum :

```yaml
experiment_id: exp-00001
campaign_id: campaign-2026-08-05-001
experiment_kind: quality
parent_commit: <sha>
parent_profile: v002 # dernière version neural active au moment de la campagne
status: accepted | rejected | failed | inconclusive | interrupted
hypothesis: "..."
allowed_changes: []
dataset: "..."
seed: 104
budget_seconds: 3600
commands: []
training_recipe: {}
training_budget_seconds: 2400
screening_budget_seconds: 750
overhead_budget_seconds: 450
architecture_fingerprint: "..."
baseline_checkpoint_sha256: "..."
candidate_checkpoint_sha256: "..."
screening: {}
validation: {}
tests: {}
decision_metrics: {}
performance:
  baseline: {elapsed_seconds: 0.0}
  candidate: {elapsed_seconds: 0.0}
performance_gate: {}
commit: <sha-or-null>
```

Le checkpoint mutable reste `NEURAL_CHECKPOINT`. Les fichiers sous `configs/neural_profiles/` ne
sont créés ou modifiés qu'au moment d'une promotion acceptée.

Pour une expérience de qualité acceptée, Codex fournit `candidate_profile` et
`candidate_checkpoint`. L'orchestrateur relance ensuite `scripts/validate_neural_profile.py` dans
le worktree, avec un panel indépendant de 200 parties par adversaire. La promotion crée les
fichiers stables et les pointeurs actifs uniquement après cette seconde validation ; une erreur ou
un rejet transforme l'expérience en échec ou rejet et le worktree candidat est abandonné.

## Flux Git et reprise

Le worktree expérimental reçoit une branche temporaire. À la fin :

- rejet, échec ou résultat non concluant : commit du rapport et des petits métadonnées, puis
  intégration sur la branche de campagne ;
- acceptation : commit du code, de la recette, du rapport et de la promotion, puis intégration sur
  la branche de campagne ;
- interruption avant le commit : aucun changement de la branche de campagne ; nettoyage du
  worktree courant ;
- interruption après le commit : le commit reste sur la branche et la reprise le détecte.

Le démarrage d'une nouvelle campagne inspecte les commits et les manifestes existants pour choisir
le prochain identifiant. Il ne réutilise jamais un identifiant déjà commité.

## Budget et exécution

Le budget nominal d'une expérience est d'une heure, réparti par défaut comme suit :

| Phase | Budget par défaut |
| --- | ---: |
| Idée, implémentation, tests et smoke test | 450 secondes inclus dans la marge |
| Entraînement ou collecte | 2 400 secondes |
| Screening en parties et analyse | 750 secondes |
| Marge d'arrêt propre et sorties | 450 secondes |

L'orchestrateur transmet ces budgets à l'agent et termine la commande au plafond global. Les
scripts d'entraînement doivent sauvegarder à des frontières cohérentes (fin d'epoch, update PPO
ou batch DAgGER) afin qu'un timeout ne produise pas un checkpoint partiellement écrit. Le rapport
enregistre `games_seen`, `transitions_seen`, `updates`, `examples_per_second`, la durée réelle et
le meilleur checkpoint atteint.

Le runner doit distinguer :

- `completed` : protocole terminé ;
- `rejected` : candidat évalué et moins bon ;
- `failed` : commande ou validation en erreur ;
- `inconclusive` : timeout ou puissance statistique insuffisante ;
- `interrupted` : arrêt demandé avant la fin.

Une expérience peut lancer un screening court, puis une validation plus large seulement si le
screening est favorable. Le nombre de parties, les seeds et les adversaires restent ceux de la
configuration de campagne, indépendamment de l'hypothèse proposée.

Les expériences de qualité utilisent le budget wall-clock pour mesurer le rendement pratique, mais
conservent aussi le nombre d'updates et de transitions. Les expériences comparant directement une
architecture doivent, lorsque c'est possible, ajouter une mesure à transitions constantes afin de
séparer qualité algorithmique et avantage de débit.

## Observabilité et rapports

Le rapport Markdown doit rendre l'expérience analysable sans relire toute la conversation :

- objectif et hypothèse ;
- idée source dans `doc/Ideas.md` ou nouvelle idée produite ;
- diff de `doc/Ideas.md`, séparé entre préparation et analyse finale ;
- parent et état initial ;
- changements effectués ;
- commandes exactes et durée ;
- résultats des tests ;
- résultats offline par phase et famille d'action ;
- résultats en parties par adversaire, avec victoires, défaites, draws et taux ;
- comparaison au baseline ;
- décision et justification ;
- limites, biais possibles et prochaine piste.

Les événements de campagne sont également écrits dans un journal append-only. Une sortie partielle
ne doit pas être confondue avec un résultat validé.

## Cas limites

- branche de lancement sale : refus sans modification ;
- branche avancée par un humain pendant la campagne : arrêt avant intégration ;
- candidat sans checkpoint ou architecture illisible : `failed`, rapport commité ;
- changement interdit détecté : rejet technique, rapport commité, aucun code candidat ;
- entraînement concurrent utilisant `checkpoint.pt` : refus ou arrêt ;
- validation interrompue : `inconclusive`, jamais promotion ;
- conflit de cherry-pick : arrêt sans résolution automatique destructive ;
- agent tué brutalement : les commits précédents restent intacts, le worktree peut être marqué
  abandonné lors du prochain lancement.

## Stratégie de tests

Avant une campagne réelle :

- tests unitaires du manifeste, des transitions d'état et de la reprise ;
- test de création et de suppression d'un worktree temporaire ;
- test d'intégration d'un commit rejeté puis d'un commit accepté ;
- test d'arrêt pendant chaque étape ;
- test de détection des fichiers interdits ;
- test de refus d'un checkout sale ;
- test de refus d'une modification de seed, panel ou seuil ;
- test de conservation d'une nouvelle idée après rejet du candidat ;
- test de conservation des suppressions et statuts `done` du catalogue ;
- test de non-régression du masque d'information ;
- campagne simulée avec commandes rapides et un faux agent ;
- une expérience réelle d'une durée très courte avant toute boucle d'une heure.

Les tests du moteur et des joueurs heuristiques restent des tests de garde et ne doivent pas être
modifiés par l'orchestrateur.

## Rollout

1. Implémenter le manifeste et le registre sans lancer d'agent.
2. Implémenter le préflight Git et les worktrees.
3. Implémenter les rapports et les commits de rejet avec un faux runner.
4. Ajouter le runner réel avec une seule famille d'expérience contrôlée.
5. Exécuter une campagne de quelques minutes et vérifier que chaque résultat est sur la branche.
6. Ajouter l'exploration libre des architectures, DAgGER, PPO et Monte Carlo.
7. Autoriser la promotion automatique après validation indépendante.

Avant d'activer les campagnes longues, exécuter une calibration du parent courant à 300, 900,
1 800, 2 700 et 3 600 secondes. Cette calibration mesure le rendement marginal du training et
permet d'ajuster les sous-budgets sans modifier le protocole de comparaison.

Une fois la boucle de qualité stabilisée, exécuter une campagne `performance` dédiée et planifier
son rappel après 8 à 12 expériences de qualité. La configuration `performance_maintenance_every`
permet d'automatiser ce rappel entre deux expériences, avec un prompt et un rapport de type
`performance`. Le rappel est un point de maintenance séparé, pas une modification automatique du
code pendant une expérience de qualité.

Le push distant reste hors périmètre de toutes les versions.

## Fichiers attendus

- `scripts/meta_improve.py` : point d'entrée de campagne ;
- `shards_ai/experimentation/` : manifestes, états, budgets, policies Git et rapports ;
- `configs/meta_improvement.yaml` : paramètres protégés de campagne ;
- `doc/Experiments/` : rapports Markdown versionnés, succès comme échecs ;
- `artifacts/experiments/` : manifestes détaillés et sorties générées ;
- `tests/experimentation/` : tests de l'orchestrateur, de la reprise et des protections ;
- `Makefile` : cible de lancement, sans contourner les cibles neural existantes.

Les chemins exacts peuvent évoluer pendant l'implémentation, mais toute modification doit rester
dans le périmètre de ce document ou recevoir une nouvelle architecture.

## Questions ouvertes

- **Non bloquante :** faut-il commencer avec un catalogue d'idées existant ou autoriser dès la V1
  des hypothèses entièrement nouvelles ? Les protections de fichiers et de validation s'appliquent
  dans les deux cas.
- **Non bloquante :** quelle puissance statistique exacte utiliser pour la confirmation finale ? La
  V1 doit conserver la configuration existante et permettre de l'augmenter sans changer le
  benchmark.
