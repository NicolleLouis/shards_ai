# Artefacts et scripts

## Politique

Le code exécutable est séparé des sorties générées :

| Besoin | Emplacement | Règle |
| --- | --- | --- |
| Préparation, entraînement, validation ou reporting | `scripts/` | Commande rejouable, avec chemins variables en arguments. |
| Mesure de performance ou campagne de référence | `benchmarks/` | Mesure uniquement ; sorties sous `artifacts/`. |
| Logique d'analyse réutilisable | `shards_ai/analysis/` | Code importable et testé. |
| Configuration ou profil publié | `configs/` | Source de vérité versionnée. |
| Dataset, checkpoint, métriques, rapport ou résultat de run | `artifacts/` | Sortie locale, ignorée par Git, classée par type. |
| Décision, comportement courant ou règle | `doc/` | Markdown uniquement ; aucun artefact d'expérience. |

## Convention `artifacts/`

- `imitation_dataset/` : datasets d'entraînement et manifests ;
- `neural_training/` : le checkpoint mutable unique et ses métriques de training ;
- `neural_benchmark/` : résultats de matchs et rapports neural ;
- `neural_validation/` : sorties de validation et promotion ;
- `analysis/` : rapports et exports d'analyse de parties, un sous-répertoire par campagne.

Les anciens chemins `analysis_output/` et `scripts/analysis_output/` sont interdits. Une sortie
temporaire sans valeur de reproduction va dans `/tmp`, pas dans le dépôt.

## Cycle de vie

On conserve un artefact s'il est référencé par une configuration ou une commande active, s'il sert
de baseline publiée, ou si sa comparaison reste utile à une décision. Sinon, il peut être supprimé
après vérification de ses références. Les datasets lourds obsolètes doivent être supprimés plutôt
que renommés ou dupliqués.

Chaque run important doit enregistrer dans son manifest la version des profils, la seed et le
schéma de données. Les scripts et benchmarks ne créent jamais leur sortie dans leur propre
répertoire.

## Campagnes autonomes neural

`scripts/meta_improve.py` exécute une campagne séquentielle dans des worktrees temporaires. Codex
CLI est l'agent par défaut (`codex exec` avec prompt sur stdin), mais la commande est remplaçable.
Le défaut utilise `--sandbox workspace-write` pour limiter l'agent au
worktree de l'expérience sans accès complet à la machine. Sur Linux, `bubblewrap` et les user
namespaces doivent être correctement configurés ; `danger-full-access` reste une surcharge
explicite réservée à un environnement isolé. Le checkout doit être propre au démarrage. Chaque expérience produit un rapport Markdown sous
`doc/Experiments/`, puis un commit de rejet, d'échec, d'interruption ou d'acceptation est intégré
sur la branche de lancement ; aucun push n'est effectué.

Les sorties détaillées sont copiées sous
`artifacts/experiments/<campaign-id>/<experiment-id>/` : manifeste, logs, tests fixes, validation
et promotion. Le seul checkpoint d'entraînement mutable reste
`artifacts/neural_training/checkpoint.pt`. Les profils stables et les pointeurs actifs sont
protégés contre l'agent et ne peuvent être créés ou modifiés que par la promotion indépendante
après validation.

La vitesse n'est pas un veto automatique à une promotion de qualité. Une amélioration de qualité
reste promouvable même si sa régression d'exécution dépasse 5 %, dès que les métriques de qualité
franchissent le gate. Le manifeste et le rapport signalent alors une dette de performance et la
prochaine expérience de la campagne est forcée en famille `performance`. Cette obligation reste
active jusqu'à une expérience de performance acceptée ; un échec ou un timeout de cette
expérience n'efface pas la dette.

Le catalogue `doc/Ideas.md` est lu au début d'une expérience et peut être mis à jour pendant la
préparation puis après l'analyse : nouvelles idées, suppressions, statuts `done` et next steps.
Les modifications sont conservées même pour un candidat rejeté et leur diff est inclus dans le
rapport.

La campagne possède aussi un mode `analysis` diagnostique. Il étudie la dernière version neural
active sans entraîner ni promouvoir de checkpoint et conserve ses observations, limites et
recommandations dans le rapport. Une analyse est déclenchée après une promotion qualité ou après
quatre expériences qualité consécutives non promues ; les expériences performance ne modifient pas
ce compteur. Une analyse réussie le remet à zéro.

Chaque agent classe aussi son expérience (`ppo`, `imitation`, `dagger`, `data`, `objective`,
`inference`, `monte_carlo`, `architecture`, `representation`, `search`, `performance` ou `other`)
et décrit sa nouveauté.
L'orchestrateur transmet l'historique des familles pour encourager une famille sous-explorée après
plusieurs essais PPO, sans interdire une nouvelle variante PPO justifiée. Lorsqu'un dataset est
déclaré, son hash, son nombre d'enregistrements, son teacher et sa recette sont conservés ; le
dataset est copié dans les artefacts de l'expérience hors Git.
