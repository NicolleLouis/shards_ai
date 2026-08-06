# Idées et priorités de travail

Ce fichier contient le catalogue courant des décisions et des prochaines expériences. Les détails,
les métriques et les commandes restent dans `doc/Experiments/`.

## Règles de décision

- Comparer chaque candidate à la référence neural active, à Random, à v007 et à la garde v008.
- Utiliser un panel complet et reproductible ; un screening court ne peut jamais accepter une candidate.
- Séparer entraînement, validation et holdout par `game_id`, et conserver la provenance des seeds.
- Mesurer séparément qualité, temps de partie, nombre d'actions et coût d'inférence.
- Modifier une seule hypothèse structurante à la fois ; ne pas modifier le moteur, les heuristiques ou le masque d'information.
- Conserver v002 comme référence tant qu'un nouveau profil n'a pas passé la gate complète.
- Ne pas reprendre une distillation locale ou une loss ranking-only sans preuve holdout indépendante et budget explicite de décisions modifiées.

## Diagnostic de situation

Les expériences `exp-00045` à `exp-00048` ont testé plusieurs corrections locales autour de
`recruit_mercenary`, de l'ancrage v002 et de la loss d'imitation. Toutes ont régressé au moins une
référence importante, malgré des gains isolés contre v001 ou v008. `exp-00049` confirme que les
faiblesses sont liées à des catégories d'actions/cartes et que certaines erreurs PLAY sont
confiantes.

Les expériences `exp-00050`, `exp-00051`, `exp-00053` et `exp-00055` ont testé quatre variations
de représentation ; elles ont toutes été rejetées pour régression ou moyenne insuffisante malgré
des gains isolés. `exp-00052` a amélioré le duel direct contre v002 mais a régressé v007/v008.
`exp-00054` confirme que les erreurs v007 sont surtout en PLAY et souvent confiantes, sans fournir
de preuve causale de force en partie.

Conclusion : v002 reste la référence active. Une prochaine mise à jour doit mesurer la dérive de
politique sur un holdout indépendant avant de modifier encore l'architecture ou l'objectif.

## Priorité immédiate : changer d'espace de solution

### 1. À privilégier — diagnostic de dérive puis correction ciblée

Construire d'abord un holdout indépendant par partie, stratifié par phase et type d'action, avec
ECE, Brier, reliability bins et un budget explicite de décisions modifiées par rapport à v002.
Après attribution, tester au plus une correction bornée sur les slices responsables, notamment
PLAY/action-card v007 et `recruit_mercenary`.

Hypothèse falsifiable : une correction bornée sur des états attribués améliore la slice ciblée sans
dépasser le budget de dérive et sans régresser v002, Random, v007 ou v008 sur le panel complet.

Protocole minimal : holdout par `game_id` non utilisé pour l'entraînement, contrôle v002, décisions
changées par phase/type/carte, puis validation complète et benchmark comparable. Ne pas sélectionner
sur l'accuracy offline, v008 seul ou le nombre d'actions.

### 2. À privilégier — changer le type d'entraînement

Tester une méthode qui ne repose pas uniquement sur l'imitation de labels heuristiques v008, par
exemple un entraînement PPO suffisamment long et profilé, du reinforcement learning offline/online
borné, ou un entraînement hybride imitation + objectif de résultat. Le choix exact doit être décrit
dans une nouvelle architecture ou une hypothèse dédiée avant implémentation.

Hypothèse falsifiable : un signal d'entraînement différent améliore la force en partie sur un panel
reproductible sans apprendre simplement les biais v007/v008 et sans perdre la garde v002.

Garde-fous : profiler d'abord la collecte si PPO est retenu, séparer les budgets collecte/entraînement/
validation, conserver un holdout non utilisé pour choisir les mises à jour et refuser toute conclusion
fondée sur une amélioration contre v008 seul.

### 3. À conserver comme diagnostics préparatoires

- Étendre l'analyse de `exp-00054` avec un holdout par partie et des intervalles d'incertitude.
- Séparer les erreurs de type d'action des erreurs de carte, en ciblant PLAY/action-card v007 et
  `recruit_mercenary`; garder les cartes rares comme slices diagnostiques tant que la couverture est faible.
- Attribuer les décisions modifiées et fixer un budget de dérive avant tout nouvel entraînement.
- Utiliser `doc/Architecture/069-protocole-analyse-informative.md` pour éviter les analyses redondantes
  et exiger une question de connaissance nouvelle.

Ces diagnostics doivent servir à choisir entre l'axe architecture/représentation et l'axe training,
pas devenir une nouvelle série d'analyses descriptives sans décision associée.

## Historique condensé

| Expérience | Statut | Conclusion opérationnelle |
|---|---|---|
| exp-00023 | Rejetée | Les reprises PPO courtes n'ont pas produit de gain exploitable. |
| exp-00024 à exp-00026 | Rejetées | DAgger uniforme et mélanges historiques/on-policy : régressions. |
| exp-00027 à exp-00035 | Rejetées | Pondérations, filtrages, marges teacher et ancrages globaux : pas de généralisation robuste. |
| exp-00036 à exp-00038 | Rejetées | Dataset équilibré, fine-tuning réduit et premières divergences : compromis non résolu. |
| exp-00039 | Analyse terminée | Premiers signaux sur `recruit_mercenary`, BUY et PLAY confiants. |
| exp-00040 à exp-00041 | Rejetées | Accuracy holdout ou marge teacher seules insuffisantes. |
| exp-00044 | Analyse terminée | Diagnostic proche d'exp-00039 ; logits inter-version non comparables. |
| exp-00045 | Rejetée | Surpondération mercenaire : régression de qualité et fort ralentissement. |
| exp-00046 | Rejetée | Ancrage local hors états mercenaires : régression v002, Random et v007 malgré un gain de temps. |
| exp-00047 | Rejetée | Ancrage dans les états achat/recrutement : régressions et trajectoires plus longues. |
| exp-00048 | Rejetée | Ranking-only : v008 protégé, mais régressions Random, v007 et v002. |
| exp-00049 | Analyse terminée | Holdout indépendant : faiblesses par action/carte et erreurs PLAY souvent confiantes. |
| exp-00050 | Rejetée | Résidu sémantique identité : régressions Random, v007 et v002 malgré v008 en hausse. |
| exp-00051 | Rejetée | Résidu sémantique borné : forte régression Random et v002 ; runtime variable. |
| exp-00052 | Rejetée | KL globale : gain contre v002 neural, mais régressions v007 et v008. |
| exp-00053 | Rejetée | Biais global par type d'action : régressions v007, v002 et coût d'inférence accru. |
| exp-00054 | Analyse terminée | 15 711 décisions : erreurs v007 concentrées en PLAY, cartes rares sous-couvertes. |
| exp-00055 | Rejetée | Interaction phase×action : gains v007/v008/v002, mais Random régresse et moyenne -0,2 point. |
| exp-00056 | Rejetée | Le résidu phase×action avec base v002 gelée protège v002, mais régresse Random et v007 ; runtime légèrement supérieur. |
| exp-00058 | Rejetée | Mélange DAgGER v002/v008 sur les états divergents : v002 et Random protégés au screening, mais aucune hausse moyenne et runtime/inférence en hausse. |
| exp-00059 | Rejetée | Pondération ciblée `play_card` depuis v002 : +4 points contre v002, mais v007 -5, v008 -1 et runtime médian +45,4 %. |
| exp-00060 | Rejetée | Pondération PLAY conservatrice (1,10, 1 000 décisions) : Random +2 et v002 +4, mais v007/v008 -8 et runtime +3,7 %. |
| exp-00061 | Analyse terminée | Holdout diagnostique masqué de 13 816 décisions : dérive v007 concentrée en PLAY/cartes, erreurs souvent confiantes, calibration ECE v007 0,171 et v008 0,027 ; aucune candidate. |
| exp-00062 | Rejetée | Biais scalaires gelés sur cinq cartes PLAY v007 : politique identique à v002 sur le panel et aucune amélioration de force ; coût médian +9,7 %. |
| exp-00063 | Rejetée | DAgGER on-policy priorisé sur quatre catégories : v008 +5 et v002 neural +2, mais Random -5, v007 -1 et runtime +3,6 %. |
| exp-00064 | Rejetée | Imitation conservatrice depuis v002 (1e-5, 10 000 décisions) : v007 +8, mais Random -4, v008 -3 et v002 -14 ; runtime +5,0 %. |
| exp-00065 | Rejetée | Conservation de politique par logits centrés v002 : Random +5 et v008 +10 au screening, mais v007 -20, v002 -10 et runtime médian +37,5 %. |
| exp-00066 | Analyse terminée | La difficulté de v002 augmente avec la compétition de l’ensemble légal : accuracy v007 64,7 %, erreurs confiantes 20,7 %, et dérive d’argmax v001/v002 7,7 % ; le signal reste surtout PLAY/gain_mastery et cartes. |
| exp-00067 | Rejetée | Continuation PPO courte depuis v002 avec horizon GAE v002, sans shaping v003 : aucun checkpoint candidat sauvegardé dans l'environnement ; le contrôle v002/v002 est neutre et la recette doit être reprise seulement avec une collecte instrumentée. |
| exp-00069 | Rejetée | Reprise PPO instrumentée depuis v002 : le processus est encore arrêté avant le premier checkpoint, donc aucune force apprise n'est mesurable ; le contrôle v002/v002 reste neutre en qualité et plus lent au second passage. |

## Expérience exp-00065 — rejetée

- [Terminé] Tester une pénalité de conservation des scores centrés de v002 pendant une imitation
  mixte v007/v008, depuis `configs/neural_profiles/v002.pt`, avec 10 000 décisions et Adam réinitialisé.
- [Résultat] Le screening de 20 parties donne Random `+5`, v007 `-20`, v008 `+10`, neural:v002
  `-10` et neural:v001 `+10` points ; la moyenne est négative, donc aucune promotion.
- [Résultat] Le benchmark médian comparable passe de `11,3577 s` à `15,6114 s`, avec `238`
  actions supplémentaires et une inférence de `4,5361 s` à `6,2102 s`.
- [Supprimé] Ne pas reprendre une conservation globale qui exécute un forward v002 à chaque
  décision ; elle ne protège pas v007/v002 et son coût est incompatible avec une candidate qualité.

## Suites après exp-00065

- [À privilégier] Revenir à l'attribution holdout par `game_id` et mesurer un budget d'argmax
  modifié avant tout nouvel objectif ; la conservation doit être appliquée seulement à une slice
  causalement confirmée.
- [À étudier] Si une slice est confirmée, pré-calculer ou embarquer la protection sans second
  forward en production, puis refaire le panel complet contre Random, v007, v008 et v002.
- [À supprimer] Toute conservation globale, toute amélioration v008 seule et toute candidate dont
  le coût comparable augmente sans gain moyen robuste.

## Pistes écartées

- Reprendre à l'identique une distillation locale ou globale, une loss ranking-only, une simple
  surpondération de `recruit_mercenary`, un résidu sémantique ou un biais global phase/action.
- Promouvoir sur la base d'un gain contre v001 ou v008 seul.
- Utiliser une accuracy holdout sans validation en partie.
- Reprendre un PPO court ou interrompu sans profilage de collecte et budget complet.
- Modifier le moteur, les heuristiques ou le masque d'information pour améliorer la candidate.

Toute nouvelle idée doit préciser l'axe choisi (architecture/représentation ou training), une
hypothèse falsifiable, la référence v002, les métriques attendues et la condition de rejet.

## Expérience exp-00066 — analyse terminée

- [Terminé] Mesurer, sur 48 parties et 14 387 observations visibles, la compétition intra-état de
  v002 : taille de l'ensemble légal, marge top-1/top-2, entropie, confiance, loss, Brier,
  accuracy, désaccord d'argmax avec v001, et attribution phase/action/carte.
- [Résultat] Sur les ensembles de 2 actions, l'accuracy est 84,4 %, contre 58,4 % à 7 actions et
  54,6 % à 9 actions. Les erreurs sur 3-4 actions sont particulièrement confiantes (23,3 % et
  27,6 % au seuil 0,8).
- [Résultat] `gain_mastery` atteint 47,8 % d'accuracy avec 44,2 % d'erreurs confiantes,
  `play_card` 72,8 % et `recruit_mercenary` 19,9 % sur 306 exemples. Les cartes faibles restent
  trop peu couvertes pour une correction isolée.
- [Résultat] v002 et v001 changent d'argmax sur 7,7 % des décisions ; le taux atteint 20,1 % pour
  les ensembles de 9 actions, mais seulement 6,9 % sur les trajectoires v007 et 8,7 % sur v008.
- [Conservé] Avant tout entraînement, fixer un budget de dérive par taille d'ensemble et vérifier
  si une slice causale conserve son effet après stratification par partie.

## Nouvelles idées après exp-00066

- [À privilégier] Construire un holdout par `game_id` équilibré sur `legal_action_set_size` et
  `phase/action`, puis estimer les intervalles de la dérive v001/v002 et de l'erreur confiante.
- [À étudier] Si le signal reste après stratification, tester une calibration bornée uniquement sur
  `gain_mastery`/PLAY et les ensembles de 3-4 actions, avec argmax v002 conservé ailleurs.
- [À supprimer] Toute calibration globale fondée sur la confiance moyenne : les états à 1 action
  gonflent artificiellement l'accuracy, et les logits inter-version ne sont pas comparables.

## Expérience exp-00067 — rejetée

- [Terminé] Tester une continuation PPO bornée depuis les poids v002, avec `gamma=0,995`,
  `gae_lambda=0,95`, sans le reward shaping de v003, deux époques d'optimisation et un budget de
  2 048 parties ; la proposition est nouvelle par la correction simultanée de l'horizon et du
  shaping, sans initialisation v001.
- [Résultat] Les paliers 2 048 puis 256 parties n'ont pas produit de checkpoint avant l'arrêt du
  processus dans cet environnement. Le contrôle batché v002 contre v002 sur 20 parties donne
  `0,00` contre Random, v007 et v008 ; aucune force nouvelle n'est démontrée.
- [Résultat] Le benchmark comparable v002 contre v002 reste à `17 527` actions, avec `14,6337 s`
  pour la baseline et `14,6008 s` pour la candidate de contrôle ; cette différence ne constitue
  pas un gain de qualité ni une candidate distincte.
- [Supprimé] Ne pas reprendre cette recette sans diagnostiquer la disparition du processus et
  sauvegarder chaque palier ; v002 reste la référence active.

## Suites après exp-00067

- [À privilégier] Instrumenter la collecte PPO avec une sauvegarde/checkpoint après chaque rollout
  et un progress file, puis reprendre uniquement depuis le dernier palier v002 vérifié.
- [À étudier] Comparer séparément `gamma=0,995` sans shaping et `gamma=1` avec shaping désactivé,
  avec le même nombre de parties et une validation complète batchée ; ne pas confondre absence de
  checkpoint avec absence d'effet de l'objectif.
- [À conserver] Le contrôle v002/v002 et les métriques de runtime restent la référence comparable.

## Expérience exp-00068 — performance — acceptée

- [Terminé] Tester une cache des embeddings statiques de cartes dans `NeuralActionScorer`, active
  uniquement en mode évaluation sans gradient ; l'entraînement conserve le chemin non caché.
- [Résultat] Sur 50 parties v002 contre v002, seeds `0..49`, un thread PyTorch et trois répétitions,
  la médiane passe de `14,7191 s` à `12,9886 s` (`-11,76 %`) et l'inférence de `5,7773 s` à
  `4,9647 s` (`-14,07 %`). Les `17 587` actions et `8 770` décisions sont identiques.
- [Conservé] Le gain est attribuable à la suppression des reconstructions répétées des embeddings
  déterministes pendant les décisions ; aucun moteur, joueur heuristique, masque, checkpoint ou
  objectif d'entraînement n'a été modifié.

## Suites après exp-00068

- [À privilégier] Profiler séparément le coût restant de `encode_observation`, du pooling et de la
  construction des représentations d'actions avant une nouvelle optimisation runtime.
- [À étudier] Évaluer une réutilisation bornée des représentations d'actions uniquement si elle peut
  respecter les identifiants et l'état observable ; mesurer d'abord son taux réel de réutilisation.
- [À supprimer] Toute modification de pooling qui change les trajectoires ou toute cache active en
  entraînement sans stratégie explicite d'invalidation.

## Expérience exp-00069 — PPO — rejetée

- [Terminé] Reprendre l'idée PPO de `exp-00067` avec une correction opérationnelle : profil dédié
  initialisé directement depuis `configs/neural_profiles/v002.pt`, Adam réinitialisé, `gamma=0,995`,
  `gae_lambda=0,95`, sans shaping, 512 parties prévues et quatre rollouts de 128.
- [Résultat] Deux tentatives, dont un smoke test d'une partie, sont arrêtées silencieusement avant
  le premier rollout sauvegardé ; aucun checkpoint candidat n'est produit. Ce n'est donc pas une
  mesure de qualité PPO et aucune promotion n'est possible.
- [Résultat] Le contrôle v002 contre v002 sur 50 parties garde `16 825` actions et `8 472`
  décisions identiques ; les temps observés sont `15,0173 s` puis `19,6409 s`, sans gain de force.
- [Supprimé] Ne pas considérer ce PPO comme un échec algorithmique : la collecte n'a pas atteint
  l'optimisation. Ne pas relancer une longue collecte dans cet environnement sans diagnostic du
  terminateur et checkpoint externe après chaque rollout.

## Suites après exp-00069

- [À privilégier] Exécuter un rollout PPO instrumenté dans un environnement qui conserve les
  processus et les sorties, avec checkpoint atomique externe après chaque rollout et reprise testée
  sur un smoke test avant toute campagne de 512 parties.
- [À étudier] Réduire temporairement le rollout à 1 partie et isoler collecte, optimisation et
  écriture du checkpoint pour identifier l'étape qui provoque l'arrêt ; reprendre ensuite exactement
  la recette `exp-00069` depuis v002.
- [À conserver] Le protocole de qualité doit toujours mesurer Random, v007, v008 et v002, avec
  validation batchée dès que le panel dépasse 20 parties et un benchmark runtime comparable.


## Décisions issues de la campagne exp-00050 à exp-00055

- [Terminé] Les cinq candidates qualité ont été rejetées ; v002 reste la référence active et aucun
  checkpoint n'a été promu.
- [Conservé] L'analyse exp-00054 justifie un holdout indépendant stratifié par partie, phase et
  type d'action, avec ECE/Brier, intervalles d'incertitude et budget de décisions modifiées.
- [À privilégier] Mesurer la dérive par phase/type/carte, puis tester au plus une correction bornée
  sur les slices PLAY/action-card v007 ou `recruit_mercenary`, avec validation complète contre v002,
  Random, v007 et v008.
- [À étudier] Relancer la famille du résidu sémantique avec un entraînement plus long et des
  évaluations intermédiaires, en donnant la priorité à `exp-00050` et en traitant `exp-00051` comme
  variante conditionnelle, seulement après mesure de la dérive.
- [À étudier] Pousser `exp-00055` : son interaction phase×action a obtenu v007 `+2` points, v008
  `+3` et v002 `+1`, mais Random `-2` et une moyenne de `-0,2` point ; le signal doit être reproduit
  et attribué avant toute nouvelle variation.
- [À supprimer] Toute promotion fondée sur v008, v001, l'accuracy offline, le nombre d'actions ou
  le runtime seul ; les résidus sémantiques, biais globaux et interaction phase×action testés sont
  déjà invalidés dans leurs protocoles respectifs.

### Relance contrôlée des résidus sémantiques (`exp-00050` / `exp-00051`)

Hypothèse : le résidu sémantique d'exp-00050 possède un signal card-sensitive réel mais n'a pas
encore appris une politique stable après une seule époque ; des évaluations intermédiaires peuvent
identifier une époque où les slices ciblées progressent sans dépasser le budget de dérive de v002.
La variante bornée d'exp-00051 est une comparaison secondaire, car elle n'a conservé aucun gain
contre v008 et a régressé Random de `10` points et v002 de `7` points.

Protocole requis :

- repartir de `configs/neural_profiles/v002.pt` pour chaque variante, sans reprendre le checkpoint
  final d'une tentative précédente ; conserver le dataset et le split `game_id` comparables ;
- entraîner exp-00050 sur plusieurs paliers courts, par exemple `1`, `2` et `4` époques, en
  sauvegardant une candidate temporaire à chaque palier ; tester exp-00051 seulement avec la même
  grille si exp-00050 montre un palier prometteur ou si l'effet de la borne doit être isolé ;
- mesurer à chaque palier la validation complète contre v002, Random, v007 et v008, ainsi que les
  décisions modifiées par phase, type d'action et carte sur un holdout indépendant ;
- mesurer séparément le runtime, le nombre d'actions et le coût d'inférence ; aucun de ces signaux
  ne peut compenser une régression de qualité ;
- conserver un budget explicite de décisions dont l'argmax peut diverger de v002, avec attribution
  des changements aux slices PLAY/action-card et `recruit_mercenary` avant toute promotion.

Critères de poursuite : un palier doit améliorer une slice ciblée ou le panel qualité sans régresser
Random, v002, v007 ou la garde v008, et sans dérive non attribuée. Critères d'arrêt : toute nouvelle
régression sur Random ou v002 sans gain ciblé reproductible, absence de progrès entre deux paliers,
ou runtime significativement dégradé. La relance ne doit produire qu'une candidate temporaire ; le
checkpoint mutable canonique reste `artifacts/neural_training/checkpoint.pt` et aucune version n'est
promue avant la gate complète.

### Suite prioritaire de l'interaction phase×action (`exp-00055`)

Constat : `exp-00055` est la candidate la plus proche d'une amélioration parmi les variations
phase/action testées, mais sa validation de `100` parties par adversaire reste insuffisante pour
conclure. La candidate a été entraînée depuis v002 avec `phase_action_interaction_v1`, Adam réinitialisé,
un taux d'apprentissage de `1e-4`, et seulement `2 000` enregistrements sur un dataset de `20 115`.
Le résultat peut donc refléter un compromis prometteur mais aussi une variance de panel ou une dérive
partielle de la politique v002.

Hypothèse : l'interaction phase×action améliore certaines décisions PLAY et les matchups v007/v008,
mais la mise à jour de base ou quelques couples phase/action introduisent la régression Random. Une
reproduction contrôlée et une attribution des décisions doivent permettre de conserver le signal
utile sans perdre la non-régression Random/v002.

Ordre des expériences :

- [Reproduction] Rejouer exactement la recette `exp-00055` avec le même dataset, split `game_id`,
  seed et checkpoint v002 ; vérifier la provenance et le contenu réel de la candidate avant de
  comparer les scores.
- [Robustesse] Répéter ensuite avec plusieurs seeds et un panel plus large ou plusieurs panels
  indépendants. Conserver les résultats par adversaire et leurs intervalles d'incertitude ; ne pas
  agréger des protocoles différents ni transformer `+1` ou `-2` points en effet établi.
- [Sous-entraînement] Produire des candidates intermédiaires après environ `2 000`, `5 000`,
  `10 000` puis la totalité des enregistrements, avec le même panel de validation à chaque palier.
  Arrêter la trajectoire si Random ou v002 se dégrade sans gain ciblé reproductible.
- [Attribution] Comparer chaque candidate à v002 sur un holdout indépendant : décisions modifiées
  par phase, type d'action et carte, avec attention aux slices PLAY et aux couples responsables de
  la hausse v007/v008. Mesurer aussi le taux de conservation de l'argmax v002.
- [Isolation] Tester une variante qui gèle tous les poids de v002 et n'entraîne que le biais
  phase×action. Cette variante doit déterminer si le signal vient de l'interaction ajoutée plutôt
  que d'une dérive générale du scorer.
- [Ciblage] Si la régression Random est localisée, limiter le biais aux couples attribués et protéger
  explicitement les autres états par conservation de l'action ou des logits v002. Ne pas élargir la
  correction à toutes les phases sans preuve issue de l'analyse.

### Expérience exp-00056 — rejetée

- [Terminé] Tester une correction d'exp-00055 avec tous les poids du scorer contextuel v002 gelés et
  seulement un résidu scalaire phase×type d'action de 72 paramètres, initialisé à zéro, entraîné sur
  20 000 décisions du dataset v008 contre Random/v007.
- [Résultat] Sur 100 parties par adversaire, Random `-1,0` point, v007 `-2,0`, v008 `+1,0`, v002
  neural `0,0` et v001 `-1,0`. La moyenne des cinq adversaires est `-0,6` point : rejet.
- [Résultat] Le benchmark médian v002-v002 de 50 parties passe de `10,1959 s` à `10,6544 s`, avec
  `18 216` contre `18 181` actions et `4,0100 s` contre `4,3518 s` d'inférence.
- [Supprimé] Ne pas promouvoir ni reprendre le résidu phase×action seul : le gain v008 ne compense
  pas les régressions Random/v007.

## Suites issues d'exp-00056

- [À privilégier] Attribuer les états où le résidu change effectivement l'argmax v002, par phase,
  action et carte, avant toute nouvelle correction ; vérifier si les régressions Random/v007 viennent
  de quelques couples ou d'un effet diffus.
- [À étudier] Si une slice causale est confirmée, tester une correction bornée qui conserve l'argmax
  v002 hors de cette slice et mesurer un budget de décisions modifiées sur un holdout par `game_id`.
- [À conserver] Le gel de la base reste un contrôle utile pour les futurs résidus, mais n'est pas une
  amélioration de force dans ce protocole.

## Expérience exp-00057 — rejetée

- [Terminé] Collecter 41 694 décisions v002 contre v007, v008 et v002, puis conserver les 25 394
  décisions PLAY appartenant à une phase dont la trajectoire v002 diverge stratégiquement de v008.
- [Terminé] Entraîner une époque depuis `configs/neural_profiles/v002.pt`, Adam réinitialisé,
  taux `1e-5`, split `game_id` seed `57057`, sans modifier le moteur, les heuristiques ou le masque.
- [Résultat] Sur 100 parties par adversaire, Random `-3` points et v002 neural `-4` points ; v007
  gagne `+8` et v008 `+5`. Le gain ciblé ne protège donc pas la force générale.
- [Résultat] Le benchmark médian v002-v002 passe de `17,9658 s` à `18,3011 s`, de `17 334` à
  `17 702` actions et de `7,1068 s` à `7,2240 s` d'inférence.
- [Supprimé] Ne pas reprendre ce filtre PLAY seul ni accepter la moyenne positive du sous-script :
  la non-régression Random/v002 reste obligatoire.

## Nouvelles idées après exp-00057

- [À privilégier] Mesurer les décisions modifiées par action et carte sur le holdout exp-00057,
  puis entraîner seulement une correction sur les couples dont le gain v007/v008 est confirmé sans
  perte Random/v002 ; fixer avant entraînement un budget d'argmax modifiés.
- [À étudier] Refaire un cycle DAgGER avec un mélange explicite des labels v002 et v008 sur les
  états divergents, afin de tester si la perte Random/v002 vient d'un déplacement trop complet vers
  le professeur heuristique.
- [À conserver] Le protocole de collecte stratégiquement divergent est informatif, mais tout
  nouveau candidat doit être comparé à v002, Random, v007 et v008 avec le même panel complet.

## Expérience exp-00058 — rejetée

- [Terminé] Collecter les états visités par v002 contre v007, v008 et v002, puis mélanger
  déterministement les labels v008 et v002 sur les phases en divergence stratégique. Cette correction
  teste directement l'hypothèse que exp-00057 a trop déplacé la politique vers le professeur v008.
- [Résultat] Sur 20 parties par adversaire, Random et v002 sont à delta `0,00`, v007 à `0,00` et
  v008 à `+0,05`; la moyenne des cinq références est `-0,01` à cause de v001 `-0,10`. Le screening
  ne montre donc pas de gain moyen suffisant et ne justifie pas une promotion.
- [Résultat] Le benchmark v002-v002 sur 50 parties passe de `22,3853 s` à `24,7165 s`, avec
  `17 661` contre `17 663` actions et `8,8046 s` contre `9,7489 s` d'inférence.
- [Supprimé] Ne pas promouvoir ce mélange ni reprendre DAgGER divergent avec la même proportion
  sans un holdout plus large et une attribution explicite des décisions modifiées.

## Suites issues d'exp-00058

- [À privilégier] Conserver le mélange v002/v008 comme contrôle, mais apprendre séparément les poids
  de mélange par phase/action sur un holdout par `game_id`, avec budget d'argmax modifiés fixé avant
  entraînement.
- [À étudier] Tester une petite fraction v008 uniquement sur les couples PLAY/cartes dont le gain
  v007 est confirmé, en protégeant Random et v002 par conservation de l'action v002 hors slice.
- [À supprimer] Toute conclusion fondée sur v008 seul, sur le screening de 20 parties seul, ou sur
  le runtime ; l'échec d'exp-00058 impose une validation complète avant toute nouvelle candidate.

## Expérience exp-00060 — rejetée

- [Terminé] Corriger exp-00059 avec une pondération `play_card` réduite à `1,10` et un budget de
  `1 000` décisions, depuis v002, afin de tester si son gain contre v002 pouvait être conservé avec
  moins de dérive et un coût comparable.
- [Résultat] La validation batchée complète de 100 parties par adversaire donne Random `+2` points,
  v002 `+4`, mais v007 `-8`, v008 `-8` et v001 `-3`; la moyenne des cinq références est `-2,6`
  points. La candidate est rejetée malgré les gains Random/v002.
- [Résultat] Le benchmark v002-v002 médian comparable passe de `14,5231 s` à `15,0631 s`, de
  `17 444` à `18 021` actions et de `5,7213 s` à `6,0329 s` d'inférence.
- [Supprimé] Ne pas poursuivre une pondération globale `play_card`, même réduite, sans attribution
  par carte et protection explicite des décisions responsables des régressions v007/v008.

## Suites issues d'exp-00060

- [À privilégier] Construire le holdout d'attribution par carte/action demandé après exp-00049 et
  mesurer quelles décisions PLAY changent réellement entre v002 et chaque candidate; fixer ensuite
  un budget d'argmax modifiés avant tout apprentissage.
- [À étudier] Tester uniquement une correction sur une carte/slice dont le gain v007 est confirmé,
  avec conservation exacte de l'action v002 hors slice et validation complète; ne plus utiliser une
  pondération de type d'action comme approximation.
- [À supprimer] Toute nouvelle variation `play_card` globale ou sélectionnée sur le duel v002,
  Random ou v008 seul. v002 reste la référence active.

## Expérience exp-00061 — analyse terminée

- [Terminé] Mesurer v002 sur 80 parties indépendantes par référence (v001, v007, v008, Random),
  avec 13 816 décisions visibles, loss/accuracy/Brier/ECE par phase, action et carte, couverture,
  confiance, logits et états représentatifs.
- [Résultat] L'accord est 83,48 % avec v001, 66,23 % avec v007 et 79,74 % avec v008. v007 est
  faible surtout en PLAY (60,27 %), `play_card` (50,95 %) et `activate_champion` (29,96 %) ; les
  erreurs carte v007 les plus nettes sont `blaster`, `infinity_shard`, `legionnaire_korvus`,
  `chevalier_le_shai` et `li_hin_la_brisee`.
- [Résultat] Les désaccords sont parfois très confiants (jusqu'à 0,987 dans les exemples v007).
  Le dataset reste PLAY-heavy (71,75 %) et les slices rares sont insuffisantes ; les logits bruts
  sont saturés et ne doivent pas être comparés entre versions.
- [Conservé] Utiliser ce diagnostic pour fixer un budget de décisions modifiées et une correction
  bornée par carte ; ne pas sélectionner une candidate sur l'accuracy offline ou v008 seul.

## Nouvelles idées après exp-00061

- [À privilégier] Construire un holdout par `game_id` plus large, équilibré par phase/action, puis
  estimer des intervalles pour les slices PLAY/cartes v007 avant entraînement.
- [À étudier] Tester une protection stricte de l'argmax v002 hors des cartes attribuées, avec un
  budget de changements fixé avant l'apprentissage et arrêt précoce sur calibration/holdout.
- [À étudier] Collecter délibérément davantage de décisions `recruit_mercenary`,
  `choose_pending_decision` et de cartes rares ; ne pas leur appliquer de pondération tant que la
  couverture ne dépasse pas un seuil explicite.

## Expérience exp-00059 — rejetée

- [Terminé] Tester une correction objective ciblée : conserver l'architecture et les poids v002,
  réinitialiser Adam, entraîner 2 000 décisions depuis v002 à `1e-5`, et multiplier par `1,25` la
  loss des actions `play_card`. Cette hypothèse est distincte des biais appris phase×action et du
  mélange DAgGER : elle réduit la portée de la mise à jour au signal PLAY déjà identifié par
  exp-00049/00054.
- [Résultat] Sur 100 parties par référence, Random est à `0,00`, v007 à `-0,05`, v008 à `-0,01`,
  v002 à `+0,04` ; la moyenne des cinq références est `+0,006`, mais la garde v008 et v007
  régressent, donc la candidate est rejetée.
- [Résultat] Le benchmark v002-v002 médian passe de `9,6928 s` à `14,0928 s` sur 50 parties,
  avec `17 517` contre `16 866` actions et `3,8002 s` contre `5,5399 s` d'inférence médiane.
- [Supprimé] Ne pas promouvoir la pondération `play_card` ni reprendre une simple pondération de
  loss ciblée sans attribution des décisions modifiées et contrainte explicite de non-régression.

## Suites issues d'exp-00059

- [À privilégier] Revenir à un diagnostic de dérive par carte dans `play_card`, puis tester au plus
  une correction sélective sur les cartes responsables, avec conservation de l'action v002 hors
  slice et budget d'argmax fixé avant l'entraînement.
- [À étudier] Comparer une mise à jour par petits paliers avec arrêt précoce sur holdout, sans
  sélectionner sur le duel v002 seul ; mesurer également le coût d'inférence à chaque palier.
- [À supprimer] Toute pondération globale ou par type d'action qui ne protège pas simultanément
  Random, v007, v008 et v002 sur le panel complet.

## Expérience exp-00062 — rejetée

- [Terminé] Tester une architecture `play_card_slice_bias_v1` depuis v002 : scorer contextuel
  entièrement gelé, cinq biais scalaires initialisés à zéro pour `PLAY × {blaster, infinity_shard,
  legionnaire_korvus, chevalier_le_shai, li_hin_la_brisee}`, entraînés sur 3 000 décisions d'un
  dataset v008/v007 de 30 062 décisions. Cette correction est nouvelle par sa surface de mise à
  jour explicitement bornée et son absence de dérive hors slice.
- [Résultat] La validation batchée complète de 100 parties par adversaire donne delta `0,00` contre
  Random, v007, v008 et v002 ; la candidate est rejetée car elle ne produit aucun gain de force.
  Les biais appris restent faibles (`0,0051`, `0,0014`, `0,0043`, `0,0027`, `-0,0015`) et n'ont
  changé aucune décision du panel de validation.
- [Résultat] Le benchmark comparable v002-v002 sur 50 parties médianes passe de `13,6957 s` à
  `15,0277 s`, avec `17 218` actions identiques et une inférence médiane de `5,365 s` à `6,092 s`.
- [Supprimé] Ne pas promouvoir cette correction ni élargir les cinq cartes sans un holdout mieux
  couvert et une preuve qu'une décision changeable est effectivement attribuée à la slice.

## Nouvelles idées après exp-00062

- [À privilégier] Construire le holdout équilibré par `game_id` demandé par exp-00061, avec comptage
  préalable des décisions `PLAY × carte` et intervalles par carte avant tout nouvel entraînement.
- [À étudier] Si une carte dispose d'une couverture suffisante, tester un seul biais borné avec un
  budget d'argmax modifiés non nul fixé à l'avance ; arrêter si l'entraînement reste sous le seuil
  de changement ou si la candidate ne bat pas v002 sans régresser Random/v007/v008.
- [À conserver] La protection stricte hors slice est un contrôle causal utile, mais son surcoût
  d'inférence doit être supprimé ou justifié avant toute future candidate.

## Expérience exp-00063 — rejetée

- [Terminé] Tester un cycle DAgGER on-policy distinct du mélange stratégique exp-00058 : collecter
  les trajectoires v002 contre v008, v007, v001 et v002, puis prioriser `play_card`,
  `recruit_mercenary`, `assign_power` et `choose_pending_decision` sans remplacer la base par une
  pondération globale PLAY.
- [Résultat] La validation batchée de 100 parties par adversaire donne Random `-5` points, v007
  `-1`, v008 `+5` et le duel v002 `+2`. La garde Random échoue ; aucune promotion n'est justifiée.
- [Résultat] Le benchmark comparable passe de `21,6276 s` à `22,4119 s`, de `17 073` à `17 592`
  actions, soit environ `+3,6 %` de temps et `+3,0 %` d'actions.
- [Supprimé] Ne pas reprendre cette priorisation seule : le gain v008/v002 neural ne compense pas
  la régression Random et la candidate n'est pas une amélioration robuste.

## Nouvelles idées après exp-00063

- [À privilégier] Avant tout nouveau DAgGER, mesurer la conservation de l'argmax v002 et les
  transitions réellement modifiées par phase/action sur un holdout séparé ; la priorisation doit
  être conditionnée à une slice causale, pas seulement à une couverture faible.
- [À étudier] Tester un mélange à faible fraction de labels teacher uniquement sur les décisions
  `recruit_mercenary` ou `choose_pending_decision` si leur couverture et leur effet causal sont
  suffisants ; protéger explicitement les décisions hors slice.
- [À supprimer] Toute nouvelle collecte DAgGER uniforme ou pondération des quatre catégories sans
  budget d'argmax modifiés et sans contrôle Random/v002 au screening.

## Expérience exp-00064 — rejetée

- [Terminé] Tester une mise à jour d'objectif conservatrice depuis `v002`, avec dataset mixte v007/v008,
  split par `game_id`, Adam réinitialisé, un seul passage effectif sur 10 000 décisions et taux
  d'apprentissage `1e-5`. Cette correction teste une dérive plus petite que les pondérations et le
  DAgGER d'exp-00063, sans changer l'architecture, le moteur, les heuristiques ou le masque.
- [Résultat] Sur 100 parties par référence, Random `-4` points, v007 `+8`, v008 `-3` et v002 neural
  `-14`. Le gain v007 est donc incompatible avec la non-régression obligatoire ; aucune promotion.
- [Résultat] Le benchmark comparable v002-v002 sur 50 parties passe de `10,6280 s` à `11,1570 s`,
  de `16 588` à `17 335` actions et de `4,2240 s` à `4,4051 s` d'inférence.
- [Supprimé] Ne pas reprendre l'imitation conservatrice seule ; elle confirme qu'un faible taux
  d'apprentissage ne suffit pas à protéger la référence active ni Random.

## Nouvelles idées après exp-00064

- [À privilégier] Mesurer explicitement les changements d'argmax v002 sur un holdout indépendant,
  puis tester une mise à jour uniquement sur une slice dont l'effet causal est confirmé, avec une
  contrainte de conservation de l'action v002 hors slice.
- [À étudier] Ajouter une loss de conservation de politique v002 dans l'entraînement, plutôt que
  seulement réduire le taux d'apprentissage ; rejeter si Random ou v002 régresse au screening.
- [À supprimer] Les recettes qui améliorent v007/v008 seul ou l'accuracy offline sans gain moyen
  sur le panel complet.

Critères de succès : gain reproductible sur une ou plusieurs slices ciblées, absence de régression
robuste contre Random, v002, v007 et v008, et dérive d'argmax v002 dans le budget fixé. Critères de
rejet : échec de reproduction, régression Random persistante, gain limité à v007/v008, ou amélioration
qui disparaît avec un panel indépendant. Le runtime, le débit et le nombre d'actions restent des
métriques secondaires et ne peuvent pas compenser un échec de qualité.
