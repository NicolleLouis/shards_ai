# Idées et priorités de travail

Ce fichier contient le catalogue courant des décisions et des prochaines expériences. Les détails,
les métriques et les commandes restent dans `doc/Experiments/`.

## Règles de décision

- Comparer chaque candidate à la référence neural active, à Random, à v007 et à la garde v008.
- Utiliser un panel complet et reproductible ; un screening court ne peut jamais accepter une candidate.
- Calculer la moyenne qualité avec les poids Random `0,5`, v007 `1`, v008 `2` et groupe neural `0,5` ; Random n'est plus une garde dure, tandis que v008 reste une non-régression obligatoire.
- Séparer entraînement, validation et holdout par `game_id`, et conserver la provenance des seeds.
- Valider sur plusieurs seeds tirés avant l'évaluation et enregistrés dans le rapport ; chaque seed doit être rejouée avec exactement les mêmes parties pour la candidate et la référence.
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

## Nouvelle architecture sémantique — prérequis d'expérimentation

L'architecture `structured_semantic_v4` transmet désormais au réseau la structure détaillée des
cartes : types d'opérations, montants, cibles, seuils, contraintes, ordre des opérations, branches
de maîtrise, capacités de champion et identifiants de passifs. Cette architecture est différente
de celle utilisée pour les checkpoints historiques ; elle ne peut donc pas être évaluée avec les
poids existants.

Toute expérience qui veut mesurer son efficacité doit commencer par un **réentraînement par
imitation** sur un dataset et un protocole comparables à ceux de la référence. Le checkpoint de
travail reste `artifacts/neural_training/checkpoint.pt`, et le résultat doit rester candidat tant
qu'il n'a pas passé la gate complète.

Les changements précédemment rejetés ne sont pas définitivement invalidés pour cette architecture :
leurs résultats concernaient l'ancien espace de représentation et l'ancien encodeur. Ils peuvent
être réessayés via le système d'expériences, un par un, avec le même panel d'adversaires, le même
holdout et les mêmes critères de qualité, afin de distinguer un effet de l'architecture d'un effet
de l'hypothèse expérimentale.

La dernière version neural stable reste **V2 (`v002`)**. Aucun changement n'est apporté à son
checkpoint, à son profil actif ou à son statut de référence. V4 est uniquement une nouvelle
candidate expérimentale jusqu'à validation.

### Qualification des expériences antérieures

Les statuts historiques du tableau ci-dessous restent inchangés : une expérience rejetée l'a été
avec les poids, le dataset et l'architecture qui étaient alors testés. La qualification suivante
précise seulement si sa conclusion peut être transposée à V4.

| Expériences | Qualification pour `structured_semantic_v4` |
|---|---|
| `exp-00023` à `exp-00026` | Résultats historiques ; les variantes d'entraînement peuvent être réessayées après imitation V4. |
| `exp-00027` à `exp-00038` | Non transposables directement lorsqu'elles modifient la représentation, la loss ou l'ancrage ; à revalider une par une. |
| `exp-00039` à `exp-00049` | Diagnostics et protocoles réutilisables ; les corrections candidates doivent être réentraînées sur V4. |
| `exp-00050` à `exp-00056` | Résultats liés à l'ancien espace de représentation ou à l'ancien scorer ; à revalider sur V4. |
| `exp-00057` à `exp-00067` | Les échecs techniques et les garde-fous restent valides ; toute conclusion de qualité doit être rejouée sur V4. |
| `exp-00068` | Toujours valide comme optimisation indépendante : le cache ne change pas les sorties ni les trajectoires. |
| `exp-00069` à `exp-00074` | Résultats historiques de training/PPO ; à revalider sur V4 après un réentraînement comparable. |

Une expérience ne doit donc pas être renommée globalement `outdated`. Pour les futures lignes, le
statut doit préciser l'un des qualificatifs suivants : `valide indépendamment de l'architecture`,
`résultat historique — à revalider sur V4`, ou `non transposable à V4`.

## Priorité immédiate : changer d'espace de solution

### 1. À privilégier — diagnostic de dérive puis correction ciblée

Construire d'abord un holdout indépendant par partie, stratifié par phase, type d'action et taille
de l'ensemble légal, avec ECE, Brier, reliability bins et un budget explicite de décisions modifiées
par rapport à v002. Après attribution, tester au plus une correction bornée sur les slices
responsables, notamment PLAY/action-card v007, `gain_mastery` et `recruit_mercenary`.

Hypothèse falsifiable : une correction bornée sur des états attribués améliore la slice ciblée sans
dépasser le budget de dérive et sans régresser v002, Random, v007 ou v008 sur le panel complet.

### 2. À privilégier — changer le type d'entraînement

Tester une méthode qui ne repose pas uniquement sur l'imitation de labels heuristiques v008, par
exemple un PPO suffisamment long et profilé ou un entraînement hybride imitation + objectif de
résultat. Le choix exact doit être décrit dans une nouvelle architecture ou une hypothèse dédiée
avant implémentation.

Garde-fous : profiler d'abord la collecte, séparer les budgets collecte/entraînement/validation,
conserver un holdout non utilisé pour choisir les mises à jour, instrumenter avantages/gradients et
refuser toute conclusion fondée sur une amélioration contre v008 ou v002 neural seul.

### 3. À conserver comme contrôles

- Le gel de la base v002 est un contrôle causal utile, mais `exp-00056` ne montre aucun gain de force.
- La cache d'embeddings de `exp-00068` est conservée comme optimisation de performance indépendante.
- Les protocoles de collecte PPO doivent conserver un checkpoint atomique externe après chaque rollout.

## Historique condensé

| Expérience | Statut | Conclusion opérationnelle |
|---|---|---|
| exp-00023 | Rejetée | Les reprises PPO courtes n'ont pas produit de gain exploitable. |
| exp-00024 à exp-00026 | Rejetées | DAgGER uniforme et mélanges historiques/on-policy : régressions. |
| exp-00027 à exp-00035 | Rejetées | Pondérations, filtrages, marges teacher et ancrages globaux : pas de généralisation robuste. |
| exp-00036 à exp-00038 | Rejetées | Dataset équilibré, fine-tuning réduit et premières divergences : compromis non résolu. |
| exp-00039 | Analyse terminée | Premiers signaux sur `recruit_mercenary`, BUY et PLAY confiants. |
| exp-00040 à exp-00041 | Rejetées | Accuracy holdout ou marge teacher seules insuffisantes. |
| exp-00044 | Analyse terminée | Diagnostic proche d'exp-00039 ; logits inter-version non comparables. |
| exp-00045 à exp-00048 | Rejetées | Corrections locales et ranking-only : régressions sur des références obligatoires. |
| exp-00049 | Analyse terminée | Holdout indépendant : faiblesses par action/carte et erreurs PLAY souvent confiantes. |
| exp-00050 à exp-00053 | Rejetées | Résidus sémantiques, KL et biais globaux : gains isolés sans généralisation. |
| exp-00054 | Analyse terminée | 15 711 décisions : erreurs v007 concentrées en PLAY, cartes rares sous-couvertes. |
| exp-00055 | Rejetée | Interaction phase×action : gains v007/v008/v002, mais Random régresse. |
| exp-00056 | Rejetée | Résidu phase×action gelé : Random -1, v007 -2, v008 +1, v002 0 point ; runtime +4,5 %. |
| exp-00057 | Échec technique | DAgGER PLAY divergent : arrêt avant validation, aucun profil/checkpoint candidat exploitable. |
| exp-00058 | Échec technique | Mélange v002/v008 : screening court sans gain moyen suffisant ; runtime +10,0 %. |
| exp-00059 | Rejetée | Pondération `play_card` : v002 +4, mais v007 -5, v008 -1 et runtime +45,4 %. |
| exp-00060 | Rejetée | Pondération PLAY réduite : Random +2 et v002 +4, mais v007/v008 -8 ; runtime +3,7 %. |
| exp-00061 | Analyse terminée | Holdout de 13 816 décisions : dérive v007 en PLAY/cartes, erreurs confiantes ; aucune candidate. |
| exp-00062 | Rejetée | Cinq biais PLAY par carte : aucune décision changée ni gain ; coût médian +9,7 %. |
| exp-00063 | Rejetée | DAgGER priorisé : v008 +5 et v002 +2, mais Random -5 et v007 -1 ; runtime +3,6 %. |
| exp-00064 | Rejetée | Imitation conservatrice : v007 +8, mais Random -4, v008 -3, v002 -14 ; runtime +5,0 %. |
| exp-00065 | Rejetée | Conservation de politique : Random +5/v008 +10, mais v007 -20/v002 -10 ; runtime +37,5 %. |
| exp-00066 | Analyse terminée | 14 387 observations : l'erreur augmente avec la compétition légale ; signal PLAY/`gain_mastery`. |
| exp-00067 | Échec technique | PPO sans checkpoint candidat avant arrêt ; aucune force apprise mesurable. |
| exp-00068 | Performance acceptée | Cache d'embeddings : temps médian -11,76 %, décisions et trajectoires identiques ; 307 tests passent. |
| exp-00069 | Échec technique | PPO instrumenté arrêté avant le premier checkpoint ; aucune mesure de qualité PPO. |
| exp-00070 | Rejetée | Petite imitation : Random -5 et v007 -25 malgré v008 +5/v002 +15 ; runtime -1,9 %. |
| exp-00071 | Rejetée | Checkpoints écrits, mais pertes quasi nulles et trajectoires inchangées ; runtime +1,44 %. |
| exp-00072 | Analyse terminée | 10 830 décisions : `gain_mastery`, `recruit_mercenary` et PLAY/cartes restent les slices critiques. |
| exp-00073 | Rejetée | PPO gamma=1 : Random +10, mais v007 -5 et v002 -10 ; campagne arrêtée après un palier. |
| exp-00074 | Rejetée | PPO sans KL : v002 neural +20, mais Random -20, v007 -5 et v008 -15 ; campagne partielle. |

## Pistes à tester

- [À privilégier] Holdout `game_id` équilibré par phase/action/taille d'ensemble légal, avec budget d'argmax v002 fixé avant tout entraînement et intervalles par slice.
- [À privilégier] Une correction bornée seulement sur une slice PLAY/carte ou `gain_mastery` causalement attribuée, avec conservation exacte de l'action v002 hors slice et validation complète.
- [À privilégier] Un diagnostic PPO d'un rollout : récompenses terminales, variance des avantages, normes de gradients, décisions modifiées et checkpoint/reprise atomiques.
- [À privilégier] Remplacer le seed unique par un panel de seeds pré-tiré et commun à la candidate et v002 ; agréger les résultats seulement après avoir conservé les deltas par seed afin de détecter les candidates dépendantes d'une seed heureuse ou malheureuse.
- [À étudier] Une continuation PPO avec KL non nul mais borné, uniquement après preuve d'un signal non dégénéré et sélection sur holdout indépendant.
- [À étudier] Augmenter la couverture de `recruit_mercenary`, `choose_pending_decision` et des cartes rares ; aucune conclusion causale sous 20 observations.

## Expérience exp-00075 — structured_semantic_v4 — rejetée

- [Terminé] Réentraîner depuis zéro l'architecture `structured_semantic_v4` par imitation sur le
  dataset normalisé, avec un budget borné de 1 000 décisions, puis comparer V4 à v002 sur les mêmes
  seeds et le panel Random/v007/v008/v001/v002.
- [Résultat] Le checkpoint est techniquement compatible avec les métadonnées V4, mais la candidate
  régresse contre Random `-15` points, v007 `-10`, v008 `-10` et v002 `-5` sur 20 parties par
  adversaire. La garde v008 et la moyenne pondérée échouent ; aucune promotion.
- [Résultat] Le runtime v002-v002 est `4,9054 s` contre `5,3725 s` pour V4 sur 20 parties ; les
  actions passent de `6 801` à `7 764`. La hausse de coût et de trajectoires ne compense pas la
  régression de qualité.
- [Supprimé] Ne pas considérer l'accuracy offline V4 (`72,4 %` top-1 sur le millier évalué) comme
  une preuve de force ; ne pas charger les poids V2 dans cette transition.

## Nouvelles idées après exp-00075

- [À privilégier] Réentraîner V4 sur un budget complet et stable avec split `game_id` explicite,
  validation holdout séparée et reprise par lots, avant d'attribuer l'échec à l'encodeur structuré.
- [À étudier] Tester une capacité V4 plus petite ou un curriculum phase/action, en gardant le même
  dataset et le même panel, afin de séparer sous-entraînement et mauvaise représentation.
- [À conserver] La contrainte d'initialisation depuis zéro, la vérification architecture profil/
  checkpoint et la validation batchée avec v002 comme référence.

## Expérience exp-00076 — structured_semantic_v4 — rejetée

- [Terminé] Réentraîner V4 depuis zéro par imitation sur le dataset V4 comparable disponible,
  avec split `game_id`, puis mesurer le runtime et le panel complet contre v002, Random, v007 et v008.
- [Résultat] La candidate régresse : Random `-10`, v007 `-10`, v008 `-5` et neural v002 `-25`
  points sur 20 parties par adversaire. La garde v008 et la moyenne pondérée échouent ; V2 reste active.
- [Résultat] Le benchmark comparable v002-v002 donne `4,3585 s` contre `4,2036 s` pour V4,
  avec `7 173` contre `7 006` actions et `1,6640 s` contre `1,5156 s` d'inférence. Le runtime
  légèrement inférieur de V4 ne compense pas la régression de force.
- [Limite] Le dataset a été interrompu par l'environnement à `64 541` décisions et l'entraînement
  borné à `1 000` décisions/une époque ; ce résultat est donc un diagnostic de sous-entraînement,
  pas une invalidation définitive de l'encodeur V4.

## Nouvelles idées après exp-00076

- [À privilégier] Relancer V4 sur un hôte permettant le budget complet, avec checkpoint/reprise par
  époque et validation holdout séparée avant toute conclusion sur la représentation.
- [À étudier] Mesurer le coût de validation offline sur le dataset complet et conserver un holdout
  par `game_id` explicitement matérialisé pour éviter les campagnes longues non reprenables.

## Expérience exp-00077 — analyse terminée

- [Terminé] Refaire le diagnostic sur 64 parties et 15 396 décisions visibles, avec v002, son parent
  v001 et les enseignants v007/v008 ; ajouter macro-accuracy phase×action, cardinalité légale,
  ECE/Brier et états représentatifs.
- [Résultat] Après rééquilibrage descriptif, v002 garde une loss meilleure (`1,1806` contre `1,3079`)
  mais une macro-accuracy plus faible (`68,96 %` contre `71,06 %`). `gain_mastery` reste à `48,70 %`
  avec ECE `0,531`, `recruit_mercenary` à `23,87 %`, et l'accuracy tombe à `60,93 %` sur sept
  actions légales.
- [Conclusion] La faiblesse n'est pas seulement un artefact PLAY-heavy ; elle persiste dans les
  slices équilibrées et est particulièrement calibrée sur `gain_mastery`/grands ensembles légaux.
  Aucun checkpoint, moteur, heuristique ou masque n'a été modifié.
- [Rapport] Les détails sont dans `doc/Experiments/exp-00077.md`; le dataset et les états bruts restent
  temporaires hors dépôt.

## Nouvelles idées après exp-00077

- [À privilégier] Tester une correction bornée et conservatrice de `gain_mastery`/`recruit_mercenary`
  avec budget d'argmax fixé avant sélection, protection exacte de v002 hors slice et holdout
  `game_id` séparé.
- [À étudier] Calibrer séparément les décisions selon la taille de l'ensemble légal, sans modifier
  les logits bruts globalement ni pondérer les cartes rares sous un seuil de couverture.
- [À supprimer] Toute nouvelle analyse globale PLAY-heavy sans macro-métriques phase×action, ECE et
  cardinalité légale ; elle répéterait exp-00072 sans réduire l'incertitude.

## Expérience exp-00078 — structured_semantic_v4 — inconclusive

- [Terminé] Réentraîner V4 avec le profil `exp00078-v4`, le dataset normalisé et un split `game_id`,
  depuis zéro, puis comparer à v002 avec le même seed et un benchmark runtime comparable.
- [Résultat] L'entraînement complet (cinq époques) a été interrompu silencieusement avant d'écrire un
  checkpoint. Un smoke-training explicite de 1 000 décisions a produit un checkpoint V4 et un screening
  de 20 parties par adversaire : Random `+5`, v007 `+25`, v008 `+10` points contre v002.
- [Limite] Ces deltas ne sont pas une preuve de qualité : le budget d'entraînement est très inférieur au
  dataset, le screening est diagnostic, et le runtime augmente d'environ 48–50 % selon l'adversaire.
  Aucune promotion ni conclusion sur l'encodeur V4 n'est autorisée.
- [Conservé] La correction testée est distincte d'exp-00075/76 : profil V4 dédié, seed de split stable,
  initialisation nulle et reprise observable ; elle doit être relancée sur un hôte qui permet le budget
  complet avant toute nouvelle variation d'architecture.

## Nouvelles idées après exp-00078

- [À privilégier] Relancer exactement `exp00078-v4` avec cinq époques et checkpoint/reprise par époque,
  puis utiliser un holdout `game_id` matérialisé et une validation longue batchée.
- [À étudier] Si le budget complet reste impossible, réduire la capacité V4 ou le dataset de manière
  pré-définie et comparer plusieurs seeds ; ne pas interpréter un smoke-training comme une expérience
  de force.
- [À supprimer] Toute acceptation fondée sur le screening positif de 20 parties, l'accuracy offline ou
  le runtime ; le coût d'inférence V4 doit être profilé séparément avant optimisation.

## Expérience exp-00079 — structured_semantic_v4 — rejetée

- [Terminé] Reprendre la piste exp-00078 avec le dataset normalisé de 100 000 décisions, un split
  `game_id`, une initialisation from-scratch et un profil V4 dédié ; le budget complet a été tenté,
  puis un fallback borné à 1 000 décisions a été exécuté après interruption technique.
- [Résultat] Le screening batché de 20 parties donne Random `+15`, v007 `+10`, v008 `0`, mais
  neural v002 `-15` et v001 `-15` points. Ce signal court ne constitue pas une preuve de force et
  la candidate n'est pas promue.
- [Résultat] Le benchmark comparable V2/V4 contre v008 donne `2,2401 s` contre `2,1853 s`,
  `1,2908 s` contre `1,2301 s` d'inférence et `6498` contre `6814` actions ; l'écart de runtime
  ne compense pas la régression contre les références neurales.
- [Échec technique] Le protocole complet (une époque sur le split train) n'a pas atteint la fin de
  l'époque après environ huit minutes et n'a produit aucun checkpoint ; le résultat mesuré est
  donc un diagnostic borné, non une validation du budget complet.

## Nouvelles idées après exp-00079

- [À privilégier] Déporter l'entraînement V4 complet sur un hôte permettant une reprise atomique par
  époque, puis refaire la validation longue avant toute nouvelle variation d'architecture.
- [À étudier] Comparer plusieurs budgets de décisions prédéclarés (1k, 10k, dataset complet) avec
  le même seed et le même holdout afin de distinguer vitesse d'apprentissage et force finale.
- [À supprimer] Toute acceptation issue du seul screening 20 parties ou d'un gain contre Random/v007
  lorsque les références neurales régressent.

## Expérience exp-00080 — structured_semantic_v4 — rejetée

- [Terminé] Correction de exp-00079 : dataset normalisé canonique de 100 000 décisions, profil V4
  dédié, initialisation from-scratch et tentative d'un passage complet borné ; le checkpoint de
  diagnostic final porte les 1 000 premières décisions uniquement.
- [Résultat] Le panel batché de 20 parties par adversaire régresse contre Random `-10`, v007 `-40`,
  v008 `-25` et v002 `-20` points. La garde v008 et la moyenne pondérée échouent ; aucune promotion.
- [Résultat] L'accuracy top-1 offline du checkpoint diagnostic est `71,1 %`, sans valeur probante
  face aux résultats en partie. Le benchmark V4 contre v008 donne `2,0065 s` contre `2,1701 s` pour
  v002, avec `6 448` contre `6 602` actions et `1,1445 s` contre `1,3081 s` d'inférence.
- [Échec technique] Le passage complet de 100 000 décisions n'a produit aucun checkpoint après plus
  de 40 minutes et a été interrompu ; le résultat de force est donc celui du fallback diagnostic,
  explicitement non assimilé à un entraînement complet.

## Nouvelles idées après exp-00080

- [À privilégier] Instrumenter et optimiser le coût d'entraînement de `structured_semantic_v4` avant
  une nouvelle campagne de qualité ; préserver les mêmes données et le même protocole de validation.
- [À étudier] Comparer un budget pré-déclaré de 1 000 puis 10 000 décisions avec checkpoints après
  chaque lot, mais seulement si les références neurales sont incluses à chaque étape.
- [À supprimer] Toute nouvelle conclusion de force fondée sur l'accuracy offline ou un panel court
  positif contre les heuristiques ; V4 n'a encore aucune preuve de promotion.

## Expérience exp-00081 — structured_semantic_v4 — rejetée

- [Terminé] Correction bornée des fallbacks à 1 000 décisions : entraînement V4 from-scratch par
  imitation sur le dataset canonique normalisé, avec budget pré-déclaré de 10 000 décisions et
  profil/checkpoint explicitement marqués `structured_semantic_v4`.
- [Résultat] Le screening batché de 20 parties améliore v007 de `+15` points et v002 neural de
  `+10`, mais régresse Random de `-20` et v008 de `-10`; moyenne pondérée `-2,5` points. La garde
  v008 et la gate de qualité échouent, V2 reste active.
- [Résultat] Le holdout offline atteint `81,01 %` top-1 sur 10 000 décisions, mais ce signal ne
  compense pas les régressions en partie.
- [Résultat] Contre v008, runtime v002 `2,0566 s` contre V4 `2,2286 s` sur 20 parties, avec
  `6 357` contre `6 833` actions et `1,1631 s` contre `1,2195 s` d'inférence.
- [Supprimé] Ne pas prolonger cette recette seule : 10 000 décisions réduisent le sous-entraînement
  des fallbacks, mais ne corrigent pas la régression Random/v008.

## Nouvelles idées après exp-00081

- [À privilégier] Construire un holdout V4 indépendant par `game_id`, puis tester une capacité ou
  un curriculum pré-déclaré ciblant `gain_mastery`/`recruit_mercenary`, avec budget de dérive V2 et
  panel complet conservés.
- [À étudier] Réduire le coût de l'encodeur structuré par profilage ciblé avant tout budget V4 plus
  long; cette optimisation ne doit pas être confondue avec une preuve de qualité.
- [À conserver] Initialisation nulle V4, dataset hashé, checkpoint mutable unique et validation
  batchée par lots de 20.

## Pistes abandonnées

- [À supprimer] Pondération globale `play_card`, conservation globale des logits, mélange DAgGER divergent ou calibration globale sans attribution causale.
- [À supprimer] Campagnes PPO de quatre parties ou moins, campagnes interrompues avant un budget comparable et promotions fondées sur la conservation de trajectoire.
- [À supprimer] Toute promotion fondée sur v008, v001, l'accuracy offline, le nombre d'actions ou le runtime seul.

Critères de succès : gain reproductible sur une ou plusieurs slices ciblées, absence de régression
contre Random, v002, v007 et v008, et dérive d'argmax v002 dans le budget fixé. Le runtime, le débit
et le nombre d'actions restent des métriques secondaires et ne peuvent pas compenser un échec de qualité.
