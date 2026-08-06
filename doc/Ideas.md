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

## Pistes écartées

- Reprendre à l'identique une distillation locale ou globale, une loss ranking-only, une simple
  surpondération de `recruit_mercenary`, un résidu sémantique ou un biais global phase/action.
- Promouvoir sur la base d'un gain contre v001 ou v008 seul.
- Utiliser une accuracy holdout sans validation en partie.
- Reprendre un PPO court ou interrompu sans profilage de collecte et budget complet.
- Modifier le moteur, les heuristiques ou le masque d'information pour améliorer la candidate.

Toute nouvelle idée doit préciser l'axe choisi (architecture/représentation ou training), une
hypothèse falsifiable, la référence v002, les métriques attendues et la condition de rejet.


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

Critères de succès : gain reproductible sur une ou plusieurs slices ciblées, absence de régression
robuste contre Random, v002, v007 et v008, et dérive d'argmax v002 dans le budget fixé. Critères de
rejet : échec de reproduction, régression Random persistante, gain limité à v007/v008, ou amélioration
qui disparaît avec un panel indépendant. Le runtime, le débit et le nombre d'actions restent des
métriques secondaires et ne peuvent pas compenser un échec de qualité.
