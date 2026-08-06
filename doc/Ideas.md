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

Conclusion : le catalogue ne doit plus privilégier une nouvelle variation locale de la même recette.
La prochaine campagne doit tester une hypothèse de rupture, tout en conservant un protocole de
comparaison contre v002.

## Priorité immédiate : changer d'espace de solution

### 1. À privilégier — architecture ou représentation

Tester une seule modification structurelle susceptible de traiter les erreurs par phase, action ou
carte que les pertes locales n'ont pas corrigées. Les options candidates sont, par exemple :

- une tête ou un encodeur conditionné par phase/type d'action ;
- une représentation des cartes combinant identité et information sémantique ;
- une dimension d'embedding ou de contexte différente, mesurée explicitement ;
- une représentation/pooling des actions légales qui conserve leur contexte sans fuite d'information.

Hypothèse falsifiable : une modification structurelle bornée améliore les slices holdout ciblées
(`play_card`, `banish_card`, `recruit_mercenary`) sans dégrader la moyenne contre v002, Random et
v007, avec un coût d'inférence mesuré.

Protocole minimal : conserver v002, le même split par partie, un contrôle sans changement, une seule
variable d'architecture, puis mesurer qualité globale, slices ciblées, décisions changées, nombre
d'actions, mémoire et temps d'inférence. Ne pas commencer par plusieurs dimensions ou plusieurs
têtes à la fois.

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

### 3. À conserver comme diagnostic préparatoire

- Étendre le holdout de `exp-00049` par phase et type d'action ; mesurer ECE, Brier et reliability bins.
- Séparer les erreurs de type d'action des erreurs de carte pour `crystal`, `moine_du_portail`,
  `ojas`, `recruit_mercenary` et `banish_card`.
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

## Pistes écartées

- Reprendre à l'identique une distillation locale ou globale, une loss ranking-only ou une simple
  surpondération de `recruit_mercenary`.
- Promouvoir sur la base d'un gain contre v001 ou v008 seul.
- Utiliser une accuracy holdout sans validation en partie.
- Reprendre un PPO court ou interrompu sans profilage de collecte et budget complet.
- Modifier le moteur, les heuristiques ou le masque d'information pour améliorer la candidate.

Toute nouvelle idée doit préciser l'axe choisi (architecture/représentation ou training), une
hypothèse falsifiable, la référence v002, les métriques attendues et la condition de rejet.

## Expérience exp-00050 — rejetée

- [Terminé] Tester une variation d’architecture `semantic_identity_v3` depuis v002 : conserver le
  contexte global des actions de v002 et ajouter un résidu sémantique linéaire sur les embeddings de
  cartes, initialisé à l’identité. Le dataset a été régénéré avec v008 et v007, séparé par `game_id`.
- [Résultat] La validation officielle de 100 parties par adversaire donne Random `-3,0` points,
  v007 `-3,0`, v008 `+5,0` et v002 neural `-12,0`. Le benchmark comparable médian passe de `14,2108 s`
  à `14,3520 s` (`-0,99 %` de débit) et de `17 073` à `16 667` actions.
- [Supprimé] Ne pas promouvoir la candidate : le gain contre la garde v008 ne compense pas les
  régressions Random, v007 et surtout v002. Le résidu sémantique est une variation effective, mais
  son transfert partiel ne suffit pas à préserver la politique active.

## Suites issues d'exp-00050

- [À privilégier] Mesurer avant entraînement la dérive de politique induite par le résidu identité,
  puis tester une version gelant ce résidu pendant le premier passage ou limitant explicitement sa
  norme ; comparer d’abord directement à v002 avant la validation complète.
- [À étudier] Évaluer séparément l’effet de la représentation sur les erreurs `play_card` par carte
  et sur `recruit_mercenary`, avec un budget de décisions modifiées et un holdout par phase/action.
- [À supprimer] Toute promotion d’une représentation qui améliore seulement v008 ou la précision
  offline sans non-régression contre Random, v007 et v002.

## Expérience exp-00051 — rejetée

- [Terminé] Tester une correction bornée de `semantic_identity_v3` : conserver le scorer contextuel
  de v002, ajouter un résidu linéaire sémantique initialisé à zéro et limiter sa contribution à
  `0,1`, avec conversion explicite des poids v002 et Adam réinitialisé.
- [Résultat] Le panel officiel de 100 parties par adversaire donne Random `-10,0` points, v007
  `-1,0`, v008 `+0,0` et v002 neural `-7,0`. Le gain contre v001 (`+11,0`) ne constitue pas une
  preuve de force contre les références actives.
- [Résultat] Le benchmark comparable médian sur 50 parties contre v002 passe de `14,2984 s` et
  `17 073` actions pour v002 à `10,7607 s` et `17 013` actions pour la candidate ; cette mesure
  candidate est très variable entre répétitions et ne compense pas la régression de qualité.
- [Supprimé] Ne pas promouvoir le résidu borné, ni reprendre une variation de représentation sans
  corriger la dérive de politique observée contre Random et v002.

## Suites issues d'exp-00051

- [À privilégier] Avant tout nouvel entraînement, mesurer la dérive décisionnelle de la candidate
  par phase et type d’action sur un holdout indépendant ; isoler les décisions changées plutôt que
  modifier encore l’amplitude du résidu.
- [À étudier] Tester une régularisation explicite des logits ou une contrainte de conservation de
  l’action v002 sur les états hors slices ciblées, avec budget de décisions modifiées et validation
  complète contre Random, v007, v008 et v002.
- [À supprimer] Les résidus bornés ou gains de runtime acceptés malgré une forte baisse contre
  Random/v002 ; ne pas utiliser v001 seul comme critère de sélection.

## Expérience exp-00052 — rejetée

- [Terminé] Tester une imitation depuis v002 avec une régularisation KL de `0,5` vers la politique
  active v002 sur chaque état et chaque action légale. Cette correction concrète des dérives des
  résidus d'exp-00050/00051 conserve le même scorer `global_candidate_context`, utilise Adam réinitialisé,
  un taux d'apprentissage réduit à `1e-4` et un split par `game_id`.
- [Résultat] Sur 100 parties par adversaire, Random progresse de `+1,0` point, mais v007 régresse de
  `-1,0` et la garde v008 de `-3,0` ; le panel direct contre v002 donne `+17,0` points mais ne
  compense pas les références heuristiques actives. Le benchmark comparable passe de `9,2263 s`
  et `16 739` actions à `9,1367 s` et `16 544` actions sur 50 parties.
- [Supprimé] Ne pas promouvoir cette régularisation globale : elle réduit le coût mesuré mais ne
  démontre pas une force supérieure contre v007 et v008.

## Suites issues d'exp-00052

- [À privilégier] Mesurer les décisions effectivement changées par la pénalité KL, par phase et type
  d'action, avant de retenter une contrainte de conservation ; conditionner éventuellement la KL aux
  états hors slices ciblées tout en conservant un budget explicite de dérive.
- [À étudier] Tester une température ou une cible de distillation calibrée sur les logits v002, avec
  une validation holdout indépendante des états changés ; ne pas confondre la KL avec une garantie de
  conservation de l'argmax.
- [À supprimer] Toute nouvelle imitation globale avec seulement un taux d'apprentissage différent,
  ou toute promotion fondée sur le gain contre v002 neural seul.

## Expérience exp-00053 — rejetée

- [Terminé] Tester une architecture `global_context_action_type_bias_v1` depuis v002 : conserver
  le scorer contextuel et ajouter un biais scalaire par type d'action, initialisé à zéro. Le dataset
  a été régénéré avec v008 comme teacher, les matchups v008/v007, séparé par `game_id`.
- [Résultat] La validation officielle de 100 parties par adversaire donne, contre la référence
  v002, Random `-2,0` points, v007 `-8,0`, v008 `+1,0`, neural v002 `-6,0` et neural v001 `+9,0`.
  La moyenne des cinq deltas vaut `-1,2` point : la candidate est rejetée.
- [Résultat] Le benchmark comparable médian sur 50 parties contre v002 passe de `9,3700 s` et
  `16 938` actions pour v002 à `14,0246 s` et `16 780` actions pour la candidate ; le coût
  d'inférence augmente également. Aucun gain de runtime n'est revendiqué.
- [Supprimé] Ne pas promouvoir ni reprendre le biais global par type d'action : il corrige
  potentiellement v008, mais régresse les références Random, v007 et v002.

## Suites issues d'exp-00053

- [À privilégier] Mesurer les décisions changées par type d'action et par phase, puis tester au
  plus une correction bornée sur les catégories réellement responsables de la régression, avec un
  budget explicite de dérive et comparaison directe à v002 avant toute validation complète.
- [À étudier] Réutiliser l'architecture sans biais comme contrôle et comparer un résidu conditionné
  par phase uniquement si l'analyse montre que le biais global mélange des phases incompatibles.
- [À supprimer] Toute nouvelle variation de biais global, toute sélection sur v008 seul, et toute
  conclusion fondée sur le gain contre v001 ou sur le nombre d'actions seul.

## Expérience exp-00054 — analyse terminée

- [Terminé] Diagnostiquer v002 sur `15 711` états visibles visités contre v007, v008 et Random,
  avec couverture phase/action/carte, loss, accuracy, confiance, désaccord v001/v002 et états
  représentatifs masqués. Aucun checkpoint n'a été modifié.
- [Résultat] L'accord v002 est de `75,35 %` avec v007 et `87,86 %` avec v008. Les erreurs v007
  sont concentrées en PLAY (`72,4 %`) et souvent confiantes (`11,97 %` globalement ; `blaster`
  `88,5 %`), tandis que v008 laisse surtout faibles `buy_card`, `recruit_mercenary` et
  `gain_mastery`.
- [Résultat] Le dataset est PLAY-heavy (`71,8 %`) et les cartes rares restent sous-couvertes ;
  les scores ne justifient pas une nouvelle pondération immédiate.
- [Conservé] Le rapport durable est `doc/Experiments/exp-00054.md` et le résultat machine est
  `result.json`.

## Suites issues d'exp-00054

- [À privilégier] Construire un holdout indépendant par partie, stratifié phase/action, avec ECE,
  Brier, reliability bins et intervalles d'incertitude avant toute nouvelle mise à jour.
- [À étudier] Tester séparément une correction bornée des slices PLAY/action/carte v007 et de
  `recruit_mercenary`, avec un budget explicite de décisions modifiées par rapport à v002.
- [À étudier] Augmenter la couverture des cartes rares avant de leur appliquer une pondération.
- [À supprimer] Toute sélection fondée sur l'accuracy offline, v008 seul ou les logits bruts
  inter-version ; ne pas reprendre une loss globale sans attribution de dérive.
