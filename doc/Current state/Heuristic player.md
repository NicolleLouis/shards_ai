# État courant — Joueur heuristique

Cette page décrit le joueur heuristique réellement disponible dans le code. Les décisions de
conception complètes sont conservées dans
[l’architecture du joueur heuristique](../Architecture/009-joueur-heuristique-pondere.md).
La séparation des valeurs immédiates et prospectives est documentée dans
[l’architecture d’évaluation des cartes](../Architecture/017-evaluation-immediate-et-prospective-cartes.md).
Le départage des égalités de bannissement est documenté dans
[l’architecture du tie-break de bannissement](../Architecture/018-departage-bannissement-deterministe.md).
La valorisation effective des mercenaires et la mesure de taille des decks sont documentées dans
[l’architecture des effets effectifs et de la taille des decks](../Architecture/025-effet-mercenaire-effectif-et-taille-deck.md).
La protection des cartes qui se remplacent est documentée dans
[l’architecture de protection du bannissement](../Architecture/026-protection-bannissement-cartes-remplacees.md).

## Composants

| Composant | Fichier | Responsabilité |
|---|---|---|
| `HeuristicPlayer` | `shards_ai/ai/heuristic_player.py` | sélection déterministe de l’action légale au meilleur score |
| `ActionFeatures` | `shards_ai/ai/heuristic_evaluator.py` | signaux numériques d’une action |
| `HeuristicWeights` | `shards_ai/ai/heuristic_evaluator.py` | coefficients manuels et calcul du score |
| `CardConstraintWeights` | `shards_ai/ai/heuristic_evaluator.py` | poids séparés des contraintes d’effets |
| extraction | `shards_ai/ai/heuristic_features.py` | traduction de l’observation et des définitions de cartes en features |
| évaluation d’état | `shards_ai/ai/state_evaluator.py` | avantages différentiels de santé, maîtrise et nombre de champions |
| profils | `shards_ai/ai/heuristic_profiles.py` | chargement et sauvegarde de profils YAML versionnés |
| optimisation | `shards_ai/optimization/heuristic.py` | recherche hybride par mutations, racing et validation appariée |
| shaping | `shards_ai/analysis/reward_shaping.py` | récompense potentielle optionnelle après chaque transition |

## Politique actuelle

`HeuristicPlayer.choose_action()` évalue uniquement les actions fournies par
`Game.legal_actions()`. Il ne modifie pas l’observation, n’appelle pas `Game.apply()` et ne
consomme pas de hasard.

Le score est le produit scalaire des features et de `HeuristicWeights`. La configuration par défaut
est le profil optimisé `v008`, confirmé sur une validation à 1 000 parties contre `v007` et
`RandomPlayer`. Les poids restent injectables explicitement dans le
constructeur pour comparer un profil historique ou expérimental ; `HeuristicWeights.zero()` permet
de composer des tests ciblés.

Un profil publié peut être chargé depuis un fichier YAML avec `load_profile()`. Le script
`scripts/optimize_heuristic.py` produit un historique JSON et un profil YAML candidat hors du vault
`doc/`. La recherche utilise des seeds déterministes, alterne les positions et peut démarrer
directement avec un mélange 50/50 de `RandomPlayer` et du profil heuristique précédent. Elle explore
les sept coefficients économiques principaux par mutations simples et conjointes, puis conserve
progressivement les meilleurs candidats avec des lots de taille croissante. Dans ce mode mixte, le
profil de référence est capturé au début de la campagne et reste fixe pendant tous les batches ; seul
le candidat optimisé évolue.

Les profils peuvent aussi contenir `card_acquisition_weights`, qui contrôle les coefficients internes
de l’estimation de valeur d’une carte (`Gems`, `Power`, maîtrise, santé, pioche, deck thinning et
target denial). `durable_replay_factor` réduit en plus la valeur d’un achat durable selon la
progression observable de la partie : maîtrise maximale atteinte ou perte de PV. La valeur est
appliquée comme un exposant sur l’opportunité restante de rejouer la carte ; `0` désactive la
décroissance et `1` applique la décroissance nominale. Les mercenaires recrutés immédiatement ne
reçoivent pas cette valeur d’acquisition durable. Les anciens profils sans cette section utilisent
les valeurs V1 par défaut. Le mode
`--acquisition-only` de `scripts/optimize_heuristic.py` gèle tous les `HeuristicWeights` et optimise
uniquement ces coefficients internes ; `--active-acquisition-fields` permet de limiter les champs.

Le mode `--combined` de `scripts/optimize_heuristic.py` optimise un candidat complet composé des
poids d’action, d’acquisition et de contraintes. Sans liste explicite, il active tous les champs
optimisables des trois familles ; les modes `--acquisition-only` et `--constraints-only` restent
spécialisés et inchangés.

Le budget de `scripts/optimize_heuristic.py` est mesuré en temps CPU du processus, et non en temps
calendaire. `--compute-seconds` est l’option recommandée ; `--duration-seconds` reste accepté pour
compatibilité et possède la même sémantique. Le compteur n’avance donc pas lorsque le processus est
suspendu ou que la machine est en veille. Sur un processus multi-cœur, le temps CPU est cumulé par
cœur utilisé.

Le mode `--combined` supporte aussi les checkpoints persistants. Un checkpoint est écrit
atomiquement au démarrage, après chaque batch et pendant la finalisation ; une interruption au milieu
d’un batch fait rejouer ce batch au prochain lancement. Le budget CPU est cumulatif entre les
sessions :

```bash
PYTHONPATH=. poetry run python scripts/optimize_heuristic.py \
  --profile configs/heuristic_profiles/v007.yaml \
  --reference-profile configs/heuristic_profiles/v007.yaml \
  --start-mixed \
  --combined \
  --compute-seconds 54000 \
  --checkpoint artifacts/heuristic_optimization/v008/checkpoint.json \
  --seed 88 \
  --publish-profile configs/heuristic_profiles/v008.yaml
```

Pour reprendre une session interrompue :

```bash
PYTHONPATH=. poetry run python scripts/optimize_heuristic.py \
  --resume artifacts/heuristic_optimization/v008/checkpoint.json \
  --publish-profile configs/heuristic_profiles/v008.yaml
```

Le checkpoint restaure le candidat courant, les échelles de mutation, la seed, les profils de
référence et le prochain batch. Les checkpoints terminés ne peuvent pas être repris directement afin
d’éviter de modifier rétroactivement un run finalisé.

En mode combiné, la recherche s’arrête également après deux batches consécutifs sans amélioration
stricte du candidat courant. Une amélioration remet ce compteur à zéro. Le checkpoint et les
métadonnées du résultat enregistrent `consecutive_failed_batches` ainsi que la raison
`two_consecutive_failed_batches` ; la validation finale reste exécutée pour mesurer le candidat
retenu.

Le poids `buy_threshold` contrôle l'admissibilité des achats durables pendant la phase `BUY`. Une
action `BuyCard` doit avoir un score strictement supérieur à ce seuil pour rester candidate ; si
aucun achat durable ne le dépasse, `StopBuying` est choisi. Les `RecruitMercenary` ne sont pas
filtrés par ce seuil, car ils ne diluent pas le deck permanent. La borne d'optimisation actuelle est
`0.0..2.0` par pas de `0.25`, et le profil `v008` utilise `0.625`.

La validation finale compare le candidat et la référence sur des seeds appariées indépendantes avec
un intervalle de confiance à 95 %. Depuis la règle `positive_confidence_lower_bound_vs_previous`,
un profil n’est publié que si la borne basse contre la référence précédente est strictement positive
et qu’il ne régresse pas significativement contre `RandomPlayer`. Le seuil de 90 % contre Random est
informatif et ne bloque plus une campagne mixte démarrée explicitement.

Une évaluation interrompue par la deadline est marquée partielle et ne peut jamais être promue ni
faire évoluer le profil courant. Les stages `initial`, `racing` et `finalist` attendent des candidats
complets ; seule la dernière évaluation éventuellement interrompue reste dans l’historique pour le
diagnostic.

Lors des évaluations où le shaping est désactivé, l'optimiseur n'instancie pas de
`RewardShapingTracker` et ne branche pas d'observateur de transition. Cela réduit le coût des
validations sans modifier les scores ni les transitions de partie.

## Lancer un cycle de training de quatre heures

Pour obtenir un débrief avant de lancer une campagne nocturne, utiliser cette commande :

```bash
PYTHONPATH=. nice -n 10 poetry run python scripts/optimize_heuristic.py \
  --profile configs/heuristic_profiles/v002.yaml \
  --start-mixed \
  --duration-seconds 14400 \
  --initial-games 200 \
  --racing-games 500 \
  --validation-games 1000 \
  --test-games 3000 \
  --seed 45 \
  --publish-profile configs/heuristic_profiles/v003.yaml
```

`14400` secondes correspondent à quatre heures. `nice -n 10` réduit la priorité CPU afin de laisser
les autres applications réactives. `--start-mixed` utilise le mélange prévu de `RandomPlayer` et de
la version heuristique de référence, figée au début de la campagne. Le profil de départ reste `v002` ;
`v003.yaml` n’est écrit que si la validation réussit.

À la fin, conserver le résumé affiché et le chemin `results=...` vers l’artefact JSON sous
`artifacts/heuristic_optimization/`. Ils servent de débrief pour décider si `v003` peut devenir le
profil de départ d’une nouvelle campagne plus longue ou si `v002` doit être conservé.

Pour calibrer uniquement la valeur intrinsèque des cartes avant une optimisation combinée :

```bash
PYTHONPATH=. nice -n 10 poetry run python scripts/optimize_heuristic.py \
  --profile configs/heuristic_profiles/v003.yaml \
  --start-mixed \
  --acquisition-only \
  --duration-seconds 3600 \
  --initial-games 200 \
  --racing-games 500 \
  --validation-games 1000 \
  --test-games 3000 \
  --seed 46 \
  --publish-profile configs/heuristic_profiles/v004.yaml
```

Cette phase conserve les poids d’action de `v003` et écrit les coefficients calibrés dans la section
`card_acquisition_weights` de `v004.yaml` si la validation réussit.

Les profils contiennent également `constraint_weights`, qui pondère séparément les contraintes des
effets conditionnels : `domination=1.5`, `union=1.0`, `echo=0.75`, `inspiration=0.5`, `mastery=1.0`
et `health=0.75` dans `v004`. Le profil publié `v005` utilise désormais `domination=1.5`,
`echo=0.9375`, `health=0.625`, `inspiration=0.5`, `mastery=0.4375` et `union=0.6875`. La pénalité graduelle des seuils de maîtrise et de santé est multipliée
par son poids ; une contrainte booléenne absente ajoute son poids une fois par opération. Les profils
historiques sans cette section utilisent explicitement six poids uniformes à `1.0`. Ces poids sont
injectés dans les voies d’évaluation du jeu, des achats, des mercenaires, du bannissement et des
décisions en attente. Le mode `--constraints-only` de `scripts/optimize_heuristic.py` permet de les
optimiser sans modifier les poids d’action ni les poids d’acquisition ; `--reference-profile` impose
le profil complet utilisé comme adversaire précédent pendant la validation.

Un wrapper prêt à lancer est disponible dans
[`scripts/train_card_acquisition_1h.sh`](../../scripts/train_card_acquisition_1h.sh). Il peut être
exécuté depuis la racine du dépôt avec :

```bash
bash scripts/train_card_acquisition_1h.sh
```

Le wrapper vérifie la présence de `v003.yaml`, utilise une priorité CPU réduite et place correctement
`PYTHONPATH` avant `poetry`, afin d’éviter l’erreur de syntaxe rencontrée avec `nice`.

Les features d’action exposent également des deltas projetés de santé, maîtrise et menace de
champions. Les projections sont partielles en V1 et signalent les opérations non couvertes ; elles
ne mutent jamais l’observation.

Le runner peut notifier un observateur après chaque `Game.apply()`. `RewardShapingTracker` calcule
alors `gamma * Phi(après) - Phi(avant)` avec `gamma=1`, puis agrège le shaping par partie. Les poids
de shaping commencent à `alpha=0.10` dans l’optimiseur et décroissent jusqu’à zéro ; la validation
finale repose uniquement sur l’issue terminale.
Par défaut, le tracker conserve seulement les agrégats nécessaires à l’optimisation ; la liste des
transitions détaillées peut être activée explicitement pour le diagnostic.

Les coûts et contraintes sont des magnitudes positives avec des coefficients négatifs. Les effets
de cartes sont lus depuis les structures déclaratives (`Effect`, `Operation`, `ChampionAbility`).
Une carte jouée depuis la main ou recrutée comme mercenaire utilise uniquement sa branche active ;
les opérations conditionnelles non validées sont ignorées sans valeur ni malus. Une carte achetée
durablement peut en revanche cumuler le potentiel de
ses branches futures ; les contraintes non encore garanties ajoutent alors un malus prospectif.

## Actions couvertes

- jeu de cartes, y compris pioche, Power, Gems, maîtrise, santé et champions ;
- achat normal et recrutement de mercenaire comme deux candidats indépendants ;
- recrutement gratuit depuis la rivière ;
- activation de champion ;
- gain de maîtrise ;
- bannissement ou refus de bannissement ;
- décisions en attente, notamment sélection de cartes et destruction de champions ;
- assignation du Power à l’adversaire ou à un champion légal ;
- passage de phase et arrêt des achats.

Les évaluations de `PlayCard` et `RecruitMercenary` ignorent les seuils futurs de la carte. Les
évaluations de `BuyCard` et `RecruitFreeCard` conservent le potentiel futur avec un malus de risque,
car ces cartes restent des actifs durables. Chaque contribution prospective est bornée à zéro après
son propre malus : un effet conditionnel défavorable ne peut pas dégrader un effet indépendant de la
même carte.

Lors d’un bannissement, l’HeuristicPlayer protège désormais les cartes dont l’effet actif jouable
maintenant produit au moins une pioche. Cela couvre les cartes à pioche directe et les Champions
avec une pioche à la pose ; les contraintes déjà satisfaites sont prises en compte. Une pioche
conditionnelle inactive ne protège pas la carte. Cette protection est une préférence de l’IA : le
moteur conserve toutes les cibles légales pour les autres joueurs et stratégies.

Lorsqu’une égalité persiste entre plusieurs `BanishCard`, la politique compare d’abord la valeur
immédiate de chaque carte comme si elle était jouée, puis le coût et un identifiant stable. L’ordre
de la main et de la défausse ne décide plus seul du candidat choisi. La valeur de bannissement est
signée : `banish_threshold - card_acquisition_value`, avec un seuil par défaut de `3.0`. Une carte
de forte valeur peut donc perdre contre `SkipBanish`. Le pouvoir activable observable d’un champion
est inclus dans sa valeur durable, en plus de sa valeur intrinsèque et de son effet de pose.

Pour une carte mercenaire, `BuyCard` reçoit sa valeur statique d’acquisition tandis que
`RecruitMercenary` ajoute les effets résolus immédiatement pendant la phase d’achat. La politique
plafonne désormais chaque opération `gain_health` immédiate à la capacité de soin restante jusqu’à
`Game.STARTING_HEALTH` ; un mercenaire qui soigne 4 lorsque le joueur est à 48 PV expose donc un
gain de 2 dans ses features. Les achats durables restent évalués prospectivement et ne sont pas
plafonnés par la santé courante.

`GainMastery` dispose aussi d’une projection inter-phase déclarative. Elle compare la meilleure
valeur d’achat avec les Gems et la maîtrise actuelles à la meilleure valeur après dépense d’une
Gems et gain d’un point de maîtrise. Les candidats incluent `BuyCard` et `RecruitMercenary` ; la
valeur postérieure utilise la maîtrise projetée afin de prendre en compte les seuils des cartes de
la rivière. En parallèle, les gains positifs de seuil sont additionnés pour toutes les cartes de
la main qui peuvent être jouées après le gain de maîtrise. Ces projections ne mutent pas
l’observation et ne simulent pas l’ordre des cartes.

Les signaux `purchase_opportunity_cost` et `mastery_threshold_value` sont sérialisables dans les
poids heuristiques. Leurs coefficients sont actuellement neutres par défaut et doivent être
calibrés dans une campagne dédiée après validation comportementale ; les profils antérieurs restent
compatibles.

## Déterminisme et diagnostic

Le tie-break est déterministe : victoire terminale, létalité, score, priorité de phase, puis ordre
des actions légales. Les scores et features sont accessibles via `score_action()` et
`features_for_action()` pour les tests et futurs modes debug, mais aucun logging ou stockage n’est
activé par défaut.

## Limites connues

- la valeur d’acquisition ne simule toujours pas le deck ni les tirages futurs ; son facteur de
  replay utilise seulement des signaux publics agrégés de progression de partie ;
- la valeur des synergies futures n’est pas apprise ;
- les cartes ou opérations déclaratives ajoutées sans extracteur correspondant doivent être couvertes
  par un test avant d’être considérées comme évaluées précisément ;
- l’optimiseur utilise une recherche hybride bornée et ne garantit pas un optimum global ;
- la recherche initiale ne couvre volontairement que sept coefficients actifs ;
- les critères de publication doivent être confirmés sur des seeds de validation indépendantes.
- la menace des champions est actuellement `1.0 × nombre de champions`, normalisée par une échelle
  fixe de `4.0` ; elle ne tient pas encore compte de la puissance individuelle des cartes ;
- `Game.preview()` n’est pas encore disponible : les deltas du joueur restent des projections
  déclaratives.

## Tests

`tests/game/test_heuristic_player.py` couvre le rejet d’une liste vide, le poids de la pioche, la
comparaison achat/recrutement mercenaire, la pénalité des seuils de maîtrise, le ciblage d’un
champion, l’arrêt des achats, la reproductibilité et l’exécution d’une partie complète.

`tests/optimization/test_heuristic.py` couvre le round-trip YAML, les coefficients gelés, le mode
mixte et la validation des bornes. Les tests de shaping couvrent la normalisation de maîtrise par
30, la menace statique par nombre de champions, le potentiel et l’observateur de transition.

## Performance

Une passe d’optimisation ciblée sur le chemin utilisé par l’entraînement avec deux
`HeuristicPlayer` a supprimé les allocations de tuples et de générateurs dans les pénalités de
contraintes, puis a calculé une fois par décision les conditions communes (`echo`, `domination` et
présence d’un champion). Sur le workload reproductible de 200 parties, seeds `0..199`, rôles et
limites par défaut identiques, la médiane de trois runs est passée de `5.580 s` à `4.475 s`, soit
`1.105 s` et `19.8 %` de réduction. Les 200 parties sont terminées sans nul dans chaque mesure.
Une passe supplémentaire n’a pas apporté de gain robuste et a été retirée.

Le script de reporting `scripts/benchmark_heuristic_report.py` capture désormais les actions via
un wrapper de joueur avant `Game.apply()` au lieu d’activer un observateur de transitions qui clone
l’état complet à chaque action. Sur 100 parties Heuristic vs Random avec `v004`, seed `1046`, et
génération JSON/HTML incluse, le temps médian du workload est passé d’environ `1.99 s` à `1.17 s`,
puis `1.11 s` après l’agrégation des compteurs par rôle, soit environ `44 %` de réduction cumulée.
Les résultats restent identiques : `73` victoires Heuristic, `27` victoires Random, aucun nul ni
erreur. Les mesures sont effectuées dans le même processus afin d’exclure le bruit du démarrage de
Python et du chargement YAML.

Le même rapport conserve maintenant, pour chaque partie terminée, les cartes de la main, de la
pioche, de la défausse et de la zone de jeu finale des rôles Heuristic et Random. Il ajoute dans
`results.json` les statistiques de moyenne, présence, faction et multiplicité centrale par rôle,
ainsi que `final_deck_delta_heuristic_minus_random` et les groupes par résultat de l’Heuristic. La
page HTML affiche des graphiques SVG des decks moyens, des factions, des groupes ×2/×3 et des plus
grands deltas Heuristic − Random. Des CSV dédiés (`final_deck_*`) permettent une analyse hors HTML.
Les deltas de choix d’actions sont également exposés par catégorie avec un nombre moyen de choix par
partie.

Le chemin de capture réutilise maintenant la `CardInstance` déjà résolue pour les compteurs
d’actions lorsqu’il construit les statistiques mercenaires. Cela évite une seconde recherche dans
la rivière pour chaque achat ou recrutement, sans modifier les résultats du rapport. Sur 100
parties Heuristic v005 contre Random, seed `1046`, le temps médian de génération est passé de
`2.56 s` à `1.90 s` (trois répétitions, soit `25.8 %` de réduction observée).

Le même rapport expose aussi une analyse comportementale limitée au joueur Heuristic : les passages
de phase de jeu avec une main non vide et les cartes alors conservées, l’utilisation de l’action
`GainMastery` (1 gemme contre 1 maîtrise), ainsi que les recrutements immédiats et achats à long
terme de cartes mercenaires, ventilés par carte et par issue (`heuristic_win`, `heuristic_loss`,
`draw`). Ces données sont disponibles dans `results.json`, dans une section HTML dédiée et dans
`heuristic_pass_play_hand_cards.csv`, `heuristic_gain_mastery_summary.csv` et
`heuristic_mercenary_choices.csv`.

La campagne du rapport est maintenant équilibrée entre deux adversaires : 50 % de parties contre
Random et 50 % contre Heuristic v007. Les statistiques sont séparées dans `results.json` sous
`opponents.random` et `opponents.v007`, avec un delta de deck et de choix calculé séparément pour
chaque adversaire. Le HTML conserve les synthèses, les decks, les deltas et les comportements de
v008, mais retire les tables HTML détaillées redondantes ; les données détaillées restent dans les
CSV et le JSON.

Sur le workload dédié de 200 parties entre deux `HeuristicPlayer`, avec seeds `0..199`, profils
actifs par défaut et limites inchangées, une passe supplémentaire a remplacé les générateurs et
reconstructions de sets de `_has_union()`, `_has_echo()` et `_has_domination()` par des boucles
directes avec retour anticipé. La médiane est passée de `4.220 s` à `3.953 s`, soit `6.3 %` de
gain supplémentaire. En incluant les passes précédentes, le gain cumulé est d’environ `29.1 %`
par rapport au benchmark historique à `5.580 s`. Les 200 parties sont terminées sans nul dans chaque mesure ; le profiling suivant n’a
pas révélé de passe locale susceptible d’apporter encore 2 % de gain robuste.

Le scoring utilise un produit scalaire explicite et ne valide pas/introspecte chaque instance de
features dans le hot path. Sur le benchmark mixte de 1 000 parties, la médiane mesurée est passée
de 25,980 s à 23,772 s dans le même environnement, avec 1 000 parties terminées dans les deux cas.
Le benchmark dédié `benchmarks/benchmark_heuristic_players.py` mesure deux `HeuristicPlayer` avec
les mêmes seeds afin d’isoler le coût de la politique heuristique.
Après la séparation entre évaluation immédiate et prospective, le chemin d’achat durable réutilise
la même agrégation prospective pour calculer la valeur et la pénalité des cartes ordinaires, au lieu
de recalculer les branches deux fois. Sur 200 parties, seeds `0..199`, la médiane est passée de
`5.195 s` à `5.115 s` (`1.5 %` de réduction observée). Le gain est positif mais inférieur au seuil
de `2 %` retenu pour une optimisation robuste ; une seconde micro-optimisation a été rejetée car
elle remontait à `5.605 s`. Les 200 parties sont terminées sans nul dans chaque campagne et la suite
complète compte `152` tests.

Lors des validations finales où `alpha=0`, la collecte détaillée du shaping est désactivée par défaut : les résultats terminaux sont
inchangés, mais les snapshots post-transition coûteux ne sont plus produits. Le diagnostic complet
reste disponible avec `--track-zero-alpha-shaping`. Sur 120 simulations fixes de validation (30
parties par profil et par adversaire), cette optimisation a réduit la médiane observée d’environ
8,37 s à 5,30 s, soit 36,7 %, sans modifier les utilités ni les différences appariées. Pendant les
simulations d’entraînement, le tracker interne
reçoit l’état post-action vivant de façon synchrone au lieu d’une seconde copie détachée ; le runner
conserve le post-état détaché par défaut pour les observateurs externes. Pendant les validations à
`alpha=0`, les politiques internes de la campagne reçoivent également l’état vivant, car elles sont
en lecture seule ; le runner reste détaché par défaut pour les autres politiques. Sur 100 parties
entre deux joueurs chargés depuis `v002.yaml`, le temps médian est passé de 4,484 s à 2,570 s après
les passes d’optimisation, soit une réduction cumulée de 42,7 %, avec 100 parties terminées à chaque
mesure. Sur le workload mixte contrôlé `RandomPlayer` contre `HeuristicPlayer` (`v002`, 100 seeds,
rôles alternés), le temps médian est passé de 2,250 s à 1,601 s après la première passe puis à
1,523 s après la seconde ; les 100 parties sont restées terminées, sans nul, avec 40 victoires de
l’heuristique à chaque mesure.
Une mesure historique annonçait 78,5 % contre `RandomPlayer` et 76,2 % contre `v003`, mais le
second résultat n'est pas reproductible avec les profils actuellement chargés. L'audit contrôlé
sur 1 000 seeds donne 79,2 % pour `v004` contre `RandomPlayer` et 49,3 % dans le duel
`v004` contre `v003` lorsque les profils YAML complets sont utilisés. La matrice d'isolation montre
que la différence vient des `constraint_weights` de `v004` : avec les poids d'action ou d'acquisition
seuls, le candidat reste à 53,0 % contre `v003`, tandis que l'activation des nouvelles contraintes
le ramène à 49,3 %. Ces poids améliorent donc le score contre RandomPlayer au prix d'une faiblesse
contre l'ancien joueur heuristique ; aucune erreur ni nul n'a été observé dans ces mesures.
Dans le chemin d’optimisation avec shaping, le tracker utilise désormais un snapshot détaché minimal
qui ne copie que les données nécessaires au potentiel d’état ; les politiques internes en lecture
seule reçoivent l’état vivant. Sur 100 évaluations fixes `v002` contre `RandomPlayer`, la médiane est
passée de 3,092 s à 2,008 s, puis à 1,963 s après la fusion du calcul des conditions de shaping, soit
environ 36,5 % de réduction cumulée, avec 52 victoires, aucun nul et aucune erreur.

Après l’ajout de `durable_replay_factor`, une passe ciblée a réutilisé le multiplicateur de replay
pour toutes les cartes d’une même décision d’achat au lieu de le recalculer pour chaque carte. Sur
le même workload de 200 parties, la médiane est passée de `5,309 s` à `5,167 s`, soit `2,7 %` de
réduction et 200 parties terminées dans chaque mesure. Une passe de profiling supplémentaire n’a
pas identifié d’optimisation locale susceptible d’atteindre encore 2 % sans modifier le périmètre
du moteur ou du joueur.

La projection inter-phase de `GainMastery` est calculée uniquement si au moins un de ses deux
coefficients (`purchase_opportunity_cost` ou `mastery_threshold_value`) est non nul. Avec le profil
`v004`, les signaux restent donc neutres sans coût de projection. Lorsque la projection est active,
elle reste une évaluation pure de l’observation projetée, réutilisant les extracteurs de features
existants pour la rivière et la main. Une tentative de réduire les allocations sur ce chemin a été
profilée, mais son gain était inférieur à 2 % et n’a pas été conservée.
