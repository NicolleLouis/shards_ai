# Joueur heuristique pondéré — Architecture

**Statut : livré** — Implémenté avec `HeuristicPlayer`, scoring pondéré, features d’actions,
comparaison achat/recrutement mercenaire, tests et benchmark dédié.

## Objective

Créer un joueur `HeuristicPlayer` valide, reproductible et configurable, capable de choisir
l’action légale ayant la meilleure valeur heuristique immédiate. L’évaluation reposera sur une
somme pondérée de caractéristiques de l’action et de son contexte, par exemple :

```text
EV(action) = a × coût
          + b × Power produit
          + c × cartes piochées
          + d × maîtrise gagnée
          + e × Gems produites
          + ...
```

La première version doit fournir une base comportementale mesurable contre `RandomPlayer` et
réutilisable par le futur script de campagnes qui fera varier les coefficients.

## Current State

- `Player` est un `Protocol` dans `shards_ai/game/players.py`.
- `RandomPlayer` implémente `choose_action(observation, legal_actions)` et ne modifie jamais
  directement le moteur.
- `GameRunner` fournit une observation détachée et les actions légales à chaque décision, puis
  valide l’action retournée via `Game.apply`.
- Les actions publiques sont des dataclasses dans `shards_ai/game/actions.py` : jeu de carte,
  achat, recrutement, activation, maîtrise, bannissement, décisions en attente et attaque.
- Les caractéristiques de cartes sont déclaratives dans `CardDefinition`, `Effect`, `Operation`
  et `ChampionAbility` (`shards_ai/game/cards/model.py`).
- `Game.legal_actions()` garantit déjà la légalité contextuelle : ressources disponibles, phase,
  cibles d’attaque, coûts et décisions obligatoires.
- `GameState` expose les ressources, zones, champions, rivière, phase et joueur adverse. La copie
  d’observation est actuellement complète ; l’information cachée n’est donc pas encore un sujet.
- Le moteur et le runner sont déterministes à seed identique. La suite couvre déjà le contrat du
  joueur random et la validité des parties.

## Target Behavior

À chaque appel, le joueur :

1. refuse explicitement une liste d’actions vide ;
2. extrait les caractéristiques de chaque action à partir de l’observation et des définitions de
   cartes ;
3. calcule un score flottant ou entier avec les coefficients configurés ;
4. retourne l’action au score maximal ;
5. applique un tie-break déterministe et stable ;
6. ne retourne jamais une action absente de `legal_actions` et ne modifie pas l’observation.

Les décisions obligatoires doivent être résolues avant les actions ordinaires. Les actions
terminales ou indispensables (`StopBuying`, `PassPlayPhase`, `AssignPower`) doivent avoir une
évaluation explicite ; elles ne doivent pas dépendre d’un score par défaut accidentel.

En cas d’égalité parfaite après le score, la priorité de phase est : jouer/activer ou gagner de la
maîtrise avant passer en phase suivante, acheter/recruter avant `StopBuying`, puis conserver l’ordre
fourni par `legal_actions()`. Cette priorité ne remplace jamais le score ; elle ne départage que les
égalités.

## Non-Goals

- Aucun réseau de neurones ni représentation tensorielle.
- Aucun look-ahead, rollout ou copie/simulation de `Game` dans la première version.
- Aucun apprentissage automatique des coefficients dans ce document.
- Aucun changement de règles, de légalité ou de transitions dans `Game`.
- Pas encore d’information cachée réaliste pour l’adversaire.
- Pas de nouvelle interface utilisateur ou de persistance des coefficients en base.

## Key Decisions

1. **Séparation politique / scoring.** `HeuristicPlayer` orchestre le choix ; un évaluateur
   indépendant transforme une action et une observation en caractéristiques puis en score.
2. **Les actions légales sont l’unique espace de choix.** L’heuristique ne recalcule pas les règles
   et ne fabrique pas d’action.
3. **Évaluation déclarative des cartes.** Les valeurs sont lues depuis `CardDefinition` et les
   opérations applicables au niveau actuel de maîtrise, plutôt que codées par nom de carte.
4. **Features par action.** Un achat doit notamment exposer coût, Gems/Power/maîtrise/santé/cartes
   piochées attendus, bouclier, type de carte, faction, mercenaire/champion et valeur de pose.
   Les features d’une action sont normalisées dans une structure stable avant pondération.
5. **Coefficients immuables pendant une partie.** `HeuristicWeights` est fourni au constructeur,
   validé, et ne change pas pendant `choose_action`. Cela garantit la reproductibilité des
   campagnes.
6. **Tie-break stable.** Le classement suit cet ordre : `terminal_win`, `lethal`, score heuristique,
   priorité explicite de la phase, puis ordre de `legal_actions()`. Aucun hasard ne doit contaminer
   la comparaison des coefficients.
7. **Baseline explicable.** Le joueur doit pouvoir retourner ou journaliser, via une API de
   diagnostic optionnelle, les features et le score de chaque action sans changer le contrat
   `Player`.
8. **Configuration manuelle.** Les coefficients sont fournis explicitement au constructeur de
   `HeuristicPlayer`, sous la forme d’un objet `HeuristicWeights`. La baseline ci-dessous constitue
   le profil initial versionné ; elle pourra ensuite être remplacée par un fichier de configuration
   ou injectée par le script de campagne.
9. **Diagnostic opt-in uniquement.** Les scores et features décision par décision ne sont pas
   conservés dans les campagnes normales. Ils seront exposés plus tard par des modes debug dédiés,
   activables explicitement pour analyser une partie ou un petit lot de parties.
10. **Métrique d’optimisation volontairement simple.** Une victoire vaut `3`, une nulle `1` et une
    défaite `0`. La santé finale, la maîtrise, la durée et le nombre d’actions sont rapportés comme
    métriques secondaires, mais ne participent pas à la fonction objectif initiale.
11. **Valeur immédiate uniquement en V1.** La valeur d’une carte est calculée à partir de ses
    effets applicables dans l’état courant. La valeur future du deck, du tempo ou d’une synergie
    qui ne produit pas encore d’effet n’est pas estimée dans cette première version. Les valeurs
    intrinsèques statiques (`card_acquisition_value`, bouclier imprimé, qualité d’un champion) sont
    autorisées ; elles ne doivent pas provenir d’une simulation de tours futurs.
12. **Valeur minimale d’acquisition.** `BuyCard` reçoit néanmoins une valeur statique de la carte
    (`card_acquisition_value`) dérivée de sa définition. Sans ce signal, un achat normal n’aurait
    aucun bénéfice au moment du choix et serait systématiquement dominé par un mercenaire ou une
    transition. Cette valeur n’est pas une simulation du futur deck ; elle représente seulement la
    qualité intrinsèque connue de la carte.
13. **Les contraintes diminuent la valeur.** Toute condition qui retarde, limite ou rend incertain
    un effet produit une feature négative. À effet identique, une carte conditionnée à 20 maîtrise
    est donc moins valorisée qu’une carte conditionnée à 10 maîtrise. La pénalité intrinsèque de la
    carte et l’état courant sont évalués séparément pour éviter de masquer une condition déjà active.

## Open Questions

Aucune question ouverte bloquante pour la V1.

## Proposed Architecture

### Composants

`shards_ai/ai/heuristic_player.py` contiendra `HeuristicPlayer` et `HeuristicWeights`.

`shards_ai/ai/heuristic_features.py` contiendra :

- `ActionFeatures`, structure complète et sérialisable des signaux numériques ;
- `features_for_action(observation, action, player_id)`, dispatcher par type d’action ;
- extracteurs de valeur de carte et d’effet, compatibles avec les branches de maîtrise,
  conditions Union/Echo/Domination et capacités de champions.

`shards_ai/ai/heuristic_evaluator.py` contiendra le calcul pur :

```text
score = weights.dot(features)
```

Cette découpe permet de tester séparément l’extraction métier, la formule et la politique de
sélection. Elle évite une classe monolithique difficile à calibrer.

### Couverture des actions

| Famille | Signaux principaux | Règle particulière |
|---|---|---|
| `PlayCard` | coût déjà payé, Gems, Power, maîtrise, soin, pioche, bannissement, contraintes, champion | estimer l’effet conditionnel dans l’état courant |
| `BuyCard` | coût, valeur d’acquisition, faction, champion, mercenaire, placement | l’achat paie le coût et ajoute la carte à la défausse |
| `RecruitMercenary` | coût, valeur d’acquisition, Power/Gems/maîtrise, placement, effet immédiat | paie le coût, joue immédiatement la carte et la remet au deck central au cleanup |
| `GainMastery` | maîtrise gagnée, coût en Gems, atteinte de seuils | comparer à une carte ou action alternative |
| `ActivateChampion` | effet actif, seuils, pioche, ressources, Power | tenir compte des champions déjà activés |
| `BanishCard` / `SkipBanish` | épuration du deck, perte d’une carte, coût d’opportunité | l’action est obligatoire/optionnelle selon l’état |
| `RecruitFreeCard` | valeur de la carte gratuite, placement main/défausse | respecter le coût maximum déjà imposé par le moteur |
| `ChoosePendingDecision` | valeur de la cible choisie | pondérer la cible plutôt que l’identifiant |
| `AssignPower` | dégâts au joueur, destruction d’un champion, létalité | privilégier victoire immédiate puis meilleur impact |
| `PassPlayPhase` / `StopBuying` | valeur de transition, Gems restantes, Power accumulé | pénaliser l’abandon lorsqu’une action rentable reste légale |

Les features de valeur d’une carte seront dérivées d’opérations connues. Les opérations non
quantifiables précisément en V1 recevront une feature dédiée ou une valeur prudente documentée,
plutôt qu’une estimation silencieuse. Les actions gagnantes et létales reçoivent une priorité
dominante indépendante des coefficients économiques.

### Pioche et sélection de cibles

La feature `card_draw` couvre les cartes ou capacités qui piochent automatiquement. Elle peut être
utilisée pour `PlayCard`, `RecruitMercenary` ou `ActivateChampion` lorsque l’action produit la
pioche. Pour `BuyCard`, elle peut seulement contribuer à `card_acquisition_value` comme propriété
intrinsèque de la carte achetée. En revanche, une pioche automatique ne crée pas un choix de carte :
l’heuristique ne choisit pas l’ordre du deck dans cette V1.

Les cartes proposées dans la rivière sont, elles, des cibles sélectionnables. Pour chaque
`BuyCard`, `RecruitMercenary` ou `RecruitFreeCard`, l’évaluateur calcule les features de la carte
ciblée et ajoute les effets applicables au contexte courant. Cela permet de cibler une carte à
pioche, une carte générant des ressources ou une carte dont les conditions sont actuellement
actives.

Pendant `ATTACK`, chaque `AssignPower` légal vers un champion adverse est évalué comme une action
distincte. La valeur de cible peut combiner :

- `champion_value` : valeur estimée du champion détruit, incluant sa capacité active et son effet
  de pose déjà consommé ;
- `target_denial` : valeur de retirer une capacité ou une protection adverse ;
- `damage_value` : dégâts ou Power qui seraient autrement assignés au joueur adverse ;
- `lethal` : priorité absolue si la cible ou l’attaque termine la partie.

Le joueur ne sélectionne que les champions retournés par `Game.legal_actions()` ; les protections,
immunités et seuils de Power restent donc de la responsabilité du moteur.

### Achat normal versus mercenaire

Lorsque la rivière expose un mercenaire abordable, le moteur peut produire `BuyCard` et
`RecruitMercenary` pour la même carte. Ces deux actions doivent être évaluées séparément, même si
elles partagent la même définition de carte :

- `BuyCard` reçoit `card_acquisition_value`, une valeur statique issue de la définition de la carte,
  ainsi que le coût payé ; aucun effet de la carte n’est résolu pendant l’achat normal ;
- `RecruitMercenary` reçoit la même `card_acquisition_value`, puis ajoute l’effet immédiat résolu
  pendant la phase d’achat dans le contexte `immediate_resolution`, avec les gains effectifs
  attendus ;
- les deux actions portent le même coût payé et la même carte, afin que la comparaison reste
  cohérente ;
- le retour du mercenaire au bas du deck central au cleanup est une conséquence de cycle, non une
  valeur de deck à optimiser en V1.

Ainsi, à coefficients identiques, un mercenaire sera choisi s’il apporte une valeur immédiate
supérieure à l’achat normal. Si l’effet immédiat ne distingue pas les deux actions, elles sont
comparées sur leur coût et leur valeur statique d’acquisition, puis le tie-break doit être explicite
et stable ; il ne faut pas choisir selon le nom de la carte ou selon l’ordre fortuit de la rivière.

Concrètement, pour chaque carte mercenaire abordable, le classement contient donc deux candidats :

```text
score(BuyCard) = score(card_acquisition_value, coût payé, placement différé)
score(RecruitMercenary) = score(card_acquisition_value, coût payé,
                                effet immédiat, placement mercenaire)
choix = argmax(scores des actions légales)
```

La comparaison inclut également les autres actions légales de la phase, notamment `StopBuying`.
Une stratégie peut donc décider de conserver ses Gems si le score de toutes les acquisitions est
inférieur au score de transition, selon les coefficients configurés.

## Data Model

Aucune migration ni nouveau stockage de partie n’est nécessaire.

`HeuristicWeights` sera un dataclass immuable contenant les coefficients nommés, avec des valeurs
par défaut explicites. Il pourra être construit depuis un mapping ou un fichier JSON/YAML dans le
futur, mais la V1 ne dépendra pas d’un fichier externe.

### Baseline manuelle V1

Les valeurs sont un point de départ pratique, calibré pour favoriser le tempo immédiat et la
victoire. Elles ne constituent pas une vérité métier et devront être évaluées expérimentalement.
Les features sont des magnitudes positives, y compris les pénalités (`cost_paid`,
`constraint_penalty`, `action_penalty`). Le signe économique est porté par le coefficient : les
coefficients de ces pénalités sont négatifs. Les coefficients non listés valent zéro.

| Feature              | Coefficient initial | Intention                                                            |
| -------------------- | ------------------: | -------------------------------------------------------------------- |
| `cost_paid`          |              `-1.0` | limiter l’investissement, sans interdire les cartes fortes           |
| `gems_produced`      |              `+1.5` | valoriser l’économie et les achats futurs                            |
| `power_produced`     |              `+2.0` | valoriser la pression offensive immédiate                            |
| `mastery_gained`     |              `+1.5` | favoriser la progression vers les seuils et la victoire par maîtrise |
| `card_draw`          |              `+2.5` | valoriser l’avantage de cartes                                       |
| `health_gained`      |              `+0.5` | valoriser la survie sans dominer l’attaque                           |
| `shield_value`       |              `+0.5` | valoriser la réduction de dégâts future                              |
| `deck_thinning`      |              `+1.0` | valoriser le bannissement des cartes faibles                         |
| `card_acquisition_value` |           `+1.0` | valoriser l’acquisition intrinsèque d’une carte                     |
| `champion_value`     |              `+2.0` | valoriser les champions et leur présence persistante                 |
| `target_denial`      |              `+1.5` | valoriser la suppression d’une capacité ou protection adverse        |
| `damage_value`       |              `+2.0` | valoriser les dégâts assignés à l’adversaire                         |
| `constraint_penalty` |              `-1.0` | pénaliser les seuils, prérequis et conditions de déclenchement       |
| `phase_progress`     |              `+0.1` | départager légèrement les actions de transition                      |
| `action_penalty`     |              `-1.0` | pénaliser une action sans bénéfice mesurable                         |
| `lethal`             |           `+1000.0` | rendre une victoire par dégâts prioritaire                           |
| `terminal_win`       |           `+1000.0` | rendre toute victoire immédiate prioritaire                          |

Les valeurs de `lethal`, `terminal_win` et des autres signaux booléens doivent être représentées de
façon cohérente, avec `0` ou `1`. `lethal` est réservé aux dégâts qui terminent la partie ;
`terminal_win` couvre également une victoire par effet de carte ou par atteinte d’un seuil de
maîtrise. Pour les cibles d’attaque, `champion_value` devra intégrer la valeur du champion détruit ;
une action gagnante reste prioritaire grâce à `terminal_win`.

La feature `constraint_penalty` agrège notamment :

- le seuil de maîtrise requis, avec une pénalité intrinsèque croissante entre 10 et 20 maîtrise ;
- la distance au seuil dans l’état courant, lorsqu’un effet reste actuellement inactif ;
- les prérequis Union, Echo, Domination et Inspiration lorsqu’ils ne sont pas satisfaits ;
- les contraintes de santé, de faction, de présence d’un champion ou de cartes disponibles dans
  une zone donnée.

Il n’existe pas de bonus positif de synergie dans la V1. Une condition satisfaite évite seulement la
pénalité correspondante ; elle n’ajoute pas de points. Elle ne doit pas non plus continuer à recevoir
sa pénalité contextuelle complète. Par exemple, une carte à 20 maîtrise reste intrinsèquement plus
exigeante qu’une carte à 10, mais si le
joueur possède déjà 20 maîtrise, la pénalité de distance au seuil devient nulle et l’effet
conditionnel est valorisé comme actif.

`ActionFeatures` sera un dataclass avec des champs numériques stables et une convention unique :

```text
cost_paid, gems_produced, power_produced, mastery_gained, health_gained,
card_draw, shield_value, deck_thinning, card_acquisition_value,
champion_value, target_denial, damage_value, constraint_penalty,
phase_progress, action_penalty, lethal, terminal_win
```

Les champs absents d’une action valent zéro. Les identifiants de carte et de cible restent des
métadonnées de diagnostic, pas des coefficients.

Les coefficients devront être versionnables dans le futur outil de campagne avec une représentation
JSON qui inclut un nom de stratégie, une version de feature schema et la liste complète des features.

## Backend Flow

`GameRunner` reste inchangé. À chaque décision, `HeuristicPlayer.choose_action` reçoit la copie de
l’état, calcule les scores, puis sélectionne l’action maximale.

Le calcul doit être pur et linéaire en nombre d’actions légales et en taille des zones inspectées.
Il ne doit pas appeler `Game.apply`, muter des cartes, modifier les decks ni consommer de hasard.
Une exception explicite est levée si une action nécessaire n’est pas supportée, afin de découvrir
les nouveaux types d’action au lieu de jouer silencieusement une stratégie incomplète.

Pour `RecruitMercenary`, l’évaluateur réutilise l’extracteur d’effets de carte dans un contexte
`immediate_resolution`. Ce contexte est un mode d’évaluation interne, pas une feature pondérée : il
indique que les opérations de la carte sont résolues maintenant. L’évaluateur ne simule pas la
mutation du moteur ; il calcule uniquement les opérations déclaratives applicables à la maîtrise
et aux conditions observables au moment du choix.

Les règles restent exclusivement dans `Game`. Si une feature exige une information que l’observation
ne contient pas, elle utilise une approximation documentée ; elle ne lit pas l’état mutable interne
du moteur.

## Frontend Flow

Sans objet pour cette étape. Le joueur est consommé par `GameRunner`, les benchmarks et les scripts
d’analyse Python.

## Authorization And Feature Gates

Sans objet. Aucun accès utilisateur ni permission n’est introduit. Une configuration de coefficients
est locale au processus et doit être passée explicitement au joueur.

## Observability And Operations

Le joueur doit offrir un mode diagnostic désactivé par défaut : dernière action choisie, score,
features et éventuellement classement des actions. Le logging ne doit pas être activé par défaut
pendant les campagnes massives, car le volume est proportionnel au nombre d’actions.

Le futur script devra enregistrer au minimum : version du code/schema, seed de départ, coefficients,
nombre de parties, adversaires, score moyen selon `3/1/0`, taux de victoire, taux de nul, durée
moyenne, nombre moyen d’actions et métriques de performance. Les campagnes doivent être
reproductibles par seeds explicites. Les métriques secondaires servent à détecter une stratégie qui
maximise artificiellement les nulles ou ralentit les parties, sans modifier l’objectif de la V1.

Le coût d’évaluation est faible pour six cartes de rivière et une main courte. Si les zones ou le
nombre de candidats augmentent, pré-calculer les agrégats de l’état une fois par décision plutôt
que rescanner les mêmes cartes pour chaque feature.

## Edge Cases

- liste d’actions vide : `InvalidActionError` identique au comportement du joueur random ;
- score `NaN` ou poids non finis : rejet à la validation des poids ;
- scores négatifs ou tous égaux : sélection toujours déterministe ;
- rivière vide ou slot `None` : aucun accès direct sans garde ;
- carte gratuite ou coût nul : ne pas confondre coût payé et valeur de la carte ;
- actions de bannissement et décisions sans candidat : suivre strictement les actions exposées ;
- égalité entre achat normal et recrutement mercenaire : tie-break documenté et testable ;
- létalité, victoire par maîtrise ou action terminale : priorité avant la formule économique ;
- seuil de maîtrise, santé maximale, boucliers et protections : ne valoriser que les effets
  applicables dans l’état courant ;
- nouvelle classe d’action ajoutée au moteur : échec explicite et test à ajouter avant usage.

## Testing Strategy

Ajouter `tests/game/test_heuristic_player.py` et, si nécessaire,
`tests/game/test_heuristic_features.py` pour vérifier :

- rejet d’une liste vide ;
- choix de la feature dominante avec des poids unitaires ciblés ;
- choix d’un achat selon coût, Power, Gems, pioche et maîtrise ;
- comparaison `BuyCard` / `RecruitMercenary` ;
- pénalisation cohérente des contraintes : 20 maîtrise < 10 maîtrise à effet égal, condition
  active moins pénalisée qu’une condition inaccessible ;
- choix de `PlayCard`, `GainMastery`, activation de champion et transitions ;
- décisions de bannissement, recrutement gratuit et choix de cible ;
- priorité d’une action létale sur un score économique ;
- égalités et reproductibilité sans hasard ;
- observation non mutée ;
- partie complète `GameRunner` avec deux joueurs heuristiques et confrontation heuristique/random ;
- compatibilité avec les cartes à effets structurés et conditions de maîtrise.

Les tests existants du moteur doivent rester inchangés. Un benchmark de référence pourra réutiliser
le format de `benchmarks/benchmark_random_players.py`, mais son implémentation appartient au second
chantier d’itération des coefficients.

## Rollout And Migration

1. Ajouter le modèle de poids, l’extraction de features, l’évaluateur et `HeuristicPlayer`.
2. Exporter `HeuristicPlayer` depuis `shards_ai/ai/__init__.py`.
3. Ajouter les tests unitaires et un test de partie complète.
4. Fournir une configuration baseline versionnée dans le code ou dans un fixture de test.
5. Ajouter ensuite le script de campagne séparé, capable d’injecter différents poids et de produire
   des métriques comparables.

Aucune migration de données, feature flag ou compatibilité rétroactive n’est requise. Le rollback
   consiste à ne pas sélectionner `HeuristicPlayer` dans un runner ; `RandomPlayer` reste inchangé.

## Files Expected To Change

- `shards_ai/ai/heuristic_player.py` — joueur et orchestration de sélection.
- `shards_ai/ai/heuristic_features.py` — extraction des signaux d’action (tentatif, selon le niveau
  de découpage retenu).
- `shards_ai/ai/heuristic_evaluator.py` — formule et validation des poids (tentatif).
- `shards_ai/ai/__init__.py` — export public.
- `tests/game/test_heuristic_player.py` — comportement de la politique.
- `tests/game/test_heuristic_features.py` — extraction et effets conditionnels (tentatif).
- `benchmarks/benchmark_heuristic_player.py` — futur second chantier, hors implémentation V1.
- `scripts/analyze_games.py` ou nouveau script dédié — futur second chantier, hors implémentation
  V1.
- `doc/Current state/Heuristic player.md` — à mettre à jour après implémentation, avec la skill de
  nettoyage de l’état courant.
