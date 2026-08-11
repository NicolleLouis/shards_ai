# Abstraction stratégique des play turns neuronaux — Architecture

## Objective

Réduire les décisions neuronales qui ne représentent qu'un ordre équivalent de
résolutions mécaniques pendant `PLAY`. Le réseau doit choisir entre les conséquences
stratégiques réelles d'un turn : construction du deck, bannissement, recrutement,
maîtrise, activation de combo et cibles importantes.

Le résultat attendu est un nouveau joueur neural qui délègue au solveur les segments
de `PLAY` déterministes ou canoniques, et choisit une **branche macro** uniquement
lorsque plusieurs conséquences stratégiques restent possibles. Cela doit réduire la
longueur des épisodes PPO et améliorer le credit assignment sans modifier les règles
du jeu ni les profils neuronaux existants.

Critères de réussite :

- l'exécution finale utilise exclusivement les actions atomiques déjà validées par
  `Game.legal_actions()` et `Game.apply()` ;
- les profils et datasets atomiques historiques restent chargeables et reproductibles ;
- chaque décision macro est rejouable à partir de sa trace atomique ;
- les permutations réellement équivalentes ne créent plus plusieurs décisions neural ;
- une différence de cible, de zone, de maîtrise, de condition, de pioche ou de ressource
  qui peut modifier la stratégie reste visible au réseau ;
- le solveur a un coût borné, déterministe et identique à chaque run.

## Current State

`Game.legal_actions()` expose actuellement toutes les actions atomiques de `PLAY` :
`PlayCard`, `ActivateChampion`, `GainMastery` et `PassPlayPhase`. Les effets peuvent
interrompre ce flux par un bannissement, un recrutement gratuit ou une
`ChoosePendingDecision`. `Game.apply()` impose ces interruptions avant toute autre
action.

`GameRunner` demande une action à chaque transition. `NeuralPlayer` représente toutes
les actions légales sous forme d'`ActionRepresentation`, les score individuellement à
partir de `NeuralObservation`, puis choisit l'argmax. `PPOTrainingPlayer` collecte une
décision et une transition par action atomique.

Les définitions de cartes exposent déjà une partie de la sémantique nécessaire :
opérations structurées, maîtrise, Union, Echo, Domination, bannissement, pioche,
récupération, recrutement gratuit et capacités de champions. Cette information est
suffisante pour guider une analyse prudente, mais ne doit pas remplacer l'exécution
réelle du moteur.

## Non-Goals

- modifier les règles, phases, actions publiques ou validations du moteur ;
- ajouter une macro-action à `shards_ai.game.actions` ou faire accepter au moteur une
  séquence non atomique ;
- imposer un ordre rigide tel que « banish toujours avant draw » ;
- résoudre exhaustivement toutes les permutations d'un turn ;
- migrer les anciens datasets, checkpoints, architectures ou profils actifs ;
- entraîner, promouvoir ou activer un checkpoint dans cette livraison ;
- utiliser l'observation masquée comme preuve que deux états internes exacts sont
  interchangeables.

## Key Decisions

1. **Abstraction dans la couche IA.** Le moteur reste l'unique autorité des actions
   atomiques ; le solveur est un consommateur de `Game.clone()`, `legal_actions()` et
   `apply()`.
2. **Nouvelle voie neural isolée.** Un nouveau profil et un nouveau schéma de dataset
   macro sont requis. Les architectures V001 à V006 et leurs collecteurs restent
   inchangés et chargeables.
3. **Branches, non règles rigides.** Une priorité heuristique guide l'exploration ;
   elle n'autorise jamais à éliminer une branche dépendante.
4. **Frontière basée sur la preuve.** Le solveur automatise l'ordre, les activations,
   les bannissements, les pending choices et les recrutements gratuits uniquement si
   les alternatives sont prouvées équivalentes ou qu'une seule action est légale.
5. **Choix neural de branche terminale.** Une candidate est une trace atomique maximale
   jusqu'à la fin de `PLAY` ou au prochain point de choix stratégique. Le réseau ne
   choisit pas une intention vague ni une unique carte intermédiaire.
6. **Observation comme regroupement, pas comme identité moteur.** Des branches peuvent
   être présentées sous une même candidate si leur résultat neural est identique et que
   l'analyse de dépendances garantit qu'aucune différence future pertinente n'est
   éliminée. La fusion de mémoïsation interne exige une signature d'état plus forte.
7. **Budget fixe et déterministe.** Le solveur n'expose aucun paramètre CLI ou profil
   réajustable par run. Ses constantes versionnées sont :

   ```text
   MAX_EXPANSIONS = 256
   MAX_MEMOIZED_STATES = 128
   MAX_MACRO_CANDIDATES = 16
   MAX_ATOMIC_ACTIONS_PER_SEGMENT = 32
   ```

   Aucun timeout n'est utilisé : sa variabilité selon la machine modifierait la
   politique et nuirait à la reproductibilité.
8. **Fallback conservateur.** Si un budget est atteint ou qu'une action n'a pas de
   descripteur fiable, le solveur applique uniquement le préfixe déjà prouvé canonique,
   puis rend la prochaine décision atomique au réseau. Il ne choisit jamais une branche
   stratégique arbitraire.
9. **Une transition PPO par segment macro.** Les actions atomiques masquées ne créent
   pas de transitions ou rewards auxiliaires. Leur nombre reste une métadonnée de trace.
10. **Validation en partie obligatoire.** Une baisse du nombre de décisions, une bonne
    couverture dataset ou une vitesse locale ne constituent pas une promotion ; le panel
    neural complet reste la preuve de qualité.
11. **Conditions déjà actives.** Une carte conditionnelle dont la condition la plus
    exigeante est déjà satisfaite peut rejoindre le groupe canonique : seuil maximal de
    maîtrise atteint, santé suffisante, Union, Domination ou présence d'un champion.
    Cette canonisation est annulée si une action stratégique légale peut modifier la
    condition avant la carte. Echo/Spectra est traité comme volatile : une pioche, une
    récupération ou une modification de défausse peut déplacer la carte Spectra et
    désactiver la condition.

## Proposed Architecture

```text
Game (règles atomiques)
        │ legal_actions / clone / apply
        ▼
PlayTurnSolver
  ├── analyse de dépendances
  ├── exploration bornée + mémoïsation
  ├── résolution canonique sûre
  └── PlayTurnCandidate[]
        │ representations macro
        ▼
Macro Neural Scorer / PPO Actor-Critic
        │ candidate choisie
        ▼
AtomicTraceReplayer
        │ PlayCard / BanishCard / ... validées une à une
        ▼
Game
```

### Solveur

Créer une couche dédiée, par exemple `shards_ai/ai/play_turn_solver.py`, sans dépendance
inverse depuis `shards_ai.game`.

Le solveur reçoit une copie du jeu au début d'un segment, jamais l'état mutable du
runner. Il explore les actions légales avec des clones du jeu. Toute transition est
donc exactement celle du moteur, y compris les cartes tirées, les mélanges et les
effets pendants.

Le solveur renvoie soit :

- un préfixe atomique canonique à appliquer immédiatement ;
- une liste de `PlayTurnCandidate` à scorer ;
- ou un fallback atomique lorsque son budget est atteint.

Une candidate contient au minimum :

```text
PlayTurnCandidate
  atomic_trace: tuple[Action, ...]
  terminal_kind: phase_end | strategic_choice | game_end
  terminal_observation: NeuralObservation
  terminal_action_representations: tuple[ActionRepresentation, ...]
  dependency_summary: PlayTurnOutcomeSummary
  atomic_action_count: int
  canonical_key: str
```

`atomic_trace` est la source de vérité pour le replay. Les observations et résumés
servent seulement à l'encodage, aux tests et au diagnostic.

### Dépendances et ordre canonique

Chaque action candidate reçoit un `DependencyDescriptor` calculé depuis l'état et la
définition de carte. Il exprime, de façon conservative :

- zones lues ou modifiées (`hand`, `discard`, `draw`, `play_zone`, champions, rivière) ;
- ressources ou flags lus/modifiés (Gems, Power, maîtrise, Focus, factions jouées,
  activation de champion, pending state) ;
- cibles créées, supprimées ou modifiées ;
- conditions susceptibles d'être activées (Union, Echo, Domination, seuil de maîtrise,
  santé, inspiration) ;
- effets non commutatifs : draw, shuffle, copie, recrutement, destruction, récupération
  et toute décision pendante.

Une condition n'est donc pas automatiquement stratégique simplement parce qu'elle est
déclarée dans la carte. Le solveur évalue la branche actuellement sélectionnée et
vérifie que sa condition est active. Il compare ensuite ses lectures de condition aux
écritures des autres actions légales. Une carte Echo active reste ainsi un choix si une
autre action disponible peut provoquer une pioche ou vider/modifier la défausse.

Deux actions ne sont commutatives que si leurs descripteurs n'ont pas de conflit et si
le moteur confirme, via l'exécution des deux ordres, une signature de résultat interne
équivalente. L'analyse sert à réduire les permutations explorées ; elle ne constitue pas
seule une preuve de règles.

L'ordre de préférence est uniquement employé pour sélectionner le représentant canonique
et explorer les candidats les plus utiles en premier :

1. effets qui augmentent ou modifient les ensembles de cibles ;
2. récupération depuis la défausse ;
3. enablers de mercenaire, champion ou combo ;
4. gains de maîtrise susceptibles de franchir un seuil ;
5. conditionnelles nouvellement activées ;
6. draws ;
7. effets indépendants ;
8. effets impossibles à activer et cleanup.

Une action de catégorie basse peut être jouée avant une action de catégorie haute si la
dépendance l'exige. En particulier, un draw, un mélange ou une récupération ne sont pas
canonisés à travers une action qui modifie leurs zones concernées.

### Signatures et regroupement

La mémoïsation interne utilise une `SolverStateKey` déterministe, plus stricte que
`NeuralObservation`. Elle doit couvrir toutes les données qui peuvent affecter les
actions futures du joueur actif : phase, ressources, zones pertinentes avec leur ordre,
champions, activations, cartes jouées, pending state, rivière, état terminal et état de
la source aléatoire si le clone l'expose de manière sérialisable.

La représentation neural masque une partie de ces détails. Elle peut réduire les
candidates présentées au modèle uniquement après que les traces aient été séparément
validées comme sans divergence stratégique future. Si cette preuve manque, les branches
restent distinctes même si leur `NeuralObservation` est identique.

Les identifiants d'instance n'ont pas de signal stratégique neural. Ils restent toutefois
dans les traces et signatures chaque fois qu'ils déterminent une cible, une zone ou une
transition moteur.

### Adaptateur de joueur et replay

Ajouter une implémentation de joueur macro séparée de `NeuralPlayer` et de
`PPOTrainingPlayer`. Elle conserve une file de `atomic_trace` : tant que le prochain
élément reste légal, elle le retourne au `GameRunner` sans nouvelle inférence. À la fin
de la trace, elle redemande au solveur les candidates du segment suivant.

Le `GameRunner` n'est pas responsable de l'abstraction et conserve son contrat : une
itération reçoit et applique une `Action` atomique. Le nouveau joueur est seul responsable
de produire cette action depuis une macro-décision antérieure.

Si la trace devient illégale, le joueur vide sa file, journalise l'anomalie et revient à
une décision fraîche ; ce cas est une erreur de solveur et doit être couvert par test.

### Modèle et données d'entraînement

Créer un schéma de record macro distinct de l'imitation atomique :

```text
MacroDecisionRecord
  schema_version
  game_id, game_seed, macro_decision_index, acting_player
  observation_before
  candidate_representations
  chosen_candidate_index
  selected_atomic_trace
  atomic_action_count
  terminal_kind
  observation_after
  outcome / sample_weight / teacher metadata
```

Chaque `candidate_representation` encode la conséquence de sa branche : état terminal
masqué, delta de ressources, maîtrise, Power, cartes ou factions jouées, flags et type
de point terminal. Elle doit aussi conserver les informations nécessaires pour distinguer
deux candidates qui mènent à la même observation globale mais ont une action stratégique
différente.

Le nouveau scorer macro est action-conditionnel : il encode l'observation initiale puis
chaque candidate de branche. Les architectures existantes ne changent pas. Le checkpoint
macro utilise une nouvelle identité d'architecture explicite et ne peut être chargé par
erreur comme un profil atomique.

Le PPO macro enregistre une décision seulement lors du choix d'une candidate. Le passage
automatique de la trace atomique ne consomme ni log-probabilité, ni estimate de valeur,
ni reward. La longueur de trace est conservée pour l'analyse, sans modifier les rewards.

## Edge Cases

- **Une seule action légale.** Le solveur la rejoue sans créer de candidate neural.
- **Bannissements multiples.** Chaque cible reste une branche jusqu'à preuve
  d'équivalence ; `SkipBanish` est une vraie alternative lorsqu'il est légal.
- **Draw et shuffle.** Ils sont traités comme dépendants des zones et ne traversent pas
  une permutation non prouvée.
- **Choix pendants.** Ils interrompent le segment ; le solveur peut les résoudre seulement
  lorsque leur unique résultat ou leur équivalence est établi.
- **Recrutement gratuit et rivière.** Les slots et identités physiques restent dans la
  trace ; le remplacement de rivière interdit toute fusion optimiste.
- **Victoire immédiate.** Une candidate qui termine la partie a `terminal_kind=game_end` ;
  aucune action suivante n'est demandée.
- **Budget atteint.** Le préfixe sûr est appliqué, puis la prochaine action atomique
  non résolue est exposée. Le résultat reste correct mais moins abstrait.
- **Carte ou effet futur non décrit.** Un descripteur inconnu interdit la canonisation et
  déclenche le fallback conservateur.

## Observability And Operations

Le solveur doit fournir des compteurs par partie et par décision :

- actions atomiques appliquées automatiquement ;
- décisions macro réellement scorées ;
- expansions, états mémoïsés et candidates retenues ;
- motifs de fallback ;
- temps de solveur séparé du temps d'inférence neural ;
- type d'effet ou de dépendance ayant empêché une fusion.

Les rapports de benchmark conserveront les décisions atomiques et macro afin de mesurer
la réduction obtenue sans confondre baisse de décisions et amélioration de qualité.
Les traces détaillées restent sous `artifacts/`, jamais dans `doc/`.

## Testing Strategy

Ajouter des tests ciblés couvrant :

- deux cartes de ressource indépendantes : une seule résolution canonique ;
- ordre dépendant via Union, Echo, Domination et seuil de maîtrise ;
- récupération de défausse avant/après draw ;
- draw avec shuffle, sans fusion abusive ;
- activation de champion ou mercenaire qui active une condition ;
- bannissement, `SkipBanish`, cibles de pending decision et recrutement gratuit ;
- égalité exacte entre l'état atteint par replay de trace et l'état simulé par le solveur ;
- déterminisme à seed identique ;
- respect immuable des quatre budgets ;
- fallback sans action illégale ni branche arbitraire ;
- compatibilité intégrale des joueurs et checkpoints atomiques existants ;
- enregistrement PPO et imitation : une seule décision par segment macro, aucune double
  attribution de reward.

Avant toute promotion, comparer le profil macro isolé au protocole neural complet : mêmes
opposants, poids, seeds et panel de parties. Publier les deltas par seed, la répartition
des décisions par type et les métriques de fallback. L'accuracy offline et le nombre de
décisions constituent des diagnostics, non une preuve de promotion.

## Rollout And Migration

1. Implémenter et tester le solveur sans modifier `NeuralPlayer`, PPO ou dataset existants.
2. Ajouter le joueur macro en inférence et mesurer la correction/reproductibilité sur des
   parties seedées.
3. Ajouter le format de collecte et le scorer macro sous une nouvelle architecture/profile.
4. Réaliser un smoke test de collecte et un entraînement court dans le checkpoint mutable.
5. Produire les analyses de couverture et benchmarks contrôlés.
6. Soumettre un candidat au panel neural complet ; ne pas modifier `active.yaml` avant
   réussite du gate.

## Files Expected To Change

- `shards_ai/ai/play_turn_solver.py` : solveur, descripteurs, signatures et budgets fixes ;
- `shards_ai/ai/neural_player.py` ou nouveau module joueur : adaptateur macro et replay ;
- `shards_ai/ai/action_representation.py` et nouveau module macro : représentation des
  consequences de branche ;
- `shards_ai/ai/rl_training.py`, `shards_ai/ai/imitation_dataset.py` et collecteurs dédiés :
  schéma et transitions macro isolés ;
- `shards_ai/ai/neural_model.py` et profils/configs dédiés : architecture macro explicite ;
- `tests/ai/` : propriétés du solveur, replay, budgets, collecte et compatibilité ;
- benchmarks et rapports neural : comptage macro, fallbacks et temps du solveur.
