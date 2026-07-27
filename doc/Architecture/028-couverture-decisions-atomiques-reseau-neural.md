# Couverture des décisions atomiques pour le réseau neural

## Objectif

Garantir que le futur réseau neural est appelé à chaque choix réel du joueur, avec exactement le
même ensemble d'actions légales que les joueurs existants. Cette étape constitue le premier travail
de la roadmap d'imitation ; elle ne construit pas encore le modèle ni le dataset.

## État actuel

Le moteur expose les décisions via `Game.legal_actions()` et applique les transitions via
`Game.apply()`. Les types d'actions actuellement présents sont :

- `PlayCard` ;
- `ActivateChampion` ;
- `GainMastery` ;
- `PassPlayPhase` ;
- `BuyCard` ;
- `RecruitMercenary` ;
- `StopBuying` ;
- `AssignPower` ;
- `BanishCard` et `SkipBanish` ;
- `RecruitFreeCard` ;
- `ChoosePendingDecision`.

Les décisions intermédiaires sont représentées par `pending_decision`, `pending_banishes` et
`pending_free_recruit_cost`. Le runner appelle déjà le joueur avec l'observation et la liste des
actions légales avant chaque transition.

`GainMastery` représente le paiement d'une gemme pour gagner une maîtrise. Il est disponible en
phase `PLAY` lorsqu'il n'a pas encore été utilisé et que le joueur possède une gemme.

## Comportement cible

Chaque appel à `Game.legal_actions()` qui retourne une liste non vide doit correspondre à une
décision pouvant être déléguée au réseau. Le réseau ne choisira jamais une action en dehors de cette
liste et ne modifiera pas lui-même l'état du jeu.

Les décisions intermédiaires sont des décisions atomiques à part entière. Par exemple, le réseau
doit pouvoir choisir séparément une carte à bannir, une cible d'attaque, un champion à détruire ou
une carte dont l'effet doit être copié.

## Hors périmètre

- encoder les observations pour le réseau ;
- générer le dataset ;
- définir ou entraîner le modèle ;
- modifier les règles du jeu sans preuve qu'une décision actuelle n'est pas exposée ;
- introduire le reinforcement learning.

## Décisions clés

- `Game.legal_actions()` reste la source de vérité des actions disponibles.
- `Game.apply()` reste la source de vérité de la validation et de la transition.
- Le réseau sera appelé après chaque transition qui produit une nouvelle décision, pas au début du
  tour pour planifier plusieurs actions.
- Les égalités de score seront gérées par le joueur neural ; elles ne changent pas le moteur.
- Les actions restent des objets structurés, et non des indices opaques dans une liste globale.
- `GainMastery` doit rester explicitement distinguable des autres actions de phase `PLAY`.

## Architecture proposée

L'audit s'appuie sur trois niveaux :

1. le catalogue des classes d'action dans `shards_ai/game/actions.py` ;
2. les branches de génération dans `shards_ai/game/game.py:legal_actions()` ;
3. les branches de validation et d'application dans `shards_ai/game/game.py:apply()`.

Pour chaque état de décision, les tests vérifieront que :

- la liste retournée contient toutes les options légales ;
- chaque action retournée est acceptée par `apply()` ;
- une action illégale est refusée ;
- les décisions en attente bloquent toute action d'un autre type ;
- la transition expose ensuite la prochaine décision atomique ;
- le runner transmet bien l'observation et les actions avant l'application.

Si l'audit révèle une décision interne non exposée, elle devra être transformée en action publique
avant de commencer le dataset. Si aucune lacune n'est trouvée, l'implémentation se limitera à
renforcer les tests et à documenter la couverture.

## Cas à tester

- phase `PLAY` : jouer une carte, activer un champion, gagner de la maîtrise, passer ;
- phase `BUY` : acheter, recruter un mercenaire, arrêter les achats ;
- phase `ATTACK` : attaquer le joueur adverse ou un champion légal ;
- bannissement obligatoire ou optionnel ;
- recrutement gratuit dans la rivière ;
- choix de cible ou de carte après un effet ;
- action non légale pendant une décision en attente ;
- partie terminée sans action légale ;
- seed identique et même état donnant la même liste d'actions.

## Questions ouvertes

Aucune question bloquante pour l'audit initial. Les éventuelles décisions non exposées devront être
traitées comme des écarts constatés par les tests, et non inventées à partir des seules classes
d'action.

## Stratégie de validation

Exécuter les tests ciblés du moteur et du runner, puis la suite complète disponible. Vérifier aussi
que la documentation reste exclusivement en Markdown et que le diff ne contient que la roadmap et
le document d'architecture de cette étape.

## Fichiers attendus

- `shards_ai/game/actions.py` — inspection, modification seulement si une action publique manque ;
- `shards_ai/game/game.py` — inspection et éventuelle correction ciblée ;
- `shards_ai/game/runner.py` — vérification du point d'appel ;
- `tests/game/test_game.py` — scénarios des actions légales ;
- `tests/game/test_runner.py` — garantie de l'appel atomique ;
- `doc/Roadmap.md` — statut de l'étape ;
- `doc/Current state/Game engine.md` — mise à jour uniquement si le comportement change.
