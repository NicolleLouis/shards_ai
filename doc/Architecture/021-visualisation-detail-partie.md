# Visualisation détaillée d’une partie — Architecture

## Objective

Créer un mode de lecture d’une partie unique, reproductible et facile à comprendre pour analyser
chaque décision de l’IA heuristique. Le rapport doit répondre rapidement à trois questions :

1. quel était l’état avant l’action ;
2. quelles actions étaient possibles et comment étaient-elles évaluées ;
3. pourquoi l’action choisie a gagné le classement.

Le mode est destiné au debug et à l’analyse humaine, pas aux campagnes statistiques.

## Current State

`GameRunner` dispose d’un `transition_observer` appelé après chaque action. Le joueur heuristique
expose `score_action()` et `features_for_action()`, mais ne conserve pas actuellement le classement
complet d’une décision. Les scripts existants produisent surtout des agrégats multi-parties ou des
rapports de decks finaux.

## Target Behavior

Ajouter une commande dédiée, par exemple :

```bash
PYTHONPATH=. poetry run python scripts/analyze_game_detail.py \
  --seed 123 \
  --player1 heuristic \
  --player2 random \
  --output-dir analysis_output/game_detail/123
```

La commande produit :

- `report.html`, rapport principal lisible dans un navigateur ;
- `game.json`, trace structurée complète pour rechercher ou rejouer un événement.

Le HTML est autonome, sans serveur ni dépendance externe. Chaque tour est regroupé dans un bloc
repliable ; chaque action affiche l’état avant/après, l’action choisie, son score, ses features et
les alternatives légales classées. Les détails des alternatives sont repliés pour garder une lecture
courte.

## Non-Goals

- construire une interface interactive permanente ou un serveur web ;
- enregistrer automatiquement toutes les campagnes d’entraînement ;
- modifier les règles ou la décision effective des joueurs ;
- afficher des informations cachées à un joueur dans une trace destinée à ce joueur ;
- remplacer les rapports statistiques existants.

## Key Decisions

1. **Une partie, une seed.** Le rapport porte sur une seule partie déterministe et indique la seed,
   les profils, les joueurs, le résultat et les limites d’exécution.
2. **HTML d’abord, JSON ensuite.** Le HTML sert à l’analyse humaine ; le JSON conserve les mêmes
   événements de façon exploitable par des scripts.
3. **Décision avant application.** Le runner expose une observation de décision avant `Game.apply()` :
   acteur, actions légales et action choisie. Cela évite de reconstruire les alternatives après coup.
4. **Réutilisation du classement.** L’explication heuristique doit appeler le même code de classement
   que `choose_action()`, afin de ne jamais expliquer une décision avec une logique différente.
5. **Information observable seulement.** L’état affiché est l’observation reçue par le joueur ; la
   trace technique complète peut afficher l’état moteur uniquement dans un mode explicitement debug.
6. **Aucun coût normal.** Le callback de décision est désactivé par défaut et n’est créé que par le
   script d’analyse.
7. **Fast path explicite.** Quand aucun observer de décision ni de transition n’est fourni, `GameRunner`
   conserve le chemin actuel : pas de copie d’état supplémentaire, pas de sérialisation, pas de
   calcul de scores d’alternatives et pas de structure de trace allouée. Le test du callback doit être
   placé autour de la capture, pas dans une abstraction exécutée pour chaque action.

## Open Questions

- **Non bloquante — affichage RandomPlayer :** afficher `score indisponible — choix aléatoire` et
  conserver seulement la liste des actions légales.
- **Non bloquante — niveau de détail :** prévoir plus tard `--compact` et `--verbose`; la première
  version utilise un format intermédiaire repliable.
- **Non bloquante — comparaison :** un futur mode pourra comparer deux traces de seeds identiques,
  mais ce n’est pas nécessaire pour la première version.

## Proposed Architecture

### Capture

Ajouter à `GameRunner.run()` un callback optionnel de décision, distinct du callback de transition :

```text
decision_observer(observation, legal_actions, chosen_action, player_id)
```

Il est appelé après `choose_action()` et avant `Game.apply()`. Le callback reçoit des copies ou des
structures sérialisables afin que le rapport ne puisse pas modifier la partie.

### Explication heuristique

Ajouter une méthode d’explication au joueur heuristique, ou une fonction dédiée partagée avec son
classement, retournant :

- score de l’action choisie ;
- features et contributions pondérées ;
- classement des actions légales ;
- priorité terminale/létalité/phase ;
- tie-break appliqué si pertinent ;
- écart avec le meilleur candidat suivant.

Pour les actions non heuristiques, l’événement conserve l’action, la phase et les actions légales,
avec une raison générique lorsque le score n’est pas disponible.

### Format d’un événement

Chaque événement JSON contient au minimum :

```text
turn, phase, actor, action, legal_actions,
state_before, state_after,
chosen_score, chosen_features, ranked_alternatives, explanation
```

Les cartes sont représentées par leur nom, identifiant stable et zone ; les états contiennent les
ressources, la santé, la maîtrise, les champions, la main observable et les tailles de deck/défausse.

### Rapport HTML

Le rapport comporte :

1. un en-tête avec seed, profils, résultat et nombre d’actions ;
2. un résumé compact des tours ;
3. un bloc `<details>` par tour ;
4. une ligne lisible par action : `acteur — phase — action — score — raison` ;
5. un tableau repliable des alternatives et de leurs contributions ;
6. un état avant/après compact.

Les valeurs numériques sont arrondies pour la lecture, tandis que `game.json` conserve la précision
utile au debug.

## Data Model

Aucun changement persistant du moteur. Les types de trace peuvent être des dataclasses locales au
module d’analyse et sérialisés immédiatement en JSON. Les fichiers générés restent hors de `doc/`.

## Backend Flow

1. Parser la seed, les types de joueurs, les profils et les limites.
2. Construire exactement une partie avec `Game.new(seed=...)` et des sources aléatoires dérivées.
3. Installer le decision observer uniquement pour cette exécution.
4. Capturer la décision avant chaque transition.
5. Exécuter la partie normalement avec `GameRunner`.
6. Ajouter le résultat terminal à la trace.
7. Écrire JSON puis HTML dans `analysis_output/game_detail/` par défaut.

Le mode normal ne passe pas par ce flux de capture : l’observer vaut `None` et le runner exécute son
chemin rapide existant. La régression de performance doit être vérifiée par un benchmark Heuristic
vs Heuristic avant/après l’ajout.

Une exception contient la dernière décision et la phase pour faciliter le diagnostic, puis est
relancée en mode strict.

## Frontend Flow

Sans frontend applicatif. Le navigateur ouvre un fichier HTML autonome ; les sections de tours et
d’alternatives utilisent les éléments HTML natifs repliables.

## Authorization And Feature Gates

Sans objet. Le mode est local et explicite.

## Observability And Operations

La sortie console indique la seed, le résultat, le nombre d’actions et les chemins `report`/`json`.
Les rapports générés ne doivent pas être déposés dans `doc/` ni versionnés automatiquement.

## Edge Cases

- joueur Random : classement heuristique absent mais choix et actions légales présents ;
- partie nulle ou limite d’actions : résultat et dernière position affichés ;
- erreur de règle : événement courant conservé et erreur affichée clairement ;
- action unique : alternatives indiquées comme inexistantes ;
- tie-break : détail du critère utilisé affiché ;
- information cachée : ne pas afficher la main ou le deck privé de l’adversaire dans l’observation
  du joueur analysé.

## Testing Strategy

- vérifier que le callback de décision ne change pas le résultat d’une partie ;
- vérifier l’ordre décision → transition et l’état avant/après ;
- vérifier une trace Heuristic vs Random et une trace Heuristic vs Heuristic ;
- vérifier les scores et alternatives d’une décision heuristique connue ;
- vérifier la sérialisation JSON et la présence des sections HTML principales ;
- vérifier les limites de partie, draw et exception stricte ;
- exécuter la suite complète.

## Rollout And Migration

Aucun changement de comportement par défaut. Le mode est ajouté comme outil local indépendant et peut
être utilisé immédiatement sur une seed connue. Les futures extensions (`--compact`, comparaison de
traces) resteront compatibles avec `game.json`.

## Files Expected To Change

- `shards_ai/game/runner.py` : callback de décision optionnel ;
- `shards_ai/ai/heuristic_player.py` : classement explicable partagé avec la décision ;
- `scripts/analyze_game_detail.py` : exécution et génération des sorties ;
- `tests/game/test_runner.py` : ordre et absence d’effet du callback ;
- `tests/analysis/test_game_detail.py` : trace JSON et HTML ;
- `doc/Current state/Analysis.md` : commande et format disponibles.
