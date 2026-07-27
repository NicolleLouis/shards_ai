# Seuil d’achat explicite — Architecture

## Objective

Empêcher le joueur heuristique d’acheter automatiquement une carte dont la valeur évaluée est trop
faible. La décision d’achat doit comparer les cartes à un seuil explicite, indépendant du score
technique de `StopBuying`.

Le problème observé est le suivant : `StopBuying` vaut actuellement `-0.9` à cause de
`phase_progress` et `action_penalty`. Une carte évaluée à `0.0`, comme `Drone Kiln` lorsque
`card_acquisition_weights.gems_produced` vaut `0.0`, est donc quand même achetée.

## Current State

`HeuristicPlayer.choose_action()` classe toutes les actions légales avec un tuple comprenant les
indicateurs de victoire, le score pondéré, la priorité de phase et l’ordre d’apparition. Pendant la
phase `BUY`, `StopBuying` est seulement une action supplémentaire avec un score de progression de
phase et de pénalité d’action.

Les poids économiques d’une carte achetée sont dans `CardAcquisitionWeights`, tandis que le score
final de l’action utilise `HeuristicWeights`. L’optimiseur sait déjà faire varier les deux familles,
mais aucun champ ne représente le minimum acceptable pour une acquisition.

## Target Behavior

Ajouter `buy_threshold` aux `HeuristicWeights`. Pendant une phase `BUY` normale :

1. calculer le score normal des actions d’achat durable et de recrutement mercenaire ;
2. appliquer le seuil uniquement aux actions `BuyCard`, dont l'acquisition dilue le deck ;
3. conserver uniquement les actions `BuyCard` dont le score est strictement supérieur à
   `buy_threshold` ;
4. laisser les actions `RecruitMercenary` dans le classement sans filtrage par seuil ;
5. si au moins une action admissible existe, choisir la meilleure selon le classement habituel ;
6. sinon choisir `StopBuying`.

Le seuil ne modifie pas le score affiché d’une carte. Il agit comme une règle d’admissibilité avant
le classement. La valeur d’achat reste donc lisible et entraînable séparément du conservatisme du
deck.

## Non-Goals

- Modifier la valeur intrinsèque ou prospective des cartes.
- Remplacer `card_acquisition_weights.gems_produced`.
- Appliquer le seuil aux recrutements mercenaires, qui ne diluent pas le deck durable.
- Appliquer le seuil aux actions obligatoires de `RecruitFreeCard` lorsqu’un choix gratuit est en
  attente et qu’aucune action de fin n’est légale.
- Modifier les phases PLAY, ATTACK ou CLEANUP.

## Key Decisions

1. **Famille du paramètre.** `buy_threshold` appartient à `HeuristicWeights`, car il contrôle une
   politique de décision et non la valeur intrinsèque d’un effet de carte.
2. **Seuil limité aux achats durables.** La comparaison utilise le même score pondéré que le
   classement de `BuyCard`, mais ne filtre jamais `RecruitMercenary` : un mercenaire ne reste pas
   dans le deck et ne crée donc pas le problème de dilution ciblé.
3. **Seuil strict.** Une carte dont le score est exactement égal au seuil n'est pas admissible. Le
   profil initial recommandé utilise `0.0`, ce qui refuse les cartes à score nul ou négatif et
   réserve l'achat aux valeurs réellement positives ; une campagne pourra ensuite déterminer si un
   seuil positif est préférable.
4. **Stop explicite.** `StopBuying` n’est pas comparé au seuil et conserve son score générique pour
   le reporting. Il devient le fallback lorsque tous les achats échouent au seuil.
5. **Actions obligatoires.** Si la liste légale ne contient pas `StopBuying` ou si la phase impose
   une résolution, le seuil ne filtre pas les actions ; le moteur doit toujours recevoir une action
   légale.
6. **Optimisation progressive.** Le champ est ajouté aux champs optimisables de `HeuristicWeights`,
   avec une borne dédiée, puis pourra être testé seul avant d’être inclus dans une campagne combinée.

## Open Questions

- **Décision enregistrée — borne initiale :** faire varier `buy_threshold` de `0.0` à `2.0` par pas
  de `0.25`. Cette plage correspond mieux aux scores d'achat observés et offre une résolution
  suffisante sans tester des seuils probablement trop conservateurs.
- **Décision enregistrée :** les mercenaires sont hors seuil. Un seuil distinct ne sera envisagé que
  pour une autre notion de qualité, pas pour corriger la dilution du deck.

## Proposed Architecture

Ajouter une fonction pure de politique, par exemple `_filter_purchase_actions_by_threshold()`, dans
`HeuristicPlayer`. Elle reçoit l’observation et les actions légales, calcule les features et scores
des `BuyCard`, puis retourne les achats durables admissibles avec les mercenaires non filtrés ; elle
retourne `StopBuying` si aucune action d'achat admissible n'existe.

Le classement existant reste ensuite responsable des priorités terminales, de la létalité et des
départages. La fonction ne doit jamais reconstruire une valeur différente de `features_for_action()`.

Le rapport de partie détaillée affichera aussi le seuil utilisé et, pour chaque ligne d’achat, le
statut `au-dessus`, `sous le seuil` ou `non achetable`. Cela permettra de distinguer une carte faible
d’une carte simplement trop chère.

## Data Model

Le profil YAML reçoit un champ dans `weights` :

```yaml
weights:
  buy_threshold: 0.0
```

`ActionFeatures` ne reçoit pas de nouvelle feature : le seuil est une règle de filtrage, pas un
effet d’action. Les anciens profils sans le champ utilisent la valeur par défaut documentée.

## Backend Flow

1. `Game.legal_actions()` produit les actions légales.
2. `HeuristicPlayer` détecte une phase `BUY` normale et la présence d’achats filtrables.
3. Les scores des `BuyCard` et `RecruitMercenary` sont calculés avec le chemin existant.
4. Les candidats sous `buy_threshold` sont exclus du classement.
5. Si aucun candidat n’est admissible, `StopBuying` est choisi lorsqu’il est légal.
6. Sinon, le classement actuel sélectionne le meilleur candidat restant.

Les actions forcées (`RecruitFreeCard`, résolution de décision ou choix de banissement) restent
inchangées. Une trace de diagnostic doit indiquer le nombre de candidats avant/après filtrage.

## Observability And Operations

Le profil sauvegardé conserve la valeur du seuil. Les rapports d’optimisation doivent enregistrer le
champ dans les candidats, l’historique et la validation. Le rapport d’une partie doit afficher le
seuil dans le bloc d’achat.

## Edge Cases

- aucune carte ne dépasse le seuil : `StopBuying` ;
- une carte exactement au seuil : achat autorisé ;
- une seule carte achetable mais sous le seuil : arrêt ;
- `StopBuying` absent : ne jamais retourner une liste vide ;
- phase d’achat avec `RecruitFreeCard` obligatoire : ne pas filtrer ;
- action terminale ou létale : conserver les priorités existantes avant le seuil lorsque cela est
  pertinent pour une future extension.

## Testing Strategy

- Drone Kiln à score `0.0` avec seuil `0.0` : `StopBuying` ;
- Drone Kiln à score `0.0` avec seuil `0.1` : `StopBuying` ;
- carte à score exactement égal au seuil : carte écartée ;
- meilleure carte sous le seuil et carte secondaire au-dessus : carte secondaire sélectionnée ;
- achat durable sous le seuil avec mercenaire légal : le mercenaire reste sélectionnable ;
- mercenaire seul disponible sous le seuil : le mercenaire est sélectionné ;
- recrutement gratuit obligatoire non filtré ;
- ancien profil sans `buy_threshold` chargé avec la valeur par défaut ;
- optimisation et sérialisation du nouveau champ ;
- non-régression complète du moteur et benchmark Heuristic vs Heuristic.

## Rollout And Migration

Ajouter le champ au profil actif avec la valeur initiale `0.0`, puis lancer une campagne courte sur
`buy_threshold` seul. Une campagne combinée ultérieure pourra tester ses interactions avec
`gems_produced`, `card_acquisition_value`, `deck_thinning` et `durable_replay_factor`.

## Files Expected To Change

- `shards_ai/ai/heuristic_evaluator.py` : champ `HeuristicWeights.buy_threshold` ;
- `shards_ai/ai/heuristic_player.py` : filtrage explicite des achats ;
- `shards_ai/optimization/heuristic.py` : bornes et optimisation du champ ;
- `shards_ai/ai/heuristic_profiles.py` et profils YAML : chargement/sérialisation ;
- `scripts/analyze_game_detail.py` : affichage du seuil et du statut de filtrage ;
- tests IA, optimisation et analyse ;
- `doc/Current state/Heuristic player.md` et `doc/Current state/Analysis.md`.
