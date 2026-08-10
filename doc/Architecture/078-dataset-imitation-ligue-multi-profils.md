# Dataset d'imitation de ligue multi-profils — Architecture

## Objective

Produire une collecte reproductible des décisions prises par Random, Heuristic V7, Heuristic
V8 et les profils neural V1 à V4, sur des matchups équilibrés. La collecte doit permettre de
construire plusieurs datasets à partir des mêmes parties : contrôle complet non pondéré, dataset
avec pondération teacher et résultat, et variante winner-only.

Le dataset doit rester compatible avec l'architecture action-conditionnelle actuelle : observation
neural masquée, actions légales représentées, scores du teacher lorsqu'ils existent et action
choisie.

## Current State

`shards_ai/ai/imitation_dataset.py` collecte actuellement les décisions des seuls joueurs
heuristiques utilisés comme teachers. Random peut jouer comme adversaire, mais ses décisions ne
sont pas enregistrées. Les profils neural sont chargés par `NeuralPlayer` depuis un checkpoint et
reçoivent déjà `NeuralObservation`.

L'entraînement utilise `action_representations`, `heuristic_scores` et
`chosen_action_index`. Les scores ne sont pas disponibles pour Random ; les décisions Random
doivent donc rester un signal de choix, avec un poids faible, sans créer une préférence paire-à-
paire artificielle.

## Target Behavior

Un nouveau script de collecte construit une ligue à partir de spécifications explicites de joueurs.
Chaque paire de profils distincts est jouée dans les deux orientations afin d'équilibrer le siège.
À chaque décision, le script enregistre la vue neural masquée du joueur actif, les actions légales,
l'action choisie, les scores disponibles, le profil du joueur et le profil adverse.

La même collecte écrit plusieurs sorties cohérentes :

- `control_full_unweighted` : toutes les décisions, poids `1.0` ;
- `weighted_moderate` : toutes les décisions avec `poids_teacher × poids_resultat` ;
- `winner_only` : décisions des joueurs gagnants uniquement, pour ablation.

Le résultat final ne doit jamais être exposé à l'observation. Il reste une métadonnée d'analyse et
un facteur de poids appliqué après la partie.

## Non-Goals

- modifier le moteur, les règles ou le masque d'information ;
- modifier un checkpoint stable ou le profil neural actif ;
- remplacer le générateur heuristique historique ;
- transformer les décisions Random en scores de préférence fiables ;
- mélanger les décisions d'une même partie entre les splits train/validation/test ;
- filtrer définitivement les perdants dans le dataset de contrôle.

## Key Decisions

1. Le nouveau collecteur est séparé du générateur historique pour préserver sa compatibilité.
2. Les matchups distincts sont joués dans les deux sens ; l'auto-match est désactivé par défaut.
3. Les joueurs sont spécifiés par type et identifiant stable : `random`, profil heuristique ou
   checkpoint neural.
4. Les observations sont sérialisées à partir de `NeuralObservation` et les actions à partir de
   `representation_for_neural_action`, afin de respecter le point de vue masqué.
5. Les scores heuristiques et neural sont stockés lorsqu'ils sont calculables. Pour Random,
   `teacher_scores` est nul et seule la loss d'action choisie peut être utilisée.
6. Les poids modérés sont : Random `0.10`, neural V1 `0.50`, V2 `0.60`, V3 `0.75`, V4 `0.90`,
   Heuristic V7 `1.00`, Heuristic V8 `1.50`; résultat : défaite `0.75`, nul `1.00`, victoire
   `1.25`.
7. Le poids est écrit dans chaque record sous `sample_weight`, avec une moyenne globale normalisée
   à `1.0` par dataset. Il ne doit pas être implémenté par duplication de lignes.
8. Le contrôle et les variantes utilisent exactement les mêmes parties et les mêmes splits par
   `game_id`.

## Open Questions

- La taille et la couverture suffisantes seront évaluées après le smoke test puis sur la collecte
  complète ; aucun seuil de volume n'est codé dans le collecteur.
- Les scores neural sont des logits du checkpoint et ne sont pas assimilés aux scores heuristiques
  pour une analyse de qualité sans métrique dédiée.

## Proposed Architecture

Le script définit une `LeaguePlayerSpec` et construit pour chaque spec le joueur correspondant.
Une instance de scorer neural est chargée une fois par checkpoint et réutilisée entre les parties.
Le hook `GameRunner.decision_observer` reçoit la décision réelle, reconstruit l'observation neural
si nécessaire, représente les actions visibles, puis calcule les annotations du player actif.

Les parties sont générées avec une seed dérivée de la seed de campagne et de l'index de matchup.
Chaque record reçoit un `game_id` stable. Les sorties sont écrites en streaming dans des fichiers
temporaires puis renommées atomiquement. Un manifeste par variante contient les profils, matchups,
seeds, comptes, exclusions, pondérations et fingerprint du catalogue de cartes.

## Data Model

Les champs d'entrée neural restent ceux du schéma `NeuralObservation`. Les annotations ajoutées
sont :

```text
teacher_type
teacher_profile_id
teacher_checkpoint
opponent_type
opponent_profile_id
teacher_scores
teacher_scores_available
sample_weight
final_outcome
```

Les champs historiques `game_id`, `game_seed`, `decision_index`, `acting_player`,
`action_representations`, `chosen_action_index` et `chosen_action` sont conservés.

## Performance And Operations

La collecte est CPU-bound et chaque décision neural effectue une inférence. Les scoreurs doivent
être chargés une seule fois ; les parties restent séquentielles et déterministes. Le JSONL est
écrit en streaming pour éviter de garder plusieurs centaines de milliers de décisions en mémoire.
Les artefacts sont produits sous `artifacts/`, jamais sous `doc/`.

## Edge Cases

- une partie terminée en nul n'est pas incluse dans `winner_only` ;
- une erreur de partie est soit fatale, soit exclue avec `--continue-on-error` ;
- une action non représentable depuis l'observation masquée arrête la partie en mode strict ;
- les scores Random sont absents et ne doivent pas être remplacés par des scores arbitraires ;
- les checkpoints neural incompatibles avec le catalogue ou l'architecture échouent avant collecte.

## Testing Strategy

- génération déterministe avec une petite ligue et une seed fixe ;
- présence des décisions des deux côtés et respect du masque adverse ;
- chargement et réutilisation des checkpoints neural ;
- absence de scores pour Random et poids correctement calculés ;
- variantes contrôle/pondérée/winner-only issues des mêmes records ;
- manifeste complet et renumérotation stable des décisions par partie ;
- smoke test de deux à quatre parties avec sortie temporaire.

## Rollout And Migration

Le script est utilisé d'abord avec un petit nombre de parties. Après analyse de la couverture,
la collecte complète est lancée par l'utilisateur dans une autre fenêtre. Aucun entraînement ni
activation de checkpoint n'est effectué par ce collecteur.

## Files Expected To Change

- `scripts/generate_imitation_league_dataset.py` : nouveau point d'entrée ;
- `shards_ai/ai/league_dataset.py` : collecte, annotations et variantes ;
- `shards_ai/ai/neural_training.py` : lecture optionnelle du poids de record ;
- `scripts/train_neural_imitation.py` : application de `sample_weight` ;
- `tests/ai/test_league_dataset.py` : tests de collecte et de pondération.
