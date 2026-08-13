# Forecast de l'horizon d'une partie — Architecture

## Objective

Construire une expérimentation supervisée capable d'estimer le nombre de tours futurs que le joueur
actif pourra encore jouer avant la fin d'une partie, à partir d'informations disponibles au moment
de la décision.

L'expérimentation doit fournir deux références comparables :

1. un baseline qui ne connaît que `turn_number` ;
2. une première architecture enrichie avec l'état des deux joueurs et les comptes factionnels
   globaux de leurs decks.

Le résultat attendu n'est pas encore une modification de la politique d'achat. Il doit d'abord
mesurer si le forecast est suffisamment précis et calibré pour être utilisé plus tard dans le
calcul de valeur d'une carte achetée.

## Current State

Le moteur stocke `GameState.turn_number` et termine les parties selon ses règles de victoire. Les
parties sont reproductibles avec une seed et les états sont observables via
`shards_ai/game/observation.py`.

`NeuralObservation` expose déjà :

- la vie, la maîtrise et les ressources des deux joueurs ;
- `turn_number` ;
- `owned_card_counts` pour le joueur actif et l'adversaire ;
- les cartes connues dans la main, la rivière et les autres zones publiques.

`owned_card_counts` est construit sur l'ensemble des zones possédées. Il ne révèle donc pas la zone
de chaque carte adverse, mais permet de calculer une composition globale du deck adverse lorsque
ce contenu fait partie de l'information connue par les règles du projet.

Le joueur neural actif est actuellement V005, avec un contrat macro/action conditionnel. Aucun
forecast d'horizon n'est encore une entrée de ce contrat et aucune pénalité d'achat ne doit être
ajoutée pendant cette expérimentation.

## Target Behavior

Chaque décision observée dans une partie terminée produit une ligne de dataset contenant :

```text
features observables au tour t
target_remaining_active_player_turns = future_turn_starts_for_active_player
terminal outcome
game_id et seed
```

Deux vues du même dataset sont générées :

### Baseline

Le baseline utilise uniquement :

```text
turn_number
```

Il doit être entraîné et évalué avec le même protocole de split que le modèle enrichi. Il mesure la
précision obtenue par une simple connaissance de l'avancement de la partie.

### Première architecture enrichie

L'entrée V1 utilise :

```text
- turn_number ;
- health du joueur actif et de l'adversaire ;
- mastery du joueur actif et de l'adversaire ;
- nombre total de cartes possédées par le joueur actif et l'adversaire ;
- quatre comptes factionnels globaux du joueur actif ;
- quatre comptes factionnels globaux de l'adversaire.
```

Les quatre factions sont `maquis`, `spectra`, `homodeus` et `order`. Les cartes neutres ne sont
pas attribuées artificiellement à une faction ; leur présence reste reflétée par le nombre total
de cartes possédées.

Le dataset peut conserver les champs `player_1_*` et `player_2_*` pour audit et reproductibilité.
Le vecteur présenté au modèle est toutefois orienté selon `active_player` et `opponent`, avec un
champ `active_player_id` si nécessaire pour reconstruire l'ordre absolu. Cela évite que le réseau
apprenne un biais de siège plutôt qu'une relation de jeu. Une variante P1/P2 stricte pourra être
mesurée comme ablation, mais n'est pas le contrat initial.

La cible est uniquement une régression entière du nombre de futurs tours du joueur actif. Le tour
actuel est exclu. Comme les joueurs jouent alternativement, cette valeur correspond au nombre de
tours globaux futurs divisés par deux, avec la convention entière appropriée. Pour éviter toute
ambiguïté au dernier tour, le générateur la calcule en comptant les futures transitions où le même
`player_id` redevient actif, plutôt qu'en déduisant uniquement une valeur arithmétique.

## Non-Goals

- modifier `Game.legal_actions()` ou `Game.apply()` ;
- modifier le joueur neural V005 ou son checkpoint actif ;
- appliquer une pénalité aux `BuyCard` ou forcer les `RecruitMercenary` ;
- utiliser les cartes futures, l'ordre réel des decks ou le résultat final comme feature ;
- utiliser des zones adverses qui ne sont pas observables ;
- conclure à la crédibilité du forecast avec une seule métrique offline ;
- entraîner simultanément un modèle d'horizon et une nouvelle politique de jeu.

## Key Decisions

1. Le premier objectif est une mesure de crédibilité, pas une amélioration immédiate du joueur.
2. Le baseline et V1 sont générés depuis les mêmes trajectoires et les mêmes `game_id`, afin que la
   comparaison mesure l'apport réel des features.
3. Le contenu factionnel est agrégé sur l'ensemble du deck possédé, jamais par zone pour le modèle
   adverse. La source est `owned_card_counts` et le catalogue des cartes.
4. Les comptes sont ordonnés de façon stable par `PLAYABLE_FACTIONS` et normalisés avec une borne
   explicite. La borne est fixée à 100.
5. Les entrées d'inférence sont relatives `active/opponent`. Les champs P1/P2 peuvent rester dans
   le dataset brut pour audit et permettre une ablation de représentation.
6. Le dataset est séparé par partie (`game_id`), jamais par lignes individuelles. Toutes les lignes
   d'une partie doivent rester dans un seul split.
7. Le dataset doit couvrir un éventail large de joueurs des deux côtés : joueurs aléatoires,
   heuristiques, versions neural disponibles et, si applicable, variantes de recherche ou de
   contrôle. Les profils et checkpoints utilisés sont équilibrés autant que possible et leur
   provenance complète est conservée.
8. Les profils doivent être permutés entre les deux sièges lorsque le protocole le permet, afin
   d'éviter un artefact `player_1/player_2` et de distinguer composition et rôle.
9. Les parties produites par des recettes différentes ne sont pas mélangées silencieusement.
10. L'erreur de régression seule ne suffit pas : l'évaluation doit inclure une erreur pondérée sur
   les horizons courts et une comparaison à la baseline.
11. Une intégration future dans la pénalité d'achat ne sera envisagée que si V1 améliore la baseline
   sur un split de parties tenu à l'écart et reste fiable sur des parties complètes contre les
   adversaires du panel.
12. Les artefacts d'expérience restent sous `artifacts/`; `doc/` ne reçoit que cette architecture
    et les éventuels documents Markdown d'état courant explicitement demandés.

## Open Questions

1. **Variété des parties d'entraînement — décision prise.** Le dataset initial couvre un éventail
   large de joueurs des deux côtés, avec plusieurs profils, stratégies et seeds. Il n'est pas
   limité à V005 ni à la performance habituelle d'un seul joueur. La matrice de collecte documente
   les profils de chaque côté, les permutations de rôles, les seeds et le nombre de parties par
   combinaison. Le split reste strictement effectué par `game_id`.
2. **Définition de la fin — décision prise.** Le tour courant est exclu. La cible est le nombre de
   futurs tours du joueur actif, donc le nombre de futures occurrences de son `player_id` avant la
   terminaison. Elle est généralement égale à `floor(total_global_turns_remaining / 2)` dans une
   alternance régulière.
3. **Modèle de sortie — décision prise.** Régression seule pour cette première expérience. La
   crédibilité sera évaluée par MAE, RMSE, biais et erreurs conditionnelles sur les horizons courts.
4. **Borne des comptes — décision prise.** La borne de normalisation factionnelle est fixée à 100,
   avec clipping explicite à cette valeur.

## Proposed Architecture

### Générateur de dataset

Ajouter un script expérimental, par exemple :

```text
scripts/generate_horizon_dataset.py
```

Le script rejoue ou observe des parties avec des seeds explicites. À chaque état avant décision,
il sérialise :

```json
{
  "schema_version": 1,
  "game_id": "...",
  "seed": 123,
  "turn_number": 7,
  "active_player_id": "player_1",
  "features": {
    "active_health": 32,
    "opponent_health": 18,
    "active_mastery": 4,
    "opponent_mastery": 7,
    "active_owned_card_count": 12,
    "opponent_owned_card_count": 15,
    "active_faction_counts": [3, 2, 0, 1],
    "opponent_faction_counts": [1, 0, 5, 2]
  },
  "target_remaining_active_player_turns": 2
}
```

Les champs bruts P1/P2 peuvent être conservés en métadonnées de ligne si la représentation
relative est utilisée pour le modèle. Les lignes doivent être déterministes pour une même partie,
configuration et seed.

Le générateur doit refuser ou signaler les parties non terminées, les parties arrêtées par une
limite de tours et les états dont le winner ou le tour terminal ne peut pas être établi. Ces lignes
ne doivent pas être utilisées comme targets normales.

### Dataset baseline

Le baseline est une projection du dataset canonique, pas une seconde collecte de parties :

```text
artifacts/horizon_forecast/horizon_v1_dataset.jsonl
artifacts/horizon_forecast/horizon_v1_baseline.jsonl
```

La projection ne conserve comme entrée que `turn_number`, tout en gardant la même cible, le même
`game_id` et le même split. Cela garantit une comparaison strictement appariée.

### Modèles

Créer un petit module supervisé dédié, par exemple :

```text
shards_ai/ai/horizon_forecast.py
```

Il contient :

- une dataclass de features et une validation de schéma ;
- l'encodeur déterministe des comptes factionnels ;
- un MLP baseline `turn_number -> remaining_active_player_turns` ;
- un MLP V1 pour les features enrichies ;
- la sauvegarde des métadonnées, de la recette et du fingerprint du dataset ;
- la prédiction bornée de `remaining_active_player_turns`.

Le baseline et V1 doivent avoir le même budget d'entraînement, le même optimizer, les mêmes seeds
et un early stopping fondé uniquement sur le split de validation. Le test final reste intouché
jusqu'à la fin de la comparaison.

### Séparation avec le joueur neural

Le forecast reste un service analytique indépendant. Aucun appel depuis `MacroNeuralPlayer` ou
`build_neural_player()` n'est introduit dans cette architecture. Une future pénalité pourra
consommer un artefact de forecast validé, mais son activation devra être une nouvelle architecture
documentée et un nouveau protocole de parties.

## Data Model

Pas de modification des règles ni de l'état métier. Le format JSONL expérimental contient :

- `schema_version` ;
- identité de partie, seed, profil et checkpoint source ;
- `turn_number` et `active_player_id` ;
- features relatives et, si nécessaire, snapshots P1/P2 d'audit ;
- `target_remaining_active_player_turns` ;
- statut de terminalité et outcome.

Les artefacts de modèle doivent contenir au minimum :

```text
feature_set
dataset_fingerprint
split_seed
training_seed
model architecture
normalization bounds
target definition
source profiles/checkpoints
```

## Backend Flow

1. Construire une matrice de collecte couvrant plusieurs profils de joueurs des deux côtés, avec
   permutations de sièges et seeds explicites.
2. Collecter des parties terminées avec seeds et provenance.
3. Construire une ligne canonique par état de décision.
4. Calculer le target uniquement après la terminaison de la partie.
5. Dédupliquer ou rejeter explicitement les doublons de `(game_id, decision_index)`.
6. Créer un split par `game_id`, en contrôlant la représentation des profils, matchups et longueurs.
7. Projeter la vue baseline.
8. Entraîner baseline et V1 séparément avec la même recette contrôlée.
9. Évaluer sur validation puis sur test tenu à l'écart.
10. Produire un rapport comparant précision et erreurs par horizon, profil et matchup.

Le dataset ne doit pas être écrit dans `doc/` et les checkpoints expérimentaux doivent respecter
la convention du dépôt : un seul checkpoint neural de travail mutable à la fois, sans créer de
checkpoints candidats arbitraires sous `artifacts/`.

## Observability And Operations

Le rapport doit inclure :

- nombre de parties, lignes et parties par split ;
- distribution des tours et des targets ;
- distribution des comptes factionnels ;
- MAE, RMSE et biais moyen ;
- MAE conditionnelle pour `remaining_active_player_turns <= 1`, `<= 3` et `> 3` ;
- MAE et biais conditionnels sur les horizons courts ;
- comparaison baseline/V1 sur les mêmes parties ;
- erreurs par profil, adversaire, tour et longueur réelle de partie.

Un forecast est considéré comme potentiellement exploitable seulement si V1 bat la baseline sur le
test tenu à l'écart, sans dégradation majeure sur les parties courtes, et si ses erreurs sont
calibrées. Une bonne MAE globale dominée par les parties longues ne suffit pas.

## Edge Cases

- partie terminée avant la première décision exploitable ;
- partie arrêtée par `max_turns` ;
- `remaining_active_player_turns = 0` au dernier état ;
- deck sans carte d'une faction ;
- cartes neutres ;
- compte dépassant la borne de normalisation ;
- définition de carte absente du catalogue ;
- partie avec actions automatiques et plusieurs décisions au même tour ;
- duplications de lignes lors d'un rejeu ;
- dataset contenant des parties issues du checkpoint évalué ;
- cible négative ou tour terminal incohérent.

## Testing Strategy

- vérifier le calcul des huit comptes factionnels à partir de `owned_card_counts` ;
- vérifier que l'ordre des tuples n'influence pas le vecteur ;
- vérifier que les neutres ne sont pas attribués à une faction ;
- vérifier la distinction actif/adversaire et l'ablation P1/P2 ;
- vérifier le calcul de `target_remaining_active_player_turns` sur une partie courte déterministe,
  en excluant le tour courant et en comptant les futures occurrences du joueur actif ;
- vérifier le rejet des parties non terminées ;
- vérifier l'absence de fuite de zones adverses ;
- vérifier la projection baseline et son identité de `game_id`/target avec V1 ;
- vérifier le split par partie ;
- vérifier la sauvegarde/restauration des modèles et métadonnées ;
- vérifier des prédictions finies et dans les bornes attendues ;
- ajouter des tests de non-régression du moteur et de l'observation existante.

## Rollout And Migration

Cette architecture ne modifie aucun profil actif, checkpoint stable ni comportement de partie.

Phase 1 : générer le dataset canonique et sa projection baseline.

Phase 2 : entraîner et évaluer baseline et V1 avec seeds et splits identiques.

Phase 3 : décider si le forecast est crédible. Si non, conserver le résultat comme diagnostic et
ne pas introduire de pénalité. Si oui, rédiger une architecture ultérieure pour la conversion en
`probability_card_seen` puis pour la pénalité douce des achats.

La promotion d'un modèle de forecast est indépendante de la promotion qualité du joueur neural.
Elle ne doit pas modifier V005 par simple amélioration offline.

## Files Expected To Change

- `scripts/generate_horizon_dataset.py` : générateur canonique, chemin exact à confirmer ;
- `scripts/train_horizon_forecast.py` : entraînement baseline et V1, chemin exact à confirmer ;
- `shards_ai/ai/horizon_forecast.py` : schéma, encodeurs et modèles ;
- `tests/ai/test_horizon_forecast.py` : unités, schémas, projections et métriques ;
- `tests/game/test_neural_observation.py` : seulement si un test de visibilité est nécessaire ;
- `configs/neural_training_profiles/candidates/` : profil expérimental si le format existant est
  réutilisé ;
- `artifacts/horizon_forecast/` : datasets, rapports et modèles, hors documentation ;
- `doc/Current state/Neural player.md` : uniquement après implémentation et si le comportement
  effectivement disponible doit être documenté.
