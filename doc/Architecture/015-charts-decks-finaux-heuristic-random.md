# Charts des decks finaux Heuristic vs Random

## Objective

Rendre `scripts/benchmark_heuristic_report.py` utile pour analyser les cartes réellement conservées
en fin de partie, et pas seulement les actions de choix. Le rapport doit permettre de repérer les
cartes privilégiées par `HeuristicPlayer` ainsi que les écarts avec `RandomPlayer`.

## Current State

Le benchmark produit déjà un JSON et un HTML avec le taux de victoire, les statistiques finales et
les cartes sélectionnées par catégorie d’action. Il ne conserve pas de snapshot des cartes des deux
joueurs à la fin d’une partie et ne peut donc pas afficher la composition moyenne des decks.
`shards_ai.analysis.campaign` possède déjà les agrégations de cartes, de factions et de deltas, ainsi
que les générateurs SVG utilisés par `analyze_games`.

## Target Behavior

Pour chaque partie terminée, le benchmark conserve les cartes de la main, de la pioche, de la
défausse et de la zone de jeu pour le rôle Heuristic et le rôle Random. Les champions restent hors
du snapshot de deck, conformément à l’analyse existante.

Le JSON expose des statistiques par rôle : moyenne de copies, taux de présence, faction et
multiplicité centrale. Il expose également `Heuristic - Random` pour le nombre moyen de cartes et le
taux de présence. Les mêmes agrégats sont disponibles séparément pour les parties gagnées et
perdues par l’Heuristic.

Le HTML ajoute des graphiques SVG cohérents avec `analyze_games` : cartes hors base, groupes par
multiplicité centrale, factions et deltas les plus importants. Les tableaux CSV/JSON restent la
source complète ; le graphique limite seulement l’affichage aux cartes non-base pertinentes.

## Non-Goals

- modifier la partie, le choix des cartes ou les règles du moteur ;
- reconstruire une liste historique des achats si elle n’est pas nécessaire au deck final ;
- ajouter une dépendance graphique JavaScript ou un serveur web ;
- mélanger les snapshots de parties incomplètes ou en erreur aux statistiques.

## Key Decisions

1. Réutiliser les fonctions d’agrégation et de génération SVG de `shards_ai.analysis.campaign` afin
   de garder les mêmes définitions de moyenne, présence et delta que `analyze_games`.
2. Capturer les deux joueurs après une partie terminée et les classer par rôle puis par résultat de
   l’Heuristic.
3. Conserver les snapshots bruts dans `results.json` et produire des CSV dédiés pour permettre une
   analyse hors HTML.
4. Trier les deltas par valeur absolue décroissante dans le graphique principal afin de montrer les
   plus gros écarts positifs et négatifs.

## Proposed Architecture

Le benchmark ajoute un snapshot final par rôle dans sa boucle de campagne. Une couche d’adaptation
convertit ces snapshots au format attendu par `build_statistics()` et `build_delta_statistics()`.
`write_reports()` sérialise les agrégats et compose les graphiques SVG existants dans la page HTML.
La collecte des actions actuelle reste indépendante des snapshots de deck.

## Data Model

Chaque snapshot contient `game_index`, `seed`, `role`, `result_group` et `cards`, où `cards` mappe
un `card_id` vers le nombre de copies présentes dans les zones de deck analysées. Les statistiques
par rôle utilisent les mêmes champs que l’analyse historique : `average_number`, `presence_rate`,
`faction`, `cost` et `central_copy_count`.

## Edge Cases

- une campagne sans victoire terminée produit des statistiques vides mais un rapport valide ;
- une carte absente d’un rôle reçoit une moyenne et une présence nulles ;
- les parties nulles sont conservées dans le résultat global mais ne contaminent pas un delta de
  gagnant/perdant ;
- une multiplicité inconnue est regroupée dans les cartes non groupées, sans inventer de copie
  centrale ;
- les profils YAML et la seed restent ceux utilisés par le benchmark existant.

## Testing Strategy

- vérifier le snapshot des deux rôles sur une partie déterministe ;
- vérifier les agrégats Heuristic/Random et le signe des deltas ;
- vérifier la présence des sections JSON, CSV et SVG dans le rapport ;
- exécuter la suite complète et un smoke test de campagne courte.

## Rollout And Migration

Les anciens rapports restent lisibles : les nouveaux champs sont ajoutés au JSON et les sorties sont
regénérées lors du prochain benchmark. Aucun fichier du vault n’est généré par la campagne.

## Files Expected To Change

- `scripts/benchmark_heuristic_report.py` — snapshots, agrégations et graphiques ;
- `tests/` — tests du reporting et des deltas ;
- `doc/Current state/Heuristic player.md` — comportement et format du reporting courant.
