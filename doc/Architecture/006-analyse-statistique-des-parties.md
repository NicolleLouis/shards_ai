# Analyse statistique des parties — Architecture

**Statut : DONE / livré** — Implémenté, validé par 85 tests et benchmarké avec deux
`RandomPlayer` symétriques.

## Objective

Ajouter un mode d'analyse reproductible permettant d'exécuter une campagne de duels et de
mesurer les cartes présentes dans le deck du gagnant. L'objectif immédiat est d'identifier des
patterns d'équilibrage et de disposer d'un outil de diagnostic pour les futures IA, sans modifier
les règles du moteur ni introduire les champions ou les mercenaires.

Une campagne doit pouvoir être lancée par un script, avec une durée maximale de 60 secondes par
défaut et une seed reproductible. Elle produit un résultat exploitable par machine et des exports
lisibles :

- gagnants, défaites, matchs nuls et parties interrompues ;
- pour chaque partie gagnée, les `card_id` et le nombre de copies de chaque carte possédée par le
  gagnant ;
- un tableau de cartes trié par nombre moyen de copies, avec nom, moyenne, faction et coût ;
- les snapshots des cartes possédées par les gagnants et les perdants ;
- une vue delta calculée comme `gagnant - perdant`, par carte et par faction ;
- une seconde vue comparative excluant les cartes de base et regroupant les cartes par nombre
  d'exemplaires disponibles dans le deck central (`×1` avec `×1`, `×2` avec `×2`, etc.) ;
- une agrégation par faction ;
- un pie chart de la composition moyenne par faction ;
- un line chart des cartes non neutres, classées par moyenne décroissante, avec le rang en abscisse.

Critère de succès : rejouer une campagne à seed et configuration identiques donne le même résultat
statistique, indépendamment de la forme d'affichage choisie.

## Current State

- `GameRunner` exécute une partie complète à partir d'un `Game` et de deux `Player` et expose le
  `GameState` final (`shards_ai/game/runner.py`).
- `GameRunner.random_duel()` construit actuellement deux `RandomPlayer` avec des flux aléatoires
  dérivés de la seed.
- `GameState` contient le statut, le gagnant et les zones de chaque `PlayerState` : main, pioche,
  défausse et zone de jeu.
- `CardDefinition` fournit les métadonnées nécessaires (`card_id`, `name`, `cost`, `faction`) et
  `CardInstance` référence une définition stable.
- `CARD_CATALOG` est indexé par `card_id`. Les cartes de départ sont neutres et les cartes du deck
  central sont réparties entre Maquis, Spectra, Ordre et Homodeus.
- Le projet ne possède pas encore de script d'analyse, de persistance de campagnes ni de
  dépendance de visualisation. Le benchmark actuel (`benchmarks/benchmark_game.py`) mesure le débit
  mais ne collecte pas de résultats.
- Les décisions d'architecture existantes demandent de ne pas conserver d'historique d'actions par
  défaut dans les simulations massives.

## Target Behavior

Un script lance une campagne avec au minimum :

```text
poetry run python scripts/analyze_games.py
poetry run python scripts/analyze_games.py --duration-seconds 30 --seed 42
```

Tous les arguments sont optionnels. Les options prévues sont `--duration-seconds` (60 secondes par
défaut), `--games` optionnel, `--seed` optionnelle, `--max-actions`, `--max-turns` et le répertoire
de sortie. Si `--seed` est absente, le script génère une seed aléatoire au démarrage. Cette seed est
affichée dans la sortie console et enregistrée dans le JSON de résultat afin de pouvoir relancer la
même campagne ultérieurement. Quand les deux limites sont présentes, la campagne s'arrête au premier
seuil atteint. Une durée seule ne promet pas un nombre exact de parties ; le résultat conserve le
nombre effectivement terminé.

Le script s'exécute séquentiellement. Les fichiers sont écrits dans `scripts/analysis_output/` par
défaut, ou dans un chemin fourni par `--output-dir`. Ce dossier est un espace de sortie local,
non versionné et facilement supprimable ; aucune base de données ni index de campagnes n'est
nécessaire.

À la fin d'une partie, si `state.winner` est défini, l'analyse parcourt les zones du gagnant et
compte les cartes par `definition.card_id`. La collection analysée est l'ensemble des cartes encore
possédées : `hand + draw_pile + discard_pile + play_zone`. Une carte bannie ou une carte encore dans
la rivière/deck central ne compte pas. Cette convention mesure le deck final possédé, pas seulement
la pioche active au moment de la victoire.

Les compteurs sont agrégés avec un zéro implicite pour toute carte absente d'une victoire. Ainsi,
`average_number` signifie « nombre moyen de copies par partie gagnée », et non « moyenne conditionnée
au fait que la carte soit présente ». Le rapport expose aussi `presence_rate` pour distinguer ces
deux lectures.

## Non-Goals

- Ajouter les champions, les mercenaires ou leurs règles.
- Modifier la stratégie de `RandomPlayer` ou construire une IA d'équilibrage.
- Conserver chaque action de chaque partie en mode massif.
- Fournir dès maintenant une interface web ou une base de données de séries temporelles.
- Déduire une causalité d'équilibrage à partir d'une simple corrélation de présence.

## Key Decisions

1. **Couche séparée du moteur.** L'analyse consomme `GameRunner` et `GameState` sans ajouter de
   logique statistique à `Game`.
2. **Reproductibilité.** Une seed racine et les paramètres complets de campagne sont enregistrés.
   La seed est fournie par `--seed` ou générée aléatoirement si l'option est absente ; dans les deux
   cas elle est affichée et persistée dans l'export. Chaque partie reçoit ensuite une seed dérivée de
   manière déterministe, sans réutiliser les mêmes flux aléatoires entre parties.
3. **Résultat canonique JSON.** Le JSON est la source de vérité de la campagne ; CSV et graphiques
   sont des projections régénérables. Cela évite de rejouer les parties pour changer un tri ou un
   graphique.
4. **Mesure par victoire.** Les moyennes de cartes sont divisées par le nombre de victoires
   analysables, pas par le nombre total de parties. Le rapport affiche séparément victoires, nuls et
   erreurs/interruptions.
5. **Cartes absentes incluses.** Le domaine statistique est tout le catalogue connu au moment de la
   campagne ; les absences valent zéro.
6. **Agrégation factionnelle.** Pour chaque victoire, les copies sont regroupées par faction puis
   moyennées. Le pie chart représente la part de chaque faction dans le deck moyen des gagnants ; les
   neutres sont conservées dans le rapport mais peuvent être masquées dans le graphique détaillé.
7. **Cartes hors base.** Le line chart exclut les cartes du deck de départ (`crystal`, `blaster`,
   `shard_reactor`, `infinity_shard`) et classe les autres cartes par `average_number` décroissant,
   avec `rank = 1` pour la carte la plus présente.
8. **Comparaison par multiplicité imprimée.** Une vue secondaire exclut les cartes de base et
  compare uniquement les cartes ayant le même nombre de copies prévu dans le deck central. La
  multiplicité vient des constantes `*_CARDS` des définitions de decks, jamais d'un comptage des
  cartes achetées dans une campagne donnée.
9. **Exécution séquentielle.** La première version ne parallélise pas les parties. Chaque `Game` est
  créé et libéré dans le même processus, ce qui simplifie la reproductibilité et suffit à la
   campagne initiale de 60 secondes.
10. **Joueurs de la V1.** Les deux participants sont deux `RandomPlayer` symétriques. La CLI ne
    configure pas encore de politiques différentes ; l'injection de joueurs IA sera ajoutée lorsque
    ces joueurs existeront.
11. **Gestion des erreurs moteur.** Le mode par défaut est tolérant : une erreur est associée à
    l'index et à la seed de la partie, comptabilisée dans le rapport, puis la campagne continue. Une
    option stricte permet d'arrêter immédiatement la campagne pour faciliter le debugging.
12. **Comparaison gagnant/perdant.** Pour chaque victoire, le snapshot du gagnant et celui du
    perdant sont conservés. Les deltas sont calculés sur les mêmes victoires et toujours dans le sens
    `moyenne_gagnant - moyenne_perdant` ; une valeur positive indique une présence supérieure chez
    les gagnants.

## Open Questions


## Proposed Architecture

Le script `scripts/analyze_games.py`, éventuellement soutenu par un module `shards_ai.analysis`, est
organisé autour de quatre responsabilités :

1. `CampaignConfig` valide les limites, la seed, les options de runner et la configuration des
   joueurs.
2. `CampaignRunner` lance séquentiellement les parties, dérive les seeds, collecte un `GameResult` léger et applique
   les limites de parties/durée.
3. `WinnerDeckCollector` convertit le gagnant final en un mapping `card_id -> count`, sans conserver
   les instances ni l'historique des actions.
4. `ReportBuilder` fusionne les snapshots, joint les métadonnées de `CARD_CATALOG`, et génère le
   JSON canonique, les CSV et les graphiques.

Flux :

```text
CampaignConfig
      ↓
CampaignRunner → GameRunner → GameState final
      ↓                       ↓
Campaign summary      WinnerDeckSnapshot
                              ↓
                       ReportBuilder
              ├── result.json
              ├── cards.csv / factions.csv
              └── charts
```

Le runner de campagne doit pouvoir accepter une factory de parties/joueurs afin que les futures IA
soient injectées sans réécrire l'analyse. Il ne doit pas dépendre d'un affichage console pendant la
boucle chaude. La parallélisation est hors périmètre de cette première version.

## Data Model

Le JSON canonique contient notamment :

```json
{
  "schema_version": 1,
  "config": {"seed": 42, "requested_games": 10000},
  "summary": {"completed": 10000, "wins": {"1": 4980, "2": 4972}, "draws": 48, "errors": 0},
  "winner_decks": [
    {"game_index": 0, "seed": 123, "winner": 1,
     "cards": {"crystal": 6, "blaster": 1, "maquis_card": 2}}
  ],
  "loser_decks": [
    {"game_index": 0, "seed": 123, "loser": 2,
     "cards": {"crystal": 7, "blaster": 1}}
  ],
  "cards": [
    {"card_id": "maquis_card", "name": "...", "average_number": 0.42,
     "presence_rate": 0.31, "faction": "maquis", "cost": 3,
     "central_copy_count": 3, "rank": 1}
  ],
  "cards_by_copy_count": {
    "3": [
      {"card_id": "maquis_card", "average_number": 0.42, "rank": 1}
    ]
  },
  "factions": [
    {"faction": "maquis", "average_number": 2.1, "share": 0.12}
  ],
  "cards_delta": [
    {"card_id": "maquis_card", "winner_average_number": 0.42,
     "loser_average_number": 0.18, "delta_average_number": 0.24,
     "winner_presence_rate": 0.31, "loser_presence_rate": 0.16,
     "delta_presence_rate": 0.15}
  ],
  "factions_delta": [
    {"faction": "maquis", "winner_average_number": 2.1,
     "loser_average_number": 1.4, "delta_average_number": 0.7,
     "delta_share": 0.04}
  ]
}
```

Les noms exacts de cartes viennent toujours du catalogue, jamais d'une saisie CLI. Les champs
`winner_decks` et `loser_decks` permettent de recalculer les agrégats ; pour de très grandes campagnes, une option
`--compact` pourra omettre ces snapshots et ne conserver que les agrégats, mais elle ne sera pas le
comportement par défaut du mode diagnostic. L'export est autonome et aucune conservation applicative
supplémentaire n'est prévue.

## Backend Flow

La boucle crée une partie avec une seed dérivée, exécute `GameRunner.run()`, puis traite le résultat
avant de libérer l'état mutable. Une exception est associée à l'index et à la seed de la partie.
L'arrêt temporel est vérifié entre deux parties, jamais au milieu d'une partie ; une partie en cours
reste donc complète ou est signalée comme interrompue selon la politique choisie.

Le builder initialise tous les `card_id` à zéro pour chaque victoire, additionne les snapshots,
calcule moyenne et présence, trie par `(-average_number, card_id)`, puis agrège par faction. Les
divisions par zéro donnent un rapport explicitement vide si aucune victoire n'est disponible.

Il applique la même agrégation aux snapshots perdants, puis joint les lignes gagnant/perdant par
`card_id` et par faction. Chaque delta conserve les quatre métadonnées d'identification et les
moyennes des deux côtés afin d'être interprétable sans recalcul. Le rapport inclut également une
vue delta comparable par multiplicité du deck central, hors cartes de base.

Il construit également une table `cards_by_copy_count` en filtrant les cartes de base puis en
regroupant selon `central_copy_count`. Chaque groupe est trié indépendamment par moyenne
décroissante, avec un rang recommençant à 1. Le tableau CSV correspondant contient au minimum
`central_copy_count`, `rank`, `card_id`, `name`, `average_number`, `presence_rate`, `faction` et
`cost`. Les graphiques comparatifs doivent être séparés par groupe de multiplicité, ou utiliser une
couleur/une facette par groupe ; une courbe unique mélangerait des cartes qui n'ont pas la même
disponibilité structurelle.

Une évolution parallèle future pourra fusionner des compteurs associatifs et conserver les seeds et
erreurs par index, mais elle ne fait pas partie de la V1.

## Frontend Flow

Il n'y a pas de frontend existant à modifier. La V1 produit des fichiers locaux :

- `result.json` pour les données ;
- `cards.csv` trié pour inspection et tableur ;
- `factions.csv` pour l'agrégation ;
- `cards_by_copy_count.csv` pour la comparaison hors cartes de base, stratifiée par multiplicité ;
- `loser_cards.csv`, `loser_factions.csv` et `loser_cards_by_copy_count.csv` pour les agrégats perdants ;
- `cards_delta.csv`, `factions_delta.csv` et `cards_delta_by_copy_count.csv` pour la comparaison
  gagnant moins perdant ;
- `report.html` avec tableaux, pie chart et line charts intégrés en SVG, consultable localement sans
   serveur. Les line charts gagnants et delta sont affichés côte à côte, y compris pour chaque
   multiplicité comparable, avec le détail d'un point accessible au survol.

Une future page pourra consommer directement le JSON sans changer la collecte.

## Authorization And Feature Gates

Sans objet : l'analyse est locale et ne manipule ni utilisateurs ni données distantes. Les options
coûteuses doivent toutefois être explicites, avec une limite de sécurité sur le nombre de parties et
la durée maximale configurable.

## Observability And Operations

La sortie console doit rester légère : progression optionnelle, débit, parties terminées, victoires,
nuls, erreurs et durée. Le JSON conserve `schema_version`, version du package si disponible, seed,
configuration, nombre de parties et seeds en erreur. Une erreur doit être suffisamment localisable
pour relancer uniquement la partie fautive.

Le rapport doit signaler quand aucune victoire n'est disponible et quand une limite a interrompu la
campagne. Les compteurs d'erreurs ne doivent pas être silencieusement mélangés aux statistiques.

## Edge Cases

- campagne avec zéro partie demandée ou durée explicitement non positive : rejet de configuration ;
- lancement sans arguments : seed aléatoire, durée de 60 secondes, sortie dans le dossier par défaut ;
- toutes les parties sont des nuls : agrégats de cartes vides, sans NaN ;
- carte bannie : absente du deck final ;
- carte jouée au moment de la victoire : comptée via `play_zone` ;
- nouvelle carte absente d'un ancien catalogue : le résultat historique reste lisible grâce au
  snapshot de métadonnées et à `schema_version` ;
- égalité de moyenne : tri secondaire stable par `card_id` ;
- campagne interrompue par erreur : mode tolérant continue, mode strict remonte l'erreur avec seed ;
- durée très courte : zéro partie terminée est un résultat valide mais explicitement signalé ;
- victoire par effet de carte ou par attaque : même collecte basée sur `state.winner`.

## Testing Strategy

- tests unitaires de validation de configuration et de dérivation de seeds ;
- tests de collecte avec cartes réparties dans chaque zone, cartes bannies et victoire par effet ;
- tests de moyenne avec présence et absence, classement et égalités ;
- tests de récupération des multiplicateurs déclarés dans les decks centraux ;
- tests vérifiant que les cartes de base sont absentes de la vue comparative et que le rang repart à
  1 dans chaque groupe `×N` ;
- tests d'agrégation factionnelle et de zéro victoire ;
- test d'intégration d'une petite campagne avec `GameRunner.random_duel()` ;
- test de reproductibilité complète du JSON hors champs temporels ;
- test du mode durée avec une horloge injectable plutôt que `sleep` ;
- test du mode tolérant/strict sur une partie qui lève une erreur ;
- test des exports CSV et de la présence des colonnes `card_id`, `name`, `average_number`, `faction`,
  `cost` ;
- test de l'export stratifié avec les colonnes `central_copy_count` et `rank` ;
- benchmark de débit et de mémoire sur 1 000, 10 000 puis 100 000 parties, sans historique d'actions.

## Rollout And Migration

Aucune migration de données ni modification de règles n'est requise. Implémenter d'abord le format
canonique et la collecte en séquentiel, puis les exports et graphiques, puis éventuellement les
workers. Le schéma JSON doit être versionné dès la V1. Une campagne ancienne reste consultable même
si le catalogue évolue, à condition de conserver les métadonnées de carte dans le rapport.

## Files Expected To Change

- `scripts/analyze_games.py` — script d'exécution de la campagne ;
- `shards_ai/analysis/` — module de configuration, campagne, collecte et rapport si le script doit
  être découpé ;
- `shards_ai/game/runner.py` — uniquement si une factory ou une interface de résultat légère est
  nécessaire (chemin tentatif) ;
- `tests/analysis/` — tests unitaires et intégration ;
- `benchmarks/benchmark_analysis.py` — débit et mémoire ;
- `pyproject.toml` — dépendance de rendu seulement si le choix de graphiques l'exige ;
- `doc/Current state/Analysis.md` — documentation de l'état réel après implémentation ;
- `README.md` — commande d'utilisation et exemple de sortie ;
- `.gitignore` — exclusion de `scripts/analysis_output/` (chemin tentatif).
