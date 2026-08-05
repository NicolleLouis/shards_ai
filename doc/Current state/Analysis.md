# État courant — Analyse statistique

Le script `scripts/analyze_games.py` exécute des duels avec deux `RandomPlayer` symétriques.
Sans argument, il utilise une durée de 60 secondes et génère une seed aléatoire. La seed est
affichée et conservée dans les résultats.

```bash
poetry run python scripts/analyze_games.py
poetry run python scripts/analyze_games.py --duration-seconds 30 --seed 42
```

Les fichiers sont écrits par défaut dans `artifacts/analysis/random_vs_random/` :

- `result.json` : résultat canonique, snapshots des decks gagnants et agrégats ;
- `cards.csv` : cartes triées par moyenne de copies dans les decks gagnants ;
- `factions.csv` : agrégation par faction ;
- `cards_by_copy_count.csv` : comparaison des cartes hors base par multiplicité du deck central ;
- `loser_cards.csv`, `loser_factions.csv` et `loser_cards_by_copy_count.csv` : mêmes agrégats pour
  les perdants ;
- `cards_delta.csv`, `factions_delta.csv` et `cards_delta_by_copy_count.csv` : différence gagnant
  moins perdant ;
- `report.html` : tableaux et graphiques SVG consultables localement. Les line charts sont
  présentés par paires, avec la vue gagnants à gauche et le delta gagnant − perdant à droite,
  y compris pour chaque multiplicité comparable ; les points affichent leur détail au survol.

Le mode est tolérant par défaut : une partie en erreur est enregistrée avec son index et sa seed,
puis la campagne continue. `--strict` arrête la campagne à la première erreur.

Les options disponibles sont `--duration-seconds`, `--games`, `--seed`, `--max-actions`,
`--max-turns`, `--strict` et `--output-dir`. Tous les arguments sont optionnels. La durée par
défaut est de 60 secondes et la seed est générée aléatoirement si elle n'est pas fournie.

## Visualisation détaillée d'une partie

Le script `scripts/analyze_game_detail.py` produit une trace déterministe d'une seule partie.
Il enregistre chaque décision avant son application, les actions légales, le score et les
contributions de l'heuristique, puis l'état résumé avant et après la transition. Le rapport HTML
regroupe les actions par tour et replie les détails volumineux pour rester lisible ; `game.json`
contient la trace complète pour une analyse automatisée.

```bash
PYTHONPATH=. poetry run python scripts/analyze_game_detail.py \
  --seed 123 \
  --player1 heuristic \
  --player2 random \
  --profile configs/heuristic_profiles/v008.yaml
```

Les sorties sont écrites par défaut dans `artifacts/analysis/game_detail/<seed>/`. Le mode
d'observation est optionnel dans `GameRunner` : sans observateur, aucune trace, sérialisation ou
copie supplémentaire n'est produite par cette fonctionnalité.

Dans les décisions de phase achat, le rapport ajoute une analyse de la rivière sur ses six slots.
Chaque carte reçoit le score de son achat théorique, même si elle est trop chère ; la colonne
`Légale` distingue les options réellement disponibles et la conclusion indique l'action retenue
par le moteur ; les achats durables indiquent également leur statut par rapport à `buy_threshold`.
Les blocs de tours sont colorés en bleu pour le joueur 1 et en violet pour le joueur 2.

## Duel heuristique contre random

Le script `scripts/analyze_heuristic_vs_random.py` exécute par défaut des parties pendant 60
secondes entre un `HeuristicPlayer` et un `RandomPlayer`. Les rôles alternent entre Player 1 et
Player 2. Il affiche les taux de victoire en pourcentage et écrit un rapport HTML autonome dans
`artifacts/analysis/heuristic_vs_random/report.html`.

```bash
poetry run python scripts/analyze_heuristic_vs_random.py
poetry run python scripts/analyze_heuristic_vs_random.py --duration-seconds 10 --seed 42
```

Pour un test reproductible court, `--games N` remplace la limite de durée. `--output` permet de
choisir le chemin du rapport HTML. Les pourcentages sont calculés sur les parties terminées ; les
erreurs éventuelles sont affichées mais exclues du dénominateur.

Les cartes statistiques d'une victoire sont les cartes encore possédées dans la main, la pioche,
la défausse et la zone de jeu du gagnant. Les cartes bannies et les cartes restées dans la rivière
ne sont pas comptées. `average_number` est calculé par victoire, avec zéro pour une carte absente.
La vue `cards_by_copy_count.csv` exclut les cartes de base et compare les cartes par multiplicité
déclarée dans le deck central.

Pour chaque victoire, le rapport conserve aussi le contenu du deck du perdant. Le delta est calculé
comme `moyenne_gagnant - moyenne_perdant`, avec un delta positif indiquant une présence supérieure
chez les gagnants. Les taux de présence et les vues comparables par multiplicité sont également
disponibles pour les deux côtés.

Le benchmark macro `scripts/benchmark_heuristic_report.py` conserve aussi `deck_size` dans chaque
snapshot final. `deck_size_by_role` compare la taille moyenne des decks Heuristic et Random ;
`heuristic_deck_size_by_result` compare les decks Heuristic dans les victoires, défaites et nuls.
Ces agrégats sont présents dans `results.json`, dans `deck_size_summary.csv` et dans la section
« Taille des decks finaux » du rapport HTML. La taille compte les cartes présentes dans la main,
la pioche, la défausse, la zone de jeu et les champions ; les cartes bannies et la rivière sont
exclues.

Le benchmark macro `scripts/benchmark_heuristic_report.py` mesure désormais v008 sur une campagne
équilibrée : les parties d’index pair opposent v008 à Random et les parties d’index impair à
Heuristic v007. Pour un nombre pair de parties, la répartition est donc exactement 50/50. Le JSON
contient une section `opponents.random` et une section `opponents.v007`, chacune avec ses résultats,
ses decks finaux, ses choix et son delta indépendant (`v008 − adversaire`). Le HTML met en avant
le résultat global, les résultats par adversaire, les decks, les deltas et les comportements de v008;
les tableaux de choix détaillés restent dans les CSV/JSON et les tableaux HTML redondants ont été
retirés.

Pour mesurer le profil courant v008 sur 1 000 parties :

```bash
PYTHONPATH=. poetry run python scripts/benchmark_heuristic_report.py \
  --games 1000 \
  --seed 87000 \
  --profile configs/heuristic_profiles/v008.yaml \
  --opponent-profile configs/heuristic_profiles/v007.yaml \
  --output-dir artifacts/analysis/heuristic_v008_mix_1000
```

## Performance de la simulation

Le benchmark `benchmarks/benchmark_analysis_campaign.py` mesure la campagne avec deux
`RandomPlayer`, seed 42 et un nombre fixe de parties. Sur trois répétitions de 10 parties, le débit
médian est passé de 1,479 à 94,975 parties/s (+6 322 %) après le remplacement de la copie générique
`deepcopy` des observations par une copie structurée détachée dans `Game.observation_for()`. Une
construction spécialisée des `CardInstance` détachées porte ensuite le débit médian à 109,059
parties/s (+12,5 % supplémentaire). Les listes, joueurs et `CardInstance` restent indépendants de
l'état réel ; les `CardDefinition` sont partagées car elles sont immuables.

Une liaison locale des fonctions de copie dans `Game._detached_state()` réduit ensuite le coût des
résolutions d'observation sans modifier leur contenu. Sur le benchmark contrôlé de 100 parties,
le débit médian est passé de 95,595 à 102,852 parties/s (+7,6 %), avec 100 victoires et aucune
erreur dans chaque mesure.

## Performance de l'agrégation

`benchmarks/benchmark_analysis.py` mesure uniquement `build_statistics()` sur 100 000 snapshots
déterministes de decks gagnants. La médiane de trois répétitions est passée de 37 042,7 à
68 262,6 snapshots/s (+84,3 %) après la préparation locale unique des métadonnées de cartes et la
suppression des compteurs de factions temporaires par snapshot. Cette optimisation ne modifie ni
les snapshots ni les agrégats produits.

Le benchmark `benchmarks/benchmark_heuristic_players.py` mesure séparément deux
`HeuristicPlayer` sur les seeds `0..N-1`, afin d’isoler le coût de l’évaluation heuristique du coût
du moteur et de la politique aléatoire.
