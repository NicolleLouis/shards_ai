# Graphiques des decks du benchmark Neural mix

## Objective

Compléter chaque section adversaire du rapport `neural_mix.html` avec les neuf graphiques de decks
déjà utilisés par les rapports heuristiques : pour les cartes dont le deck central contient ×1, ×2
ou ×3 exemplaires, afficher séparément le deck moyen NeuralPlayer, le deck moyen adverse et leur
écart.

## Current State

`benchmarks/benchmark_neural_mix.py` agrège déjà, pour chaque rôle, le nombre moyen de copies et le
taux de présence des cartes finales. Son HTML affiche ces données sous forme de tableaux, mais ne
reprend pas les graphiques SVG de `shards_ai.analysis.campaign`. Le résumé ne contient pas les
cartes absentes des decks, ce qui suffit aux tableaux actuels mais empêcherait une comparaison
complète par multiplicité centrale.

## Target Behavior

Pour chaque adversaire, le rapport affiche neuf graphiques autonomes :

- `×1` : NeuralPlayer, adversaire, delta ;
- `×2` : NeuralPlayer, adversaire, delta ;
- `×3` : NeuralPlayer, adversaire, delta.

Chaque point correspond à une carte. Les graphiques Neural et adversaire sont triés par nombre
moyen de copies décroissant. Le graphique delta est trié par valeur absolue décroissante. Les cartes
sans occurrence dans un rôle sont incluses avec une moyenne nulle. Les détails de la carte restent
accessibles dans le titre SVG au survol, comme dans les rapports existants.

## Non-Goals

- Ajouter une dépendance JavaScript ou un serveur de graphiques.
- Modifier les statistiques JSON existantes ou les règles de constitution des decks.
- Afficher les cartes de base `crystal`, `blaster`, `shard_reactor` et `infinity_shard` dans ces
  graphiques ; elles ne possèdent pas de multiplicité centrale ×1/×2/×3.

## Key Decisions

- Réutiliser `_line_svg` et `central_copy_counts` de `shards_ai.analysis.campaign` pour conserver
  les mêmes conventions visuelles et la même définition de multiplicité centrale.
- Produire les graphiques à partir des agrégats déjà calculés par `_summary`, sans recalculer les
  parties ni modifier le JSON par partie.
- Le delta est le delta du nombre moyen de copies (`NeuralPlayer - adversaire`), cohérent avec
  l’ordonnée demandée et les graphiques historiques ; la présence reste disponible dans les
  tableaux et les tooltips peuvent être enrichis ultérieurement.
- Les neuf graphiques sont répétés dans chaque section adversaire, car les populations Random,
  v007 et v008 sont agrégées séparément.

## Open Questions

Aucune question bloquante. Les noms de cartes sont affichés dans les tooltips SVG et non sur chaque
abscisse afin de conserver la lisibilité lorsque plusieurs cartes appartiennent à un groupe.

## Proposed Architecture

Ajouter une fonction de préparation des lignes de graphique qui, pour une multiplicité donnée,
fusionne les cartes centrales avec les statistiques Neural/adversaire. Elle crée trois listes
indépendantes : copies moyennes Neural, copies moyennes adverses et delta. Chaque liste est triée
selon sa métrique avant d’être passée à `_line_svg`. `_render_report` ajoute les trois graphiques de
chaque multiplicité dans une grille HTML.

## Performance And Operations

La génération est en mémoire et porte seulement sur le catalogue de cartes, pas sur le nombre de
parties. Elle n’ajoute aucun coût au benchmark de jeu ; le coût intervient uniquement lors du rendu
HTML.

## Testing Strategy

- Vérifier les neuf titres de graphiques dans le HTML d’un résumé synthétique.
- Vérifier que les cartes absentes d’un rôle sont représentées à zéro.
- Vérifier les ordres décroissants des groupes Neural, adverse et delta absolu.
- Exécuter la suite complète.

## Files Expected To Change

- `benchmarks/benchmark_neural_mix.py` — préparation et rendu des SVG ;
- `tests/ai/test_neural_benchmark.py` — tri, zéros et présence des neuf graphiques ;
- `doc/Current state/Neural player.md` — description du rapport enrichi.
