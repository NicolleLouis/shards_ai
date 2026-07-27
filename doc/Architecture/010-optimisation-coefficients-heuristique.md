# Optimisation des coefficients du HeuristicPlayer — Architecture

**Statut : proposé** — Document d’architecture préalable à l’implémentation.

## Objective

Rendre `HeuristicPlayer` plus performant en recherchant automatiquement un profil de
`HeuristicWeights` qui maximise sa performance contre `RandomPlayer`, tout en conservant la
reproductibilité, l’explicabilité des décisions et la séparation entre moteur, politique et
expérimentation.

Le résultat attendu est un profil de coefficients versionné, accompagné de preuves suffisantes
pour comparer ce profil à la baseline manuelle. Une optimisation ne sera pas considérée comme
réussie parce qu’un échantillon ponctuel donne un meilleur win rate : le gain devra être confirmé
sur des seeds indépendantes et par alternance de la position de départ.

## Current State

- `HeuristicPlayer` sélectionne l’action légale de score maximal.
- `ActionFeatures` et `HeuristicWeights` sont des dataclasses immuables dans
  `shards_ai/ai/heuristic_evaluator.py`.
- Le score est un produit scalaire des features et des coefficients.
- Les priorités `terminal_win` et `lethal` sont déjà placées avant le score économique dans le
  classement ; elles ne doivent pas être laissées à une optimisation aveugle.
- `GameRandom` et `GameRunner` permettent de rejouer une partie à seed identique.
- `scripts/analyze_heuristic_vs_random.py` réalise déjà une campagne HeuristicPlayer contre
  RandomPlayer, avec seeds dérivées, alternance de `PlayerId`, limites d’exécution et rapport HTML.
- `shards_ai/analysis/campaign.py` fournit une infrastructure de campagnes et de résultats JSON,
  mais son chemin générique actuel concerne principalement les campagnes de parties random.
- Les benchmarks existants mesurent le débit des parties, mais ne pilotent pas encore une recherche
  de coefficients.
- La politique est non différentiable du point de vue des coefficients : une petite variation ne
  change souvent aucune décision, puis peut provoquer un changement brutal d’action. Le résultat
  d’une campagne est également bruité par le tirage des cartes et les décisions du joueur random.

## Target Behavior

Une campagne d’optimisation doit :

1. recevoir un profil initial, des bornes et un budget explicite ;
2. générer des candidats valides sans modifier le moteur ni l’état partagé ;
3. évaluer chaque candidat sur les mêmes seeds que les autres candidats d’un même round ;
4. alterner le joueur heuristique entre les deux positions ;
5. agréger victoires, défaites, nulles, erreurs, durée et nombre de parties terminées ;
6. comparer les candidats avec une métrique et un niveau de confiance documentés ;
7. conserver le meilleur candidat confirmé, pas simplement le dernier candidat testé ;
8. pouvoir faire évoluer la population d’adversaires au cours de la nuit : `RandomPlayer` au
   démarrage, puis mélange de `RandomPlayer` et d’une version heuristique précédente ;
9. réévaluer le champion sur un jeu de seeds de validation séparé, puis sur un jeu de test jamais
   utilisé pendant la recherche ;
10. écrire un artefact d’expérience hors de `doc/`, contenant la configuration, les seeds, les
   candidats testés, les résultats et le profil retenu.

Le profil optimisé doit rester injectable dans `HeuristicPlayer` comme n’importe quel autre
`HeuristicWeights`. Aucun coefficient ne doit être modifié pendant une partie.

## Non-Goals

- Modifier les règles, les transitions ou la légalité du moteur.
- Ajouter un look-ahead, une recherche d’actions ou une nouvelle feature dans la même évolution.
- Optimiser `terminal_win` ou `lethal`, qui sont des priorités sémantiques et non des préférences
  économiques.
- Déduire la force générale de l’heuristique à partir du seul adversaire `RandomPlayer`.
- Déployer automatiquement un profil en production ou remplacer silencieusement la baseline.
- Déposer des logs, JSON, HTML ou résultats bruts dans `doc/`.

## Key Decisions

1. **Optimisation boîte noire.** Le score de campagne est bruité et la politique est par morceaux
   constante ; les méthodes à gradient ne sont donc pas le choix initial. L’optimiseur manipule
   seulement des mappings de coefficients et appelle une fonction d’évaluation pure du point de
   vue de l’expérience.

2. **Descente coordonnée comme méthode de référence.** Pour chaque coefficient optimisable, on
   teste une perturbation positive et négative autour du profil courant, puis on conserve la
   meilleure modification si elle est suffisamment prometteuse. On répète le passage sur les
   coefficients jusqu’à absence d’amélioration ou épuisement du budget. Cette méthode est simple,
   explicable, peu dépendante du nombre de dimensions et constitue un oracle de comparaison avant
   d’introduire une méthode plus complexe.

3. **Recherche multi-échelle.** Une coordonnée ne sera pas modifiée avec un incrément fixe unique.
   Les pas commenceront relativement grands dans les bornes, puis seront réduits, par exemple par
   division par deux après un passage sans amélioration. Pour un coefficient de valeur `w`, les
   candidats initiaux seront `w - step` et `w + step`, bornés par `[min, max]`. Un coefficient qui
   atteint une borne sera signalé, car cela peut indiquer une borne trop étroite ou une feature mal
   calibrée.

4. **Racing statistique.** Une comparaison ne lancera pas nécessairement le même grand nombre de
   parties pour tous les candidats. Tous commencent sur un petit lot de seeds appariées ; seuls les
   candidats encore compatibles avec le meilleur sont promus vers des lots supplémentaires. Cela
   réduit le coût d’une recherche tout en évitant de déclarer vainqueur un profil sur un bruit
   d’échantillonnage.

5. **Seeds communes par round.** Les candidats d’un même round utilisent les mêmes seeds et le même
   ordre d’alternance des positions. Cette comparaison appariée réduit la variance due au tirage
   initial et rend les deltas entre profils plus informatifs. Les seeds d’entraînement, validation
   et test sont générées séparément et persistées dans la configuration de l’expérience.

6. **Objectif principal et métriques secondaires.** L’objectif initial est une utilité par partie :
   `victoire = 1`, `nulle = 0.5`, `défaite = 0`. Le win rate sans les nulles reste rapporté pour
   comparaison avec les outils existants. Seront aussi rapportés le taux de nulles, la durée
   moyenne, le nombre d’actions et le taux d’erreurs. Une optimisation ne pourra pas échanger une
   hausse du win rate contre un taux d’erreurs ou de parties interrompues non maîtrisé.

7. **Validation avant acceptation.** Le champion d’entraînement est comparé à la baseline sur un
   jeu de validation indépendant. Le test final utilise un troisième jeu de seeds verrouillé. Le
   profil n’est accepté que si le gain dépasse une marge minimale configurable et si son intervalle
   de confiance est compatible avec un gain réel ; sinon la baseline reste le profil recommandé.

8. **Contraintes sur les coefficients.** Les bornes, le signe attendu des pénalités et les
   coefficients gelés sont définis explicitement. Les coefficients qui correspondent à une priorité
   sémantique (`terminal_win`, `lethal`) sont gelés. Les bornes évitent qu’un coefficient compense
   artificiellement une échelle de feature ou rende la politique illisible.

9. **Normalisation avant recherche large.** Avant une recherche globale, les distributions des
   features seront mesurées sur un corpus de décisions reproductible. Si deux features ont des
   échelles très différentes, les coefficients ne sont pas comparables et la recherche devient
   instable. La première implémentation pourra conserver les features brutes avec des bornes
   explicites, mais une normalisation documentée est préférable pour une évolution ultérieure.

10. **Expérience séparée du code de production.** Le moteur et `HeuristicPlayer` ne connaissent ni
    le budget de recherche ni le protocole statistique. Un module d’optimisation construit des
    joueurs avec un profil donné et réutilise `GameRunner`.

11. **Curriculum d’adversaires.** La campagne commence contre `RandomPlayer`, puis peut ajouter la
    version heuristique précédente dans la population adverse. Le candidat est toujours évalué
    séparément contre chaque adversaire afin qu’un score mixte ne masque pas une régression.

12. **Rotation des seeds.** Les seeds communes sont conservées à l’intérieur d’un batch pour
    comparer les candidats, mais chaque batch reçoit de nouvelles seeds dérivées du seed racine et
    de son numéro. Les jeux de validation et de test sont séparés et ne servent jamais à choisir un
    coefficient.

13. **Seuil de promotion plutôt que cible absolue.** Un seuil de 90 % contre `RandomPlayer` peut
    déclencher la phase mixte, mais ne constitue pas seul un critère de qualité : il peut être
    irréaliste ou refléter un sur-apprentissage. La promotion exige aussi un gain confirmé contre
    la baseline et un taux d’erreur acceptable. Un candidat qui bat clairement l’ancienne
    heuristique peut être promu même s’il n’atteint pas 90 % contre random.

14. **Profil publié dans un fichier explicite.** Les coefficients retenus seront chargés depuis un
    fichier YAML versionné et lisible par un humain. Le fichier contiendra le profil actif, son
    identifiant, sa provenance, le profil parent, les adversaires utilisés, les seeds et les
    résultats de validation. Les rapports détaillés de campagnes seront exportés en JSON afin de
    conserver l’historique complet sans coder les résultats dans Python. La baseline Python reste
    disponible comme fallback si aucun fichier n’est fourni.

15. **Composition de la phase mixte.** Après promotion depuis la phase initiale contre random,
    chaque batch utilise 50 % de parties contre `RandomPlayer` et 50 % contre une version
    heuristique de référence figée au début de la campagne. Les résultats restent calculés
    séparément pour chaque adversaire, puis agrégés uniquement pour le pilotage du budget et de la
    recherche. La référence ne suit jamais les évolutions du candidat pendant cette campagne.

## Open Questions

- **Non bloquante :** quelle valeur minimale de gain sur le jeu de validation doit déclencher
  l’acceptation automatique ? Une valeur initiale raisonnable est +3 points d’utilité sur 100
  parties, à confirmer par le premier benchmark.
- **Non bloquante :** faut-il optimiser d’abord les 15 coefficients économiques ou inclure des
  profils avec certains coefficients à zéro pour mesurer la valeur réelle des features ? La phase
  initiale devrait tester l’ablation de chaque feature avant une recherche exhaustive.

## Proposed Architecture

### Composants

`shards_ai/ai/heuristic_evaluator.py` reste propriétaire de `HeuristicWeights` et du calcul du
score. Il pourra recevoir une méthode de sérialisation minimale si nécessaire, sans dépendre de
l’optimiseur.

`shards_ai/optimization/heuristic.py` (chemin proposé) contiendra :

- une configuration immuable de recherche : profil initial, bornes, coefficients gelés, pas,
  budgets, seuils et seeds ;
- un générateur de candidats qui applique une perturbation à une seule coordonnée ;
- une stratégie `CoordinateDescentOptimizer` indépendante de la simulation des parties ;
- un résultat de recherche avec historique des rounds, profil courant, champion et budget consommé.

`shards_ai/analysis/heuristic_campaign.py` (chemin proposé) contiendra l’évaluation d’un profil :

- construction déterministe d’une partie par seed ;
- alternance de la position de `HeuristicPlayer` ;
- `RandomPlayer` avec une source aléatoire dérivée de façon stable ;
- résultat agrégé par profil et par lot de seeds ;
- estimation d’intervalle de confiance et comparaison appariée entre profils.

`scripts/optimize_heuristic.py` (chemin proposé) sera l’entrée CLI. Il affichera un résumé court
et écrira les résultats expérimentaux dans un répertoire d’artefacts hors de `doc/`.

### Comment reconnaître un minimum local

Dans cet espace, un minimum local n’est pas observé directement par une dérivée. On parle plutôt
de maximum local de la performance : pour un profil `w`, aucun voisin testé dans les directions
autorisées n’améliore suffisamment l’objectif.

Pour une descente coordonnée, après un passage complet :

```text
pour chaque coefficient i:
    mesurer f(w - step_i * e_i)
    mesurer f(w + step_i * e_i)
    prendre le meilleur voisin si son gain est confirmé
si aucun voisin n’est meilleur:
    réduire tous les step_i
si tous les steps < tolérance ou budget épuisé:
    arrêter
```

Cette condition ne prouve pas un optimum global. Elle signifie seulement qu’aucune variation
unidimensionnelle de la taille testée n’a produit un gain suffisamment robuste. Elle peut manquer
un optimum accessible uniquement en modifiant simultanément deux coefficients, et elle peut
déclarer un faux plateau si l’échantillon est trop petit. Pour limiter ces risques :

- répéter les derniers rounds avec davantage de parties ;
- effectuer des redémarrages depuis quelques profils perturbés ;
- comparer les meilleurs profils des redémarrages sur le même jeu de validation ;
- conserver l’historique et les intervalles de confiance plutôt qu’un seul score arrondi.

### Choisir le changement de coefficients

Le changement est choisi en trois étapes :

1. **Direction :** `+step` et `-step` sur une seule feature, sauf coefficient gelé ou borne
   atteinte.
2. **Amplitude :** pas relatif au domaine, par exemple 10 à 25 % de l’intervalle autorisé au
   premier round, puis réduction après un plateau. Un pas trop petit ne change aucune action ; un
   pas trop grand mélange plusieurs effets de politique.
3. **Acceptation :** conserver le meilleur voisin seulement si son gain estimé dépasse une marge
   de bruit. Une comparaison peut utiliser la différence d’utilité sur les mêmes parties et un
   intervalle de confiance bootstrap ou une approximation binomiale ; en cas de recouvrement
   important, le candidat est promu pour un lot plus grand plutôt qu’accepté immédiatement.

Le choix de `+step` ou `-step` n’est donc pas basé sur le signe intuitif d’un coefficient. Par
exemple, augmenter `card_draw` peut pousser l’heuristique à acheter trop tôt et réduire le taux de
victoire ; seule la campagne appariée tranche. Le signe et les bornes restent des contraintes de
lisibilité et de sécurité, pas une preuve de performance.

### Méthodes standards considérées

| Méthode | Intérêt | Limite dans ce projet | Décision |
|---|---|---|---|
| Descente coordonnée / pattern search | simple, explicable, efficace sur peu de dimensions | peut manquer les interactions entre coefficients | méthode V1 et référence |
| Recherche aléatoire bornée | très robuste comme baseline, parallélisable | gaspille des évaluations si le budget est faible | utilisée pour contrôler la qualité de V1 |
| CMA-ES / stratégies évolutionnaires | adaptée aux objectifs bruités et non lisses, explore les interactions | plus de parties, hyperparamètres et explicabilité moindre | option après benchmark |
| Optimisation bayésienne | économise les évaluations coûteuses | modèle moins naturel avec 15+ dimensions et plateaux discrets | non prioritaire |
| SPSA / méthodes de gradient sans dérivée | peu d’évaluations par itération | variance élevée et signal discontinu | non prioritaire |

La littérature d’optimisation boîte noire justifie surtout de comparer plusieurs familles sur le
même budget. Le projet ne doit pas supposer qu’une méthode sophistiquée sera meilleure avant cette
mesure. La priorité est un optimiseur coordinate/pattern-search reproductible, puis un benchmark
contre recherche aléatoire et éventuellement CMA-ES.

### Entraînement par batches pendant la nuit

Le script CLI peut exécuter une boucle de batches avec une deadline globale :

```text
profil = baseline
adversaires = [RandomPlayer]

tant que budget non épuisé:
    seeds = nouvelles seeds pour le batch
    candidat = optimiser(profil, adversaires, seeds, budget_du_batch)
    mesurer candidat contre chaque adversaire séparément

    si candidat atteint le seuil de promotion contre random
       et bat la baseline avec une marge confirmée:
        adversaires = [RandomPlayer, HeuristicPlayer(profil)]

    si candidat bat HeuristicPlayer(profil) sur validation fraîche:
        profil = candidat
```

Le changement de population prend effet au début du batch suivant. Ainsi, tous les candidats d’un
batch affrontent exactement la même distribution d’adversaires. La version heuristique de référence
est capturée au lancement et reste identique jusqu’à la fin de la campagne. La composition, le seuil de
promotion, la durée de chaque batch et le nombre maximal de batches sont configurables.

Le seuil de 90 % est un déclencheur pratique, pas une promesse de résultat. Un profil à 90 % sur
1 000 parties mais 84 % sur des seeds fraîches ne doit pas être considéré comme promu. À l’inverse,
un profil qui bat clairement l’ancienne heuristique sans atteindre 90 % contre random peut être
retenu : l’objectif final est la progression de l’heuristique, pas un score arbitraire contre un
adversaire faible.

## Data Model

Aucun changement du modèle de partie ni migration n’est nécessaire.

Les types proposés sont sérialisables :

- `CoefficientBound(name, minimum, maximum, step)` ;
- `OptimizationConfig(initial_weights, bounds, frozen, train_seed, validation_seed, test_seed,
  batch_sizes, max_rounds, restarts, confidence_level, minimum_gain)` ;
- `CandidateResult(weights, seed_set, games, wins, draws, losses, utility, confidence_interval)` ;
- `OptimizationResult(baseline, champion, accepted_profile, history, config)`.

Les sorties JSON/CSV/HTML sont des artefacts d’expérience et restent hors de `doc/`. Le profil
retenu est un fichier YAML explicite, par exemple `configs/heuristic_profiles/v002.yaml`. Il ne
remplace pas automatiquement les valeurs par défaut : le profil est activé par configuration ou
par un argument CLI. Chaque campagne conserve un rapport JSON distinct, par exemple dans
`artifacts/heuristic_optimization/<run-id>/`.

Le YAML publié doit au minimum contenir :

```yaml
schema_version: 1
profile_id: heuristic-v002
parent_profile_id: heuristic-v001
weights:
  gems_produced: 1.7
  power_produced: 2.2
metadata:
  optimizer: coordinate_descent
  opponent_mix:
    random: 0.5
    previous_heuristic: 0.5
  validation_seed: 123456
  validation_games: 10000
```

Les champs absents de `weights` conservent la valeur de la baseline ou du profil parent selon le
mode de chargement retenu ; le chargeur doit néanmoins rejeter les coefficients inconnus, non finis
ou hors bornes.

## Backend Flow

1. Charger la baseline et vérifier les bornes.
2. Générer le seed racine et réserver des domaines distincts pour les batches, la validation et le
   test.
3. Évaluer la baseline contre `RandomPlayer` sur le premier lot.
4. Générer les voisins du profil courant, lancer le racing sur les seeds communes et retenir les
   voisins prometteurs.
5. Répéter par coordonnées, réduire les pas sur plateau et exécuter les redémarrages configurés.
6. À la fin du batch, évaluer le champion contre chaque adversaire sur des seeds fraîches.
7. Si le seuil et les critères de gain sont satisfaits, ajouter l’ancienne heuristique à la
   population pour le batch suivant.
8. Si le candidat bat l’ancienne version sur validation, le rendre nouveau profil courant ; sinon
   conserver l’ancienne version.
9. Comparer le profil final et la baseline sur le jeu de test verrouillé.
10. Écrire le rapport et retourner un code d’échec si aucune amélioration confirmée n’est obtenue.

Les erreurs de partie sont comptées et attachées au seed. En mode strict, elles interrompent
l’expérience ; en mode tolérant, un candidat n’est comparable que si son taux d’erreur respecte le
seuil configuré. Chaque exécution doit pouvoir être rejouée avec la configuration et le seed racine
du rapport.

## Frontend Flow

Sans objet : cette feature est une expérimentation CLI et ne crée pas d’interface utilisateur.

## Authorization And Feature Gates

Sans objet. L’activation d’un profil optimisé est un choix de versionnement local, pas une
fonctionnalité utilisateur.

## Observability And Operations

Chaque round doit journaliser ou enregistrer : round, coordonnée, pas, profil candidat, nombre de
parties, utilité, win rate, nulles, erreurs, intervalle de confiance, temps et décision
accepté/rejeté. Le rapport doit permettre de répondre à : « pourquoi ce coefficient a-t-il changé
? » et « le gain survit-il à de nouvelles seeds ? ».

Les campagnes longues doivent être interrompables sans perdre l’historique déjà écrit. Les
artefacts volumineux et rapports générés sont exclus du dépôt par `.gitignore` si nécessaire.

## Edge Cases

- coefficient inconnu, non fini ou hors bornes ;
- pas nul ou coefficient déjà à une borne ;
- candidat produisant des parties invalides ou dépassant `max_actions`/`max_turns` ;
- uniquement des parties nulles ;
- nombre de parties terminé différent du nombre demandé ;
- seed ou ordre d’alternance mal réutilisé entre train et validation ;
- apparent gain dû à un seul round ou à la position Player 1 ;
- plateau où aucune modification ne change une décision ;
- deux profils distincts donnant exactement la même politique sur toutes les seeds ;
- amélioration contre `RandomPlayer` mais dégradation nette contre un autre profil.

## Testing Strategy

- Tester la génération de candidats, le respect des bornes, les coefficients gelés et la réduction
  des pas.
- Tester que les mêmes seeds et positions sont utilisées pour comparer deux candidats.
- Tester l’agrégation victoire/nulle/défaite et la gestion des erreurs.
- Tester la reproductibilité complète d’une campagne avec seed racine identique.
- Tester que `HeuristicPlayer` reste inchangé quand aucun profil optimisé n’est chargé.
- Tester le protocole de validation : les seeds de validation et de test ne sont jamais entraînées.
- Ajouter un test d’intégration court avec quelques parties contre `RandomPlayer`, sans dépendre d’un
  score statistique exact fragile.
- Comparer sur un budget fixe la descente coordonnée et la recherche aléatoire ; conserver cette
  comparaison comme benchmark, pas comme test unitaire.

## Rollout And Migration

1. Ajouter l’infrastructure d’évaluation et les tests sans changer la baseline.
2. Implémenter la descente coordonnée et produire un premier rapport local versionné uniquement
   pour sa configuration et son profil candidat.
3. Comparer baseline, champion d’entraînement et champion de validation.
4. Publier le profil retenu dans un fichier explicite ou une constante nommée, après revue du
   rapport.
5. Garder la baseline et permettre un retour arrière par configuration.

Aucune migration de données n’est prévue. Les anciennes parties restent rejouables car le profil
utilisé doit être enregistré avec les expériences et, à terme, dans les métadonnées de benchmark.

## Files Expected To Change

- `shards_ai/optimization/__init__.py` — nouveau package, si l’infrastructure est retenue.
- `shards_ai/optimization/heuristic.py` — optimiseur et types de configuration, chemin proposé.
- `shards_ai/analysis/heuristic_campaign.py` — campagne spécialisée, population d’adversaires,
  rotation des seeds et statistiques, chemin proposé.
- `scripts/optimize_heuristic.py` — CLI et export d’artefacts, chemin proposé.
- `shards_ai/ai/heuristic_evaluator.py` — éventuellement sérialisation/validation complémentaire,
  sans modifier le contrat de score.
- `shards_ai/ai/heuristic_profiles.py` — chargeur YAML et validation des profils, chemin proposé.
- `configs/heuristic_profiles/*.yaml` — profils publiés, chemin proposé.
- `tests/optimization/` et `tests/analysis/` — tests unitaires et intégration.
- `.gitignore` — éventuellement exclusion des sorties locales d’expériences.

La présente architecture ne modifie volontairement pas `doc/Current state/` : cette mise à jour
sera faite après implémentation, lorsque le comportement réellement livré sera connu.
