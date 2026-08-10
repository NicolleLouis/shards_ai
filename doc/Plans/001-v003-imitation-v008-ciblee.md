# Plan d'action — V003 initialisé, imitation V008 ciblée

## Objectif

Tester si une imitation de Heuristic V008, initialisée avec les poids de Neural V003,
peut améliorer la force de jeu sans remplacer globalement la politique déjà promue.

L'expérience doit expliquer un gain éventuel par une slice de décision précise. Une
amélioration offline globale ou un gain contre un seul adversaire ne suffit pas pour
promouvoir le checkpoint.

## État initial

V003 est le profil neural actif. Le benchmark panel du 7 août 2026 a été exécuté avec
2 000 parties par adversaire, soit 12 000 parties au total, seed `105` :

| Adversaire | Victoires V003 | Défaite adverse |
|---|---:|---:|
| Random | 90,95 % | 9,05 % |
| Heuristic V007 | 37,55 % | 62,45 % |
| Heuristic V008 | 18,80 % | 81,20 % |
| Neural V001 | 46,90 % | 53,10 % |
| Neural V002 | 47,65 % | 52,35 % |
| Neural V003 | 51,05 % | 48,95 % |

Ce résultat confirme une faiblesse spécifique contre V008. Il ne permet pas encore de
choisir entre `play_card`, `activate_champion` ou une autre slice ; la prochaine étape
doit donc être un diagnostic de trajectoires V003/V008.

## Diagnostic initial des états visités

Le premier diagnostic contrefactuel, exécuté sur les mêmes 2 000 seeds, contient `318 492`
 décisions V003 :

| Slice | Décisions | Accord top-1 V003/V008 | Divergence |
|---|---:|---:|---:|
| Global | 318 492 | 72,22 % | 27,78 % |
| Phase `play` | 228 415 | 67,71 % | 32,29 % |
| `play_card` choisi par V003 | 154 175 | 61,47 % | 38,53 % |
| `activate_champion` choisi par V003 | 12 403 | 64,59 % | 35,41 % |
| `banish_card` choisi par V003 | 5 491 | 45,16 % | 54,84 % |
| `buy_card` choisi par V003 | 28 448 | 53,05 % | 46,95 % |

V008 choisit `recruit_mercenary` dans `5 523` états où V003 choisit toujours une autre
action. Ce signal est important, mais il ne prouve pas que recruter est causalement meilleur :
le diagnostic évalue V008 contrefactuellement sur des états visités par V003 et mesure un
regret selon le score heuristique, pas une variation de win-rate.

Le script a donc été étendu pour ventiler aussi par cardinalité légale et par carte. Cette
version étendue doit être exécutée avant de choisir la slice du fine-tuning.

## Décision après le diagnostic enrichi

La slice prioritaire retenue pour le premier pilote est :

```text
action = play_card
cardinalité de l'ensemble légal = 5, 6, 7 ou 8
```

Elle représente environ `100 000` décisions visitées dans le diagnostic, avec des divergences
de `54,0 %` à 5 actions légales, `41,0 %` à 6, `45,8 %` à 7 et `47,6 %` à 8. À l'inverse,
les décisions `play_card` avec 2 ou 3 actions légales sont presque stables (`0 %` et `5,1 %`
de divergence) et doivent rester protégées.

Les cartes V008 `initie_de_l_ordre`, `drone_kiln` et `drone_reacteur` montrent aussi une
divergence élevée, mais la première expérience ne les isolera pas : une slice par carte serait
plus difficile à attribuer et risquerait de sur-apprendre des corrélations de dataset.

`recruit_mercenary`, `banish_card` et `buy_card` restent des pistes secondaires. Leur divergence
est forte, mais leur couverture est plus faible et le diagnostic contrefactuel ne prouve pas que
le choix V008 améliore le résultat de partie.

## Contraintes invariantes

- Ne pas modifier le moteur, les heuristiques V007/V008 ou le masque d'information.
- Conserver V003 et V002 inchangés comme références.
- Utiliser l'architecture `structured_semantic_v4` et initialiser les poids depuis
  `configs/neural_profiles/v003.pt`.
- Réinitialiser l'optimiseur lors du fine-tuning ; ne pas reprendre l'état Adam de V003.
- Écrire le seul checkpoint de travail dans `artifacts/neural_training/checkpoint.pt`.
- Conserver les datasets, manifests, hashes, métriques et rapports sous `artifacts/`.
- Ne jamais promouvoir depuis un pilote ou un screening court.

## Étape 1 — Adapter l'infrastructure d'initialisation

Ajouter au script d'imitation une distinction explicite entre :

- `--initialize-from` : charge les poids et la configuration modèle d'un parent, sans
  reprendre son profil ni son optimiseur ;
- `--resume-from` : reprend un entraînement du même profil avec son état d'optimiseur.

Le profil candidat déclarera `parent_profile_id: v003`, un nouvel identifiant d'expérience,
la même architecture et une sortie mutable. Cette modification doit être couverte par des
tests : chargement des poids V003, optimiseur neuf, architecture identique et rejet des
architectures incompatibles.

## Étape 2 — Diagnostiquer la dérive contre V008

Produire un holdout indépendant par `game_id`, avec V008 comme teacher et des parties
contre Random et V007. Conserver la provenance complète : seed, dataset, hash, parties,
matchups et split.

Comparer V003 et V008 par :

- phase et type d'action ;
- carte et capacité ;
- cardinalité de l'ensemble légal ;
- confiance/calibration ;
- accord d'argmax ;
- couverture des slices rares.

Les slices candidates sont, par ordre initial, `play_card` avec 4–8 actions légales,
puis `activate_champion`. Le choix final doit être déterminé par la couverture et la
concentration de la dérive, pas par préférence préalable.

## Étape 3 — Construire la candidate ciblée

Générer suffisamment de décisions V008 pour couvrir la slice sélectionnée et conserver
un holdout qui n'a jamais servi à choisir les poids.

Le premier pilote sera volontairement limité à 5 000–10 000 décisions ciblées :

- learning rate faible, initialement `1e-4` ;
- initialisation V003 ;
- optimiseur neuf ;
- pondération limitée à la slice retenue ;
- régularisation ou contrainte de conservation de V003 hors slice ;
- budget explicite de décisions dont l'argmax peut changer.

Le pilote est diagnostique et ne peut pas être promu.

Avant de lancer ce pilote, ajouter au pipeline d'imitation un filtre de slice explicite et testé.
Le filtre doit conserver les décisions hors slice dans le holdout et permettre de mesurer combien
de décisions V003 changent hors de `play_card` avec 5–8 actions légales.

## Résultat du pilote exp00100

Le pilote a été exécuté avec :

- `5 000` décisions d'entraînement filtrées ;
- initialisation depuis `configs/neural_profiles/v003.pt` ;
- learning rate `1e-4` et optimiseur neuf ;
- holdout non filtré de `15 127` décisions ;
- dataset DAgGER on-policy de `171 913` décisions, issu de `1 000` parties Random/V007,
  sans erreur.

Le screening de `20` parties par adversaire a été rejeté sans promotion :

| Adversaire | Delta candidate − V003 |
|---|---:|
| Random | `+5` points |
| V007 | `-20` points |
| V008 | `-10` points |
| Neural V001 | `-15` points |
| Neural V002 | `-20` points |
| Neural V003 | `-20` points |

La candidate progresse uniquement contre Random et régresse contre les références neural,
V007 et la garde V008. Le résultat est suffisamment défavorable pour arrêter cette recette,
même si le screening reste trop court pour une conclusion de promotion.

## Suite après le pilote rejeté

Ne pas augmenter le nombre de décisions ni lancer la validation longue. Le filtrage de dataset
n'empêche pas la dérive des paramètres partagés, et la validation offline V008 ne suffit pas à
la détecter.

La prochaine modification doit ajouter une ancre de politique V003 pendant l'entraînement :
sur les records ciblés, contraindre les scores/logits de la candidate à rester proches de V003,
et ne relâcher cette contrainte que lorsque V008 fournit un signal de classement différent.
Cette ancre devra mesurer séparément la dérive d'argmax sur les records hors slice et sur la slice,
avec un budget pré-déclaré. Un nouveau pilote ne sera justifié qu'après ce changement et de
nouveaux tests.

## Étape 4 — Valider offline

Sur le holdout indépendant, comparer V003 et la candidate sur :

- accord avec V008 dans la slice ciblée ;
- cross-entropy/top-1 ;
- dérive d'argmax V003 → candidate ;
- nombre et proportion de décisions modifiées hors slice ;
- ECE et Brier ;
- phase, action, carte et cardinalité.

Arrêter si le gain ciblé s'accompagne d'une dérive hors slice supérieure au budget
pré-déclaré ou d'une dégradation nette sur les slices protégées.

## Étape 5 — Screening en partie

Jouer la candidate contre Random, V007, V008 et Neural V001–V003, avec environ 20 parties
par adversaire. Ce screening sert uniquement à détecter une régression manifeste.

Le benchmark doit conserver les mêmes seeds entre V003 et la candidate lorsque la
comparaison est appariée.

## Étape 6 — Validation complète

Si le screening est propre, lancer 100–200 parties par adversaire, idéalement sur plusieurs
seeds de campagne. Appliquer la gate actuelle :

- groupe neural de poids total `1`, soit `1/3` pour V001, V002 et V003 ;
- poids Random `0,5`, V007 `1`, V008 `2` ;
- garde dure de non-régression contre V008 ;
- comparaison directe de la candidate contre V003.

## Décision

- **Promouvable** : gain de qualité en partie, pas de régression V008, dérive maîtrisée
  et amélioration expliquée par la slice ciblée.
- **Rejetée** : amélioration offline sans gain en partie, régression V008 ou dérive hors
  slice excessive.
- **À reprendre** : signal positif mais couverture ou puissance statistique insuffisante ;
  aucune variation de recette ne doit être introduite avant d'avoir fermé cette incertitude.

## Commandes attendues

Les commandes exactes seront écrites après l'étape d'infrastructure et le choix de la
slice. Le benchmark baseline actuellement conservé est :

```text
artifacts/neural_benchmark/neural_panel.json
artifacts/neural_benchmark/neural_panel.html
```
