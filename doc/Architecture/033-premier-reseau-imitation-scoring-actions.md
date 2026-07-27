# Premier réseau neural d'imitation et scoring des actions

## Objectif

Implémenter et entraîner le premier modèle neural capable de calculer un score pour chaque action
légale à partir d'une observation partiale :

```text
score(observation, action) -> scalaire
```

Le joueur neural évaluera toutes les actions légales du point de décision, choisira le meilleur
score et départagera les égalités avec sa source aléatoire contrôlée. Cette phase reste de
l'imitation supervisée ; le reinforcement learning et le self-play sont hors périmètre.

## État actuel

Les fondations suivantes sont disponibles :

- observation partielle versionnée avec `Game.neural_observation_for()` ;
- représentation sémantique des cartes, incluant identité et effets structurés ;
- représentation action-conditionnelle versionnée ;
- générateur JSONL d'imitation avec scores heuristiques, actions légales et résultats de partie ;
- profils heuristiques versionnés et benchmarks reproductibles.

Le dépôt ne contient actuellement ni PyTorch, ni NumPy, ni autre framework neural dans Poetry. Le
premier modèle nécessite donc une dépendance explicite et un protocole d'installation reproductible.

## Comportement cible

Pour une observation et `N` actions légales :

1. encoder l'observation une seule fois ;
2. encoder les `N` actions en batch ;
3. concaténer l'état encodé avec chaque action ;
4. produire `N` scores scalaires ;
5. retourner l'action originale ayant le meilleur score.

Le modèle ne génère pas d'action nouvelle et ne contourne jamais `Game.legal_actions()`.

## Hors périmètre

- reinforcement learning, PPO et self-play ;
- recherche d'arbre ou simulation prospective ;
- ajout d'une récompense intermédiaire ;
- modification des règles du moteur ;
- modèle qui planifie un tour complet ;
- publication automatique d'un modèle comme adversaire de référence.

## Décisions clés

- PyTorch est le framework recommandé pour le premier modèle, avec exécution CPU compatible et GPU
  optionnelle.
- Le modèle est action-conditionnel : il ne possède pas une tête de sortie statique par action.
- La représentation de l'observation et de l'action reste la source de vérité des entrées ; le
  modèle ne reçoit jamais `GameState` complet.
- L'encodeur de carte réutilise l'identité `card_definition_id` et les attributs/effets structurés
  du step 8.c.
- L'identité de carte est complétée par une voie structurée, avec token `UNK` pour une carte absente
  du vocabulaire appris.
- Les zones de cartes utilisent des agrégations de type ensemble/multiensemble ; aucune information
  d'ordre caché n'est reconstruite.
- Le classement relatif est la cible principale. La régression directe des scores bruts reste une
  expérience optionnelle car leur échelle dépend du profil et du contexte.
- La première loss recommandée combine une préférence paire-à-paire dérivée des scores heuristiques
  et une loss de classification vers l'action réellement choisie. Les poids seront configurables.
- Les paires d'actions sont échantillonnées lorsque leur nombre devient trop élevé ; les actions
  légales restent toutes présentes dans le dataset et à l'inférence.
- Les partitions train/validation/test sont faites par `game_id` ou seed, jamais par décision.
- Le modèle et le vocabulaire sont sauvegardés avec les versions des schémas, du catalogue et du
  dataset.

## Architecture du modèle

### Encodeur de carte

Construire un encodeur partagé pour toutes les cartes :

- embedding catégoriel de `card_definition_id` avec ligne `UNK` ;
- embeddings ou encodages catégoriels pour faction, type et passif ;
- projection des attributs numériques normalisés : coût, bouclier, PV de champion et indicateurs ;
- encodage des effets structurés et de leurs opérations ;
- agrégation des branches et opérations en conservant les champs conditionnels.

Le même encodeur est utilisé pour les cartes de l'observation et les cartes ciblées par une action.
Les représentations de définitions immuables sont mises en cache avant l'entraînement et l'inférence.

### Encodeur d'observation

Produire un vecteur d'état à partir de :

- ressources et métriques scalaires normalisées ;
- somme pondérée des embeddings des cartes de la main active ;
- somme pondérée par quantité pour les pioches, défausses et compositions globales ;
- encodage slot-par-slot de la rivière ;
- agrégation des zones de jeu et champions visibles ;
- masque des factions jouées ;
- phase, statut et contexte de décision en embeddings catégoriels ;
- `turn_number` normalisé comme signal de progression public.

Les zones ayant une structure de comptage utilisent leur quantité comme multiplicité ou poids. Le
modèle ne reçoit pas d'ordre de pioche, de défausse ou de main adverse.

### Encodeur d'action

Chaque action est encodée avec :

- embedding du `action_type` ;
- embedding de la phase ;
- embedding de la carte ciblée via l'encodeur partagé, si présente ;
- embedding de la cible (`opponent`, champion, choix générique) ;
- slot de rivière normalisé ;
- montant normalisé ;
- `choice_id` résolu vers une carte publique lorsque possible, sinon token catégoriel générique.

L'`instance_id` est conservé pour faire correspondre le score à l'action moteur, mais n'est pas une
entrée stratégique principale du modèle.

### Scorer action-conditionnel

Le vecteur d'état est concaténé à chaque vecteur d'action puis passé dans un petit MLP partagé :

```text
observation -> state_encoder -> state_vector
action[i]   -> action_encoder -> action_vector[i]
(state_vector, action_vector[i]) -> scorer -> score[i]
```

Le scorer est partagé entre toutes les actions et tous les types d'action. Le batch d'actions permet
de calculer toutes les sorties d'une décision en une seule passe.

## Objectifs d'entraînement

Pour une décision contenant les scores heuristiques `h_i` et les prédictions `p_i` :

### Préférence paire-à-paire

Former des paires où `h_i > h_j` et entraîner `p_i > p_j` avec une loss logistique ou hinge. Les
paires peuvent être pondérées par l'écart de score ou limitées à un échantillon déterministe pour
contrôler le coût.

### Imitation de l'action choisie

Ajouter une loss de classification sur `chosen_action_index`, calculée sur les scores du modèle
parmi les actions légales de la décision. Cette cible reproduit la politique finale, y compris les
priorités qui ne sont pas entièrement représentées par le score brut.

### Configuration initiale

Le premier entraînement comparera au minimum :

- préférence seule ;
- préférence + imitation choisie ;
- préférence + imitation choisie + régression de scores normalisés par décision.

Les poids, la température et la normalisation ne seront pas figés dans le format du dataset. Chaque
expérience enregistrera sa configuration.

## Dataset et partitions

Le lecteur JSONL doit fonctionner en streaming ou via un cache prétraité, sans charger un million de
décisions en mémoire. La partition par parties doit être déterministe à partir de `game_id` ou d'une
seed de split indépendante.

Le split initial recommandé est :

- 80 % train ;
- 10 % validation ;
- 10 % test.

Les profils et types de rencontres doivent être représentés dans les trois partitions autant que
possible, sans déplacer des décisions d'une même partie entre partitions.

## Validation et critères de réussite

### Validation offline

Mesurer :

- top-1 agreement avec l'action heuristique choisie ;
- rang de l'action choisie ;
- exactitude paire-à-paire ;
- corrélation ou erreur des scores normalisés si cette sortie est activée ;
- métriques par phase, type d'action, profil et type de rencontre ;
- absence d'action illégale sélectionnée ;
- temps d'inférence par décision et par nombre d'actions.

### Validation en jeu

Après validation offline, intégrer un joueur neural et l'évaluer sur des seeds séparées contre :

- RandomPlayer ;
- le meilleur profil heuristique publié ;
- les profils présents dans le dataset mais absents de certains splits.

Le modèle ne devient pas la référence active sans comparaison reproductible et décision explicite.

## Performance et évolutivité

Le hot path est le scoring de toutes les actions d'une décision. Il doit :

- encoder l'observation une fois ;
- batcher les actions ;
- réutiliser les représentations de cartes ;
- éviter les conversions JSON pendant l'inférence ;
- utiliser `torch.no_grad()` en production ;
- mesurer séparément coût de préparation et coût du forward.

Un modèle compact est préféré au premier essai afin de mesurer la qualité avant d'augmenter la
profondeur. Les batchs d'entraînement peuvent regrouper des décisions de tailles différentes via un
collator et des listes d'actions aplaties avec offsets.

## Reproductibilité et artefacts

Chaque run d'entraînement doit enregistrer :

- seed Python/framework ;
- configuration du modèle ;
- configuration de loss ;
- split et seed de partition ;
- manifest du dataset ;
- version du catalogue et des schémas ;
- métriques train/validation/test ;
- checkpoint du meilleur modèle selon la métrique de validation.

Les checkpoints et métriques restent hors de `doc/`, sous `artifacts/neural_imitation/`.

## Questions ouvertes

- La version minimale de PyTorch et le support GPU devront être fixés dans `pyproject.toml` au début
  de l'implémentation. Recommandation : PyTorch stable compatible Python 3.11, CPU obligatoire et
  CUDA optionnelle.
- La taille exacte des couches et des embeddings sera choisie après un benchmark sur 5 000 décisions.
- Le traitement final des scores heuristiques de profils différents sera comparé offline avant le
  lancement à un million de décisions.

## Fichiers attendus

- `pyproject.toml` — dépendances neural explicites ;
- `shards_ai/ai/neural_model.py` — encodeurs et scorer ;
- `shards_ai/ai/neural_training.py` — dataset reader, collator, losses et boucle d'entraînement ;
- `scripts/train_neural_imitation.py` — entraînement et checkpoints ;
- `tests/ai/test_neural_model.py` — formes, gradients et scoring variable ;
- `tests/ai/test_neural_training.py` — dataset, loss, split et reproductibilité ;
- `artifacts/neural_imitation/` — checkpoints et métriques ;
- `doc/Roadmap.md` — passage du step 8.f à `DONE` après validation.

## Validation attendue

Installer la dépendance neural explicitement, exécuter un entraînement court sur le dataset de
validation, vérifier les métriques offline et la latence d'inférence, puis seulement envisager
l'entraînement sur le dataset d'un million de décisions.
