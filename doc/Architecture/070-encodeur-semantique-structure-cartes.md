# Encodeur sémantique structuré des cartes

## Objective

Remplacer la réduction actuelle de la sémantique d'une carte par des compteurs de types d'effets
par un encodeur qui transmet au réseau la structure utile des effets : ordre des opérations,
branches de maîtrise, montants, cibles, seuils et contraintes.

Le résultat attendu est un `card_embedding` exploitable par le joueur neural pour les cartes
présentes dans l'état et pour les cartes candidates d'une action. Une carte qui pioche une carte et
une carte qui en pioche trois doivent notamment produire des signaux différents.

Le succès sera évalué par une comparaison contrôlée avec l'encodeur actuel : même dataset, split,
seed, calendrier d'entraînement, architecture de scorer et protocole de parties.

## Current State

`shards_ai/ai/card_representation.py` construit déjà un descripteur immuable et sérialisable
`CardSemanticRepresentation`. Il conserve les attributs statiques, les effets principaux et de
pose, les étapes conditionnelles, les opérations détaillées, les capacités de champion et les
passifs.

`shards_ai/ai/neural_model.py` réduit ensuite ce descripteur dans `_semantic_features` : 12 scalaires,
4 indicateurs de faction, 17 compteurs de types d'opérations et 12 compteurs de types de capacités,
soit 45 valeurs. Les montants et paramètres de chaque opération ne sont donc pas transmis. Ainsi,
`draw_card(amount=1)` et `draw_card(amount=3)` ont le même signal sémantique si le reste de la carte
est identique.

Le modèle utilise ensuite un MLP `45 -> semantic_hidden_dim -> card_embedding_dim`, fusionné avec
un embedding appris de `card_definition_id`. Les cartes sont ensuite poolées dans l'observation ou
encodées comme carte d'une action. Les architectures historiques sont distinctes par leur contexte
d'actions : V001 utilise `independent_action`, V002 utilise `global_candidate_context`, et V003
utilise `semantic_identity_v3`. Leur comportement doit rester figé même si V4 reçoit un nouvel
encodeur.

Les profils et checkpoints historiques V001/V002/V003 doivent rester chargeables et inchangés.

## Target Behavior

Un encodeur structuré produit un embedding déterministe pour une définition de carte donnée, à
poids de réseau fixés, sans dépendre de `CardInstance.instance_id`, de la zone de la carte ou du
joueur qui l'observe.

Il encode séparément :

- les attributs statiques de la carte ;
- l'effet principal ;
- l'effet de pose éventuel ;
- la capacité de champion éventuelle ;
- le passif éventuel ;
- l'identité de définition, par la voie d'identité déjà existante.

L'ordre des opérations et des étapes est conservé. Les séquences de longueur variable sont traitées
par padding et masques, sans introduire de faux effets.

## Non-Goals

- Modifier les règles du moteur ou les définitions du catalogue.
- Ajouter une description textuelle ou utiliser `CardDefinition.name`.
- Encoder l'état dynamique dans l'embedding statique d'une carte.
- Remplacer le scorer action-conditionné ou son espace d'actions.
- Réentraîner ou réécrire les checkpoints stables existants.
- Garantir qu'une carte totalement nouvelle sera jouée correctement sans réentraînement.

## Key Decisions

1. La structure complète de `CardSemanticRepresentation` est la source d'entrée ; le réseau ne
   reconstruit pas la sémantique à partir de noms ou de fichiers.
2. Chaque opération est encodée individuellement. Son type, sa cible et sa faction utilisent des
   embeddings catégoriels ; ses montants, seuils et contraintes utilisent des valeurs numériques
   normalisées.
3. Les `EffectStep` sont encodés dans l'ordre avec `mastery_at_least`. Les opérations d'une étape
   restent ordonnées, car cet ordre est celui de résolution du moteur.
4. `effect`, `on_play_effect` et `champion_ability` ont des voies ou tokens de type distincts afin
   qu'un effet de pose ne soit pas confondu avec l'effet permanent.
5. Les champs absents utilisent un token ou un masque explicite. Ils ne sont pas remplacés par une
   opération fictive.
6. `card_definition_id` reste une voie d'identité séparée. Une carte inconnue utilise l'index `UNK`
   tout en conservant son encodage structuré.
7. L'encodeur statique est pré-calculé et mis en cache par `card_definition_id` en inférence. Le
   chemin d'entraînement reste différentiable et ne réutilise pas un embedding calculé avant une
   mise à jour des poids.
8. L'architecture reçoit un nom et des dimensions explicites dans les métadonnées du checkpoint,
   par exemple `structured_semantic_v4`. Elle ne remplace pas implicitement une architecture
   historique.
9. Le premier modèle utilise une attention multi-tête locale sur les opérations et les étapes,
   plutôt qu'un Transformer global. Les séquences de cartes étant courtes, cette attention reste
   limitée aux tokens d'un effet ou d'une carte et son nombre de têtes est configurable.
10. `passive_kind` est encodé comme une catégorie, avec un token `NONE` pour l'absence de passif et
    un token `UNK` pour un type non présent dans le vocabulaire. Le réseau peut ainsi distinguer
    les protections et immunités concrètes du catalogue.
11. Les implémentations des encodeurs historiques sont isolées des nouveaux encodeurs. V001
    conserve son scorer et son contexte `independent_action`; V002 conserve son scorer et son
    contexte `global_candidate_context`; V003 conserve son architecture explicitement versionnée.
    V4 ne modifie aucune de ces classes et est construit par un chemin dédié.
12. Les contrats stables partagés restent limités à `NeuralObservation`, `ActionRepresentation`,
    `NeuralModelConfig`, le format de checkpoint et le chargeur. Le moteur de jeu, les observations,
    les actions et les benchmarks génériques ne sont pas dupliqués.

## Open Questions

- Non bloquant : quelles bornes exactes retenir pour les montants extrêmes du catalogue ? Les valeurs
  doivent être documentées et écrêtées de façon déterministe.

## Proposed Architecture

### Séparation des générations

Le code des modèles est organisé par génération, sans copier l'ensemble du joueur neural :

```text
shards_ai/ai/neural/
  loader.py                 # sélection par architecture du checkpoint
  legacy_v001.py            # scorer indépendant, contexte independent_action
  legacy_v002.py            # scorer avec contexte global des candidats
  legacy_v003.py            # architecture semantic_identity_v3
  structured_v004.py        # encodeur sémantique structuré et scorer V4
```

Les noms de fichiers sont indicatifs ; la séparation logique est obligatoire. `neural_player.py`
reste l'API publique et délègue la construction au loader. Les classes historiques ne doivent pas
hériter d'une base dont le chemin d'encodage est modifié par V4. Une petite base de contrats ou des
fonctions utilitaires sans état peuvent être partagées, mais jamais l'implémentation mutable de
l'encodeur sémantique.

Le choix du contexte d'actions est une propriété du checkpoint :

| Version | Architecture | Contexte des actions |
| --- | --- | --- |
| V001 | `independent_action` | chaque action est encodée indépendamment |
| V002 | `global_candidate_context` | chaque action reçoit le résumé des actions candidates |
| V003 | `semantic_identity_v3` | chemin historique explicitement versionné |
| V004 | `structured_semantic_v4` | encodeur de carte structuré ; contexte défini par le profil candidat |

Le loader refuse une architecture inconnue et reconstruit le modèle correspondant aux métadonnées
du checkpoint. Aucun fallback implicite vers V4 ne doit être ajouté.

### Attributs statiques

`StaticCardEncoder` encode coût, bouclier, statut champion/mercenaire, PV de champion, faction et
passif. Les catégories utilisent des vocabulaires versionnés ; `passive_kind` possède les tokens
`NONE` et `UNK`. Les scalaires sont normalisés et écrêtés.

### Opération

`OperationEncoder` reçoit :

```text
kind, target, faction
amount, mastery_at_least, health_at_least, recruit_to_hand_at_mastery
requires_union, requires_echo, requires_domination, requires_inspiration
```

Les valeurs `None` sont accompagnées d'un masque ou d'un token absent. Une opération
`draw_card(amount=3)` transmet donc explicitement son montant `3`, en plus de son type.

### Étape et effet

`StepEncoder` combine le seuil de maîtrise de l'étape et la séquence d'embeddings d'opérations avec
une attention multi-tête masquée. Chaque tête peut apprendre une relation différente entre les
opérations, par exemple leur ordre, leurs seuils ou leur complémentarité. Le masque interdit de
prendre en compte les positions de padding.
`EffectEncoder` combine les valeurs plates `flat_gems`/`flat_power` et la séquence d'étapes. Les
effets principal et de pose partagent les poids d'encodage, mais reçoivent un token de rôle
différent.

### Capacité et carte

`ChampionAbilityEncoder` encode `kind`, `amount`, `threshold`, `faction`,
`secondary_amount`, `draw_amount` et `requires_domination`. Un token de rôle absent masque la voie
pour les cartes ordinaires.

`CardSemanticEncoder` concatène les sorties statiques, effet principal, effet de pose, capacité et
passif, puis applique un MLP vers `card_embedding_dim`. Ce vecteur est fusionné avec
`card_id_embedding` comme dans le modèle actuel.

```text
CardDefinition
  -> CardSemanticRepresentation
  -> static / operation / step / effect / ability encoders
  -> structured semantic embedding
  + card_id embedding
  -> fused card embedding
  -> observation pooling or action encoder
```

## Data Model

Aucune table ni donnée persistée n'est nécessaire. Le contrat existant
`CardSemanticRepresentation` est étendu seulement si un champ du catalogue manque ; aucune
information ne doit être supprimée pour simplifier l'encodeur.

Les vocabulaires catégoriels, les bornes numériques, la version de schéma de représentation et le
nom d'architecture sont enregistrés dans les métadonnées du profil et du checkpoint. Les tenseurs
statiques pré-calculés restent des caches runtime et ne sont pas ajoutés au vault `doc/` ni au dépôt
comme artefacts générés.

## Backend Flow

À l'initialisation du scorer V4, le catalogue est converti en représentations sémantiques puis en
entrées tokenisées statiques. En inférence, ces entrées sont encodées une fois par carte et
conservées dans un cache non persistant. Pour l'entraînement, elles sont reconstruites dans le graphe
courant afin que les gradients traversent l'encodeur.

Le loader sélectionne d'abord la génération. Les chemins V001/V002/V003 restent inchangés et
continuent d'utiliser leurs contextes d'actions historiques. Le chemin V4 demande un embedding par
`card_definition_id`; son pooling des zones et son encodage des actions sont isolés dans son
implémentation. Les contrats d'observation et d'action restent communs, sans fuite d'informations
privées.

Une représentation contenant un type d'opération ou de capacité inconnu doit échouer explicitement,
comme le fait déjà `card_representation.py`, plutôt que d'être silencieusement ignorée.

## Frontend Flow

Sans objet : cette évolution concerne exclusivement le moteur neural Python et ses artefacts
d'entraînement.

## Authorization And Feature Gates

Sans objet côté autorisation. L'activation se fait par le choix explicite de l'architecture dans le
profil neural. `configs/neural_profiles/active.yaml` reste inchangé jusqu'à validation comparative.

## Observability And Operations

Les rapports d'entraînement et de benchmark doivent enregistrer :

- nom d'architecture et version de représentation ;
- dimensions et vocabulaires de l'encodeur ;
- nombre maximal d'étapes/opérations et taux de padding ;
- temps d'encodage et temps total d'inférence ;
- métriques offline et résultats en parties.

Les diagnostics doivent permettre de vérifier qu'un montant de pioche différent produit bien des
entrées distinctes avant l'entraînement. Une anomalie de dimension ou de vocabulaire doit bloquer
le chargement du checkpoint.

## Edge Cases

- Carte sans effet : tokens et masques absents, mais attributs statiques conservés.
- Carte avec plusieurs branches de maîtrise : toutes les étapes sont conservées dans leur ordre.
- Carte sans faction, passif ou capacité : token absent explicite.
- Effet avec plusieurs `draw_card` : chaque opération et son montant restent distincts.
- Montant supérieur à la borne : écrêtage documenté, avec métrique de détection.
- Opération ou capacité inconnue : erreur explicite lors de la construction de la représentation.
- Carte inconnue du vocabulaire d'identité : embedding `UNK` et voie structurée conservée.
- Ajout du catalogue : le cache est invalidé et un checkpoint compatible avec le nouveau catalogue
  doit être produit.

## Testing Strategy

Ajouter des tests unitaires couvrant :

- différence d'entrée entre `draw_card(amount=1)` et `draw_card(amount=3)` ;
- différence d'entrée entre deux opérations de même type mais de cibles ou montants différents ;
- conservation de l'ordre des opérations et des étapes ;
- conservation des seuils de maîtrise/santé et des contraintes booléennes ;
- conservation de `draw_amount` dans une capacité de champion ;
- masquage correct des effets absents ;
- déterminisme du pré-encodage et invalidation du cache ;
- longueur variable, padding et batch ;
- compatibilité de chargement des architectures historiques ;
- chargement et score d'une observation avec les checkpoints V001 et V002, en vérifiant leurs
  contextes d'actions respectifs ;
- vérification qu'une modification de l'encodeur V4 ne modifie pas les sorties des chemins
  historiques à poids et entrées identiques ;
- égalité des sorties entre chemin cache inférence et chemin non mis en cache.

Un benchmark contrôlé comparera ensuite l'ancien et le nouvel encodeur sur le même protocole. La
validation doit inclure une partie smoke test, les benchmarks offline et des parties contre les
adversaires de référence. Une amélioration inférieure à un seuil robuste de 2 % ne sera pas attribuée
à l'encodeur sans répétitions et analyse de variance suffisantes.

## Rollout And Migration

1. Isoler les classes et contextes V001/V002/V003 derrière le loader, sans modifier leur chemin
   d'encodage ni leur format de checkpoint.
2. Implémenter l'encodeur V4 derrière une nouvelle architecture explicite.
3. Ajouter les tests de représentation, d'encodage et de non-régression historique.
4. Produire un checkpoint candidat dans le chemin de travail mutable canonique
   `artifacts/neural_training/checkpoint.pt`, sans créer de checkpoint de session supplémentaire.
5. Comparer qualité, vitesse, mémoire et stabilité avec les profils existants.
6. Si le candidat est validé, le copier vers `configs/neural_profiles/vNNN.pt` et enregistrer son
   profil ; le fichier mutable reste réservé à la version suivante.
7. Ne modifier `configs/neural_profiles/active.yaml` qu'après validation explicite.

Le rollback consiste à sélectionner un profil historique. Aucun dataset historique ne doit être
réécrit ; les nouveaux datasets enregistrent la version de représentation et l'architecture qui les
ont produits.

## Files Expected To Change

- `shards_ai/ai/neural/loader.py` — sélection explicite par architecture (chemin indicatif).
- `shards_ai/ai/neural/legacy_v001.py` — isolation du scorer V001 (chemin indicatif).
- `shards_ai/ai/neural/legacy_v002.py` — isolation du scorer V002 (chemin indicatif).
- `shards_ai/ai/neural/legacy_v003.py` — isolation du scorer V003 (chemin indicatif).
- `shards_ai/ai/neural/structured_v004.py` — encodeur structuré (chemin indicatif).
- `shards_ai/ai/neural_model.py` — refactor éventuel du loader et des contrats existants.
- `shards_ai/ai/card_representation.py` — uniquement si un champ sémantique requis manque.
- `tests/ai/test_neural_model.py` — dimensions, masques, cache et sorties structurées.
- `tests/ai/test_card_representation.py` — cas de pioche et champs détaillés si nécessaire.
- `configs/neural_training_profiles/candidates/` — profil candidat versionné.
- `doc/Current state/` — mise à jour uniquement après implémentation et validation.

Références historiques : `doc/Architecture/030-representation-semantique-cartes-reseau-neural.md`
et `doc/Architecture/066-architecture-embedding-cartes-v003.md`.
