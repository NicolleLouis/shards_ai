# Représentation sémantique des cartes pour le réseau neural

## Objectif

Définir une représentation canonique, sérialisable et extensible des cartes que le futur réseau
pourra transformer en card embedding. Cette représentation doit conserver l'identité explicite des
cartes tout en exposant leurs propriétés et leurs effets structurés, afin de ne pas limiter la
généralisation à des identifiants opaques.

Ce step prépare la donnée d'entrée du futur encodeur. Il ne définit pas encore la dimension des
embeddings, les poids appris, la fonction de perte ni le modèle `score(observation, action)`.

## Résumé du besoin

Le réseau doit reconnaître une carte précise comme `drakonarius`, car certains effets ou protections
concernent une autre carte identifiée explicitement. En parallèle, une nouvelle carte doit pouvoir
être rapprochée de cartes existantes grâce à ses propriétés : faction, coût, ressources produites,
conditions, effets, capacités et rôle de champion.

Les champions et les cartes ordinaires utilisent le même format de représentation. Le contexte de
l'action (`PlayCard`, `ActivateChampion`, achat, cible, etc.) sera encodé séparément au step 8.d.

## État actuel

Le moteur possède déjà une représentation déclarative et immuable :

- `CardDefinition` contient l'identité, le coût, la faction, le bouclier, le type champion, les PV
  imprimés, l'effet de pose, la capacité de champion, le passif et le statut mercenaire ;
- `Effect` contient des effets simples ou des branches conditionnelles ;
- `EffectStep` contient un seuil de maîtrise et une séquence d'opérations ;
- `Operation` contient le type d'effet, ses valeurs, sa cible, ses seuils, ses contraintes de
  faction et ses conditions (`union`, `echo`, `domination`, `inspiration`) ;
- `ChampionAbility` contient le type de capacité, ses valeurs, seuils, faction éventuelle et
  contraintes de domination ;
- `CARD_CATALOG` fournit une définition par `card_id`.

La définition contient aussi `name`, mais aucune description humaine structurée n'est disponible ni
nécessaire pour cette phase. Les noms ne seront pas utilisés comme signal principal.

## Comportement cible

Chaque `CardDefinition` doit pouvoir être convertie en un descripteur neural stable et déterministe.
Deux instances physiques de la même définition doivent produire exactement le même descripteur. Le
descripteur ne doit jamais dépendre de `CardInstance.instance_id` ni de la zone dans laquelle la
carte se trouve.

Le descripteur conservera :

- `card_definition_id` comme identité directe ;
- les attributs statiques de la carte ;
- les effets immédiats et conditionnels ;
- les capacités et passifs de champion ;
- une version de schéma pour rendre les datasets et checkpoints interprétables.

## Hors périmètre

- apprendre les poids d'un embedding ;
- choisir la taille des vecteurs ou l'architecture MLP/Transformer ;
- encoder une action légale ;
- encoder l'état complet d'une partie ;
- ajouter des descriptions textuelles humaines ;
- modifier la sémantique des règles ou corriger une carte dont la définition serait incorrecte ;
- assurer le zéro-réentraînement complet après ajout d'une carte.

## Décisions clés

- La représentation sera hybride : identité explicite + attributs structurés.
- `card_definition_id` est conservé pour les relations précises entre cartes, par exemple
  Drakonarius et Général Décurion.
- L'identité ne doit pas être la seule voie d'information. Les attributs structurés doivent rester
  exploitables si une carte est nouvelle ou si ses effets sont modifiés.
- L'identité d'une carte est un identifiant de définition, jamais un `instance_id` physique.
- Les cartes ordinaires et les champions utilisent le même descripteur racine.
- Les différences des champions sont exprimées par `is_champion`, `champion_health`, `on_play_effect`,
  `champion_ability` et `passive_kind`.
- Les effets sont représentés par leurs données structurées, pas par une description humaine.
- L'ordre des opérations dans une étape est conservé, car le moteur les résout dans cet ordre.
- L'ordre des `EffectStep` est conservé dans le format sémantique, même si le moteur les normalise
  déjà par seuil de maîtrise.
- Les catégories et valeurs sont sérialisées avec les valeurs stables des enums et des chaînes de
  `OperationKind`/`ChampionAbilityKind`.
- Le descripteur est immuable après construction et peut être mis en cache par `card_definition_id`.
- Une nouvelle carte ou une carte modifiée nécessite au minimum la régénération du descripteur et un
  réentraînement minimal du modèle ; elle ne doit pas rendre le format incompatible.

## Représentation proposée

Les types proposés sont des structures de données, indépendantes des tenseurs :

```text
CardSemanticRepresentation
  schema_version
  card_definition_id
  cost
  faction
  shield
  is_champion
  champion_health
  is_mercenary
  effect
  on_play_effect
  champion_ability
  passive_kind

EffectRepresentation
  flat_gems
  flat_power
  steps: tuple[EffectStepRepresentation, ...]

EffectStepRepresentation
  mastery_at_least
  operations: tuple[OperationRepresentation, ...]

OperationRepresentation
  kind
  amount
  target
  mastery_at_least
  health_at_least
  faction
  requires_union
  requires_echo
  requires_domination
  requires_inspiration
  recruit_to_hand_at_mastery

ChampionAbilityRepresentation
  kind
  amount
  threshold
  faction
  secondary_amount
  draw_amount
  requires_domination
```

Les champs absents sont représentés par `None`, `False`, `0` ou une séquence vide selon leur type.
Cette normalisation permet au futur encodeur de traiter une carte ordinaire et un champion à travers
la même interface.

Le champ `card_definition_id` sera ensuite traité par une voie d'identité apprise. La voie
structurée encodera les autres champs. Pour une carte inconnue du vocabulaire appris, l'encodeur
pourra utiliser un token d'identité `UNK` tout en exploitant son descripteur structuré.

## Architecture proposée

Créer un module dédié, proposé sous le nom `shards_ai/ai/card_representation.py`, avec :

- des dataclasses immuables de représentation ;
- une fonction `representation_for_definition(definition)` ;
- une conversion explicite de `Effect`, `EffectStep`, `Operation` et `ChampionAbility` ;
- un cache optionnel par `card_definition_id` ou par définition immuable ;
- une fonction de sérialisation déterministe destinée aux datasets et diagnostics.

Le catalogue reste la source de vérité des règles et des cartes. Le module neural ne doit pas
reconstruire la sémantique en inspectant des noms de fichiers ou des descriptions humaines.

Le chemin d'utilisation sera :

```text
CardDefinition
    ↓ conversion déterministe
CardSemanticRepresentation
    ↓ futur encodeur neural
card embedding
    ↓ observation/action model
score(observation, action)
```

La conversion doit être pure : elle ne modifie ni `CardDefinition`, ni `CardInstance`, ni le
catalogue. Le cache ne doit pas permettre qu'une modification d'un objet mutable influence une
représentation déjà publiée ; les définitions actuelles sont immuables.

## Identité et généralisation

L'identité directe est nécessaire, mais crée un risque de mémorisation : un embedding appris par
`card_definition_id` peut traiter une nouvelle carte comme inconnue. La voie structurée limite ce
risque, mais ne garantit pas qu'un modèle non réentraîné saura jouer correctement avec une carte
totalement nouvelle.

Le contrat retenu est donc :

- identité disponible pour les cartes connues et les relations précises ;
- représentation structurée disponible pour les cartes nouvelles ou modifiées ;
- réentraînement minimal accepté après évolution du catalogue ;
- version de schéma et version de catalogue conservées dans les artefacts d'entraînement.

## Performance et évolutivité

La conversion d'une carte ne doit pas être exécutée à chaque décision pour chaque copie physique.
Elle doit partir de `CardDefinition` et être réutilisable pour toutes les instances ayant le même
`card_definition_id`.

Le cache des représentations est prioritaire sur une duplication de la représentation dans chaque
`CardInstance`. La conversion est suffisamment petite pour être calculée au chargement du catalogue
ou paresseusement au premier accès.

Les effets étant structurés et de taille faible, une représentation en tuples imbriqués est
préférable à du JSON reconstruit à chaque décision. La sérialisation JSON pourra être dérivée pour
les datasets et diagnostics.

## Cas particuliers

- carte ordinaire sans effet structuré : `effect` contient ses valeurs plates ;
- carte avec plusieurs branches de maîtrise : toutes les branches sont conservées ;
- opération avec condition de faction : conserver `faction` et la contrainte associée ;
- champion avec effet de pose et capacité active : conserver les deux séparément ;
- champion avec passif protecteur : conserver `passive_kind` ;
- carte sans faction explicite : conserver `None`, notamment pour les cartes neutres si applicable ;
- effet de victoire : conserver `kind="win"` sans inventer de valeur numérique ;
- nouveau type d'opération ou de capacité : le convertisseur doit échouer explicitement tant que le
  schéma ne le supporte pas, afin d'éviter un dataset silencieusement incomplet.

## Stratégie de test

Ajouter des tests qui vérifient :

- déterminisme de la représentation pour une même définition ;
- indépendance vis-à-vis de `CardInstance.instance_id` ;
- présence de l'identité exacte de `drakonarius` ;
- conservation des attributs statiques de toutes les cartes du catalogue ;
- conservation de toutes les branches de maîtrise et de l'ordre des opérations ;
- conservation des contraintes `union`, `echo`, `domination`, `inspiration`, santé et maîtrise ;
- conservation des capacités et passifs de tous les champions ;
- représentation correcte des effets de pose ;
- sérialisation stable et versionnée ;
- cache équivalent à une conversion sans cache ;
- échec explicite d'un type sémantique non supporté.

## Questions ouvertes

Aucune question bloquante. La dimension des embeddings, la stratégie de perte et le traitement
précis du token `UNK` seront décidés lors des steps modèle et entraînement.

## Fichiers attendus

- `shards_ai/ai/card_representation.py` — descripteurs sémantiques et conversion ;
- `shards_ai/ai/__init__.py` — exports publics éventuels ;
- `tests/ai/test_card_representation.py` — tests du descripteur et du cache ;
- `doc/Current state/Heuristic player.md` — aucune modification attendue sauf impact indirect ;
- `doc/Current state/Game engine.md` — aucune modification attendue si le moteur ne change pas ;
- `doc/Roadmap.md` — passage du step 8.c à `DONE` après validation.

## Validation attendue

Exécuter les tests ciblés de représentation, les tests du moteur et la suite complète. Vérifier que
la conversion ne modifie aucune règle, n'ajoute aucun texte humain requis et n'introduit pas de
coût par instance physique dans les parties existantes.
