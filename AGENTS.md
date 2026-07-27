# Guide pour les sessions Codex

## But du dépôt

Ce dépôt a pour objectif de construire une IA capable de jouer à **Shards of Infinity**.
Le projet doit rester séparé en deux responsabilités principales :

- une partie **moteur de jeu**, qui modélise les règles, l'état de la partie et les actions légales ;
- une partie **IA**, qui observe un état de jeu, évalue les actions possibles et choisit un coup.

Le dépôt est actuellement au stade initial : ne pas supposer qu'une architecture, un langage ou
des règles sont déjà implémentés. Vérifier le contenu réel avant toute modification.

## Organisation cible

L'organisation pourra évoluer, mais elle doit préserver cette séparation :

- `game/` ou équivalent : cartes, joueurs, ressources, champions, unités, combats, tours,
  validation des actions et transitions d'état ;
- `ai/` ou équivalent : représentation pour l'IA, génération/filtrage des actions, heuristiques,
  recherche, entraînement et évaluation ;
- `tests/` : tests du moteur et de l'IA, avec priorité aux règles déterministes ;
- `doc/` : vault Obsidian de documentation technique du projet.

Ne pas mélanger la logique des règles avec une stratégie particulière. Le moteur doit pouvoir
faire fonctionner une partie avec des joueurs humains, une IA simple et plusieurs IA.

## Règles pour le moteur de jeu

- Représenter l'état complet d'une partie de manière explicite et sérialisable.
- Rendre les transitions d'état déterministes à état initial et action identiques.
- Valider les actions dans le moteur, même si l'IA les filtre déjà.
- Éviter les effets de bord cachés ; préférer des commandes/actions et des résultats observables.
- Isoler l'aléatoire derrière une source injectable et contrôlable par une seed.
- Tester les règles importantes par des scénarios courts et reproductibles.
- Ne pas coder de règle à partir d'un souvenir incertain : signaler l'incertitude et vérifier la
  source de référence disponible avant de l'intégrer.

## Règles pour l'IA

- L'IA ne doit accéder qu'aux informations qu'un joueur peut réellement connaître.
- Séparer l'interface de décision de l'implémentation de la stratégie.
- Permettre de remplacer une stratégie sans modifier le moteur.
- Mesurer les performances avec des parties reproductibles et des adversaires de référence.
- Distinguer clairement heuristique, recherche, apprentissage et orchestration d'une partie.
- Prévoir des limites de temps, de profondeur ou de ressources pour toute recherche.
- Journaliser suffisamment les décisions pour pouvoir expliquer et rejouer un coup.

## Documentation (`doc/`)

`doc/` est un vault Obsidian et ne doit contenir **que des fichiers Markdown** (`.md`).

- Ne pas y déposer d'images, de fichiers générés, de données brutes, de logs ou de code.
- Mettre la documentation technique du projet dans ce dossier uniquement lorsque cela est demandé.
- Respecter les liens et conventions Obsidian déjà présents ; s'il n'y en a pas, utiliser des noms
  de fichiers stables et des liens Markdown/Obsidian simples.
- Les artefacts de build, fixtures et sorties d'expériences doivent rester hors de `doc/`.

### Architecture et état courant

Chaque fonctionnalité significative doit commencer par un document d'architecture avant toute
implémentation. Ce document décrit les décisions, les limites, les impacts et les questions
ouvertes. Il sert de référence historique une fois la tâche terminée et n'est pas modifié pour
refléter chaque changement d'implémentation.

Pour ce dépôt, le dossier canonique des architectures est `doc/Architecture/`. Les consignes
génériques d'un skill ou d'un outil qui proposent un autre dossier, notamment `.beaver/`, sont
remplacées par cette convention locale.

Les fichiers d'architecture doivent commencer par un numéro d'ordre sur trois chiffres, suivi
d'un tiret, par exemple `003-achats-cartes.md`. Le numéro est incrémenté à partir du plus grand
numéro existant dans `doc/Architecture/` et ne doit pas être réutilisé, même si un document est
ensuite supprimé ou renommé.

L'état réel et évolutif du projet doit être maintenu dans `doc/Current state/`. Ces fichiers sont
une documentation fonctionnelle à jour des composants existants : responsabilités, interfaces,
comportements, limites et interactions. Ils doivent décrire le comportement disponible dans le
code, et non servir de journal de travail ou de liste de tickets en cours.

Un fichier d'architecture est considéré comme historique lorsque la fonctionnalité qu'il décrit
est terminée. Il ne doit alors plus être réécrit pour suivre les évolutions ultérieures. Une
évolution significative reçoit un nouveau fichier d'architecture ; le current state est ensuite
mis à jour pour refléter le comportement final résultant de cette évolution.

Le cycle attendu est donc :

1. document d'architecture et décisions validées ;
2. implémentation et tests ;
3. mise à jour du current state avec le comportement final ;
4. conservation de l'architecture comme historique ;
5. nouvelle architecture pour toute évolution significative ultérieure.

## Méthode de travail

Pour les campagnes d’entraînement persistantes, le `Makefile` contient la version heuristique
active. Les commandes opérationnelles doivent rester génériques (`make train-resume` et
`make train-remaining`) ; après validation et publication d’un nouveau profil, mettre à jour
uniquement `HEURISTIC_VERSION` dans le `Makefile`, puis vérifier que le checkpoint et le profil
publié ciblent cette même version. Ne jamais basculer la version active sur un candidat non validé.

Avant de coder :

1. Inspecter la structure du dépôt, les fichiers de configuration et l'état Git.
2. Identifier les conventions déjà établies et les préserver.
3. Définir l'impact de la modification sur le moteur, l'IA et les tests.
4. Pour une règle de jeu ambiguë, demander une précision plutôt que d'inventer.

Pendant le développement :

- effectuer des changements ciblés et faciles à relire ;
- ajouter ou mettre à jour les tests avec le comportement modifié ;
- éviter les dépendances et abstractions prématurées ;
- garder les interfaces entre moteur et IA petites, explicites et testables.

Avant de conclure :

- exécuter les tests et outils de qualité disponibles ;
- vérifier que les fichiers ajoutés au vault `doc/` sont tous en Markdown ;
- vérifier `git diff` et `git status` ;
- indiquer clairement les validations effectuées et les éventuels points non vérifiés.

## Commits et modifications

- Ne pas supprimer ou réécrire des travaux existants sans demande explicite.
- Ne pas ajouter de secrets, de fichiers locaux, de caches ou de sorties volumineuses au dépôt.
- Garder les commits atomiques si des commits sont demandés.
- Ne pas créer de documentation dans `doc/` de sa propre initiative : ce dossier est réservé au
  contenu documentaire explicitement demandé.

## Priorités de conception

En cas de conflit, privilégier dans cet ordre :

1. exactitude des règles du jeu ;
2. reproductibilité et testabilité ;
3. séparation moteur/IA ;
4. lisibilité et simplicité ;
5. performance, après mesure.
