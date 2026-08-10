# TODO — Expérience 1 : cardinalités de l'observation neuronale

## Statut

À étudier. Aucun changement de code, de checkpoint actif ou de règle du moteur n'est demandé par
cette note.

## Question

Le pooling moyen des cartes perd-il une information utile parce qu'il ne distingue pas deux zones
homogènes de tailles différentes ? L'ajout de cardinalités explicites améliore-t-il la politique
neuronale sans dégrader les décisions existantes ?

## Hypothèse falsifiable

À représentation de cartes identique, les cardinalités des zones permettent au réseau de mieux
traiter les achats, bannissements et décisions qui dépendent de la vitesse de recyclage du deck.
Le gain doit rester visible hors échantillon et en partie, pas uniquement dans la loss offline.

## Périmètre de la candidate

Ajouter uniquement des scalaires dérivés de `NeuralObservation`, sans modifier le moteur ni le
masque d'information :

- joueur actif : tailles de `draw_pile_counts`, `discard_counts` et `champions` ;
- joueur actif : total de `owned_card_counts` ;
- adversaire : total de `owned_card_counts`, taille de `discard_counts`, nombre de champions ;

Les cardinalités de la rivière et du deck central sont volontairement exclues : elles décrivent
l'offre publique disponible, pas la structure du deck du joueur. Leur utilité stratégique et leur
interaction avec les achats constituent une autre question expérimentale, hors de cette étape.

La taille de la main et la taille de `play_zone` sont également exclues. La main est déjà
représentée par son contenu et les actions légales ; `play_zone` est largement redondante avec le
total possédé et les autres zones actives. Elles pourront faire l'objet d'une ablation ultérieure
si une analyse montre une information résiduelle.

Les cartes restent poolées comme aujourd'hui. L'expérience ne contient aucun compte de faction,
aucun indicateur Union/Echo/Domination et aucune nouvelle information adverse cachée.

## Décisions à prendre dans l'architecture

Avant le code, documenter :

1. la liste et l'ordre exacts des scalaires ;
2. les bornes théoriques, le clipping et la normalisation de chaque scalaire ;
3. la distinction entre taille de zone et total possédé ;
4. un `observation_feature_set` explicite dans la configuration du modèle ;
5. la nouvelle identité d'architecture et l'incompatibilité avec les anciens checkpoints.

Le feature set historique doit rester disponible pour charger V001–V004. Le nouveau modèle doit
être entraîné comme candidate depuis le checkpoint actif uniquement selon une stratégie
d'initialisation explicitement définie ; le checkpoint stable actif ne doit jamais être écrasé.

## Tests techniques

- Deux observations avec le même pooling mais des cardinalités différentes produisent des
  scalaires différents.
- Les comptes vides, les valeurs maximales et les valeurs hors borne sont déterministes.
- Les cardinalités adverses ne proviennent que des zones déjà visibles.
- Les observations et datasets historiques restent désérialisables.
- Les architectures historiques chargent leur ancienne dimension d'état.
- Un forward candidate produit des scores finis pour toutes les actions légales.

## Évaluation

Même dataset, split par `game_id`, teacher, recette et seeds que V004. Ventiler au minimum par
phase, type d'action, taille de deck et cardinalité de l'ensemble légal. Mesurer : top-1, regret,
dérive d'argmax, calibration, taille finale du deck, nombre d'actions et latence séparée de
l'encodage d'état et du scoring.

Une candidate ne passe au panel que si le smoke test et le holdout sont propres. La promotion suit
la gate panel en vigueur : moyenne pondérée strictement positive contre tous les adversaires
pondérés, sans transformer V008 en garde dure indépendante.
