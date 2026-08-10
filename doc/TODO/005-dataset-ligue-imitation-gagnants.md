# TODO — Dataset de ligue pour l’imitation des joueurs gagnants

## Statut

Idée à évaluer. Ne pas remplacer le dataset d’imitation de Heuristic V8 avant une
comparaison contrôlée et une validation en partie complète.

Cette note décrit une expérience de données. Si une variante est retenue et nécessite une
évolution du modèle, une architecture dédiée devra être écrite avant son implémentation.

## Hypothèse

Faire jouer plusieurs joueurs distincts les uns contre les autres, puis utiliser les
trajectoires des joueurs ayant gagné pour entraîner un nouveau joueur neural par imitation.

L’hypothèse est qu’une ligue diversifiée couvre davantage de styles et de situations
stratégiques que l’imitation de Heuristic V8 seul, notamment pour :

- les décisions de début de partie ;
- les choix de deckbuilding ;
- les décisions de recrutement et d’activation de maîtrise ;
- les réponses à des adversaires qui ne jouent pas comme V8.

Le risque principal est de confondre « décision prise par le gagnant » et « bonne décision ».
Le résultat final d’une partie ne permet pas d’attribuer automatiquement la victoire à chacune
des décisions de la trajectoire.

## Pourquoi ne pas filtrer uniquement les gagnants

Un dataset winner-only peut introduire plusieurs biais :

- les premières décisions sont étiquetées positives parce que la partie a été gagnée plus tard ;
- une mauvaise décision d’un gagnant est conservée, tandis qu’une bonne décision d’un perdant est
  supprimée ;
- les parties longues et les joueurs qui produisent beaucoup de décisions peuvent dominer le
  dataset ;
- deux joueurs gagnants peuvent choisir des actions différentes dans des états proches ;
- les parties gagnées contre un adversaire faible peuvent peser autant que les parties gagnées
  contre un adversaire fort ;
- les joueurs les plus souvent vainqueurs peuvent imposer leur style et réduire la diversité
  recherchée.

Le winner-only doit donc être une ablation ou une variante de sampling, pas le dataset de
référence initial. Les décisions des perdants doivent rester disponibles comme contre-exemples,
comme données de diversité et pour analyser les situations où le résultat a basculé.

## Principe de collecte

Construire une ligue composée de joueurs versionnés et explicitement identifiés, par exemple :

- Heuristic V8, conservé comme teacher et ancrage principal ;
- anciennes versions heuristiques et neural validées ;
- candidats PPO ou autres joueurs d’exploration, uniquement s’ils sont jouables ;
- joueurs volontairement plus faibles pour diversifier les états rencontrés ;
- RandomPlayer comme adversaire de couverture, mais pas comme source d’expertise.

Chaque partie doit être reproductible et enregistrer au minimum :

- la seed de la partie et la version du moteur ;
- les profils des deux joueurs et leur siège ;
- un `game_id` stable ;
- l’observation masquée du joueur actif avant chaque décision ;
- toutes les actions légales et leurs représentations ;
- l’action choisie ;
- la phase, le tour et le type d’action ;
- le résultat final, le vainqueur et la marge disponible ;
- le numéro de décision dans la partie ;
- la provenance complète du dataset et sa configuration de collecte.

L’observation donnée au modèle ne doit jamais contenir le résultat futur de la partie. Le résultat,
le profil adverse et la provenance servent à pondérer, filtrer ou analyser les exemples, mais ne
doivent pas devenir des informations accessibles au joueur pendant la partie.

## Variantes de dataset à comparer

Conserver le dataset V8 actuel comme contrôle. Produire ensuite, avec les mêmes seeds et les mêmes
splits par `game_id`, les variantes suivantes :

1. **Toutes les décisions de la ligue** : aucune sélection par résultat final.
2. **Ligue équilibrée** : toutes les décisions, mais avec un nombre équilibré de parties par
   matchup, profil, siège, phase et type d’action.
3. **Gagnants sur-échantillonnés** : toutes les décisions restent disponibles, mais les décisions
   des gagnants reçoivent un poids modéré ou sont échantillonnées plus souvent.
4. **Winner-only** : seules les décisions du vainqueur sont conservées, comme expérience de
   sélection-biais explicite.
5. **V8 + ligue** : le dataset V8 reste majoritaire et la ligue apporte une fraction contrôlée
   de diversité.

La comparaison doit commencer par les variantes 1, 3 et 5. La variante 4 ne doit être lancée que
pour mesurer directement le risque de perturbation.

## Pondération proposée

Préférer une pondération explicite à un label binaire gagnant/perdant. Une première recette
possible est :

```text
poids = poids_teacher
      × poids_matchup
      × poids_phase_action
      × facteur_resultat
```

Contraintes de la première expérience :

- V8 conserve un poids de référence fixe ;
- aucun profil non validé ne peut avoir un poids supérieur à V8 ;
- le facteur gagnant/perdant reste modéré et doit être configurable ;
- le nombre maximal de décisions par partie est plafonné ;
- les poids sont enregistrés dans le manifeste et ne sont pas ajustés après avoir vu les résultats ;
- une ablation sans pondération est obligatoire.

Il faut conserver à la fois le résultat binaire et la marge, même si la première loss n’utilise
que le résultat binaire. Une future expérience pourra tester une pondération par marge, mais elle
doit rester séparée pour éviter de mélanger deux hypothèses.

## Protocole expérimental

### 1. Préparer le panel de joueurs

Définir une liste courte de profils avec leurs checkpoints, leurs hashes et leur statut
(`reference`, `validated`, `exploratory` ou `weak`). Définir également les matchups et leur poids
avant la collecte.

Ne pas mélanger silencieusement des architectures ou des checkpoints incompatibles. Chaque ligne
doit permettre de retrouver le joueur qui a produit l’action.

### 2. Collecter un pilote réduit

Lancer une collecte suffisamment grande pour observer les distributions, mais assez courte pour
être analysée avant tout entraînement long. Vérifier :

- le nombre de parties par matchup et par siège ;
- la proportion de victoires par profil ;
- les décisions par phase et par type d’action ;
- les longueurs de parties ;
- les états dupliqués ou quasi identiques avec des actions différentes ;
- la présence d’informations interdites dans l’observation ;
- la stabilité du format JSONL et de la provenance.

Arrêter la collecte si un profil domine le volume, si un matchup produit des parties invalides ou
si une catégorie importante d’action disparaît.

### 3. Construire les splits

Partitionner par `game_id`, jamais par décision individuelle. Utiliser le même split de base pour
toutes les variantes afin que la différence mesure le dataset et non un holdout différent.

Prévoir au minimum :

- un train commun ;
- une validation indépendante ;
- un holdout par partie ;
- un holdout de matchups ou de profils si le volume le permet.

### 4. Entraîner les variantes

Réutiliser la recette d’imitation existante et ne changer qu’une dimension à la fois : sélection
des lignes ou pondération. Garder V8 comme contrôle et comparer les checkpoints intermédiaires,
pas uniquement le dernier epoch.

Chaque entraînement doit utiliser le checkpoint mutable canonique du dépôt et produire un rapport
reproductible sous `artifacts/`. Aucun checkpoint stable ou profil actif ne doit être modifié avant
la validation.

### 5. Évaluer

Mesurer séparément :

- accord d’imitation et regret par rapport aux teachers ;
- performance par phase et type d’action ;
- dérive d’argmax par rapport à V8 ;
- conservation des décisions V8 sur les situations où V8 est performant ;
- comportement de début de partie et de deckbuilding ;
- runtime et coût d’inférence ;
- victoires, défaites et nuls contre le panel complet.

Les métriques offline et l’accord avec un teacher ne suffisent pas pour accepter la variante. La
promotion suit la gate de validation en partie du dépôt, sur le panel complet, avec plusieurs seeds
et conservation des résultats par adversaire.

## Critères de décision

La variante est intéressante si elle démontre, avec plusieurs seeds :

- un gain reproductible contre au moins un adversaire difficile ou une slice stratégique ciblée ;
- aucune dégradation non expliquée contre V8, Random et les références neural ;
- une couverture ou une robustesse supérieure à V8 seul ;
- un coût d’inférence acceptable ;
- une amélioration qui ne dépend pas exclusivement d’un matchup ou d’un profil gagnant.

La variante est rejetée si :

- elle améliore le holdout mais régresse sur le panel réel ;
- elle apprend principalement le style du profil le plus fréquent ;
- elle augmente fortement la dérive d’argmax sans gain de partie ;
- elle supprime des catégories importantes de décisions ;
- le résultat positif disparaît avec une autre seed ou un autre équilibre de matchups ;
- le winner-only fait moins bien que toutes les variantes conservant les décisions des perdants.

## Extensions possibles

Si le mélange simple n’est pas suffisant, tester séparément :

- une loss de préférence utilisant les actions légales et les scores heuristiques lorsqu’ils sont
  disponibles ;
- une pondération par avantage local ou par différence de résultat entre actions, plutôt que par
  vainqueur final ;
- de l’Expert Iteration : le neural produit des parties, puis les situations difficiles sont
  rejouées ou annotées par V8 ;
- une ligue avec matchmaking adaptatif, seulement après avoir obtenu une collecte équilibrée ;
- un warm-start V8 puis un fine-tuning sur la ligue, comparé à un entraînement depuis zéro.

Ces extensions doivent rester des expériences distinctes. Elles ne doivent pas être ajoutées à la
première collecte pour rendre le résultat impossible à attribuer.

## Livrable attendu

Un rapport d’expérience doit contenir :

- l’hypothèse et les variantes exactes ;
- la liste des joueurs, checkpoints et matchups ;
- les seeds, volumes, splits et poids ;
- les statistiques de dataset avant entraînement ;
- les métriques offline et les résultats du panel complet ;
- les dérives d’argmax par phase et action ;
- le résultat : accepté, rejeté ou inconclusif ;
- la condition précise de la prochaine étape.

Le dataset de ligue ne devient une nouvelle source par défaut qu’après cette comparaison. V8 reste
le contrôle et l’ancrage tant qu’un candidat n’a pas démontré un gain reproductible en partie.
