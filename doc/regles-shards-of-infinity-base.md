# Règles de Shards of Infinity — boîte de base

Ce document est une synthèse de référence des règles de la boîte de base de **Shards of
Infinity**, sans extensions. Il sert à vérifier les futures implémentations du moteur. La V0 du
projet n'implémente qu'un sous-ensemble simplifié décrit dans
[`001-architecture-moteur-duel-v0.md`](Architecture/001-architecture-moteur-duel-v0.md).

## But du jeu

Chaque joueur dirige un personnage et cherche à gagner en maîtrisant son Infinity Shard ou en
éliminant les autres joueurs. Dans la boîte de base, un joueur peut gagner en atteignant 30 points
de maîtrise et en activant son Infinity Shard, ou en étant le dernier joueur encore en vie.

## Matériel

La boîte de base contient :

- 4 cartes Personnage avec suivis de santé et de maîtrise ;
- 88 cartes du deck central ;
- 40 cartes de decks de départ, soit quatre decks de 10 cartes composés de 7 Crystals,
  1 Blaster, 1 Shard Reactor et 1 Infinity Shard.

Le jeu est prévu pour 2 à 4 joueurs.

## Mise en place

1. Chaque joueur reçoit une carte Personnage.
2. Chaque joueur commence avec 50 points de santé.
3. Chaque joueur commence avec 0 point de maîtrise.
4. Le premier joueur est déterminé aléatoirement ; les tours passent ensuite dans le sens
   horaire.
5. Chaque joueur mélange son deck de départ et pioche 5 cartes, laissant 5 cartes dans sa pioche.
6. Les cartes à bordure noire sont mélangées pour former le deck central.
7. Six cartes du deck central sont révélées pour former la rivière centrale.

## Ressources

- Les **Gems** servent à recruter des cartes depuis la rivière centrale.
- Le **Power** sert à attaquer les champions et les joueurs adverses.
- La **Mastery** augmente la puissance de certaines cartes et permet d'atteindre la condition de
  victoire liée à l'Infinity Shard.
- La santé représente les points de vie du personnage et ne peut pas dépasser 50.

La maîtrise est permanente pendant la partie, ne se dépense pas et ne peut pas dépasser 30.

## Zones de cartes

Chaque joueur utilise :

- une main ;
- une pioche personnelle ;
- une défausse ;
- une zone de jeu.

Les cartes jouées restent dans la zone de jeu jusqu'à la fin du tour, sauf règle contraire.

Si la pioche personnelle est vide lorsqu'il faut piocher ou révéler une carte, toute la défausse
est mélangée pour reformer la pioche. Si la pioche s'épuise au milieu d'une pioche multiple, la
défausse est remélangée et la pioche continue carte par carte.

## Structure d'un tour

Le tour officiel comporte trois phases.

### 1. Phase de jeu

Le joueur peut effectuer les actions autorisées dans l'ordre de son choix :

- jouer des cartes de sa main pour produire des ressources et appliquer leurs effets ;
- utiliser les capacités de ses champions ;
- recruter des Allies, Champions ou Mercenaries en payant leur coût ;
- jouer immédiatement des Mercenaries selon leur règle de fast-play ;
- utiliser une fois par tour la capacité Focus de son personnage en dépensant 1 Gem pour gagner
  1 Mastery ;
- utiliser du Power pour détruire des champions.

Les cartes Ally jouées produisent leur effet puis restent dans la zone de jeu jusqu'à la fin du
tour. Les ressources non dépensées disparaissent à la fin du tour.

### 2. Phase d'attaque

Le joueur assigne tout son Power restant aux joueurs adverses. Les dégâts peuvent être répartis
librement entre les adversaires en multijoueur.

Les joueurs attaqués peuvent révéler autant de cartes Shield de leur main qu'ils le souhaitent afin
de réduire les dégâts reçus. Les cartes révélées restent dans leur main.

Après réduction par les Shields, la santé de chaque joueur est diminuée. Un joueur dont la santé
atteint 0 ou moins est éliminé.

Les dégâts contre les joueurs sont appliqués ensemble à la fin du tour. Les dégâts contre les
champions peuvent être appliqués pendant la phase de jeu.

### 3. Phase de fin

Les actions sont effectuées dans cet ordre :

1. placer les Mercenaries joués rapidement sous le deck central ;
2. placer les Allies de la zone de jeu dans la défausse ;
3. laisser les Champions en jeu ;
4. placer les cartes restantes de la main dans la défausse ;
5. piocher 5 cartes ;
6. passer le tour au joueur suivant.

## Types de cartes

### Personnages

Les cartes Personnage suivent la santé et la maîtrise du joueur. Chaque personnage possède une
capacité Focus utilisable une fois par tour, en dépensant 1 Gem pour gagner 1 Mastery.

### Allies

Un Ally recruté depuis la rivière est placé dans la défausse du joueur. Il pourra être pioché plus
tard. Lorsqu'il est joué, son effet est appliqué dans l'ordre indiqué, puis la carte reste dans la
zone de jeu jusqu'à la fin du tour avant d'être défaussée.

### Mercenaries

Un Mercenary peut être recruté normalement dans la défausse ou être joué immédiatement depuis la
rivière en payant son coût. Dans ce second cas, son effet est appliqué immédiatement et la carte
est placée sous le deck central à la fin du tour. Un Mercenary joué rapidement ne peut pas ensuite
être recruté et compte comme un Ally joué.

### Champions

Les Champions sont recrutés comme les Allies mais restent dans la zone de jeu lorsqu'ils sont
joués. Ils peuvent être épuisés une fois par tour pour utiliser leur capacité. Ils restent prêts
ou épuisés jusqu'au reset de fin de tour.

Pour détruire un Champion, il faut lui infliger au moins autant de Power que sa santé pendant un
même tour. Les dégâts infligés aux Champions ne persistent pas d'un tour à l'autre. Un Champion
détruit rejoint la défausse de son contrôleur.

## Maîtrise et seuils

La maîtrise est gagnée grâce à certaines cartes et grâce à la capacité Focus du personnage. Elle
reste acquise jusqu'à la fin de la partie.

Certaines cartes ont des bonus de seuil de maîtrise. Le seuil est vérifié au moment où la carte est
jouée ou au moment où une capacité du Champion est utilisée. La maîtrise gagnée plus tard ne modifie
pas rétroactivement l'effet déjà déclenché. La maîtrise gagnée par la carte elle-même peut compter
pour le seuil de cette même carte.

À 30 points de maîtrise, le joueur peut atteindre la condition liée à l'Infinity Shard selon les
règles de la carte.

## Santé, attaque et défense

La santé d'un joueur ne peut pas dépasser 50. Lorsqu'elle atteint 0, le joueur est éliminé.

Le Power restant sert à attaquer. Les Shields réduisent uniquement les dégâts reçus par les
joueurs ; ils ne protègent pas les Champions.

## Banish

Lorsqu'un effet demande de bannir une carte, celle-ci est placée dans une zone de bannissement et
est retirée du jeu. Elle ne retourne ni dans la main, ni dans la pioche, ni dans la défausse.

Une carte déjà jouée ne peut pas être bannie depuis la main pendant le même tour, car elle n'est
plus dans la main.

## Fin de partie

Un joueur est éliminé lorsque sa santé est réduite à 0 ou moins. Le dernier joueur encore en jeu
gagne. Un joueur peut également gagner en atteignant la condition de maîtrise de son Infinity Shard.

## Principes de priorité

- Une carte qui contredit une règle générale applique son propre texte.
- Les effets d'une carte sont exécutés dans l'ordre où ils sont écrits.
- Les choix imposés par une carte sont effectués au moment où elle est jouée ou activée.
- Les ressources non dépensées disparaissent à la fin du tour.
- Les cartes de la zone de jeu ne vont pas dans la défausse avant la phase de fin, sauf effet
  contraire.

## Source

Cette synthèse est basée sur le livret officiel de règles de la boîte de base publié par IELLO :

- [Shards of Infinity — page officielle IELLO](https://iellogames.com/games/shards-of-infinity/)
- [Rulebook officiel PDF](https://iellogames.com/wp-content/uploads/2021/02/SOI_001_Rules_LOCA_V1_Light.pdf)

Les futures décisions propres au projet doivent être ajoutées à l'architecture et à l'état des
features, plutôt que de modifier silencieusement les règles de référence.
