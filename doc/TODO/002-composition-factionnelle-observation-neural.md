# TODO — Expérience 2 : état du deck neural et composition factionnelle

## Statut

À implémenter avec les cardinalités de l'expérience 1 dans un feature set unique. Aucun changement
du moteur, des heuristiques ou du masque d'information n'est autorisé.

## Question

Les comptes factionnels globaux donnent-ils au réseau une information stratégique que le pooling
des cartes ne restitue pas assez clairement, notamment lors des achats, recrutements et
bannissements ?

## Hypothèse falsifiable

À taille de deck comparable, la composition factionnelle du deck permet de mieux estimer le
potentiel futur de synergies. Cette information doit améliorer des slices identifiables, et non
seulement augmenter la capacité du modèle.

## Périmètre de la candidate

Conserver les sept cardinalités de zones de l'expérience 1 et ajouter quatre scalaires pour le
joueur actif : nombre total de cartes possédées des factions `maquis`, `spectra`, `homodeus` et
`order`, toutes zones possédées confondues.

Il n'existe pas de candidate « factions seules » : cardinalités et composition factionnelle sont
une seule variation architecturale et un seul nom de feature set.

Ces comptes décrivent un potentiel structurel. Ils ne représentent pas l'activation immédiate
d'une carte et ne doivent pas être mélangés avec les features de l'expérience 3.

La composition doit être dérivée de `owned_card_counts` et du catalogue connu. Elle ne doit pas
ajouter de champ dans l'observation sérialisée et ne doit pas exposer les cartes adverses cachées.

## Décisions à prendre dans l'architecture

Avant le code, documenter :

1. si les cartes neutres sont exclues ou comptées dans un groupe distinct ;
2. si le compte inclut les champions et les cartes en défausse ;
3. les bornes et la normalisation des quatre comptes ;
4. le nom du feature set et la compatibilité des checkpoints ;
5. les slices où le potentiel global est censé être utile.

## Tests techniques

- Deux decks de même taille mais de composition différente produisent des comptes différents.
- Les cartes neutres ne sont pas attribuées arbitrairement à une faction.
- La permutation des cartes dans une zone ne change pas les comptes.
- Les comptes adverses ne sont jamais calculés ni exposés.
- Les comptes sont cohérents avec le catalogue et les cartes possédées.
- Le modèle historique reste chargeable sans ces features.

## Évaluation

Comparer à V004 avec un protocole identique. Ventiler les résultats par action (`buy_card`,
`recruit_mercenary`, `banish_card`, `play_card`), faction de la carte candidate, taille du deck et
présence d'opérations conditionnelles dans la carte.

Un gain limité aux achats mais accompagné d'une dérive hors slice doit être considéré comme un
signal incomplet, pas comme une promotion. La gate finale reste la moyenne pondérée du panel
officiel ; V008 est pondéré mais n'est pas une garde dure séparée.
