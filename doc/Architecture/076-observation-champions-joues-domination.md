# Architecture — Observation des champions joués pour Domination

## Décision

L'observation neural expose un masque factionnel séparé pour les champions joués pendant le tour
courant. Ce masque est distinct de `played_faction_mask`, qui décrit toutes les factions jouées ce
tour.

```python
played_champion_faction_mask: tuple[bool, bool, bool, bool]
```

L'ordre reste `maquis`, `spectra`, `homodeus`, `order`. Les cartes neutres sont ignorées.

## Règle couverte

Pour Domination, les factions prises en compte sont l'union des cartes présentes dans la main, dans
la `play_zone`, et des champions dont l'identifiant appartient à `played_card_ids_this_turn`.
Un champion seulement activé ne compte pas. La carte candidate est exclue par le calcul moteur
lorsqu'elle est déjà dans une zone concernée.

Le champ ajoute uniquement une information que le joueur actif connaît déjà. Il ne révèle aucune
zone adverse ni aucun identifiant technique supplémentaire.

## Compatibilité

Le schéma d'observation passe à la version 3. La lecture des anciens JSONL accepte l'absence du
champ et utilise un masque nul, afin de conserver la compatibilité des anciens datasets. Les
nouveaux datasets sérialisent toujours le champ.

Cette évolution ne modifie ni les règles du moteur, ni les actions légales, ni le masque
d'information. Elle prépare l'encodage actionnel de l'expérience tactique ; aucun entraînement
n'est lancé par cette modification.

## Tests

- un champion joué ce tour active sa faction dans le nouveau masque ;
- un champion seulement activé n'y contribue pas ;
- un mélange de cartes en main et en `play_zone` reste représenté par les masques factionnels ;
- le masque est remis à zéro au nettoyage du tour ;
- les anciens JSONL sans ce champ restent désérialisables.
