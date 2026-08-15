# État courant — RandomPlayer

`RandomPlayer` est une stratégie indépendante du moteur, implémentée dans
`shards_ai/ai/random_player.py`. Elle reçoit l'observation et la liste des actions légales, puis
retourne une action sans modifier directement l'état. Elle est déclarée en lecture seule ; en
l'absence d'observateur de transitions, `GameRunner` peut donc lui transmettre l'état vivant pour
éviter une copie. L'observation détachée reste disponible explicitement et reste le comportement
par défaut dès qu'un observateur de transitions est actif.

Dans `GameRunner`, `RandomPlayer` est automatiquement enveloppé par `LegacyActionMiddleware`. Le
joueur conserve son contrat historique `PLAY/BUY` alors que le moteur sous-jacent peut intercaler
les actions modernes. `PassPlayPhase` est consommée par le middleware et `StopBuying` est traduite
en `EndMainPhase`.

## Politique actuelle

### `PLAY`

- lorsqu'un bannissement optionnel est en attente, choisit `SkipBanish` dans 50 % des cas ; sinon
  choisit une carte légale à bannir ;
- lorsqu'un recrutement gratuit est en attente, choisit uniformément une carte légale de la rivière ;
- sélectionne aléatoirement parmi les actions `PlayCard`, `ActivateChampion` et `GainMastery` légales ;
- `GainMastery` dépense 1 Gem et ajoute 1 maîtrise, au maximum une fois par tour ;
- retourne `PassPlayPhase` seulement lorsqu'il ne reste ni carte jouable ni gain de maîtrise légal.

### `BUY`

- récupère les actions `BuyCard` et `RecruitMercenary` légales fournies par le moteur ;
- s'il n'y en a aucune, retourne `StopBuying` ;
- sinon, avec une probabilité de 10 %, retourne `StopBuying` ;
- dans les autres cas, choisit uniformément entre achat normal et recrutement ;
- recommence après chaque achat avec la nouvelle observation.

La probabilité d'arrêt et le choix de carte utilisent le flux `GameRandom` dédié au joueur. Cette
politique est volontairement arbitraire et constitue une stratégie de référence, pas une règle du
moteur.

### `ATTACK`

Choisit aléatoirement parmi le joueur adverse et les champions dont les PV sont inférieurs ou égaux
au Power disponible, selon les actions légales exposées par le moteur. Il choisit aussi aléatoirement
les décisions en attente et les ordres de résolution proposés par le moteur.

## Reproductibilité

`GameRunner.random_duel(seed)` dérive des flux indépendants pour le moteur et chaque joueur. Une
même seed, une même configuration et les mêmes versions de code produisent donc le même résultat.

## Limites

Le joueur random ne connaît pas encore de stratégie économique. Les boucliers sont résolus
automatiquement par le moteur ; les bannissements optionnels sont sélectionnés aléatoirement à 50 %.
Il peut toutefois résoudre les cartes dont les effets sont conditionnés par la maîtrise ou la
défausse via les actions produites par le moteur. Une phase ou une liste d'actions incohérente avec
cette politique provoque une `InvalidActionError` explicite.

## Tests

Les tests de `tests/game/test_random_player.py` vérifient le jeu des cartes et le gain de maîtrise,
l'achat via le runner, l'arrêt à 10 %, l'assignation complète du Power, la reproductibilité des
parties, le rejet d'actions illégales et les limites d'actions et de tours. Les tests Spectra de
`tests/game/test_spectra_cards.py` vérifient en plus la décision de bannissement à 50 % ; les tests
Homodeus et Ordre couvrent les décisions de recrutement et les effets de leurs cartes. Les tests
`tests/game/test_mercenaries.py` vérifient le choix aléatoire d'une récupération de mercenaire et
les transitions de recrutement.
