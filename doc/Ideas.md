# TODO et idées d’amélioration

## Joueur heuristique

- [ ] Investiguer pourquoi l’heuristique bannit très peu `Blaster` alors que des cartes bannies
  auparavant semblent nettement plus fortes. Comparer les scores de bannissement, les cartes
  conservées, les seuils de victoire protégés et les résultats victoire/défaite afin de déterminer
  si le signal de `deck_thinning`, la protection des cartes de victoire ou la valeur d’acquisition
  favorise à tort la conservation de `Blaster`.
- [x] Évaluer `GainMastery` avec son impact inter-phase : la projection compare désormais la
  meilleure valeur d'achat avant et après la dépense de la gemme.
- [x] Valoriser explicitement le franchissement d'un seuil de maîtrise, en additionnant les gains
  positifs pour toutes les cartes de la main qui peuvent être jouées après le `+1`.
- [x] Définir une projection pure et déterministe de la valeur des achats avant/après `GainMastery`,
  couvrant l'absence d'achat, les achats de mercenaires et les égalités de valeur.
- [x] Couvrir cette décision inter-phase par des scénarios reproductibles : achat perdu, seuil de
  maîtrise franchi et conservation de la maîtrise malgré la perte d'un achat.
- [ ] Calibrer les coefficients `purchase_opportunity_cost` et `mastery_threshold_value` dans une
  campagne dédiée ; ils restent neutres dans `v004`.
- [ ] Optimiser `cost_paid`, actuellement à `0.0` dans `v004`, afin que le coût réel des achats
  influence la décision.
- [x] Réévaluer `card_acquisition_weights.power_produced`, actuellement à `0.0` : la campagne du
  22/07/2026 n'a pas validé son activation, seule ou combinée à `gems_produced`.
- [x] Réévaluer séparément `card_acquisition_weights.gems_produced` avec une campagne dédiée et
  une validation finale d'au moins 5 000 parties par adversaire ; `0.5` est retenu dans `v006`.
- [ ] Étendre la valeur d’acquisition aux effets conditionnels futurs : valoriser l’effet courant
  ainsi qu’un bonus pour les branches accessibles plus tard avec une pénalité de contrainte adaptée.
- [ ] Ajouter au reporting les scores comparés de `BuyCard` et `RecruitMercenary`, le coût moyen,
  la progression de partie et le multiplicateur `durable_replay_factor`.
- [ ] Vérifier sur des campagnes indépendantes si la suppression de `card_acquisition_value` pour
  les mercenaires suffit à éviter leur sur-utilisation, sans ajouter de pénalité artificielle.
- [ ] Corriger la valorisation des effets immédiats des mercenaires pour utiliser le gain réellement
  obtenu dans l'état courant plutôt que l'effet nominal complet. Exemple : `Clerc Aux spores` est
  affiché comme un soin de 4 alors qu'avec 48 PV il ne peut rendre que 2 PV avant le plafond de 50.
  Vérifier également les éventuelles saturations de Gems, Power et autres ressources.
- [x] Ajouter un mode de campagne combinée permettant de faire évoluer les poids d’action,
  d’acquisition et de contraintes dans un même candidat complet.
- [ ] Lancer une première campagne combinée longue et mesurer si les corrélations entre familles
  apportent un gain réel par rapport à `v005`.
- [ ] Si la recherche locale hybride plafonne ou devient instable avec les nombreux paramètres,
  évaluer une optimisation bayésienne ou une recherche hybride plus exploratoire.

## Stratégie de transition vers le neural

- [ ] Définir un critère de suffisance pour l’heuristique : adversaire stable, reproductible et
  suffisamment qualifié pour générer un partenaire d’entraînement, sans chercher l’optimum absolu
  de la stratégie heuristique.
- [ ] Une fois ce seuil atteint, prioriser la collecte de parties et le premier réseau neural par
  imitation plutôt que de poursuivre indéfiniment l’optimisation des poids heuristiques.
