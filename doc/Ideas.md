# Idées et priorités de travail

Le profil neural stable actuel est `v002`, dans `configs/neural_profiles/v002.pt`. Les cycles
DAgGER et le reward shaping de deckbuilding sont déjà implémentés ; la faiblesse principale reste
la performance contre `v008`.

Les expériences doivent rester comparables : une seule variation importante à la fois, les mêmes
seeds de validation et les mêmes adversaires. Un checkpoint promu reste gelé ; seul
`artifacts/neural_training/checkpoint.pt` peut être entraîné.

## Priorité 1 — évaluer le candidat PPO v003

- [x] Lancer un entraînement court du profil candidat
  `configs/neural_training_profiles/candidates/v003.yaml`, après un smoke test d'une partie.
- [ ] Si la progression est reproductible, poursuivre l'entraînement jusqu'au budget du profil,
  puis effectuer une validation finale d'au moins 200 parties avant toute promotion.
- [ ] Vérifier que le potentiel de deckbuilding n'est pas exploité artificiellement par le banish de
  cartes faibles ou par la réduction de la taille du deck.
- [ ] Comparer séparément l'effet du shaping sur les achats, puis sur l'ensemble des décisions de
  deckbuilding, sans modifier simultanément les autres hyperparamètres.

## Méthodes réutilisables

- [ ] Réutiliser l'enrichissement DAgGER pour les comportements faibles identifiés par les
  analyses : collecter les états réellement visités, demander le choix du teacher, conserver les
  actions légales et leurs métriques, puis fusionner avec l'historique selon un échantillonnage
  contrôlé. Cette méthode peut cibler `play_card`, les achats ou toute autre famille de décisions.
- [ ] Ajouter une tête d'incertitude comme mécanisme early stage pour sécuriser l'exploration,
  sélectionner des états pour DAgGER et diagnostiquer les décisions fragiles. Une délégation
  temporaire à Heuristic V8 peut être testée, mais la performance finale doit être mesurée avec le
  réseau seul afin de ne pas créer un plafond d'imitation.

## Priorité 2 — expériences d'architecture

- [ ] Évaluer l'architecture expérimentale `semantic_identity_v3` sur un split hors entraînement,
  puis en parties contre les adversaires de référence si les métriques d'imitation progressent.
- [ ] Tester l'effet des dimensions des représentations de cartes, en particulier l'embedding
  d'identité `card_definition_id` et l'embedding sémantique. Comparer les variantes une par une,
  puis vérifier en parties si un gain offline se confirme.
- [ ] Ajouter un contexte global de la liste d'actions légales : encoder chaque action, agréger les
  alternatives disponibles, puis injecter ce contexte dans le score de chaque action. Mesurer le
  gain sur les décisions où les alternatives se définissent mutuellement.
- [ ] Étudier une version plus aboutie où le réseau reçoit toute la liste d'actions en une fois et
  retourne directement un logit par action. Le modèle doit rester invariant à l'ordre de la liste,
  tout en permettant un échantillonnage par température plutôt qu'un choix systématique du premier
  logit.
- [ ] Ne tester une nouvelle architecture PPO qu'après avoir obtenu une baseline stable et
  reproductible.

## Priorité 3 — pistes complémentaires

- [ ] **Rééchantillonnage DAgGER par regret** : prioriser les états où l'écart entre l'action du
  NeuralPlayer et celle du teacher est le plus coûteux, plutôt que de rééchantillonner toutes les
  divergences de manière uniforme. L'objectif est de concentrer le dataset sur les erreurs qui
  changent réellement l'issue ou la qualité de la partie.
- [ ] **Loss listwise sur les actions légales** : apprendre le classement complet des actions d'une
  décision, et pas seulement l'action choisie ou des comparaisons par paires. Cette loss pourra être
  comparée à la sortie conjointe de tous les logits, avec des métriques top-1, top-3, regret et
  calibration de la distribution.
- [ ] **Recherche courte pour reranker** : utiliser le réseau pour sélectionner quelques actions,
  simuler un ou deux coups pour chacune, puis reranker les résultats. Commencer par les achats,
  attaques et décisions de fin de phase afin de mesurer le gain sans remplacer toute la politique.
- [ ] **Ensemble de checkpoints** : faire voter plusieurs checkpoints promus ou plusieurs scorers
  indépendants. Les désaccords entre modèles peuvent fournir une mesure d'incertitude et des états
  intéressants pour DAgGER, tandis que le benchmark doit aussi mesurer chaque modèle séparément.
- [ ] **Curriculum par phase de jeu** : comparer l'entraînement uniforme à un curriculum donnant
  progressivement plus de poids aux achats, au jeu des cartes, aux combats et aux décisions
  conditionnelles. Vérifier que le gain sur une phase ne dégrade pas les autres.
- [ ] **Crédit temporel par décision** : mesurer l'impact d'un achat, recrutement ou bannissement
  plusieurs tours plus tard, au lieu de ne regarder que la récompense immédiate ou finale. Ce signal
  devra rester séparé du reward terminal pour éviter de masquer la vraie performance en victoire.
- [ ] **Belief state de l'adversaire** : estimer les cartes probablement détenues par l'adversaire à
  partir des informations publiques et de l'historique observable, sans exposer sa main réelle.
  Cette piste vise l'information cachée et ne doit être introduite qu'après stabilisation de
  l'observation actuelle.

## Priorité 4 — alternatives à PPO

- [ ] Tester un apprentissage Monte-Carlo à partir du résultat final de la partie, en conservant la
  représentation état-action actuelle. Comparer cette valeur `Q(s, a)` à PPO sur la stabilité, la
  qualité des décisions et le coût d'entraînement.
- [ ] Évaluer une exploration contrôlée pendant la collecte Monte-Carlo, puis comparer une sélection
  gloutonne à un échantillonnage par température pendant l'inférence.

## Plus tard

- [ ] Ajouter un self-play avec snapshots adverses après validation d'un agent fixe performant.

## Règles de mesure

- Utiliser 64 parties par adversaire pour la sélection périodique et au moins 200 pour la validation
  finale.
- Comparer chaque expérience à `v002` sur les mêmes seeds ; conserver `v001`, `v002` et les autres
  profils promus comme références gelées.
- Garder séparées les métriques de rollout, les évaluations gloutonnes et les validations finales.

## Expérience exp-00003 — résultat et suite

- **Idée sélectionnée :** évaluer le candidat PPO `v003`.
- **Statut :** entraînement court terminé après smoke test ; validation comparative courte acceptée,
  sans promotion du checkpoint.
- **Idées retirées :** aucune ; les pistes non sélectionnées restent dans ce catalogue.
- **Prochaine étape :** refaire une validation finale d'au moins 200 parties avec les mêmes adversaires
  et seeds avant toute promotion de `v003`. Vérifier aussi l'exploitation réelle du deckbuilding
  (banish et taille du deck), comme prévu par la piste initiale.
- **Nouvelle idée future :** comparer, dans une expérience isolée, la robustesse de `v003` sur un
  second seed de validation avant d'autoriser une promotion.
