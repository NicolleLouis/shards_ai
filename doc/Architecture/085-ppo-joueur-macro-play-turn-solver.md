# Adaptation PPO du joueur macro avec PlayTurnSolver

## Objectif

Entraîner le joueur unifié V4 (`structured_semantic_v5_macro_tactical_action_v1`) avec PPO. Une décision PPO est soit un choix de branche PLAY exposé par `PlayTurnSolver`, soit une décision atomique hors PLAY. Les actions atomiques de la trace choisie sont rejouées par le moteur et ne constituent pas des transitions supplémentaires.

## État actuel

`MacroNeuralPlayer` sait résoudre les branches PLAY, vérifier la légalité de chaque action rejouée et représenter les décisions atomiques avec le schéma V4. `NeuralActorCritic` et `PPOTrainingPlayer` ne savent toutefois scorer que des `ActionRepresentation` atomiques et le collecteur ne consomme pas les payloads macro.

## Comportement cible

- Le backbone PPO est `MacroActionScorerV4`, initialisé par `configs/neural_profiles/v005.pt`.
- L’actor score une liste variable de `MacroActionRepresentation`; le champ `decision_kind` distingue macro PLAY et atomique.
- Le critic dépend uniquement de l’observation encodée.
- Le joueur échantillonne une candidate avec `Categorical`, conserve log-probabilité, valeur, observation et index, puis délègue le replay au joueur macro.
- Le reward est `0` jusqu’à la fin, puis `+1`, `-1` ou `0` sur la dernière transition. `gamma=1` et `gae_lambda=1`.
- Le mélange candidat est V8 45,4545 %, V7 27,2727 %, V1/V2/Random 9,0909 % chacun.

## Décisions

1. Le moteur reste la source d’autorité pour les actions légales et les transitions.
2. Les replays automatiques et les branches singleton ne sont pas des décisions PPO.
3. Le checkpoint de travail unique reste `artifacts/neural_training/checkpoint.pt`; aucune promotion ou modification des pointeurs actifs n’est effectuée par ce changement.
4. Le reward shaping est refusé pour ce profil, même si une configuration non vide était ajoutée accidentellement.
5. Les adversaires V1/V2 sont chargés comme joueurs atomiques existants; l’évaluation du learner utilise le même adaptateur macro que le rollout.

## Hors périmètre

Pas de campagne PPO longue, de promotion de checkpoint, de modification du moteur de règles, ni de remplacement du profil actif.

## Architecture proposée

`PPOTrainingMacroPlayer` spécialise `MacroNeuralPlayer`. Son scorer échantillonne la liste de candidats V4 et place un payload PPO dans une file consommée immédiatement par le collecteur. Le solver conserve la trace choisie; chaque retour d’action est vérifié contre les actions légales du moteur.

`NeuralActorCritic` choisit automatiquement `macro_scorer` pour l’architecture V4 et encode la valeur depuis `encode_observation`. L’export de l’actor conserve les clés `macro_scorer.*`, ce qui permet à `build_neural_player` de reconstruire le même contrat macro lors de l’évaluation.

## Tests et validation

Tests ciblés: chargement V005, logits finis pour des listes V4 variables, mélange macro/atomique, replay légal sans transition supplémentaire, reward terminal et GAE sans fuite entre épisodes, adversaires V1/V2 et évaluation gloutonne macro. La suite PPO atomique est conservée. Un smoke test court doit précéder toute campagne.

## Questions ouvertes

Aucune question bloquante pour l’implémentation. La qualité compétitive du candidat devra être décidée ultérieurement avec le panel de parties complet; les métriques PPO et le smoke test ne sont pas une preuve de promotion.
