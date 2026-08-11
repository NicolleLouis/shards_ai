# Représentation et apprentissage macro neural V2 — Architecture

## Objective

Corriger les limites de l'expérience `exp00109-macro-v8-deck-state` avant de poursuivre
l'entraînement du joueur macro. Le réseau doit pouvoir distinguer les actions stratégiques que
`PlayTurnSolver` lui présente, sans réintroduire les permutations atomiques déjà prouvées
interchangeables et sans accéder à une information cachée.

Le travail doit pouvoir être réalisé bloc par bloc. Chaque bloc possède son propre contrôle, ses
métriques et une condition d'arrêt. Une feature rejetée ou inconclusive n'est pas automatiquement
incluse dans le bloc suivant.

Ordre d'exécution retenu : blocs 1, 2, 3, 4, 6, 5, 7, puis 8. La couverture atomique unifiée du
bloc 6 est stabilisée avant la régénération et l'entraînement du dataset complet du bloc 5.

Critères de réussite globaux :

- aucune décision neural lorsqu'une seule branche légale subsiste ;
- aucun candidat stratégiquement distinct rendu indistinguable par absence d'identité ou de cible ;
- compatibilité de lecture complète des checkpoints et datasets macro V1 ;
- mesures offline séparant les décisions triviales des vrais choix ;
- amélioration attribuable d'abord sur les choix macro, puis extension progressive du même réseau
  aux décisions atomiques jusqu'à zéro recours au fallback historique ;
- validation finale par le panel pondéré complet, et non par l'accuracy offline seule.

## Current State

`PlayTurnSolver` produit un préfixe automatique puis, au premier choix stratégique, un candidat par
action racine légale. Chaque candidat conserve sa trace atomique exacte dans `PlayTurnCandidate`,
mais `MacroActionRepresentation` ne transmet au modèle que :

- l'histogramme des types d'actions de la trace ;
- le type terminal et la phase ;
- Gems, maîtrise, Power, tailles de main/défausse/play zone et longueur de trace.

L'identité de l'action racine, la définition de carte, la cible, `pending_kind` et les activations
tactiques ne sont pas encodées. Deux bannissements de cartes différentes ou deux pending choices
peuvent donc produire exactement le même vecteur.

L'audit du dataset
`artifacts/imitation_dataset/macro_v8_vs_v7_v4_100k.jsonl` donne :

- 39 193 décisions `macro_play` et 61 061 décisions `atomic` ;
- au moins une collision exacte entre candidats dans 40,4 % des décisions macro ;
- candidat teacher en collision dans 83,7 % des `BanishCard` et 81,9 % des
  `ChoosePendingDecision` ;
- label non premier parmi des scores nécessairement identiques dans 53,3 % des `BanishCard` et
  42,0 % des `ChoosePendingDecision` ;
- 5 586 records à candidat unique dans le dataset complet ; ces records ont une loss nulle mais
  comptent comme top-1 corrects ;
- top-1 holdout epoch 5 de 77,96 % toutes décisions confondues, mais 74,11 % sur les seuls choix à
  plusieurs candidats et 25,71 % sur `BanishCard`.

Le trainer macro filtre actuellement les records `atomic`. À l'inférence contre V008, le joueur
effectue en moyenne 70,2 décisions macro et 47,9 décisions de fallback atomique par partie. Ce
fallback reste le checkpoint V004 non réentraîné. Le panel apparié de 100 parties donne 19 % contre
V008 pour le candidat macro et 23 % pour V004 ; le benchmark de 1 000 parties contre V008 donne
21,9 % au candidat.

Les tests ciblés actuels valident le replay, le bornage du solveur, le chargement du modèle et le
training nominal, mais ne testent ni l'absence de collisions sémantiques ni la qualité informative
de chaque candidat.

## Target Behavior

Pour chaque vrai choix macro, le modèle reçoit :

1. l'observation masquée au point de décision ;
2. une représentation neural de l'action stratégique racine ;
3. un résumé borné de ses conséquences déterministes et connues ;
4. éventuellement un groupe de features tactiques explicitement activé par configuration ;
5. la trace atomique uniquement comme provenance et comme résumé de longueur/types, jamais comme
   source d'identifiants d'instances appris.

Les décisions à candidat unique sont rejouées automatiquement. À terme, les choix atomiques hors
périmètre macro sont présentés au même réseau comme des branches de longueur 1. Le réseau utilise
ainsi un contrat candidat commun dans toutes les phases : branche macro pendant `PLAY`, action
atomique ailleurs ou lorsque le solveur ne peut pas abstraire davantage.

Le fallback V004 est uniquement une étape de migration et un mécanisme de rollback. Le contrat
d'inférence final ne l'appelle plus sur une partie supportée : toutes les décisions non automatiques
sont prises par le réseau unifié.

## Non-Goals

- modifier les règles, `Game.legal_actions()` ou `Game.apply()` ;
- augmenter les budgets fixes de `PlayTurnSolver` ;
- donner au modèle l'ordre interne des cartes interchangeables ;
- entraîner ou modifier en place `configs/neural_profiles/v004.pt` ;
- remplacer le checkpoint mutable canonique `artifacts/neural_training/checkpoint.pt` ;
- exposer l'identité d'une future carte piochée, l'ordre caché de la pioche ou une information
  adverse masquée ;
- regrouper immédiatement toutes les features sans ablation ;
- maintenir durablement deux politiques neuronales concurrentes, une macro et une de fallback ;
- promouvoir sur un seul seed, le top-1, la loss, la vitesse ou le seul résultat contre V008.

## Key Decisions

1. **V1 reste immuable.** `structured_semantic_v5_macro_deck_state_v1` et le schéma dataset 1
   restent chargeables. La nouvelle représentation reçoit un nouvel identifiant d'architecture et
   un nouveau numéro de schéma.
2. **Action racine comme identité stratégique.** Le candidat V2 contient une
   `ActionRepresentation` dérivée de `representation_for_neural_action()` sur l'observation du point
   de choix.
3. **Pas d'embedding d'instance.** `card_instance_id` et `choice_id` peuvent être conservés dans le
   record pour l'alignement et le replay, mais ne sont jamais transformés en signal appris. Le
   modèle utilise `card_definition_id`, le type de cible, les scalaires et les catégories stables.
4. **Pas de look-ahead caché.** Une conséquence simulée n'est encodée que si elle est déductible de
   l'observation et de l'action. L'identité exacte d'une carte révélée par une future pioche est
   exclue, même si le clone moteur la connaît.
5. **Tactique action-conditionnée.** Union, Echo et Domination dépendent du couple
   `(observation, action racine)` et ne deviennent pas des scalaires d'état partagés.
6. **Vrais choix uniquement.** Zéro ou un candidat ne produit ni inférence, ni record
   `macro_play`, ni contribution à l'accuracy ou à la loss.
7. **Un réseau final unique.** Le même scorer action-conditionné traite les branches macro et les
   candidats atomiques de longueur 1. Un champ `decision_kind` explicite distingue les deux espaces,
   et les features macro non applicables valent zéro pour une action atomique. V004 reste seulement
   un contrôle et un rollback pendant la migration.
8. **Une variation majeure à la fois.** Identité racine, conséquences tactiques, couverture
   atomique unifiée et DAgGER sont évaluées séparément avant regroupement.
9. **Données causales.** Toute modification du schéma candidat exige un dataset régénéré ; une
   migration synthétique du JSONL V1 ne peut pas reconstruire les identités et cibles absentes.
10. **Gate inchangée.** La décision finale utilise le panel pondéré courant. Aucun adversaire
    individuel n'est transformé en garde dure supplémentaire.

## Open Questions

1. **Non bloquant — nom de l'adaptateur final.** `MacroNeuralPlayer` peut rester un nom transitoire.
   Après suppression du fallback, décider si la classe devient `UnifiedNeuralPlayer` ou si le nom
   historique est conservé pour limiter la migration d'API.
2. **Non bloquant — pending choices non cartes.** Définir une petite taxonomie stable par
   `pending_kind` et type de choix. Ne pas apprendre la chaîne arbitraire de `choice_id`.
3. **Non bloquant — regroupement final.** Les features de conséquence et tactiques ne seront
   réunies que si leurs ablations isolées montrent une complémentarité reproductible.

## Proposed Architecture

```text
NeuralObservation masquée
          │
          ├── encodeur d'état V5 + deck_state_v1
          │
PlayTurnSolver ── candidat racine + trace validée
          │
          ├── RootActionEncoder
          │     type, card_definition_id, sémantique, cible stable, scalaires
          ├── KnownConsequenceEncoder (default V3 contract)
          │     deltas publics/déterministes, pending_kind, flags de branche
          └── TacticalEncoder (default V4 contract)
                Union, Echo, Domination pour l'action racine
          │
          ▼
UnifiedDecisionScorer V2 ── score par candidat non trivial
          │
          ▼
replay atomique validé par Game.apply()

Hors PLAY / solveur sans abstraction
          │ actions légales transformées en traces de longueur 1
          └──────────────────────────────► UnifiedDecisionScorer V2
```

### Bloc 1 — Contrat de décision et métriques fiables

**Statut : terminé côté implémentation et tests.** La régénération du rapport baseline reste une
validation expérimentale à exécuter avec le dataset retenu.

**Objectif.** Supprimer les pseudo-décisions et produire une baseline qui ne masque pas les erreurs.

Travail :

- dans `MacroNeuralPlayer`, rejouer directement l'unique candidat sans appeler
  `candidate_scorer` ;
- ne pas incrémenter `macro_decisions` et ne pas émettre de `MacroDecisionPayload` dans ce cas ;
- faire ignorer aux readers historiques les records V1 à moins de deux candidats ;
- séparer dans les métriques `all_records`, `non_trivial_records`, cardinalité légale, action racine,
  collision du candidat choisi et collision parmi les alternatives ;
- corriger le dénominateur de loss : un record à loss nulle parce qu'il ne contient aucun choix ne
  compte pas comme record entraîné ;
- enrichir le rapport contrefactuel en séparant `macro_choice`, `macro_replay` et
  `atomic_fallback`. Un ordre canonique différent ne doit pas être présenté comme une erreur
  stratégique sans preuve.

Critères d'acceptation :

- zéro appel au scorer avec moins de deux candidats ;
- zéro record V2 à candidat unique ;
- reproduction des trajectoires moteur avant/après sur un panel déterministe ;
- rapport baseline V1 régénéré avec top-1 non trivial et taux de collisions.

Condition d'arrêt : tout changement de vainqueur ou de transition atomique indique un bug de replay
et bloque les blocs suivants.

### Bloc 2 — Schéma candidat V2 unifié et identité de l'action racine

**Statut : terminé côté implémentation et tests.** La régénération du dataset V2, le réentraînement
et la comparaison V1/V2 restent des validations expérimentales ; aucun checkpoint n'est promu à ce
stade.

**Objectif.** Rendre distinguables les actions stratégiques sans ajouter encore les conséquences
tactiques.

Ajouter un contrat de données versionné, conceptuellement :

```text
DecisionCandidateRepresentationV2
  schema_version: 2
  decision_kind: macro_play | atomic
  root_action: ActionRepresentation
  trace_action_type_counts
  terminal_kind, phase
  gems, mastery, power
  hand_size, discard_size, play_zone_size
  atomic_action_count
```

`root_action` conserve l'identité de définition et les paramètres publics. Le dataset peut garder
la trace exacte pour le replay, mais le tenseur ne doit pas contenir `card_instance_id`.

Le nouveau `RootActionEncoder` réutilise autant que possible l'encodage action-conditionné existant :

- embedding et représentation sémantique de `card_definition_id` ;
- one-hot du type d'action et de la phase ;
- type de cible stable, `river_slot` et `amount` normalisés lorsqu'ils sont pertinents ;
- catégorie du choix pending, sans hash ni vocabulaire ouvert des IDs d'instance.

Critères d'acceptation :

- deux `BanishCard` visant deux définitions différentes produisent des tenseurs différents ;
- deux `PlayCard` de définitions différentes produisent des tenseurs différents même avec les
  mêmes ressources terminales ;
- deux instances réellement interchangeables d'une même définition ne reçoivent pas des embeddings
  différents ;
- aucune carte cachée ne peut être résolue par `representation_for_neural_action()` ;
- les checkpoints V1 se chargent toujours avec leur classe historique.

Le schéma est commun aux futures décisions atomiques, mais ce bloc entraîne encore **identité racine
macro seulement**, avec les conséquences V1 inchangées. Comparer à exp00109 sur le même split, les
mêmes seeds et les mêmes records de parties régénérées. L'activation du mode atomique reste isolée
dans le bloc 6.

Condition d'arrêt : si les collisions `BanishCard`/`ChoosePendingDecision` persistent à cause d'un
champ absent, corriger le schéma avant tout entraînement long.

### Bloc 3 — Conséquences connues et cibles complexes

**Statut : terminé côté implémentation et tests unitaires.** Le screening en partie et la mesure
d'ablation finale sont volontairement reportés à la validation de l'architecture complète.

**Objectif.** Distinguer les branches dont l'action racine seule ne décrit pas tout le résultat
stratégique.

Ajouter, derrière un feature set explicite :

- deltas `gems`, `mastery`, `power`, santé active et santé adverse ;
- deltas de cardinalité des zones déjà publiques ;
- `pending_kind` et nombre de choix restants ;
- définition des cartes explicitement bannies, activées, récupérées, copiées ou recrutées par la
  trace lorsque cette identité est connue au point de décision ;
- masques de factions jouées et de champions joués résultant d'actions connues ;
- type de fin de branche et présence d'une victoire immédiate.

Ne jamais encoder l'identité d'une carte obtenue par une pioche encore inconnue. Pour les branches
contenant un effet aléatoire ou une révélation, encoder l'intention et les deltas de cardinalité
connus, pas le résultat secret du clone.

Critères d'acceptation :

- tests dédiés aux bannissements main/défausse, copies, recrutements gratuits, champions et draws ;
- test négatif prouvant que deux ordres cachés de pioche donnent le même input pré-décision ;
- normalisations bornées et valeurs finies sur tout le catalogue ;
- mesure d'ablation `root_action_v2` contre `root_action_plus_known_consequence_v1`.

Condition d'arrêt : rejeter ce groupe si son gain offline est limité aux collisions déjà résolues par
le bloc 2, s'il fuit une information ou s'il ne survit pas au screening en partie.

### Bloc 4 — Activation tactique macro

**Statut : terminé côté implémentation et tests unitaires.** L'ablation tactique et la validation
finale sont reportées à la fin de la todo globale.

**Objectif.** Porter les contrats V6 au candidat macro sans les confondre avec l'état partagé.

Pour les racines `PlayCard`, calculer :

```text
requires_union, union_active,
requires_echo, echo_active,
requires_domination, domination_active,
domination_missing_count
```

Le calcul réutilise exactement les règles de l'architecture 077, dont
`played_champion_faction_mask`. Les actions non `PlayCard` reçoivent des zéros.

Critères d'acceptation :

- mêmes fixtures et mêmes résultats que `StructuredSemanticV6TacticalActionScorer` ;
- tests de candidates distinctes dans une observation identique ;
- ablation tactique isolée contre le meilleur checkpoint du bloc 2 ou 3 ;
- mesure par slice Union/Echo/Domination, pas seulement un top-1 global.

Condition d'arrêt : ne pas regrouper ce bloc si les slices tactiques ne progressent pas de façon
reproductible ou si la dérive hors slice dépasse le budget déclaré avant l'expérience.

### Bloc 5 — Dataset V2 et entraînement macro reproductible

Ce bloc est volontairement traité après le bloc 6 ; son contrat de dataset devra donc reprendre la
représentation unifiée et le schéma candidat effectivement stabilisés.

Le contrat retenu est le dataset schema 3 avec des candidats schema 4 : les décisions `macro_play`
et `atomic` partagent la même représentation, et les pondérations `decision_kind` sont déclarées
dans le profil avant l'entraînement.

**Objectif.** Régénérer des exemples causaux et entraîner seulement une architecture compatible.

Travail :

- incrémenter `MACRO_DATASET_SCHEMA_VERSION` pour les nouveaux fichiers sans réécrire le dataset V1 ;
- conserver `game_id`, seed, teacher, adversaire, siège, résultat et provenance ;
- ajouter `root_action`, feature-set candidat et fingerprint du catalogue au manifest ;
- refuser explicitement les datasets antérieurs au schema 3 dans le trainer unifié V4 ;
- entraîner sur les records `macro_play` et `atomic`, avec des pondérations de `decision_kind`
  fixées dans le profil avant le run ;
- split par `game_id`, mêmes seeds pour les ablations, validation naturelle non rééquilibrée ;
- journaliser top-1 non trivial, rang, collision, cardinalité légale et résultats par action racine,
  phase, matchup et `decision_kind` ;
- conserver `artifacts/neural_training/checkpoint.pt` comme unique checkpoint mutable ;
- initialiser uniquement les modules réellement compatibles depuis V004. Les nouveaux encodeurs sont
  initialisés explicitement et leur provenance est inscrite dans le checkpoint.

Avant un run long, faire un smoke test, mesurer le temps par epoch et vérifier que le coût cumulé
reste sous 5 heures.

Critères d'acceptation :

- dataset sans erreur et sans décision macro triviale ;
- zéro collision inexpliquée du candidat teacher sur les actions ciblées ;
- commandes de génération, training, rapport et benchmark reproductibles via le `Makefile` et
  `NEURAL_CHECKPOINT` ;
- checkpoint portant l'architecture, le feature set, le schéma dataset, le profil et les seeds.

### Bloc 6 — Couverture atomique unifiée et suppression du fallback

**Statut : terminé côté implémentation et tests unitaires.** La validation sur corpus complet,
l'absence définitive de fallback et le panel final sont reportés à la validation globale.

**Objectif.** Étendre le même réseau aux décisions BUY, combat et pending qui représentent environ
40 % des décisions neural actuelles, puis réduire le recours au fallback V004 jusqu'à zéro.

Introduire un adaptateur commun de candidats :

- pendant `PLAY`, il transmet les branches produites par `PlayTurnSolver` ;
- hors `PLAY`, il transforme chaque action légale en `DecisionCandidateRepresentationV2` avec une
  trace atomique de longueur 1 ;
- lorsque le solveur atteint un budget ou refuse une abstraction, il présente les actions atomiques
  légales au même réseau au lieu d'appeler un second modèle ;
- avec zéro ou un choix, il conserve le contrat automatique du bloc 1.

Le scorer partage l'encodeur d'état, l'encodeur d'action racine et la tête de score. Le
`decision_kind` et les features applicables permettent de distinguer macro et atomique ; les champs
de conséquence macro non applicables sont nuls et masqués. Il ne doit pas exister une tête dont la
sortie contourne silencieusement l'apprentissage des autres phases.

Entraîner ce mode sur les records `atomic` du dataset V2, tout en conservant les records macro. Le
sampling ou la pondération doit être fixé avant l'expérience afin que les décisions atomiques ne
noient pas les choix macro. V004 sert de contrôle de dérive ; V008 reste le teacher cible lorsque les
labels disponibles proviennent de V008.

Les mesures prioritaires sont `buy_card`, `stop_buying`, `recruit_mercenary`, `assign_power` et les
pending choices non absorbés par le solveur. Rapporter aussi la part des décisions exécutées par :

- replay automatique ;
- branche macro du réseau unifié ;
- candidat atomique du réseau unifié ;
- fallback V004 historique.

Critères d'acceptation :

- couverture et accuracy par phase/action/matchup ;
- argmax drift contre V004 sur les slices non entraînées ;
- benchmark avec les poids macro gelés lors du premier apprentissage atomique, afin d'attribuer le
  changement ;
- toutes les classes d'actions légales supportées par la représentation unifiée ;
- zéro appel au fallback V004 sur un corpus déterministe couvrant les parties complètes et les
  limites du solveur ;
- compteur runtime `legacy_fallback_decisions == 0` sur le screening puis sur le panel complet ;
- checkpoint unique et explicite pour le réseau unifié.

Condition d'arrêt : ne pas retirer V004 tant qu'une action légale peut rester non représentable, que
le fallback est encore appelé, ou que le mode atomique provoque une régression non attribuée sur les
choix macro. Une fois ces conditions levées, V004 sort du chemin nominal et reste uniquement un
rollback explicite de profil.

### Bloc 7 — États visités et DAgGER macro

**Objectif.** Réduire le décalage entre les états visités par le teacher macro V8 et ceux visités par
le candidat après ses premières divergences.

Ce bloc ne commence qu'après stabilisation du meilleur encodeur macro offline. Collecter les états
réellement visités par le candidat, demander à V8 le choix contrefactuel parmi les mêmes racines, et
conserver le parent checkpoint, la seed, le fingerprint de politique et la provenance de chaque
record.

Critères d'acceptation :

- séparation des records imitation initiale et DAgGER ;
- pondération et volume déclarés avant entraînement ;
- budget d'argmax drift sur le holdout initial ;
- comparaison `sans DAgGER` contre `avec DAgGER` à architecture et seeds identiques.

Condition d'arrêt : ne pas augmenter mécaniquement le volume si le holdout initial régresse ou si le
gain vient uniquement des états collectés par la même politique.

### Bloc 8 — Composition, validation et promotion

**Objectif.** Composer uniquement les blocs ayant fourni une preuve indépendante.

Ordre de comparaison :

1. V004 atomique seul ;
2. macro V1 exp00109 + fallback V004 ;
3. meilleur macro V2 + fallback V004 ;
4. réseau V2 unifié couvrant macro et atomique, avec mesure du fallback résiduel ;
5. réseau V2 unifié sans fallback nominal, seulement si le bloc 6 atteint zéro appel ;
6. variante DAgGER, seulement si le bloc 7 est accepté.

Rapporter par seed les taux de victoire, deltas, décisions macro/fallback, désaccords stratégiques,
latence, collisions résiduelles et dérive d'argmax. La promotion suit le panel complet et ses poids
courants. Une baisse de décisions ou une amélioration contre V008 seule n'est pas une preuve
suffisante.

## Data Model

Le schéma V2 ajoute au record `macro_play` :

- `candidate_schema_version` ;
- `candidate_feature_set` ;
- `root_actions`, alignées positionnellement avec `candidates` ;
- conséquences connues optionnelles et versionnées ;
- diagnostic d'équivalence/collision ;
- métadonnées de provenance nécessaires aux ablations.

Les records `atomic` adoptent le même contrat candidat : chaque action légale devient un candidat
de longueur 1, `decision_kind=atomic`, avec les champs macro non applicables masqués. Le chosen index
reste aligné sur la liste des candidats, ce qui permet un trainer et un scorer communs sans perdre
la ventilation des métriques par espace de décision.

Les traces atomiques exactes restent disponibles pour le replay et le diagnostic. Les IDs
d'instance ne sont pas des features du modèle. Le manifest porte les versions observation, action,
candidat, catalogue et solveur.

## Information Mask And Safety

Toute feature est construite depuis `NeuralObservation` et les actions légales visibles. Une donnée
présente uniquement dans `Game.clone()` n'est pas automatiquement autorisée. Les tests doivent
comparer des états internes différents donnant la même observation masquée et vérifier que les
tenseurs candidats restent identiques lorsque seule l'information cachée varie.

Le moteur reste l'autorité : chaque élément de trace est revalidé par `legal_actions()` avant
`Game.apply()`. Une trace devenue illégale reste une erreur de solveur, jamais une occasion de
forcer l'action.

## Observability And Operations

Chaque rapport macro doit inclure :

- schémas et feature sets effectifs ;
- nombre total de records, vrais choix et décisions automatiques ;
- cardinalité moyenne et distribution des candidats ;
- collisions totales, collisions du teacher et labels impossibles ;
- top-1/rang par action racine et par nombre de candidats ;
- taux de désaccord séparé pour choice/replay/fallback ;
- décisions automatiques, macro, atomiques unifiées et fallback historique par partie ;
- taux de recours au fallback historique, dont la cible finale est zéro ;
- latence moyenne et p95 du scorer unifié dans chaque mode ;
- résultats de partie par adversaire et par seed.

Une métrique absente n'est pas interprétée comme zéro. Les rapports doivent identifier le checkpoint
par profil, architecture et hash, pas seulement par le chemin mutable.

## Edge Cases

- zéro candidat macro : présentation des actions atomiques légales au même réseau ;
- un candidat : replay automatique, aucune inférence ;
- plusieurs instances de la même définition : aucune identité d'instance apprise ;
- deux définitions à conséquence numérique identique : distinguées par l'action racine ;
- choix pending non carte : catégorie stable ou feature inconnue, jamais vocabulaire libre ;
- draw/shuffle : aucune identité future révélée au modèle ;
- candidat terminant immédiatement la partie : flag connu et borné ;
- checkpoint/dataset de mauvais schéma : erreur explicite avant training ;
- budget solveur atteint : décision atomique par le même réseau, sans candidat macro partiel ;
- collision résiduelle : rapportée et bloquante si elle concerne des branches non prouvées
  équivalentes.

## Testing Strategy

- tests unitaires du replay automatique à candidat unique et absence de payload ;
- tests de sérialisation V1/V2 et refus des mélanges incompatibles ;
- tests des tenseurs racine par type, définition, cible et pending kind ;
- tests de non-utilisation des IDs d'instance ;
- tests du masque d'information avec états internes jumeaux ;
- tests de collisions sur bannissement, pending, activation et play card ;
- tests des deltas et de chaque feature tactique ;
- tests de métriques excluant les pseudo-décisions ;
- tests de toutes les actions légales dans l'adaptateur candidat unifié ;
- tests prouvant que les limites du solveur basculent vers le mode atomique du même réseau ;
- test de partie complète avec `legacy_fallback_decisions == 0` ;
- smoke generation/training/reload du checkpoint mutable ;
- tests de trajectoires déterministes avant/après le bloc 1 ;
- screening en partie puis panel complet uniquement pour les candidats ayant passé leurs contrôles
  offline préenregistrés.

## Rollout And Migration

1. Conserver exp00109 et son dataset V1 comme contrôle historique.
2. Livrer et valider le bloc 1 sans modifier de checkpoint.
3. Introduire le schéma et l'architecture V2 sans changer les pointeurs actifs.
4. Régénérer un dataset V2 ; ne pas convertir artificiellement le JSONL V1.
5. Exécuter les ablations identité, conséquences et tactique séparément.
6. Geler les poids macro avant la première expérience de couverture atomique unifiée.
7. Conserver V004 dans le chemin transitoire jusqu'à couverture de toutes les actions et zéro appel
   observé ; le retirer ensuite du chemin nominal sans supprimer le checkpoint stable.
8. N'introduire DAgGER qu'après une baseline V2 unifiée reproductible.
9. Composer les blocs acceptés et exécuter le panel complet sans fallback nominal.
10. Copier vers `configs/neural_profiles/vNNN.pt` et modifier les pointeurs actifs uniquement après
   acceptation explicite de la promotion.

Un bloc rejeté conserve son résultat, ses limites, sa condition d'arrêt et les changements requis
avant toute relance. Il n'est pas répété avec seulement plus d'epochs ou plus de données.

## Files Expected To Change

Chemins probables, à confirmer au début de chaque bloc :

- `shards_ai/ai/play_turn_solver.py` ;
- `shards_ai/ai/macro_player.py` ;
- `shards_ai/ai/macro_model.py` ;
- `shards_ai/ai/macro_imitation_dataset.py` ;
- `shards_ai/ai/macro_training.py` ;
- `shards_ai/ai/action_representation.py` ;
- `shards_ai/ai/structured_v006.py` ou un helper tactique partagé ;
- `scripts/generate_macro_imitation_dataset.py` ;
- `scripts/train_macro_imitation.py` ;
- `scripts/compare_macro_vs_heuristic_actions.py` ;
- `scripts/validate_macro_neural_profile.py` ;
- `benchmarks/benchmark_neural_mix.py` ;
- `configs/neural_training_profiles/candidates/` ;
- `tests/ai/test_play_turn_solver.py` ;
- `tests/ai/test_macro_observer.py` ;
- `tests/ai/test_macro_imitation_dataset.py` ;
- `tests/ai/test_macro_model.py` ;
- `tests/ai/test_macro_training.py` ;
- `doc/Current state/Neural player.md`, seulement après implémentation effective de chaque
  comportement.
