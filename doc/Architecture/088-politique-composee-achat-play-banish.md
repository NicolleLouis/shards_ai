# Politique composée achat / play / banish — Architecture

## Objective

Permettre d'entraîner uniquement la politique d'achat et de recrutement pendant que
le `PLAY` et le bannissement utilisent des algorithmes fixes. Chaque famille doit
ensuite pouvoir être remplacée indépendamment par une politique neuronale,
heuristique, déterministe ou de recherche, sans modifier le moteur ni les deux
autres familles.

La première composition cible est :

```text
BUY / recrutement  -> AcquisitionNeuralPolicy
PLAY                -> FixedPlayPolicy
BANISH              -> FixedBanishPolicy
```

Les choix initiaux sont désormais fixés ainsi :

```text
PLAY fixe   -> Heuristic V008
acquisition -> apprentissage par imitation de Neural V006
BANISH fixe -> Blaster prioritaire, puis Crystal depuis la défausse, sinon SkipBanish
```

Le remplacement progressif prévu est :

```text
étape 1 : neural acquisition + play fixe + banish fixe
étape 2 : neural acquisition + play fixe + banish neural
étape 3 : neural acquisition + play neural + banish neural ou fixe
```

Une seule sous-politique est entraînée ou modifiée à la fois au début. Les
comparaisons de qualité utilisent des parties complètes, avec les mêmes seeds,
adversaires et rôles.

## Current State

Le moteur expose les actions atomiques via `Game.legal_actions()` et reste
l'autorité de la légalité et des transitions. Les familles pertinentes sont :

- achat : `BuyCard`, `StopBuying` ;
- recrutement : `RecruitMercenary`, `RecruitFreeCard` ;
- play : `PlayCard`, `ActivateChampion`, `GainMastery`, `PassPlayPhase` et les
  décisions pendantes non couvertes par le banissement ;
- banissement : `BanishCard`, `SkipBanish`.

`NeuralPlayer` reçoit `NeuralObservation` et score les actions légales. La
représentation masque les identifiants d'instance dans le modèle, tout en les
conservant dans l'action appliquée. `HeuristicPlayer` reçoit un `GameState` et
implémente déjà les achats, recrutements, sorties de phase et bannissements.

`HybridPlayer` permet déjà trois ablations ponctuelles, mais son contrat est
centré sur une délégation réseau/heuristique et ne constitue pas encore une
composition générique de trois politiques remplaçables.

`MacroNeuralPlayer` et `PlayTurnSolver` peuvent produire une trace atomique de
plusieurs actions PLAY. Le solver connaît aussi `BanishCard` et
`SkipBanish`. Une trace PLAY qui traverse un bannissement empêcherait donc de
remplacer le bannissement indépendamment : cette frontière doit être renforcée.

Le profil neural actif est `v005`, architecture
`structured_semantic_v5_macro_tactical_action_v1`. Le checkpoint mutable unique
reste `artifacts/neural_training/checkpoint.pt`. Il ne faut pas créer trois
checkpoints de travail simultanés.

## Target Behavior

Le runner conserve un joueur unique présentant une méthode `choose_action` au
moteur. Ce joueur contient un routeur et trois modules de politique.

À chaque décision :

1. le moteur fournit l'état et les actions légales ;
2. le routeur identifie la famille à partir des actions légales et de l'état
   pendant la décision ;
3. le module correspondant choisit une action parmi les actions qui lui sont
   présentées ;
4. le moteur valide et applique l'action ;
5. le joueur journalise la famille, l'identifiant de politique et la raison du
   choix.

La priorité de routage est :

```text
pending banishment       -> BanishPolicy
pending free recruitment -> AcquisitionPolicy
buy / recruitment        -> AcquisitionPolicy
remaining PLAY            -> PlayPolicy
other phases              -> politique existante ou adaptateur dédié
```

Le routeur ne doit pas déduire la famille uniquement de `Phase.PLAY`, car un
bannissement peut être demandé pendant PLAY ou BUY.

## Non-Goals

- modifier les règles du moteur, les actions publiques ou leur validation ;
- entraîner simultanément les trois politiques dans la première livraison ;
- créer trois variantes incompatibles de `NeuralObservation` ;
- attribuer au réseau les actions automatiques rejouées par le solver ;
- utiliser le score offline d'une seule tête comme preuve d'amélioration en
  partie ;
- faire du play fixe une dépendance cachée de l'algorithme de bannissement ;
- créer un checkpoint mutable par module.

## Key Decisions

1. **Frontière stable de composition.** Les politiques choisissent une action
   légale ; elles ne mutent pas directement le jeu et ne contournent jamais
   `Game.apply()`.

2. **Contrat commun.** Chaque politique expose au minimum :

   ```python
   choose_action(
       observation: GameState,
       legal_actions: Sequence[Action],
   ) -> Action
   ```

   Une politique neuronale reçoit en interne l'observation masquée construite
   par `Game.neural_observation_for(player_id)`. Une politique déterministe ou
   heuristique peut utiliser l'état complet déjà autorisé par son contrat.

3. **Remplacement par configuration.** La composition référence trois
   identifiants indépendants, par exemple `acquisition=neural_candidate`,
   `play=heuristic_v008`, `banish=deterministic_blaster_crystal`. Le moteur et le runner ne
   connaissent pas les classes concrètes.

4. **Achat et recrutement dans une même tête.** `BuyCard`,
   `RecruitMercenary`, `RecruitFreeCard` et `StopBuying` appartiennent à
   `AcquisitionPolicy`, car le recrutement modifie le même compromis de deck et
   intervient dans le même horizon de décision.

5. **Banish prioritaire et indépendant.** Dès que les actions légales sont
   `BanishCard`/`SkipBanish`, le routeur appelle exclusivement
   `BanishPolicy`. Une politique PLAY ne peut ni choisir ni rejouer un
   bannissement.

6. **Barrière du solver PLAY.** `PlayTurnSolver` doit s'arrêter avant une
   décision de bannissement et rendre un préfixe atomique au routeur. Une trace
   PLAY ne peut pas contenir une action de bannissement non encore choisie par
   `BanishPolicy`. Cela peut réduire certaines macros, mais préserve la
   remplaçabilité et la causalité expérimentale.

7. **Play fixe en première étape.** Le play initial est `Heuristic V008`.
   `FixedPlayPolicy` reste un adaptateur remplaçable, afin de pouvoir tester
   ensuite V005 macro ou un solver déterministe sans modifier le routeur.

8. **Enseignant acquisition initial.** Le modèle d'acquisition commence par
   imiter `Neural V006`. Le dataset conserve `teacher_id=v006` et mesure
   séparément `BuyCard`, `RecruitMercenary`, `RecruitFreeCard` et `StopBuying`.

9. **Banishment fixe initial.** `FixedBanishPolicy` choisit dans cet ordre :
   un Blaster de la main ou de la défausse ; à défaut un Crystal de la défausse ;
   à défaut `SkipBanish`. Les définitions sont identifiées par `card_id`
   (`blaster` et `crystal`). En cas de plusieurs correspondances, le choix est
   stable et documenté.

10. **Apprentissage isolé.** Le premier modèle neural ne score que les candidats
   d'acquisition. Le dataset est filtré sur les décisions d'achat/recrutement,
   partitionné par `game_id`, et ne contient pas de labels PLAY ou banishment.
   Le modèle peut réutiliser un encodeur commun, mais son checkpoint doit
   déclarer explicitement sa famille et son contrat d'inférence.

11. **Une seule cible mutable.** Pendant l'entraînement de l'acquisition,
   `NEURAL_CHECKPOINT` désigne le seul checkpoint de travail. Les politiques
   fixes chargent des profils stables ou des paramètres heuristiques protégés.
   Une version validée est ensuite copiée dans `configs/neural_profiles/`.

12. **Diagnostics explicites.** Chaque décision expose au moins
   `policy_id`, `decision_family`, `action_type`, `fallback_used` et, pour le
   neural, le nombre de candidats et le score ou rang choisi. Un fallback
   silencieux est interdit.

## Open Questions

- **Non bloquante :** choisir le prochain remplacement du `FixedPlayPolicy`
  entre V005 macro et un solver déterministe, après l'expérience acquisition.
- **Non bloquante :** déterminer si une future cible d'acquisition issue d'une
  recherche dépasse suffisamment Neural V006 pour justifier un nouveau dataset.

Le choix du critic PPO est décidé pour la trajectoire cible : encodeur partagé,
trois têtes de politique spécialisées et critic de valeur partagé au départ.
Une spécialisation ultérieure des critics reste une ablation, pas une précondition
de l'architecture.

## Proposed Architecture

```text
                         GameRunner
                             │
                 GameState + legal_actions
                             │
                    ComposedPlayer
                             │
                    DecisionRouter
              ┌──────────────┼──────────────┐
              │              │              │
    AcquisitionPolicy    BanishPolicy    PlayPolicy
       neural/fixed      fixed/neural    fixed/neural
              │              │              │
       BuyCard /        BanishCard /   PlayCard /
       Recruit* /       SkipBanish     Champion /
       StopBuying                      Mastery / Pass
              └──────────────┬──────────────┘
                             │
                    Game.apply(action)
```

### Cible PPO multi-tête

Lorsque les trois politiques seront réintroduites dans un apprentissage commun,
le modèle cible sera :

```text
NeuralObservation + candidats légaux
              │
       encodeur partagé
              │
    ┌─────────┼─────────┬─────────┐
    │         │         │         │
 buy head  play head  banish head  V(s)
    │         │         │         │
 acquisition PLAY    banish    valeur état
```

Chaque tête conserve son espace de candidats et son masque de légalité. Le
critic partagé estime la valeur de l'état global, car les trois familles
modifient la même partie et partagent la récompense finale. Les transitions
restent étiquetées par `decision_family`, ce qui permet des statistiques et
pertes séparées sans fabriquer trois épisodes indépendants.

Je recommande ce critic partagé au départ plutôt que trois critics spécialisés :
il évite de dupliquer la valeur sur des états identiques et facilite le crédit
à long terme entre achat et utilisation future des cartes. Une ablation avec
`V_buy`, `V_play` et `V_banish` pourra être mesurée après obtention d'une
baseline stable.

Le PPO multi-tête n'est pas activé pendant l'entraînement initial de
l'acquisition. Cette campagne reste une imitation isolée de V006 contre PLAY
Heuristic V008 et Banish déterministe.

### `DecisionRouter`

Le routeur est responsable uniquement de la classification de la décision et
de la délégation. Il vérifie que le module sélectionné est compatible avec les
actions légales. Une réponse d'un mauvais type ou une action absente de la liste
légale produit une erreur explicite avant le retour au runner.

Les actions de bannissement ont priorité sur les autres classifications. Une
décision de recrutement gratuit pendante a priorité sur l'achat normal. Les
classes d'action sont préférées à des tests de chaîne sur `phase`.

### `AcquisitionPolicy`

Le module neural représente et score uniquement les actions d'acquisition
présentes dans l'état courant. Il doit inclure l'alternative de sortie
(`StopBuying`) afin que le réseau apprenne quand conserver ses ressources.

Les recrutements gratuits et mercenaires sont encodés comme des actions
d'acquisition, avec leurs différences de coût, de destination et d'effet
immédiat conservées dans la représentation. Le module ne décide jamais de
jouer une carte ou de bannir une carte.

### `FixedPlayPolicy`

Ce module encapsule le comportement PLAY gelé. Il peut utiliser
`PlayTurnSolver`, mais ses traces doivent respecter la barrière de banissement.
Les actions rejouées automatiquement par le solver sont des transitions moteur,
pas de nouvelles décisions de l'acquisition.

Le module doit être réinitialisable au début d'une partie et vider toute trace
en attente lors d'un changement de joueur, d'une erreur de légalité ou d'une
nouvelle frontière de décision.

Le choix initial est `Heuristic V008`. L'adaptateur conserve une configuration
permettant de le remplacer par V005 macro ou un solver déterministe.

### `FixedBanishPolicy`

Ce module applique la priorité `blaster` dans la main ou la défausse, puis
`crystal` dans la défausse, puis `SkipBanish`. À égalité, il choisit la carte
correspondant au plus petit identifiant d'instance afin que la règle soit
reproductible. Il ne reçoit jamais une trace PLAY déjà composée contenant un
banissement.

### Adaptateur de compatibilité

L'interface publique actuelle de `GameRunner` reste inchangée. Un adaptateur
convertit l'observation `GameState` en `NeuralObservation` uniquement pour le
module neural. `HybridPlayer` peut être conservé pour les benchmarks
historiques, mais la nouvelle composition doit avoir un nom et des diagnostics
distincts afin de ne pas confondre une ablation ancienne avec cette
architecture.

## Data Model

Les enregistrements de décision d'acquisition doivent ajouter ou normaliser :

```text
decision_family: acquisition
decision_kind: atomic
action_type: buy_card | recruit_mercenary | recruit_free_card | stop_buying
game_id
observation
legal_candidates
chosen_candidate
teacher_id
```

Les décisions PLAY et banishment peuvent rester dans les datasets généraux pour
les benchmarks, mais sont exclues du dataset d'entraînement initial de la tête
d'acquisition. Les identifiants de carte et les candidats doivent respecter les
règles d'observation masquée existantes ; les identifiants d'instance ne doivent
pas devenir une feature stratégique.

Un profil d'acquisition doit déclarer au minimum : architecture, famille,
dataset, parent éventuel, seeds, split `game_id`, checkpoint mutable, enseignant
et politique de validation. Le profil ne doit pas être chargé par
`build_neural_player()` comme s'il s'agissait d'un checkpoint macro PLAY.

## Backend Flow

1. Construire le joueur composé avec trois modules et leurs `policy_id`.
2. À chaque tour, demander les actions légales au moteur.
3. Router une décision de bannissement ou de recrutement pendante avant la
   décision de phase générale.
4. Appeler le module choisi et valider son résultat.
5. Appliquer l'action par `Game.apply()`.
6. Enregistrer un événement de décision pour les choix réels ; marquer les
   replays automatiques comme `replay`, sans les compter comme décisions
   neural.
7. Réinitialiser l'état local du module en fin de partie ou lors d'une erreur.

Si une politique renvoie une action illégale, le joueur lève une erreur
diagnostique. Un fallback optionnel doit être explicite dans la configuration,
compter dans `fallback_used` et être séparé des résultats du réseau.

## Observability And Operations

Les benchmarks doivent enregistrer par partie et par famille : nombre de
décisions, actions choisies, replays, fallbacks, temps d'inférence neural,
identifiants des trois politiques, seed, adversaire, résultat, tours, santé,
maîtrise, dégâts et composition finale du deck.

Les rapports doivent séparer au minimum :

- performance de la composition complète ;
- performance avec une seule tête remplacée ;
- accord offline de la tête acquisition ;
- résultat contre le play et le banish fixes ;
- première divergence entre deux compositions sur les mêmes seeds.

Une amélioration de l'accord achat ne suffit pas à promouvoir le modèle si le
score de partie complète baisse.

## Edge Cases

- une action `BanishCard` apparaît pendant BUY : elle est routée vers
  `BanishPolicy`, pas vers `AcquisitionPolicy` ;
- un recrutement gratuit apparaît après un effet PLAY : il est routé vers
  `AcquisitionPolicy` uniquement si le moteur expose `RecruitFreeCard` comme
  décision légale ;
- une trace PLAY fixe arrive à une frontière banishment : elle est interrompue,
  puis reprise après résolution légale du banissement ;
- une seule action légale : elle peut être appliquée automatiquement, sans
  inférence ni transition d'apprentissage ;
- un module neural sans candidat compatible, un score non fini ou un checkpoint
  de mauvaise famille doit échouer explicitement ;
- une politique fixe utilisant une heuristique qui voit `GameState` ne doit pas
  transmettre cet état au modèle neural ni aux métriques d'observation masquée.

## Testing Strategy

- tests du routeur pour chaque famille et pour les priorités banishment /
  recrutement pendante ;
- test garantissant qu'une trace PLAY ne contient pas de bannissement traversant
  la frontière ;
- tests de compatibilité de chaque implémentation avec les actions légales ;
- tests d'équivalence moteur : la composition ne modifie aucune transition
  autorisée ;
- tests de dataset vérifiant l'exclusion des décisions PLAY et banishment du
  train acquisition ;
- test de chargement refusant un checkpoint macro PLAY dans
  `AcquisitionPolicy` ;
- smoke tests des quatre compositions : neural acquisition, acquisition
  heuristique, acquisition déterministe et baselines fixes ;
- benchmark apparié sur un petit panel avant toute campagne longue, puis panel
  complet contre le profil actif V005 et les adversaires de référence. La
  composition initiale de référence est explicitement
  `acquisition=v006`, `play=heuristic_v008`,
  `banish=deterministic_blaster_crystal`.

## Rollout And Migration

### Étape 1 — acquisition isolée

Implémenter le routeur et les adaptateurs sans modifier le comportement du
moteur. Figer PLAY et banishment. Générer un dataset spécialisé achat /
recrutement, entraîner dans `NEURAL_CHECKPOINT`, puis comparer la composition
  à une composition identique où l'acquisition est Neural V006, puis à une
  composition identique où l'acquisition est Heuristic V008 si cette baseline
  est disponible pour la même trajectoire.

### Étape 2 — banishment remplaçable

Ajouter le module neural ou une nouvelle heuristique banishment. Vérifier d'abord
la barrière du solver et les scénarios de bannissement pendant BUY et PLAY.
Comparer `neural acquisition + banish neural` à la même acquisition avec
banishment fixe, en gardant PLAY strictement inchangé.

### Étape 3 — play remplaçable

Brancher une politique PLAY neural ou améliorée, sans réentraîner simultanément
l'acquisition. Utiliser d'abord l'acquisition et le banishment validés comme
composants fixes, puis mesurer la nouvelle politique PLAY.

Chaque étape est réversible en changeant les trois identifiants de politique.
Une promotion technique du module et une promotion qualité de la composition
complète restent deux décisions séparées.

## Files Expected To Change

- `shards_ai/ai/composed_player.py` — nouveau joueur composé, nom indicatif ;
- `shards_ai/ai/decision_policy.py` — protocole, résultats et diagnostics,
  nom indicatif ;
- `shards_ai/ai/decision_router.py` — classification et délégation,
  nom indicatif ;
- `shards_ai/ai/macro_player.py` et `shards_ai/ai/play_turn_solver.py` —
  interruption des traces avant bannissement ;
- `shards_ai/ai/neural_player.py` ou nouveau module acquisition — adaptation du
  scorer et du chargement de profil ;
- `shards_ai/ai/player_factory.py` — construction depuis une composition ;
- `shards_ai/ai/hybrid_player.py` — conservation ou adaptation de compatibilité,
  à confirmer pendant l'implémentation ;
- `scripts/normalize_imitation_dataset.py` et scripts de génération/formation —
  dataset acquisition filtré ;
- `benchmarks/benchmark_neural_hybrids.py` ou nouveau benchmark de compositions ;
- `tests/ai/test_hybrid_player.py` et nouveaux tests du routeur/solver ;
- `configs/neural_training_profiles/candidates/` — profil acquisition ;
- `Makefile` — cibles spécialisées utilisant `NEURAL_CHECKPOINT` ;
- `doc/Current state/Neural player.md` — uniquement après implémentation et
  validation du comportement final.
