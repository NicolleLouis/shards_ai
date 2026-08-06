# Idées et priorités de travail

Ce fichier contient les décisions utiles pour choisir les prochaines expériences. Les détails
complets, les métriques et les commandes sont conservés dans `doc/Experiments/`.

## Règles de décision

- Comparer chaque candidate à la référence neural active, à Random, à v007 et à la garde v008.
- Utiliser un panel complet et reproductible ; un screening court sert uniquement à détecter une
  régression évidente, jamais à accepter une candidate.
- Séparer les parties entre entraînement, validation et holdout. Le split doit être fait par partie,
  pas seulement par décision.
- Mesurer simultanément la qualité et le temps de partie. 
- Modifier une seule hypothèse à la fois. Ne pas modifier le moteur, les heuristiques ou le masque
  d’information pour une expérience neural.
- Conserver v002 comme référence tant qu’un nouveau profil n’a pas passé la gate complète.

## Décisions issues des dernières expériences

| Expérience | Statut | Conclusion opérationnelle |
|---|---|---|
| exp-00023 | Rejetée | Les reprises PPO courtes n’ont produit aucun gain ; réduire le coût de collecte avant de retenter PPO. |
| exp-00024 à exp-00026 | Rejetées | Le DAgger uniforme et les mélanges historiques/on-policy dégradent Random, v007 ou v002. |
| exp-00027 à exp-00031 | Rejetées | Les pondérations globales ou ciblées améliorent parfois v008, mais ne généralisent pas au panel neural et peuvent ralentir les parties. |
| exp-00032 à exp-00035 | Rejetées | Le filtrage de désaccords, les marges teacher seules et l’ancrage global ne suffisent pas à préserver la politique. |
| exp-00036 | Gate qualité positive, gate finale non retenue | Le dataset équilibré est prometteur, mais le coût de partie est proche de la limite et la régression contre v002 reste notable. |
| exp-00037 | Rejetée | Réduire seulement la taille du fine-tuning ne corrige pas le compromis qualité/performance. |
| exp-00038 | Rejetée | Les premières divergences stratégiques sont trop rares et favorisent `play_card`. |
| exp-00039 | Analyse terminée | Le diagnostic met en évidence `recruit_mercenary`, BUY et les erreurs PLAY confiantes ; aucun checkpoint n’a été modifié. |
| exp-00044 | Analyse terminée | Sur 6 246 états principaux visités par v002, `recruit_mercenary` reste à 22,38 %, les erreurs PLAY confiantes persistent et les logits v001/v002 ne sont pas comparables entre versions. Aucun checkpoint n’a été modifié. |
| exp-00040 | Rejetée | La bonne accuracy holdout et le gain contre Random ne compensent pas la régression v007 ni le ralentissement de 13,57 %. |
| exp-00041 | Rejetée | Une marge teacher élevée seule ne garantit pas une meilleure politique en partie. |
| exp-00045 | Rejetée | La correction ciblée de `recruit_mercenary` améliore v001 mais régresse v007, v008 et v002, avec un ralentissement médian de 61,3 %. |
| exp-00046 | Rejetée | La distillation locale ancrée sur v002 hors états mercenaires réduit le temps de partie, mais régresse Random, v007 et v002 sur la validation longue. |
| exp-00047 | Rejetée | L’ancrage local de v002 sur les alternatives hors achat/recrutement ciblés ne corrige pas la dérive : Random, v007, v008 et v002 régressent malgré un gain contre v001. |

Les expériences antérieures sont également disponibles dans `doc/Experiments/`, mais ne doivent pas
être reprises à l’identique sans nouvelle hypothèse ou nouveau protocole.

## Priorité immédiate

### 1. Construire un diagnostic offline exploitable

- Construire un holdout par partie et par seed, avec provenance complète des décisions.
- Comparer v002 et les candidates par phase et type d’action : `buy_card`, `recruit_mercenary`,
  `banish_card`, `play_card` et `attack`.
- Mesurer les décisions changées, les alternatives légales, les marges teacher et les logits v002
  sur ces mêmes états.
- Examiner en priorité les cas où la candidate améliore v008 mais régresse Random, v007 ou v002.
- Mesurer la calibration des erreurs PLAY à forte confiance, notamment pour `crystal`,
  `ermite_fongique` et les champions.

Rapport de départ : `doc/Experiments/exp-00039.md`.

### 2. Tester une correction locale et attribuable

Après le diagnostic uniquement :

- sélectionner un sous-ensemble équilibré par phase et action ;
- conserver explicitement les décisions v002 hors des états ciblés ;
- comparer une loss de classement, une distillation ou une régularisation locale par ablation ;
- conserver un holdout qui n’a jamais servi à choisir les exemples ;
- arrêter la piste si le gain robuste reste inférieur à 2 % pour une optimisation de performance.

### 3. Stabiliser les protocoles

- Utiliser la validation par lots avec reprise pour les panels de plus de 20 parties par adversaire :
  `scripts/validate_neural_profile_batched.py`.
- Conserver les fichiers de progression hors de `doc/`.
- Comparer les mêmes seeds, le même panel et le même nombre de parties entre candidate et référence.
- Ne jamais conclure à partir d’un rapport interrompu ou d’un panel court.

## Pistes à conserver, mais non prioritaires

### Recherche de décision

- Comparer la décision gloutonne à une recherche bornée par temps, avec le réseau comme prior.
- Tester une politique hybride réseau + heuristique ou réseau + recherche uniquement si le budget de
  temps est mesuré séparément.
- Vérifier que toute recherche respecte strictement les informations visibles par le joueur.

### Objectif et apprentissage

- Tester une loss pairwise/margin ou une distillation contrôlée, avec ablation de chaque composante.
- Comparer récompense terminale, shaping borné et shaping par phase, sans modifier plusieurs
  composantes dans la même expérience.
- Reprendre PPO seulement après profilage de la collecte, avec une durée suffisante et une sélection
  monotone protégeant v002, Random, v007 et v008.
- Tester Monte-Carlo à la place de PPO

### Architecture et représentation

- Comparer un encodeur partagé à l’architecture actuelle, avec têtes par phase ou type d’action.
- Tester des variantes d’embeddings, de pooling invariant et de représentation des actions légales.
- Mesurer séparément qualité, mémoire et coût d’inférence ; une architecture plus grande n’est pas
  présumée meilleure.
- Changer le nombre de dimenson notamment pour la représentation d'une carte entre l'aspect sémantique et l'aspect id

### Performance d’inférence

- Explorer le batching sûr, la compilation, la réduction des allocations et les caches sans changer
  la politique observable.
- Pour toute optimisation, conserver le protocole benchmark → profilage → changement ciblé →
  re-mesure, avec au moins trois répétitions lorsque le gain est faible.

## Pistes écartées jusqu’à nouvel élément

- Reprises PPO courtes ou interrompues sans update complet.
- Pondérations globales d’actions ou de phases sans analyse offline.
- DAgger uniforme, premières divergences seules et sélection fondée uniquement sur la marge teacher.
- Promotion fondée sur l’accuracy holdout ou sur un gain contre v008 seul.
- Mélanges historiques/on-policy sans filtrage et sans garde explicite de v002.
- Modification du moteur, des heuristiques ou du masque d’information pour améliorer une candidate
  neural.

Toute nouvelle idée doit être ajoutée dans une section courte avec une hypothèse falsifiable, les
métriques attendues et les gardes à respecter. Une fois testée, elle doit être résumée dans le tableau
ci-dessus et détaillée dans son rapport d’expérience.

## Expérience exp-00044 — analyse terminée

- [Terminé] Diagnostiquer v002 sur 6 246 états principaux visités, stratifiés par v001/v007/v008,
  phase, action, carte, loss, confiance et désaccord avec v001.
- [Résultat] `recruit_mercenary` reste à `22,38 %` d'accuracy ; PLAY représente `72,1 %` des états ;
  174 erreurs v002 ont une confiance intra-état au moins égale à `0,8`.
- [Corrigé] Le holdout par partie et la calibration restent à construire. Un libellé `random` mal
  configuré a été exclu car il utilisait le profil heuristique v001, pas `RandomPlayer`.
- [Suite] Confirmer `buy_card`/`recruit_mercenary` et les erreurs PLAY confiantes sur un holdout
  indépendant avant tout entraînement.

## Expérience exp-00045 — rejetée

- [Terminé] Tester une imitation depuis v002 sur `15 052` décisions de `103` parties, séparées par
  `game_id`, avec teacher v008 contre Random/v007 et une surpondération limitée aux décisions
  `BuyCard`/`RecruitMercenary` ciblées (`2,0`); BUY/PLAY et les autres actions restent uniformes.
- [Résultat] Le panel apparié de 20 parties donne Random `0,0`, v007 `-5,0`, v008 `-5,0` et v002
  `-5,0` points; le gain contre v001 (`+15,0`) ne généralise pas. La médiane du benchmark v002-v002
  passe de `27,9335 s` à `45,0660 s` sur 50 parties (`-38,04 %` de débit).
- [Supprimé] Ne pas promouvoir v009; ne pas reprendre la surpondération mercenaire seule sans
  corriger la dérive de politique et le coût d'inférence/partie.

## Suites issues d'exp-00045

- [À privilégier] Construire un holdout par partie avec attribution des décisions changées par
  catégorie et matchup avant tout nouvel entraînement; vérifier si la correction mercenaire déplace
  des décisions PLAY ou BUY hors de la cible.
- [À étudier] Tester une distillation locale qui conserve explicitement les sorties de v002 hors des
  états mercenaires ciblés, avec une ablation sans surpondération et un budget de changements borné.
- [À étudier] Mesurer séparément le coût d'inférence de la candidate et le nombre d'actions avant de
  poursuivre une piste qualité; toute candidate doit rester comparable à v002 sur le benchmark.

## Expérience exp-00046 — rejetée

- [Terminé] Tester une distillation locale depuis v002 avec une pénalité MSE (`0,25`) sur les
  sorties v002 pour les décisions non ciblées ; les états offrant simultanément `buy_card` et
  `recruit_mercenary` restent entraînés sans surpondération.
- [Résultat] Sur 100 parties par adversaire, Random `-4` points, v007 `-7`, v008 `+4` et v002
  `-9` ; la moyenne des trois gardes demandées est `-2,33` points. Le benchmark comparable de 50
  parties passe de `46,4307 s` à `39,1018 s`, mais avec `18 636` actions contre `17 247`, donc ce
  gain de temps ne constitue pas un gain de force.
- [Supprimé] Ne pas promouvoir v010 ni reprendre cet ancrage global ; conserver l'idée d'un
  ancrage conditionnel seulement après attribution des décisions changées et un budget explicite
  de dérive par catégorie.

## Suites issues d'exp-00046

- [À privilégier] Construire une distillation réellement locale sur les états mercenaires ciblés,
  avec pénalité v002 sur toutes les alternatives légales non ciblées dans le même état, puis
  mesurer les décisions changées par action avant la validation longue.
- [À mesurer] Séparer le temps d'inférence par décision du nombre total d'actions ; le benchmark
  v010 est plus rapide mais suit des trajectoires plus longues.
- [À garder] Rejeter toute candidate qui régresse v002, Random ou v007, même si elle améliore v008
  et l'accuracy holdout.

## Expérience exp-00047 — rejetée

- [Terminé] Tester une imitation depuis v002 sur les états offrant simultanément l’achat et le
  recrutement du même mercenaire, avec teacher v008 sur ces deux actions et ancrage MSE v002 sur
  toutes les autres alternatives légales du même état. Le split est séparé par `game_id`.
- [Résultat] Sur 100 parties par adversaire, la candidate régresse Random de `-5,0` points, v007 de
  `-3,0`, v008 de `-1,0` et v002 de `-2,0`; le gain contre v001 (`+9,0`) ne généralise pas. Le
  benchmark comparable de 50 parties est légèrement plus lent (`17,9898 s` contre `17,7198 s` en
  médiane) et suit `17 691` actions contre `17 247`.
- [Supprimé] Ne pas promouvoir v011 ni reprendre cet ancrage local sans attribution des décisions
  changées et garde explicite de la trajectoire; cette correction concrète d’exp-00046 est rejetée.

## Suites issues d'exp-00047

- [À privilégier] Construire l’attribution offline des changements de politique par action et par
  matchup avant tout nouvel entraînement, puis borner le nombre de décisions modifiées dans les
  états ciblés.
- [À étudier] Tester une correction de représentation ou une loss locale seulement sur les erreurs
  `recruit_mercenary` confirmées par un holdout indépendant; conserver un contrôle sans correction.
- [À supprimer] Tout ancrage MSE local ou global qui ne protège pas Random, v007, v008 et v002 sur
  la validation longue; le gain contre v001 seul ne constitue pas une preuve de qualité.
