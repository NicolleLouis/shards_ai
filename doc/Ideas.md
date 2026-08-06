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
