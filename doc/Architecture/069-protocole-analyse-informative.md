# Protocole d'analyse informative du joueur neural

## Objective

Empêcher les expériences `analysis` de répéter un diagnostic déjà réalisé sans produire de
connaissance nouvelle. Une analyse doit réduire une incertitude utile pour choisir une future
expérience de qualité ou de performance.

Le protocole distingue deux cas :

- si le profil neural étudié a changé, les analyses précédentes peuvent être rejouées, car le
  comportement à expliquer est différent ;
- si le profil est inchangé, une analyse doit répondre à une question encore ouverte avec une
  méthode, une population ou une métrique apportant une information nouvelle.

## Current State

Le mode `analysis` est traité dans `scripts/meta_improve.py`. Il vérifie actuellement seulement
que l'agent produit des observations et que les tests fixes passent. Le prompt demande de lire les
rapports historiques, mais ne force pas l'agent à expliciter ce qui reste inconnu ni ne rejette une
analyse redondante.

Les rapports `exp-00039` et `exp-00044` illustrent cette limite : ils étudient tous deux v002 sur
des trajectoires de type v008, avec des découpages proches, et aboutissent aux mêmes conclusions
sur BUY, `recruit_mercenary` et les erreurs PLAY confiantes. `exp-00049` apporte une information
réellement nouvelle grâce à un holdout indépendant par partie et à l'inclusion de `RandomPlayer`.

Les données durables disponibles sont le manifeste JSON, le rapport Markdown, le profil parent,
le commit parent, le dataset et son hash, ainsi que le catalogue `doc/Ideas.md`.

## Target Behavior

Avant de choisir une analyse, l'agent doit construire un bref inventaire des connaissances déjà
établies sur le profil sujet : conclusions, limites, jeux de données, panels, métriques et
questions ouvertes.

Le rapport doit ensuite expliciter :

- la question inconnue ciblée ;
- pourquoi sa réponse peut modifier le choix d'une future expérience ;
- la différence avec les analyses précédentes ;
- la nouvelle source de preuve, population, comparaison ou métrique ;
- le résultat attendu selon les hypothèses possibles ;
- la décision future associée à chaque résultat possible.

Une analyse qui reprend le même profil, les mêmes données, les mêmes métriques et les mêmes
questions sans information additionnelle est refusée comme redondante avant ou pendant la décision
déterministe. Elle doit être classée `inconclusive` ou `failed` selon qu'elle est insuffisante ou
techniquement incorrecte, et non `accepted` simplement parce que ses tests passent.

## Non-Goals

- Interdire de reproduire une analyse après une modification réelle du profil neural.
- Transformer une analyse en validation de qualité ou en mécanisme de promotion.
- Déduire automatiquement qu'une hypothèse est vraie à partir d'un simple champ textuel.
- Exiger une métrique unique pour tous les profils ou toutes les familles de problèmes.
- Conserver les datasets bruts, logs ou sorties volumineuses dans `doc/`.

## Key Decisions

1. **Le sujet est identifié par profil et commit.** Le profil étudié comprend au minimum
   `analysis_subject_profile` et `parent_commit`. Un changement de l'un de ces éléments autorise
   la reprise d'une analyse historique, mais le rapport doit toujours préciser ce qui est repris.

2. **Une analyse commence par une question de connaissance.** Le champ `analysis_question` doit
   être formulé comme une inconnue falsifiable, par exemple : « Les erreurs PLAY confiantes de
   v002 viennent-elles d'une mauvaise représentation de carte ou d'un mauvais choix d'action ? ».
   « Mesurer à nouveau les erreurs PLAY » n'est pas une question suffisante si elle est déjà
   couverte par un rapport antérieur.

3. **La nouveauté doit être structurée.** Le résultat doit fournir `novelty_basis` avec au moins
   un élément nouveau parmi `profile`, `dataset`, `population`, `opponent`, `metric`, `comparison`
   ou `decision_question`. Pour un profil inchangé, `profile` ne peut pas être le seul élément.

4. **La répétition utile est une réplication explicitement motivée.** Une réplication avec les
   mêmes paramètres peut être acceptée uniquement si elle teste une limite signalée auparavant,
   par exemple un échantillon trop petit, une variance excessive ou une absence de holdout. Elle
   doit référencer la conclusion et la limite concernées.

5. **Le résultat doit être actionnable.** `future_decision` décrit comment le résultat changera la
   sélection d'une prochaine expérience. Une analyse qui ne peut conduire à aucune décision
   différente est un diagnostic descriptif, pas une analyse informative.

6. **Le gate reste déterministe et borné.** Le code vérifie la présence des champs, compare les
   identifiants de provenance aux analyses antérieures et bloque les répétitions manifestes. Le
   jugement scientifique sur la pertinence de la question reste documenté et peut être contrôlé
   par revue humaine.

7. **Le changement de profil ne dispense pas d'apprendre.** Même lorsqu'une analyse est autorisée
   parce que v003 remplace v002, elle doit expliquer quelles observations de v002 sont reconduites,
   lesquelles sont invalidées et quelles nouvelles décisions deviennent possibles.

## Proposed Architecture

### Inventaire historique

Ajouter une lecture des rapports et manifestes `analysis` précédents portant sur le profil sujet.
L'inventaire extrait une empreinte légère composée de :

- profil et commit sujets ;
- hash du dataset ou description de la population ;
- adversaires et seeds ;
- phases, actions, cartes et métriques analysées ;
- question, base de nouveauté et décision future ;
- limites encore ouvertes.

Le prompt reçoit cet inventaire résumé et doit sélectionner une question encore ouverte. Les
rapports complets restent la source de référence lorsque le résumé est insuffisant.

### Contrat de résultat

Le résultat `analysis` ajoute les champs suivants :

```json
{
  "analysis_question": "...",
  "prior_findings_considered": ["exp-00044: ..."],
  "novelty_basis": {
    "dataset": "independent per-game holdout",
    "metric": "ECE and reliability bins"
  },
  "expected_information": "...",
  "future_decision": "..."
}
```

`analysis_subject_profile` reste obligatoire. Le rapport affiche ces éléments avant les
observations afin de séparer la question, la preuve et l'interprétation.

### Décision de redondance

Pour un profil et un commit identiques, le gate rejette ou marque inconclusive une analyse si :

- elle ne fournit pas de `analysis_question` ;
- sa question est absente ou équivalente à une question déjà couverte ;
- elle ne fournit aucune base de nouveauté admissible ;
- son dataset, sa population, ses métriques et ses comparaisons sont tous déjà utilisés ;
- elle ne décrit pas de décision future conditionnée au résultat.

La comparaison textuelle ne doit pas prétendre résoudre la similarité scientifique. Une empreinte
exacte détecte les répétitions manifestes ; les cas ambigus sont signalés pour revue et restent
documentés.

## Data Model

Le modèle reste sans base de données. Les champs sont ajoutés au manifeste ou au résultat JSON :

- `analysis_subject_commit` ;
- `analysis_question` ;
- `prior_findings_considered` ;
- `novelty_basis` ;
- `expected_information` ;
- `future_decision` ;
- `analysis_fingerprint` calculée à partir des éléments de protocole, pas du texte libre.

Les rapports historiques ne sont pas réécrits. Les anciennes analyses dépourvues de ces champs sont
traitées comme des références historiques incomplètes et peuvent être citées, mais ne constituent
pas une preuve de nouveauté pour une nouvelle analyse.

## Observability And Operations

Le manifeste et le rapport doivent permettre de répondre rapidement à trois questions :

1. Quelle inconnue l'analyse cherchait-elle à réduire ?
2. Quelle information nouvelle a-t-elle effectivement produite ?
3. Quelle expérience future cette information autorise-t-elle ou écarte-t-elle ?

Le message de rejet doit nommer l'élément redondant : profil, dataset, population, métrique,
comparaison ou question. Les artefacts détaillés restent sous `artifacts/experiments/`.

## Edge Cases

- Un nouveau profil avec exactement le même protocole est autorisé, mais doit comparer les
  conclusions valides et invalidées entre les deux profils.
- Une réplication avec de nouvelles seeds mais sans justification statistique est insuffisante.
- Un dataset différent mais issu des mêmes parties et contenant les mêmes observations n'est pas
  considéré comme nouveau.
- Une analyse peut découvrir qu'une question précédente est invalide ; elle doit alors citer la
  limite ou l'erreur méthodologique corrigée.
- Si les rapports historiques sont absents ou illisibles, l'analyse est techniquement inconclusive,
  pas automatiquement nouvelle.
- Une analyse interrompue ne consomme pas la question si elle n'a produit aucune preuve exploitable.

## Testing Strategy

- tester l'acceptation d'une analyse avec une question et une base de nouveauté nouvelles ;
- tester le refus d'une analyse identique sur le même profil, commit, dataset et métriques ;
- tester l'autorisation d'un protocole identique après changement de profil ;
- tester le cas de réplication justifiée par une limite antérieure ;
- tester l'absence de base de nouveauté et l'absence de décision future ;
- tester le rendu des nouveaux champs dans le rapport Markdown ;
- tester la reconstruction de l'inventaire à partir des manifestes historiques anciens et nouveaux.

## Rollout And Migration

1. Ajouter les champs et la validation au mode `analysis`.
2. Mettre à jour le prompt pour imposer l'inventaire, la question et la décision future.
3. Ajouter les tests de redondance et de changement de profil.
4. Conserver les rapports existants sans migration destructive.
5. Après déploiement, considérer `exp-00039` et `exp-00044` comme un exemple de répétition à éviter,
   et `exp-00049` comme exemple d'analyse informative.

## Files Expected To Change

- `scripts/meta_improve.py` — inventaire historique, prompt et décision déterministe ;
- `shards_ai/experimentation/manifest.py` — champs de provenance analytique ;
- `shards_ai/experimentation/report.py` — sections question/nouveauté/décision ;
- `tests/experimentation/test_campaign_git.py` — gate et historique ;
- `tests/experimentation/test_primitives.py` — rendu du rapport ;
- `doc/Current state/Artifacts and scripts.md` — protocole courant après implémentation.
