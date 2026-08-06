# Analyses diagnostiques de la boucle neural

## Objective

Ajouter un troisième mode d'expérience, `analysis`, qui étudie la dernière version neural active
sans entraîner ni promouvoir de candidate. L'analyse produit un rapport Markdown exploitable par
les expériences suivantes pour choisir des hypothèses mieux ciblées.

Une analyse est déclenchée immédiatement après une promotion qualité, ou après quatre expériences
qualité consécutives non promues. Elle ne remplace pas l'analyse humaine et ne constitue pas une
preuve de progression.

## Current State

`scripts/meta_improve.py` ne distingue actuellement que les expériences `quality` et `performance`.
Le prompt demande déjà de lire `doc/Ideas.md` et `doc/Experiments`, mais le gate attend pour une
expérience qualité des résultats de validation et une candidate promouvable. Les rapports sont
commités pour chaque tentative et les artefacts détaillés restent sous `artifacts/experiments/`.

## Target Behavior

Une expérience `analysis` :

- charge la dernière version neural active comme sujet d'étude ;
- compare si utile son parent neural actif, V008 et les autres adversaires visibles ;
- mesure des erreurs et désaccords par phase, action, carte, dataset et confiance ;
- peut exécuter des analyses offline et des benchmarks diagnostiques reproductibles ;
- écrit un résultat JSON et un rapport Markdown avec observations, limites et recommandations ;
- peut enrichir `doc/Ideas.md` avec des hypothèses futures ;
- ne demande ni `candidate_profile`, ni checkpoint candidat, ni promotion.

Les expériences suivantes lisent ces rapports au même titre que les autres rapports historiques.

## Non-Goals

- Modifier le moteur, les joueurs heuristiques ou le masque d'information.
- Promouvoir un checkpoint depuis une analyse.
- Faire de l'analyse une nouvelle métrique de qualité cachée dans le gate.
- Conserver dans `doc/` des datasets bruts, logs ou sorties volumineuses.

## Key Decisions

- `analysis` devient une valeur explicite de `experiment_kind`, distincte de `quality` et
  `performance`.
- Une analyse est acceptée si elle produit un rapport exploitable et que les tests fixes passent ;
  elle n'est pas évaluée par la gate qualité ou la gate performance.
- Le compteur porte sur les expériences qualité non promues consécutives. Les expériences
  performance n'augmentent ni ne réinitialisent ce compteur.
- Une promotion qualité remet le compteur à zéro puis programme immédiatement une analyse.
- Quatre échecs qualité consécutifs déclenchent une analyse avant la qualité suivante. Une analyse
  terminée remet le compteur à zéro ; un timeout ou une erreur conserve le compteur pour éviter de
  perdre le diagnostic attendu.
- Le sujet de l'analyse est toujours le profil neural actif au démarrage de l'expérience, enregistré
  dans le manifeste avec son commit et sa version.
- Une analyse peut modifier `doc/Ideas.md`, et cette modification est conservée même si l'analyse
  est techniquement non concluante ; son code temporaire est supprimé comme pour un rejet.

## Proposed Architecture

`Campaign` maintient un compteur local reconstruit au démarrage à partir des manifests historiques.
Après chaque expérience qualité, il met à jour ce compteur selon le statut final. La boucle principale
insère une analyse dans les deux cas prévus, en utilisant la commande agent dédiée ou la commande
par défaut.

Le prompt `analysis` impose une analyse du profil actif, des mesures reproductibles et un résultat
avec `analysis`, `observations`, `limitations` et `recommendations`. Il rappelle que toute
information utilisée doit être visible par le joueur neural.

Le manifeste ajoute les champs nécessaires au suivi (`analysis_subject_profile`, résumé et
recommandations éventuels). Le rapport affiche une section dédiée. Les analyses détaillées ou
datasets intermédiaires sont archivés sous l'artefact de l'expérience, jamais dans `doc/`.

## Data Model

Pas de migration ni de base de données. Les données durables sont :

- `experiment_kind: analysis` dans `manifest.json` ;
- `parent_profile` comme sujet neural actif ;
- le JSON de résultat archivé ;
- le rapport `doc/Experiments/exp-NNNNN.md` ;
- le diff éventuel de `doc/Ideas.md`.

## Observability And Operations

Le rapport doit distinguer clairement mesures, interprétations et recommandations. Chaque mesure
doit indiquer seed, profil, dataset, panel, nombre de parties ou d'états et limites statistiques.
Un timeout conserve les logs et le rapport mais ne remet pas le compteur d'échecs à zéro.

## Edge Cases

- Aucun dataset ou checkpoint analytique n'est copié dans Git.
- Une analyse sans observation exploitable est `failed`, mais reste documentée.
- Une promotion performance ne déclenche pas d'analyse de qualité.
- Si une analyse est forcée mais interrompue, l'expérience suivante reste une analyse jusqu'à
  production d'un diagnostic complet, sauf interruption répétée nécessitant une intervention.
- Une analyse ne peut pas contourner la validation des chemins protégés.

## Testing Strategy

- Tester la reconstruction du compteur depuis les manifests.
- Tester le déclenchement après quatre statuts qualité non promus.
- Tester le déclenchement après une promotion et l'absence de déclenchement après performance.
- Tester qu'une analyse ne demande pas de candidate et produit un commit de rapport.
- Tester la conservation des modifications `doc/Ideas.md` et la suppression du code temporaire.
- Exécuter la suite complète d'expérimentation et les tests de validation neural.

## Files Expected To Change

- `scripts/meta_improve.py`
- `shards_ai/experimentation/manifest.py`
- `shards_ai/experimentation/report.py`
- `configs/meta_improvement.yaml`
- `tests/experimentation/`
- `doc/Current state/Artifacts and scripts.md`

## Open Questions

- Non bloquante : quelles analyses spécialisées seront ajoutées en premier ; la V1 peut accepter
  une analyse générale produite par l'agent puis enrichir le catalogue progressivement.
