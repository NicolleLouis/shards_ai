# Idées et priorités de travail

## Expérience exp-00004 — rejetée

- [Terminé] Reprendre PPO v003 avec reward shaping depuis v001, avec reprise explicite du checkpoint
  mutable canonique et validation complète contre v002, Random, v007 et v008.
- [Résultat] v003 améliore v008 de `+6,5` points et égale v007, mais régresse contre Random de `-3,0`
  points et contre v002 de `-5,0` points ; la promotion est rejetée.
- [Supprimé] Ne pas promouvoir ce checkpoint et ne pas conserver une cible v003 comme nouvelle référence.

## Suites possibles

## Expérience exp-00005 — rejetée

- [Terminé] Reprendre PPO sans reward shaping depuis les poids v002, avec Adam réinitialisé et un
  run exploratoire court (128 parties), puis validation indépendante contre v002, Random, v007 et
  v008.
- [Résultat] La candidate régresse contre Random de `-4,5` points, v007 de `-1,0` point et v008 de
  `-1,0` point ; elle progresse contre v001 de `+8,5` points. La performance passe de `9,378 s` à
  `11,780 s` sur 50 parties contre v002.
- [Supprimé] Ne pas promouvoir la candidate ni reprendre ce protocole court comme preuve de gain.

## Suites possibles

- [À étudier] Comparer les décisions par phase et catégorie entre v002 et la candidate, en particulier
  deckbuilding et banish, avant tout nouvel entraînement.
- [À étudier] Si une nouvelle reprise PPO est tentée, conserver les poids v002 comme référence,
  augmenter la durée d'entraînement et utiliser une sauvegarde de meilleure évaluation qui protège
  explicitement Random et v008.

## Expérience exp-00006 — rejetée

- [Terminé] Corriger le protocole court en reprenant explicitement les poids v002, sans reward shaping,
  avec Adam réinitialisé et une sélection monotone ; le run a atteint un update complet de 122 parties
  avant arrêt à la frontière de sauvegarde.
- [Résultat] La candidate gagne `+0,5` point contre Random, mais perd `-1,5` point contre v007,
  `-1,0` point contre la garde v008 et `-2,0` points contre v002 ; le score pondéré est `+0,1` point
  et la promotion est rejetée.
- [Supprimé] Ne pas promouvoir v006 ni considérer l’amélioration du benchmark 50 parties comme une
  preuve de qualité ; le run ne justifie pas non plus une continuation identique sans analyse.

## Suites possibles

- [À étudier] Comparer les décisions par phase et catégorie de v002 et v006, en priorisant PLAY,
  deckbuilding et banish, pour identifier les régressions v007/v008.
- [À étudier] Reprendre PPO seulement après cette analyse, avec une durée suffisante et une sélection
  qui protège simultanément v002 et la garde heuristique v008.
- [À étudier] Tester une correction ciblée de la représentation ou de la pondération des actions PLAY,
  sans modifier le moteur ni le masque d’information.

## Expérience exp-00007 — non concluante

- [Terminé] Tester une reprise PPO depuis v002 avec Adam réinitialisé, un taux d’apprentissage réduit
  (`5e-5`), un horizon de crédit plus court (`gamma=.99`, `gae_lambda=.90`), 4096 parties prévues et
  sélection monotone.
- [Résultat] La collecte PPO n’a produit aucun update ni checkpoint avant l’interruption pour durée
  excessive. Un contrôle v002 copié vers le checkpoint mutable a été validé sur 200 parties par
  adversaire : delta `0,0` contre Random, v007 et v008, sans changement de politique.
- [Supprimé] Ne pas considérer le contrôle comme une candidate entraînée ni comme une preuve de gain ;
  ne pas promouvoir v009.

## Suites possibles

- [À étudier] Profiler la collecte PPO et réduire son coût par update avant de relancer un entraînement
  long, sans changer le protocole de qualité ni le nombre de parties de validation.
- [À étudier] Comparer offline les décisions v002 par phase et catégorie, en particulier PLAY,
  deckbuilding et banish, puis choisir une correction ciblée de représentation ou de pondération.
- [À étudier] Si PPO est relancé, reprendre exclusivement v002 avec une durée suffisante et une
  sauvegarde monotone protégeant simultanément v002, Random, v007 et v008.

Les rapports sous `doc/Experiments/` servent à comprendre les essais précédents et à éviter les
répétitions inutiles ; ils n'empêchent pas l'agent d'inventer une nouvelle expérience ou de corriger
une piste existante.

## Expérience exp-00008 — rejetée

- [Terminé] Tester une reprise PPO depuis v002 avec un taux d'apprentissage très réduit (`2e-5`),
  des lots de 32 parties et une évaluation fréquente, afin de corriger l'absence d'update de exp-00007.
- [Résultat] Un seul update a été sauvegardé (`32` parties, `5220` transitions), puis la collecte
  n'a pas terminé les `256` parties prévues. La sélection monotone a restauré v002 ; la validation
  exploratoire donne un delta de `0,0` contre Random, v007, v008 et v002.
- [Supprimé] Ne pas promouvoir v009 ni considérer ce protocole interrompu comme une preuve de gain.

## Suites possibles

- [À étudier] Profiler la collecte PPO pour expliquer le blocage après le premier petit lot, puis
  réduire son coût sans modifier le moteur, les heuristiques ou le masque d'information.
- [À étudier] Comparer offline les décisions v002 par phase et catégorie, en priorité PLAY,
  deckbuilding et banish, avant toute nouvelle reprise PPO.
- [À étudier] Si PPO reprend, conserver v002 comme initialisation et référence, avec une durée
  suffisante et une sélection monotone protégeant simultanément Random, v007 et v008.

## Expérience exp-00009 — rejetée

- [Terminé] Tester une continuation PPO bornée à deux petits updates depuis v002, avec Adam
  réinitialisé, taux d'apprentissage `1e-5`, horizon `gamma=.99` / `gae_lambda=.90` et évaluation
  après chaque update.
- [Résultat] Un seul update a été sauvegardé (`32` parties, `4866` transitions) ; l'évaluation
  monotone a restauré v002 et la collecte du second update n'a pas terminé dans le budget. La
  validation donne `0,0` de delta contre Random, v007, v008 et v002. Le benchmark comparable est
  `13,709 s` contre `14,326 s`, mais les `8573` décisions et `17247` actions sont identiques.
- [Supprimé] Ne pas promouvoir v010, ne pas interpréter le gain de temps du checkpoint restauré
  comme un gain de qualité, et ne pas relancer cette variation PPO bornée à l'identique.

## Suites possibles

- [À étudier] Profiler séparément la collecte PPO et la construction des observations pour expliquer
  pourquoi un update de 32 parties dépasse ensuite le budget ; toute optimisation doit rester hors
  du moteur, des heuristiques et du masque d'information.
- [À étudier] Réaliser l'analyse offline v002 par phase et catégorie, en priorité PLAY, deckbuilding
  et banish, puis choisir une seule correction de représentation ou de pondération avant un nouvel
  entraînement.
- [À étudier] Si PPO est relancé après ce diagnostic, utiliser une durée mesurable suffisante et une
  sélection monotone protégeant simultanément v002, Random, v007 et v008.

## Expérience exp-00023 — rejetée

- [Terminé] Tester une continuation PPO conservatrice depuis v002, sans reward shaping, avec un
  taux d'apprentissage réduit (`5e-5`), un horizon `gamma=.99` / `gae_lambda=.90`, deux epochs par
  update et trois updates bornés.
- [Résultat] La candidate a produit 256 parties et 41 248 transitions, mais la sélection monotone
  a restauré v002. Le panel indépendant de 200 parties par adversaire donne `0,0` de delta contre
  Random, v007, v008 et v002; le benchmark médian passe de `9,7551 s` à `9,8206 s`.
- [Supprimé] Ne pas promouvoir v011, ne pas créer de profil neural versionné et ne pas reprendre ce
  protocole PPO borné sans réduire d'abord le coût de collecte.

## Suites possibles

- [À étudier] Profiler séparément la collecte PPO, l'encodage des observations et les mises à jour
  avant toute nouvelle expérience PPO; les quatre dernières reprises n'ont établi aucun gain.
- [À étudier] Réaliser enfin l'analyse offline v002 par phase et catégorie, en priorité PLAY,
  deckbuilding et banish, puis choisir une correction unique de représentation ou de pondération.
- [À étudier] Réserver les prochaines expériences de qualité à une candidate produite par une recette
  non testée et suffisamment longue pour éviter de confondre checkpoint restauré et apprentissage.

## Expérience exp-00024 — rejetée

- [Terminé] Tester une correction DAgger on-policy : collecter les états visités par v002 contre la
  garde heuristique v008, puis effectuer une époque d’imitation depuis les poids v002 avec Adam
  réinitialisé.
- [Résultat] Le lot contient `4 987` décisions, dont `986` divergences v002/v008. La candidate
  régresse contre Random (`-0,5` point), v007 (`-3,0`), v008 (`-3,0`) et v002 (`-3,5`) sur 200
  parties par adversaire ; elle progresse seulement contre v001 (`+9,0`). Le score pondéré est
  `-1,6` point et le benchmark passe de `9,609 s` à `10,077 s` sur 50 parties.
- [Supprimé] Ne pas promouvoir v012 ni reprendre cette recette DAgger à l’identique ; aucune
  modification du moteur, des heuristiques ou du masque d’information n’a été conservée.

## Suites possibles

- [À étudier] Analyser les divergences DAgger par catégorie et par phase avant tout nouvel
  entraînement ; le lot v008 a probablement surpondéré des décisions qui ne généralisent pas à la
  référence neural.
- [À étudier] Tester un mélange pondéré de données historiques et on-policy, avec validation
  séparée des catégories PLAY, deckbuilding et banish, en partant toujours de v002.
- [À étudier] Ne pas accepter une candidate DAgger qui améliore v001 ou v008 court mais régresse la
  référence v002 ou la garde v008 sur le panel complet.
