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

## Expérience exp-00025 — rejetée

- [Terminé] Tester une correction DAgger conservatrice : 2 562 décisions visitées par v002 contre
  v008, un seul passage d'imitation depuis v002, Adam réinitialisé et taux d'apprentissage `1e-5`
  au lieu de `1e-4` dans exp-00024.
- [Résultat] La candidate régresse contre Random de `-1,5` point, v007 de `-4,5` points et v008 de
  `-0,5` point sur 200 parties ; elle régresse aussi de `-4,0` points contre v002. Le score pondéré
  est `-0,65` point. Le benchmark passe de `9,276 s` à `9,841 s` sur 50 parties et le taux de gain
  neural passe de `52 %` à `46 %`.
- [Supprimé] Ne pas promouvoir v013 ni reprendre cette recette DAgger, même avec un taux
  d'apprentissage plus faible.

## Suites possibles

- [À étudier] Analyser les 986 désaccords v002/v008 d'exp-00024 et les désaccords d'exp-00025 par
  phase et catégorie avant de créer un nouveau dataset ; filtrer les décisions sans divergence et
  les divergences non stratégiques plutôt que d'imiter v008 uniformément.
- [À étudier] Tester un mélange explicite de données historiques et on-policy avec une petite
  pondération des labels v008, en gardant v002 et v008 comme gardes dures.
- [À étudier] Évaluer séparément PLAY, deckbuilding et banish avant tout nouvel entraînement ; ne
  pas utiliser l'amélioration contre v001 comme critère de promotion.

## Expérience exp-00026 — rejetée

- [Terminé] Tester un mélange déterministe de `6 000` décisions historiques et `2 562` décisions
  on-policy v002/v008, avec une époque d'imitation à `1e-5` depuis v002 et Adam réinitialisé.
- [Résultat] Sur 200 parties par adversaire contre la référence active v002, les deltas sont
  Random `-3,0` points, v007 `-2,5` points et v008 `+0,5` point ; le delta direct contre v002 est
  `-2,5` points. Le benchmark comparable passe de `9,276 s` à `9,894 s` sur 50 parties.
- [Supprimé] Ne pas promouvoir v014 ni conserver le checkpoint mutable comme référence active.

## Suites possibles

- [À étudier] Refaire l'analyse offline par phase et catégorie sur ce mélange, en isolant PLAY,
  deckbuilding et banish pour déterminer si la régression vient de la proportion historique ou des
  labels v008.
- [À étudier] Tester un mélange plus conservateur avec moins de décisions on-policy et filtrage des
  divergences stratégiques, uniquement après avoir mesuré ces catégories.
- [À étudier] Garder v002, Random et v008 comme gardes dures ; ne pas promouvoir une candidate qui
  améliore seulement la garde v008.

## Expérience exp-00027 — acceptée par le protocole déterministe

- [Terminé] Réentraîner une époque depuis v002 sur le dataset historique v008, avec une pondération
  modérée (`1,25`) des décisions `PLAY` et `BANISH`, sans ajouter de divergences DAgger on-policy.
- [Résultat] Le panel de 200 parties donne Random `-3,5` points, v007 `+2,0`, v008 `+2,5` et la
  référence v002 `-4,0`; le score pondéré atteint `+0,5` point, juste au seuil, et le benchmark passe
  de `9,3099 s` à `9,4056 s` (`+1,03 %`, sous le seuil de régression de 5 %).
- [Conservé] La candidate et son checkpoint mutable sont conservés comme résultat expérimental;
  aucun profil actif n'est promu automatiquement.

## Suites possibles

- [À étudier] Vérifier la stabilité de cette pondération ciblée sur un second seed avant promotion;
  le gain est au seuil et la régression contre v002 reste de `4,0` points.
- [À étudier] Comparer offline les catégories `PLAY`, `BANISH`, deckbuilding et recrutement entre
  v002 et v015 pour déterminer si le gain contre v007/v008 est concentré dans les décisions pondérées.
- [Supprimé] Ne pas augmenter davantage la pondération ciblée ni ajouter simultanément des données
  DAgger tant que cette attribution n'est pas isolée.

## Expérience exp-00028 — rejetée

- [Terminé] Reproduire sur une seconde seed la pondération ciblée `1,25` de `PLAY` et `BANISH` de
  l'exp-00027, depuis les poids v002, avec Adam réinitialisé et une époque sur 5 000 décisions.
- [Résultat] Le panel indépendant reproduit le compromis : Random `-3,5` points, v007 `+2,0`, v008
  `+2,5` et la référence v002 `-4,0`. Le benchmark comparable passe de `9,5435 s` à `9,6003 s`;
  la candidate est donc légèrement plus lente et ne corrige pas la régression neural.
- [Supprimé] Ne pas promouvoir cette recette ni considérer la répétition comme une validation stable
  de l'exp-00027; ne pas conserver de profil v016.

## Suites possibles

- [À étudier] Analyser les décisions v002/candidates par phase et catégorie pour déterminer pourquoi
  la pondération PLAY/BANISH améliore v007/v008 mais dégrade Random et v002.
- [À étudier] Tester une pondération plus sélective ou une contrainte de conservation des logits sur
  les catégories non ciblées, uniquement après cette analyse offline.
- [À supprimer] Les reprises PPO courtes, le DAgger uniforme et les mélanges historiques/on-policy
  sans filtrage restent écartés par les régressions documentées.

## Expérience exp-00029 — rejetée

- [Terminé] Tester depuis les poids v002 une époque d'imitation historique avec une pondération
  ciblée `1,10` des décisions `recruit_mercenary`, sans modifier le moteur, les heuristiques ou le
  masque d'information.
- [Résultat] Sur le panel apparié de 16 parties par adversaire, Random `-6,25` points, v007 `0,0`
  et v008 `+25,0`; la candidate régresse fortement contre la référence v002 malgré le gain v008.
  Le benchmark comparable passe de `9,304 s` à `25,326 s` sur 50 parties.
- [Supprimé] Ne pas promouvoir la candidate ni reprendre cette pondération; le checkpoint reste un
  artefact expérimental hors profil actif.

## Suites possibles

- [À étudier] Analyser offline les décisions de recrutement et la cause de l'allongement des parties
  avant toute nouvelle pondération d'action.
- [À privilégier] Tester une contrainte de conservation des logits ou une correction de représentation
  avec un protocole explicitement comparé à v002, plutôt que d'augmenter encore une loss ciblée.

## Expérience exp-00030 — rejetée

- [Terminé] Tester une imitation historique v008 depuis les poids v002 avec une régularisation de
  conservation des scores du v002 (`lambda=0,10`), afin de corriger les décisions enseignées sans
  déplacer excessivement la politique neural active.
- [Résultat] Le screening apparié de 16 parties par adversaire donne Random `0,0`, v007 `-6,25`
  points et v008 `0,0`; la candidate est rejetée. Le benchmark comparable sous la même limite de
  1 000 actions est `9,7685 s` pour v002 contre `9,7641 s` pour la candidate, sans gain de qualité.
- [Supprimé] Ne pas promouvoir la candidate ni reprendre cette régularisation à l'identique. Le
  panel officiel de 200 parties par adversaire a été interrompu car la candidate rendait le protocole
  standard trop long; ce résultat ne prétend pas être une acceptation issue d'un protocole court.

## Suites possibles

- [À étudier] Mesurer les logits v002 et v008 par phase et catégorie avant tout nouvel entraînement,
  notamment pour vérifier si l'ancrage doit être appliqué seulement aux actions non stratégiques.
- [À étudier] Tester une régularisation de divergence bornée avec un coefficient beaucoup plus faible,
  uniquement sur un dataset équilibré par phase, si l'analyse offline montre un déplacement localisé.
- [À supprimer] Les pondérations PLAY/BANISH, recrutement, DAgger uniforme et reprises PPO courtes
  restent écartées par les régressions déjà documentées.

## Expérience exp-00031 — rejetée

- [Terminé] Tester une imitation historique v008 depuis v002 avec une pondération de phase `buy` à
  `1,15`, afin de cibler le deckbuilding sans répéter les pondérations PLAY/BANISH ou recrutement.
- [Résultat] Le panel de 200 parties donne Random `-2,0` points, v007 `-1,5` point et v008
  `-1,5` point contre la référence active v002. Le benchmark neural comparable passe de `9,182 s`
  à `9,516 s` (+3,64 %), donc la candidate n'est pas retenue.
- [Supprimé] Ne pas conserver cette pondération de phase ni promouvoir la candidate.

## Suites possibles

- [À étudier] Comparer offline les logits et décisions par phase `buy`, `play` et `banish` avant une
  nouvelle pondération ; la pondération globale de `buy` ne suffit pas et dégrade les trois gardes.
- [À privilégier] Tester seulement après cette analyse une correction locale sur les états de
  deckbuilding réellement divergents, avec un dataset équilibré par phase et une contrainte de
  non-régression contre v002.
- [À supprimer] Les pondérations de phase globales sans analyse offline, les pondérations d'action
  isolées, l'ancrage global des logits et les reprises PPO courtes restent écartés.

## Expérience exp-00032 — rejetée

- [Terminé] Filtrer les décisions historiques v008 où v002 choisit une action différente, équilibrer
  les phases `attack`, `buy` et `play`, puis faire une époque d'imitation depuis v002 à `1e-5` avec
  Adam réinitialisé.
- [Résultat] Le filtre a trouvé `18 241` désaccords sur `100 000` décisions et retenu `1 980` états.
  Sur 200 parties par adversaire, Random `-6,0` points, v007 `-2,5`, v008 `+0,5` et v002 `+2,0` ;
  la candidate est rejetée malgré le gain v008/v002. Le benchmark comparable passe de `9,493 s` à
  `9,557 s` (+0,67 %).
- [Supprimé] Ne pas promouvoir `exp032` ni conserver ce checkpoint comme profil actif.

## Suites possibles

- [À étudier] Analyser les désaccords retenus par type d'action et par phase pour isoler ceux qui
  améliorent v002 sans sacrifier Random ; le filtrage seul est insuffisant.
- [À privilégier] Tester un sous-ensemble encore plus sélectif, limité aux désaccords stratégiques
  dont le score teacher dépasse clairement le score v002, avec validation complète identique.
- [À supprimer] Le filtrage uniforme des désaccords équilibrés par phase et toute promotion fondée
  sur le gain contre v008 seul.

## Expérience exp-00033 — rejetée

- [Terminé] Tester un filtre de désaccords plus sélectif que exp-00032 : conserver les états où v002
  choisit une autre action que v008 et où la marge heuristique de v008 sur l'action choisie par v002
  est au moins `1,0`, avec un plafond de `800` états par phase, puis entraîner une époque depuis v002.
- [Résultat] Le dataset contient `1 183` décisions (`attack=113`, `buy=270`, `play=800`). Le
  benchmark comparable passe de `9,5451 s` à `10,0903 s` sur 50 parties contre v002, avec un taux
  de victoire de `52 %` à `36 %`. Le screening donne Random `-12,5`, v007 `0,0`, v008 `+12,5` et
  v002 `-12,5` points ; la candidate est rejetée.
- [Supprimé] Ne pas promouvoir cette candidate ni reprendre ce seuil et ce plafond à l'identique.

## Suites possibles

- [À étudier] Analyser les états sélectionnés par carte et type d'action : le filtre conserve trop
  d'états `play` et déplace la politique malgré une marge teacher élevée.
- [À privilégier] Tester une sélection équilibrée par type d'action avec une marge élevée et une
  contrainte de conservation des décisions v002 sur les états non sélectionnés, après analyse offline.
- [À supprimer] Les filtres de désaccords fondés uniquement sur la marge heuristique et les panels
  courts comme preuve de promotion.

## Expérience exp-00034 — rejetée

- [Terminé] Tester une imitation historique v008 depuis v002 avec un ancrage local faible des scores
  centrés de v002 (`lambda=0,02`), afin de limiter les déplacements de politique observés dans les
  filtres de désaccords exp-00032/33 sans répéter l'ancrage global exp-00030.
- [Résultat] Le dataset contient `5 133` décisions v008 contre Random/v007. Sur 50 parties par
  adversaire contre la référence active v002, le delta est Random `0,0`, v007 `+8,0`, v008 `-4,0`
  et v002 `+6,0` points. Le score pondéré est `+0,2` point, sous le seuil, et le panel 200 parties
  n'a pas produit de rapport exploitable dans le budget ; aucune promotion n'est revendiquée.
- [Supprimé] Ne pas promouvoir cette candidate ni reprendre l'ancrage `lambda=0,02` à l'identique.

## Suites possibles

- [À étudier] Comparer offline les déplacements de logits par phase et type d'action pour déterminer
  pourquoi l'ancrage protège Random mais dégrade encore v008 et augmente le temps de partie.
- [À privilégier] Tester une régularisation de conservation uniquement sur les décisions non ciblées,
  avec un dataset train/validation séparé et un panel complet avant toute nouvelle variation.
- [À supprimer] Les mises à jour historiques uniformes, les filtres de désaccords sans contrainte de
  conservation et les pondérations globales de phase/action restent écartés par les régressions
  documentées.
