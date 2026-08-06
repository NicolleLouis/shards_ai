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

## Expérience exp-00035 — rejetée

- [Terminé] Tester une époque d'imitation historique uniforme très conservatrice depuis v002, avec
  Adam réinitialisé, un taux d'apprentissage de `1e-5` et au plus `10 000` décisions, afin d'isoler
  l'effet de l'amplitude de mise à jour sans nouvelle pondération d'action ni filtrage des données.
- [Résultat] Le dataset v008 contient `100 000` décisions. Sur le screening contre v002, la candidate
  régresse de `-10,0` points contre Random, de `-40,0` points contre v007 et reste à `0,0` contre v008
  (10 parties par adversaire). Le benchmark comparable contre v002 passe de `9,475 s` à `10,355 s`
  (`+9,29 %`) et le taux de victoire de `52 %` à `46 %`.
- [Supprimé] Ne pas promouvoir v035 ni reprendre cette mise à jour uniforme à faible amplitude ;
  elle ne corrige pas les régressions observées et augmente la durée des parties.

## Suites possibles

- [À étudier] Mesurer offline les déplacements de logits par phase et type d'action avant tout nouvel
  entraînement, avec un vrai split train/validation et un panel complet de 200 parties.
- [À privilégier] Tester une régularisation de conservation seulement sur les décisions non ciblées,
  ou une architecture/représentation candidate depuis v002, mais uniquement avec une attribution
  distincte des pondérations et filtres déjà rejetés.
- [À supprimer] Les mises à jour historiques uniformes, même à faible taux d'apprentissage, les
  filtres de désaccords sans contrainte de conservation et les pondérations globales restent écartés.

## Expérience exp-00036 — acceptée par le panel officiel

- [Terminé] Tester une imitation sélective depuis v002 sur un dataset déterministe équilibré par
  catégorie : `300` décisions `buy_card`, `300` `recruit_mercenary`, `300` `banish_card` et `300`
  `play_card`, toutes avec un label v008 de rang 1 et une marge heuristique d'au moins `1,0`.
  Les divergences faibles et les décisions `attack` sans marge suffisante ont été exclues.
- [Résultat] Le panel officiel de `200` parties par adversaire donne Random `-1,5` point, v007
  `-3,5`, v008 `+2,0` et v002 `-2,5`. Le score pondéré est `+1,05` point et le validateur accepte
  la candidate ; le benchmark comparable passe toutefois de `9,24685 s` à `9,68588 s`
  (`-4,53 %` de débit), proche de la limite de performance.
- [Conservé] Le checkpoint expérimental et le profil `exp036` sont conservés pour la gate finale ;
  aucun profil stable v003 n'est créé et v002 reste la référence active.

## Suites possibles

- [À étudier] Vérifier la stabilité du dataset équilibré sur un second seed avant toute promotion,
  en surveillant particulièrement la régression v002/v007 et le coût de partie.
- [À privilégier] Si la gate finale rejette la candidate, réduire la taille du fine-tuning ou
  ajouter une sélection par holdout sans reprendre les filtres uniformes exp-032/033.
- [À supprimer] Les datasets déséquilibrés dominés par `play`, les mises à jour uniformes et les
  reprises PPO courtes restent écartés par l'historique.

## Expérience exp-00037 — rejetée

- [Terminé] Réduire de moitié le fine-tuning sélectif équilibré d'exp-00036 : `600` décisions,
  `150` par catégorie stratégique, depuis v002, une époque, Adam réinitialisé et `1e-5`.
- [Résultat] Le panel de 200 parties donne Random `+0,0`, v007 `-2,5` et v008 `+3,0` points ;
  le benchmark passe de `9,278599 s` à `9,757413 s` sur 50 parties (`+5,16 %`). La gate qualité
  interne est positive, mais la gate de performance rejette la candidate.
- [Supprimé] Ne pas promouvoir exp-00037 ni considérer la réduction de taille seule comme une
  correction suffisante d'exp-00036.

## Suites possibles

- [À étudier] Mesurer offline les déplacements de logits et les décisions modifiées par catégorie,
  avec un holdout séparé, avant tout nouvel entraînement.
- [À privilégier] Tester une représentation ou une contrainte locale qui protège le débit et v002,
  plutôt qu'une nouvelle pondération ou un simple changement de taille.

## Expérience exp-00038 — rejetée

- [Terminé] Tester un DAgger sélectif depuis v002 : collecter les états visités contre v007/v008,
  ne conserver que la première divergence de v002 qui rend la phase `PLAY` stratégiquement différente
  de v008, puis entraîner une époque à `1e-5` depuis v002 avec Adam réinitialisé.
- [Résultat] Sur `3 170` décisions collectées, seulement `18` premières divergences stratégiques ont
  été retenues, toutes `play_card`. Le panel de screening donne Random `-6,25` points, v007 `0,0`,
  v008 `+12,5` et v002 `0,0`; le benchmark passe de `9,2801 s` à `9,4403 s` (+1,73 %).
- [Supprimé] Ne pas promouvoir exp-00038 ni reprendre le filtre « première divergence » seul : il
  est trop rare et ne protège pas Random malgré un gain sur v008.

## Suites possibles

- [À étudier] Conserver l'idée de divergence stratégique, mais construire un vrai split train/holdout
  et inclure plusieurs types d'action au lieu de laisser `play_card` monopoliser les exemples.
- [À privilégier] Tester une correction DAgger pondérée qui mélange une petite base historique v008
  avec les divergences stratégiques, en imposant une garde Random/v002 avant toute promotion.
- [À supprimer] Les lots composés de quelques divergences `play_card` seulement et toute conclusion
  fondée sur le gain v008 sans gain contre une référence neural.

## Expérience exp-00039 — analyse terminée

- [Terminé] Analyser v002 sans entraînement : loss, accuracy, couverture/imbalance, confiance,
  cartes, comparaison v001 et panels visibles v008/v007.
- [Résultat] Sur 12 114 décisions visibles, l’accuracy est `85,03 %`; BUY est à `77,04 %`,
  `recruit_mercenary` à `25,64 %`, et PLAY représente `73,36 %` des labels. Les erreurs PLAY
  peuvent rester très confiantes; le panel 20 parties donne v002 à `15 %` contre v008 et `50 %`
  contre v007.
- [Conservé] Le diagnostic et ses recommandations sont dans `doc/Experiments/exp-00039.md`;
  aucun checkpoint n’a été modifié.

## Suites issues de l’analyse exp-00039

- [À privilégier] Construire un holdout par partie stratifié par phase/action avant tout nouvel
  entraînement, puis traiter séparément `recruit_mercenary` et les choix BUY.
- [À étudier] Mesurer la calibration des probabilités intra-état et les erreurs PLAY à confiance
  élevée, notamment `crystal`, `ermite_fongique` et les champions.
- [À supprimer] Toute pondération globale ou promotion fondée sur les seuls labels v008 sans
  protection explicite de v002, Random et la garde v008.

## Pistes générales à explorer

## Expérience exp-00040 — rejetée

- [Terminé] Tester un fine-tuning sélectif depuis v002 sur `12 105` décisions de `82` parties,
  avec split déterministe par `game_id` (`10 318` train, `829` holdout), et pondération `1,25` des
  actions `buy_card`, `recruit_mercenary` et `banish_card`. La candidate a été initialisée par les
  poids v002 avec Adam réinitialisé; le moteur, les heuristiques et le masque sont restés inchangés.
- [Résultat] Le holdout atteint `89,14 %` top-1 et `92,63 %` pairwise, mais le screening apparié de
  `20` parties donne Random `+10,0` points, v007 `-15,0`, v008 `0,0` et v002 `+5,0` (référence
  neural v002). Le benchmark comparable de `50` parties contre v002 passe de `9,7297 s` à
  `11,0499 s` (`-13,57 %` de débit), avec `17 214` contre `17 240` actions.
- [Supprimé] Ne pas promouvoir la candidate ni reprendre la pondération sélective sans corriger la
  faiblesse v007 et le ralentissement. Les panels demandés de `100` et `200` parties n'ont pas
  produit de sortie exploitable dans la fenêtre de l'orchestrateur; aucun statut accepté n'en est
  déduit.

## Suites issues d'exp-00040

- [À privilégier] Construire le prochain dataset avec un holdout par partie effectivement séparé et
  un filtrage par marge teacher vérifié sur chaque alternative légale, puis ablater la pondération
  des actions rares au lieu de la conserver implicitement.
- [À étudier] Comparer les erreurs holdout de `v007` et Random par action avant tout entraînement,
  en particulier les décisions `play_card` qui peuvent expliquer le compromis observé.
- [À supprimer] Toute promotion fondée sur la seule accuracy offline, le screening de 20 parties,
  ou un gain contre v008 sans non-régression simultanée contre v007 et la référence v002.

Ces pistes sont volontairement larges. Elles ne constituent pas un plan imposé : l'agent peut les
décomposer, les combiner avec prudence, les reformuler après une analyse offline ou les supprimer
si les rapports montrent qu'elles n'ont plus de sens. Chaque expérience doit toutefois isoler une
hypothèse et comparer la candidate à la dernière version neural active.

### Recherche et décision

- [À étudier] Ajouter une recherche Monte Carlo ou Monte Carlo Tree Search limitée par un budget de
  temps, avec le réseau comme politique/prior et éventuellement comme estimateur de valeur.
- [À étudier] Comparer une décision gloutonne, une moyenne de rollouts, une recherche par action et
  une recherche limitée aux états visités, sans exposer à l'IA des informations masquées.
- [À étudier] Tester une politique hybride : réseau pour sélectionner les actions candidates puis
  recherche Monte Carlo ou heuristique uniquement comme outil de décision autorisé.
- [À étudier] Mesurer séparément le coût et le gain de la recherche afin de distinguer une
  amélioration de qualité d'une simple dégradation du temps de partie.

### Objectif d'apprentissage et pertes

- [À étudier] Remplacer ou compléter la cross-entropy d'imitation par une loss de classement
  pairwise/margin entre l'action du teacher et les alternatives légales.
- [À étudier] Tester une loss focalisée ou pondérée par la difficulté, la rareté de l'action, la
  phase et le désaccord entre le réseau et le teacher.
- [À étudier] Ajouter une distillation KL vers v008, v002 ou un ensemble de teachers, avec une
  pondération contrôlée et une comparaison séparée des labels et des logits.
- [À étudier] Tester une contrainte de conservation de la politique v002, locale aux états non
  ciblés, plutôt qu'une régularisation uniforme de tout le dataset.
- [À étudier] Comparer imitation pure, actor-critic, PPO, A2C et une loss hybride imitation plus
  renforcement, en conservant un protocole d'évaluation commun.

### Variantes PPO et renforcement

- [À étudier] Tester une variante PPO réellement distincte : clipping, coefficient de valeur,
  entropie, GAE, horizon, mini-batches, nombre d'epochs ou stratégie de sélection du meilleur état.
- [À étudier] Tester PPO avec un réseau valeur séparé, une tête politique/valeur partagée ou une
  initialisation de la valeur par une estimation Monte Carlo.
- [À étudier] Comparer PPO depuis v002 avec Adam réinitialisé à une continuation contrôlée avec
  état d'optimiseur, en documentant l'effet propre de l'optimizer.
- [À étudier] Tester une politique d'exploration différente : température, epsilon, bruit de logits,
  entropie adaptative ou exploration ciblée des actions rarement choisies.
- [À étudier] Tester des retours Monte Carlo complets, n-step ou TD(lambda) au lieu de la seule
  combinaison actuelle des retours PPO.

### Reward shaping et objectif de jeu

- [À étudier] Comparer plusieurs shaping potentiels : maîtrise, dégâts, shards, champions, unités,
  deckbuilding et contrôle de phase, avec une seule composante ajoutée par expérience.
- [À étudier] Tester un shaping potentiel invariant par état, borné et nul à l'état terminal, afin
  d'éviter de modifier artificiellement la préférence entre trajectoires gagnantes.
- [À étudier] Tester un shaping distinct par phase (`attack`, `buy`, `play`, `banish`) ou par type
  d'action, avec ablation de chaque composante.
- [À étudier] Comparer récompense terminale seule, récompense terminale plus shaping dense, et
  shaping utilisé uniquement pour l'avantage ou la sélection des trajectoires.
- [À étudier] Mesurer si le shaping améliore la qualité réelle ou seulement des métriques offline
  corrélées aux choix de v008.

### Architecture et dimensions

- [À étudier] Varier la largeur et la profondeur du tronc et des têtes d'action, avec une petite,
  moyenne et grande configuration évaluées sous le même budget d'entraînement.
- [À étudier] Comparer les réseaux indépendants actuels à un encodeur partagé avec têtes spécialisées
  par phase ou par type d'action.
- [À étudier] Tester des embeddings de cartes plus larges ou plus compacts, des embeddings séparés
  par zone et une projection commune des cartes similaires.
- [À étudier] Tester une architecture avec attention sur la main, la rivière, les champions et les
  unités visibles, sans ajouter d'information inaccessible au joueur.
- [À étudier] Comparer une architecture MLP, une architecture résiduelle légère, une architecture
  avec pooling invariant et éventuellement une architecture récurrente pour l'historique visible.
- [À étudier] Tester une tête de valeur distincte, des sorties factorisées par action et une
  représentation des actions légales adaptée à leur cardinalité variable.
- [À étudier] Mesurer séparément la qualité, la mémoire et le temps d'inférence de chaque dimension
  ou architecture ; une architecture plus grande ne doit pas être présumée meilleure.

### Données et imitation

- [À étudier] Construire des datasets train/validation/holdout distincts par partie et par seed,
  plutôt qu'un simple split de décisions individuelles.
- [À étudier] Comparer imitation v008, imitation v002, mélange de teachers et labels issus d'un
  vote entre teachers, en conservant la provenance de chaque décision.
- [À étudier] Tester DAgGER avec plusieurs itérations courtes, un budget de divergences contrôlé et
  un mélange explicite avec le dataset historique.
- [À étudier] Réimpliquer le réseau sur des datasets distincts par phase, type d'action, difficulté
  ou désaccord, avec une validation holdout correspondante.
- [À étudier] Tester le dédoublonnage, l'équilibrage, le curriculum et le filtrage par confiance du
  teacher, sans laisser une seule phase ou action dominer le dataset.

### Représentation et inférence

- [À étudier] Tester de nouvelles représentations des ressources, de l'état des cartes, des effets
  persistants et des actions légales tout en vérifiant explicitement le masque d'information.
- [À étudier] Comparer normalisation, encodage ordinal, one-hot, pooling invariant et features
  relatives au joueur, avec une ablation par groupe de features.
- [À étudier] Tester calibration des logits, température, seuil de confiance et fallback interne
  lorsque le réseau est incertain, sans consulter l'état caché.
- [À étudier] Explorer des optimisations d'inférence indépendantes de la qualité : batch de
  décisions, compilation, réduction des allocations, cache sûr et nombre de threads contrôlé.
