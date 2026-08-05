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

Les rapports sous `doc/Experiments/` servent à comprendre les essais précédents et à éviter les
répétitions inutiles ; ils n'empêchent pas l'agent d'inventer une nouvelle expérience ou de corriger
une piste existante.
