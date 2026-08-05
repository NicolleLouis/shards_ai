# Idées et priorités de travail

## Expérience exp-00004 — rejetée

- [Terminé] Reprendre PPO v003 avec reward shaping depuis v001, avec reprise explicite du checkpoint
  mutable canonique et validation complète contre v002, Random, v007 et v008.
- [Résultat] v003 améliore v008 de `+6,5` points et égale v007, mais régresse contre Random de `-3,0`
  points et contre v002 de `-5,0` points ; la promotion est rejetée.
- [Supprimé] Ne pas promouvoir ce checkpoint et ne pas conserver une cible v003 comme nouvelle référence.

## Suites possibles

- [À étudier] Tester une reprise plus conservatrice depuis v002, avec une durée d'entraînement suffisante
  et un contrôle explicite de la régression contre la référence neural.
- [À étudier] Analyser les décisions et les catégories d'actions qui expliquent le gain contre v008
  mais la perte contre v002 avant de relancer un PPO.

Les rapports sous `doc/Experiments/` servent à comprendre les essais précédents et à éviter les
répétitions inutiles ; ils n'empêchent pas l'agent d'inventer une nouvelle expérience ou de corriger
une piste existante.
