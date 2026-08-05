# Sélection monotone des états PPO v002

## Objective

Empêcher le trainer PPO de retenir un état qui améliore la moyenne grâce à Random tout en
régressant contre v008, l'adversaire cible. Le checkpoint mutable doit conserver le meilleur état
réellement acceptable par adversaire.

## Current State

`evaluate_greedy_model` joue un panel fixe contre Random, v007 et v008, mais son score est une
moyenne simple des trois taux de victoire. Le trainer remplace l'état retenu dès que cette moyenne
augmente. Ce critère peut masquer une régression contre v008 ou v007.

## Target Behavior

Chaque évaluation produit toujours un résultat détaillé par adversaire et un score pondéré avec les
poids du profil (`20 %` Random, `30 %` v007, `50 %` v008). Un candidat n'est retenu que si son taux
de victoire est au moins égal à celui du meilleur état retenu pour chacun des trois adversaires et
qu'il améliore strictement le score pondéré. Sinon, le meilleur état est restauré.

Le meilleur résultat détaillé est sauvegardé dans le checkpoint mutable afin qu'une reprise conserve
la même référence de sélection. Les anciens checkpoints qui ne contiennent pas ce résultat sont
compatibles : le trainer réévalue leur politique au démarrage.

## Non-Goals

- augmenter automatiquement le nombre de parties d'évaluation ;
- remplacer la validation finale, qui doit rester plus large que le panel périodique ;
- modifier la loss PPO, le reward terminal ou le mélange de collecte ;
- créer un deuxième fichier de checkpoint.

## Key Decisions

- **Pas de régression :** comparaison indépendante sur Random, v007 et v008 par rapport au meilleur
  état déjà retenu.
- **Départage :** le score pondéré par le profil favorise v008 sans permettre à une autre victoire de
  compenser une régression.
- **Égalité :** une égalité par adversaire n'est pas une amélioration ; le score pondéré doit donc
  progresser strictement pour accepter un état.
- **Reprise :** `best_evaluation` devient la source de vérité de la sélection, avec repli sur une
  évaluation immédiate pour les checkpoints produits avant cette évolution.

## Proposed Architecture

Ajouter une fonction pure de comparaison des évaluations. `evaluate_greedy_model` calcule le score
pondéré à partir des poids positifs du profil et retourne les résultats par adversaire. Le trainer
conserve `best_evaluation`, `best_evaluation_score`, ainsi que les états modèle/optimiseur en
mémoire. À chaque évaluation, il accepte uniquement un candidat monotone et strictement meilleur
selon le score pondéré ; sinon il recharge l'état retenu.

## Testing Strategy

- vérifier le score pondéré 20/30/50 ;
- rejeter une évaluation qui améliore Random mais baisse v008 ;
- accepter une évaluation sans régression qui améliore strictement v008 ou le score pondéré ;
- vérifier la compatibilité d'un checkpoint sans `best_evaluation` ;
- exécuter les tests RL, la suite complète et un smoke test du trainer.

## Files Expected To Change

- `shards_ai/ai/rl_training.py` ;
- `scripts/train_neural_rl.py` ;
- `tests/ai/test_rl_training.py` ;
- `doc/Current state/Neural player.md` ;
- `README.md`.
