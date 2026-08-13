# Idées et priorités de travail

Ce fichier contient le catalogue courant des décisions et des prochaines expériences. Les détails,
les métriques et les commandes restent dans `doc/Experiments/`.

## Règles de décision

- Comparer chaque candidate à v007, v008 et aux profils neural v001 à v005 ; Random reste un benchmark
  diagnostique hors gate.
- Utiliser un panel complet et reproductible ; un screening court ne peut jamais accepter une candidate.
- Calculer la moyenne qualité sans Random avec les poids v007 `1,5`, v008 `2`, neural v001/v002 `0,5` chacun,
  v003/v004 `0,25` chacun et v005 `1` ; aucun adversaire, y compris v008, n'est une garde dure : seule la
  moyenne pondérée strictement positive décide.
- Séparer entraînement, validation et holdout par `game_id`, et conserver la provenance des seeds.
- Valider sur plusieurs seeds tirés avant l'évaluation ; chaque seed doit être rejouée avec exactement les mêmes parties pour la candidate et la référence.
- Pour un entraînement long multi-seed, ne pas réentraîner un contrôle expérimental à chaque seed :
  comparer la candidate au checkpoint stable du profil neural actif (`configs/neural_profiles/v005.pt`
  actuellement), qui est déjà la référence de la gate. Un contrôle réentraîné n'est justifié que si
  l'expérience modifie le dataset, le protocole d'entraînement ou une autre variable qui rend le
  checkpoint actif incomparable.
- Avant toute campagne longue, estimer le coût cumulé de toutes les seeds et variantes ; aucune
  commande ne doit engager une campagne dont la durée attendue dépasse 15 heures. Réduire d'abord
  le nombre de variantes, les epochs ou le volume de données, puis mesurer la candidate retenue sur
  plusieurs seeds avec le panel paired.
- Mesurer séparément qualité, temps de partie, nombre d'actions et coût d'inférence.
- Modifier une seule hypothèse structurante à la fois ; ne pas modifier le moteur, les heuristiques ou le masque d'information.
- Conserver v001 à v004 comme références historiques protégées ; le profil actif est v005.
- Ne pas reprendre une distillation locale ou une loss ranking-only sans preuve holdout indépendante et budget explicite de décisions modifiées.

## Diagnostic de situation

Les expériences `exp-00045` à `exp-00074` ont montré des régressions répétées malgré des gains isolés : les corrections locales, pondérations, filtrages, DAgGER et PPO testés n'ont pas généralisé au panel complet. Les diagnostics indiquent des faiblesses par catégories d'actions/cartes, des erreurs PLAY confiantes et une sensibilité à la cardinalité légale.

La campagne `exp-00075` à `exp-00099` a produit une seule promotion qualité : `exp-00084` a promu `v003` sur une validation de 200 parties par adversaire, malgré les régressions du screening court. `exp-00104` a ensuite promu `v004`, une V5 avec une préférence sémantique (`id_scale=0,5`, `semantic_scale=1,5`), sur la gate en parties. `v003` et `v002` restent les références historiques protégées.

## Nouvelle architecture sémantique — prérequis d'expérimentation

L'architecture `structured_semantic_v4` transmet au réseau la structure détaillée des cartes : types d'opérations, montants, cibles, seuils, contraintes, ordre des opérations, branches de maîtrise, capacités de champion et identifiants de passifs. Elle est incompatible avec les poids historiques et doit être réentraînée depuis zéro.

Toute expérience V4 doit utiliser un dataset hashé, un split `game_id` séparé, un holdout non utilisé pour sélectionner les poids, le checkpoint de travail canonique `artifacts/neural_training/checkpoint.pt`, puis une validation batchée longue. Un checkpoint de diagnostic, un top-1 offline ou un gain contre un seul adversaire ne justifie ni promotion ni conclusion de force.

Les changements précédemment rejetés peuvent être réessayés sur V4 uniquement comme nouvelles hypothèses, une à la fois, avec le même panel et les mêmes critères. Le cache d'embeddings de `exp-00068` et le cache d'encodages d'actions de `exp-00099` sont des optimisations indépendantes de la qualité.

### Qualification des expériences antérieures

| Expériences | Qualification |
|---|---|
| `exp-00023` à `exp-00026` | Résultats historiques ; variantes d'entraînement à revalider après imitation V4. |
| `exp-00027` à `exp-00067` | Diagnostics, garde-fous et échecs historiques ; conclusions de qualité à revalider sur V4. |
| `exp-00068` | Valide indépendamment de l'architecture : cache sans changement de sorties ni trajectoires. |
| `exp-00069` à `exp-00074` | Résultats historiques de training/PPO ; à revalider avec un protocole comparable. |
| `exp-00075` à `exp-00098` | Résultats historiques V4 ; `exp-00084` a promu `v003`, les autres résultats sont rejetés, bornés ou diagnostiques. |
| `exp-00099` | Valide indépendamment de l'architecture : optimisation d'inférence à trajectoire identique. |

## Priorité immédiate : changer d'espace de solution

### 1. À privilégier — diagnostic de dérive puis correction ciblée

Construire d'abord un holdout indépendant par partie, stratifié par phase, type d'action, carte et taille de l'ensemble légal, avec ECE, Brier, reliability bins, intervalles par slice et budget explicite de décisions modifiées par rapport au profil actif. Après attribution, tester au plus une correction bornée sur une slice `play_card`/carte ou `activate_champion`, avec conservation exacte de la politique hors slice.

Hypothèse falsifiable : la correction améliore la slice ciblée sans dépasser le budget de dérive et sans régresser v002, Random, v007 ou v008 sur le panel complet et plusieurs seeds.

### 2. À privilégier — reprendre V4 seulement avec une preuve complète

La recette `exp-00098` a atteint `20 000` décisions avec reprise atomique, mais son panel de 4 parties est insuffisant et régresse Random de `25` points. La rejouer inchangée sur un hôte permettant `100` à `200` parties par adversaire (`batch_games=20`), puis comparer l'argmax V4/v003 sur un holdout indépendant par phase/action/cardinalité avant toute variation.

Ne pas sélectionner une réduction de capacité, une pondération globale ou un budget de `1 000` décisions sur le runtime, le top-1 offline ou un gain isolé : `exp-00095` et `exp-00097` ont encore régressé v008.

### 3. À étudier — changer le type d'entraînement

Tester un PPO suffisamment long et profilé ou un entraînement hybride imitation + objectif de résultat, après une nouvelle architecture ou hypothèse dédiée. Séparer collecte, entraînement et validation ; instrumenter récompenses terminales, variance des avantages, normes de gradients, décisions modifiées et checkpoints atomiques.

### 4. À conserver comme contrôles et performance

- Le gel de la base v002/v003 et la comparaison intra-ensemble des probabilités sont des contrôles causaux utiles ; les logits bruts inter-architectures ne sont pas comparables.
- Les protocoles de collecte PPO conservent un checkpoint atomique externe après chaque rollout.
- `exp-00099` accepte le cache d'encodages d'actions : sur 50 parties V3 contre V3, seeds `0..49`, trois répétitions et un thread PyTorch, `8,0331 s` devient `6,9034 s` (`-14,06 %`), avec `8 684` décisions, `17 396` actions et trajectoires identiques. Reprofiler maintenant le pooling d'observation et l'encodage d'état, puis mesurer le taux de réutilisation par type d'action et matchup.

## Historique condensé

| Expérience | Statut | Conclusion opérationnelle |
|---|---|---|
| `exp-00023` à `exp-00038` | Rejetées | PPO/DAgGER, pondérations et ancrages : pas de généralisation robuste. |
| `exp-00039` à `exp-00049` | Diagnostics/rejetées | Faiblesses `recruit_mercenary`, BUY/PLAY et erreurs PLAY confiantes. |
| `exp-00050` à `exp-00056` | Rejetées/diagnostics | Variations de représentation et résidus : gains isolés, références régressées. |
| `exp-00057` à `exp-00067` | Échecs/rejetées/diagnostics | DAgGER/PPO et corrections PLAY sans preuve de force ; garde-fous conservés. |
| `exp-00068` | Performance acceptée | Cache d'embeddings : temps médian `-11,76 %`, sorties et trajectoires identiques. |
| `exp-00069` à `exp-00074` | Échecs/rejetées/diagnostics | PPO et imitation courte : aucune amélioration qualité robuste. |

## Résumé de la campagne `exp-00075` à `exp-00100`

| Expériences | Type | Décision | Résultat principal |
|---|---|---|---|
| `exp-00075` à `exp-00076`, `79` à `80`, `83`, `85`, `87` à `89`, `92` à `95`, `97` à `98` | Qualité | Rejetées ou échecs techniques | V4 régresse une garde ou reste limitée à un fallback/screening ; aucune promotion dans ces expériences. |
| `exp-00077`, `91`, `96` | Analyse | Terminées | La dérive active est localisée ; elle ne fournit pas de causalité de win-rate. |
| `exp-00078`, `81`, `82`, `90` | Qualité | Inconclusives/échecs | Budget complet ou checkpoint exploitable absent ; aucun signal de force retenu. |
| `exp-00084` | Qualité | Acceptée/promue | Le screening de 20 parties régressait v002/v001 de `25` points, mais la promotion paired de 200 parties par adversaire a donné une moyenne pondérée `+1` point et v008 `+3` points ; `v003` a été créé. |
| `exp-00104` | Qualité | Acceptée/promue | La V5 `id_scale=0,5`, `semantic_scale=1,5` a été promue en `v004` sur trois batches de 200 parties par adversaire, avec `+0,290 pp` de moyenne pondérée ; la seed longue `50100` a été rejetée. |
| `exp-00086` | Performance | Rejetée | Index de vocabulaires : `-0,96 %` sur 100 parties, trajectoires identiques ; patch retiré. |
| `exp-00099` | Performance | Acceptée | Cache d'actions : `-14,06 %` médian à workload et trajectoires identiques. |
| `exp-00100` | Qualité | Rejetée | Fine-tuning V008 depuis V003 sur `play_card`/5–8 actions : gain offline mais régression de `-10` points contre V008 et de `-12,41` points pondérés au screening. |
| `exp-00101` | Qualité | Rejetée partielle | L'ablation V5 sans identité régresse de `3,54` points top-1 offline sur deux seeds supplémentaires ; conserver V5 avec identité et passer aux scales. |

## Pistes à tester

- [À privilégier] Holdout `game_id` équilibré par phase/action/carte/cardinalité, avec budget d'argmax fixé avant tout entraînement, intervalles par slice et plusieurs seeds communes candidate/référence.
- [À privilégier] Correction bornée sur une seule slice `play_card`/carte ou `activate_champion`, avec conservation exacte de la politique hors slice et validation complète.
- [À privilégier] Rejouer sans changement la candidate `exp-00098` sur un hôte permettant le panel long, puis mesurer l'argmax V4/v003 sur holdout indépendant.
- [À privilégier] Augmenter la couverture visible de `recruit_mercenary`, `choose_pending_decision` et des cartes rares avant toute pondération ; aucune conclusion causale sous un seuil de couverture prédéclaré.
- [À étudier] Correction exacte par masque de décision ou conservation hors slice, avec budget de changements fixé avant entraînement et plusieurs seeds.
- [À privilégier] Ajouter une ancre de politique V003 à la loss avant toute nouvelle imitation V008 ; mesurer séparément la dérive d'argmax dans et hors de la slice ciblée.
- [Terminé] Étape B V5 : la candidate `id_scale=0,5, semantic_scale=1,5` a été promue en `v004` sur la gate en partie ; l'exploration `id_scale=1,5, semantic_scale=0,5` reste non promue.
- [À étudier] PPO avec KL non nul mais borné ou objectif hybride, seulement après preuve d'un signal non dégénéré et sélection sur holdout indépendant.
- [À étudier] Reprofiler pooling d'observation et encodage d'état avec le cache d'actions actif ; mesurer le taux de hit par action et matchup avant toute nouvelle optimisation.

## Pistes abandonnées

- [À supprimer] Promotions fondées sur un screening court, un fallback `1k`, le top-1 offline, un gain contre Random/v007 ou le runtime seul.
- [À supprimer] Pondérations globales, budgets de `1 000` décisions sélectionnés sur un gain isolé et nouvelles analyses PLAY-heavy sans macro-métriques, ECE et cardinalité légale.
- [À supprimer] L'optimisation des index de vocabulaires de `exp-00086`, mesurée plus lente de `0,96 %` sur le workload contrôlé.
- [À supprimer] Toute modification de qualité, dataset, objectif ou architecture motivée par `exp-00099`, qui est exclusivement une expérience de performance.
- [À supprimer] Relancer `exp-00100` avec le même fine-tuning ciblé sans ancre V003 ; augmenter le dataset, les epochs ou la pondération ne corrige pas la dérive des paramètres partagés.
