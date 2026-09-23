# Rétroaction DRT et Choix Discret (DMC) - Méthode MSA

Ce dossier contient la sauvegarde de référence de tous les fichiers modifiés et ajoutés dans le sous-module `eqasim-java` ainsi que la documentation complète des modifications apportées pour implémenter la boucle de rétroaction **MSA (Method of Successive Averages)** dans le modèle de choix de mode discret (**Discrete Mode Choice**).

---

## 1. Contexte et Diagnostic du Problème Initial

Lors des simulations de sensibilité sur la flotte de transport à la demande (DRT), des résultats contre-intuitifs apparaissaient :
- La part modale du DRT dans `modestats.csv` restait élevée (ex. 6,9 %) même en cas de sous-dimensionnement extrême de la flotte.
- L'analyse détaillée de `output_trips.csv.zst` a révélé que la part modale réellement servie n'était en réalité que de 0,39 % : plus de 91 % des demandes DRT étaient rejetées (`PassengerRequest rejected`).
- Les agents rejetés subissaient un `stuckAndAbort` dans MATSim, ce qui faisait disparaître leurs trajets du réseau sans qu'aucun mécanisme de replanning ou de report modal ne soit déclenché vers la marche, le vélo, les TC ou la voiture.
- Dans le modèle de choix discret Eqasim, les paramètres marginaux du DRT (`betaTravelTime_u_min` et `betaWaitingTime_u_min`) étaient initialisés à `0.0`, et aucune pénalité de rejet n'existait. Le temps d'attente prédit était figé à une valeur théorique statique.

---

## 2. Architecture de la Solution Implémentée

### 2.1. Nouvelle fonction d'utilité systématique du DRT
L'utilité du mode DRT dans le modèle Multinomial Logit de DMC est désormais formulée ainsi :

```text
V_DRT = alpha_DRT + beta_tt * t_tt + beta_wait * t_wait + beta_cost * c + beta_rej * P_rej
```

Où :
- $\alpha_{\text{DRT}} = -0.5$ : Constante spécifique du mode (ASC).
- $\beta_{\text{tt}} = -0.06\text{ util/min}$ ($-3.6\text{ util/h}$) : Désutilité marginale du temps à bord (alignée sur la voiture).
- $\beta_{\text{wait}} = -0.09\text{ util/min}$ ($-5.4\text{ util/h}$) : Désutilité du temps d'attente (pénalisée 1,5× plus que le temps à bord).
- $\beta_{\text{cost}} = -0.13\text{ util/€}$ : Sensibilité au coût monétaire.
- $\beta_{\text{rej}} = -5.0\text{ util}$ : Pénalité directe liée au risque de rejet.

> **Équivalence temps de trajet de $\beta_{\text{rej}}$ :**  
> Une pénalité de $-5.0$ pour un rejet certain ($P_{\text{rej}} = 1.0$) équivaut à un allongement perçu du temps de trajet de :  
> $$\Delta t_{\text{tt}} = \frac{-5.0}{-0.06} \approx 83.3\text{ minutes}$$  
> Si $P_{\text{rej}} = 50\%$, la pénalité équivaut à $\approx 41.7\text{ minutes}$ supplémentaires. Dès que la flotte sature, la chute drastique de $V_{\text{DRT}}$ force les agents à se reporter sur les modes alternatifs (TC, marche, vélo, voiture).

---

### 2.2. Mécanisme de lissage par Méthode des Moyennes Successives (MSA)
Pour éviter les phénomènes de sur-réaction, de résonance ou d'oscillations en cycles limites autour du seuil de capacité de la flotte, les observations de chaque itération sont lissées avec un pas décroissant conforme aux conditions de convergence de Robbins-Monro ($\sum \gamma_n = \infty$, $\sum \gamma_n^2 < \infty$) :

$$\gamma_n = \frac{1}{n^{0.75}}$$

À chaque itération $n$ (calculé lors de l'événement `notifyAfterMobsim`) :
$$P_{\text{rej}}^{(n)} = (1 - \gamma_n) \cdot P_{\text{rej}}^{(n-1)} + \gamma_n \cdot p_{\text{obs}}^{(n)}$$
$$t_{\text{wait}}^{(n)} = (1 - \gamma_n) \cdot t_{\text{wait}}^{(n-1)} + \gamma_n \cdot \bar{t}_{\text{wait, obs}}^{(n)}$$

Ce lissage garantit une adaptation rapide lors des premières itérations tout en assurant une convergence rigoureuse vers l'équilibre offre/demande au fil des itérations.

---

## 3. Correspondance et Rôle des Fichiers Modifiés

| Fichier dans `src/java/` | Emplacement d'origine dans `eqasim-java` | Modifications apportées |
|---|---|---|
| [`DrtFeedbackService.java`](file:///home/mmastio/popsynth/src/java/DrtFeedbackService.java) | `core/src/main/java/org/eqasim/core/simulation/modes/drt/mode_choice/predictors/DrtFeedbackService.java` | **Nouveau service singleton**. Implémente `BasicEventHandler`, `AfterMobsimListener`, `StartupListener`. Collecte les requêtes servies/rejetées et les temps d'attente réels via les événements MATSim. Calcule $P_{\text{rej}}$ et $t_{\text{wait}}$ via l'algorithme MSA à chaque fin de mobsim. |
| [`DrtVariables.java`](file:///home/mmastio/popsynth/src/java/DrtVariables.java) | `core/src/main/java/org/eqasim/core/simulation/modes/drt/mode_choice/variables/DrtVariables.java` | Ajout du champ public final `rejectionProbability`, avec surcharge des constructeurs et getter `getRejectionProbability()`. |
| [`DefaultDrtPredictor.java`](file:///home/mmastio/popsynth/src/java/DefaultDrtPredictor.java) | `core/src/main/java/org/eqasim/core/simulation/modes/drt/mode_choice/predictors/DefaultDrtPredictor.java` | Injection de `DrtFeedbackService`. Remplace le temps d'attente fixe théorique par le temps lissé réel et transmet $P_{\text{rej}}$ au conteneur `DrtVariables`. |
| [`DrtUtilityEstimator.java`](file:///home/mmastio/popsynth/src/java/DrtUtilityEstimator.java) | `core/src/main/java/org/eqasim/core/simulation/modes/drt/mode_choice/utilities/estimators/DrtUtilityEstimator.java` | Intégration du terme de pénalité de rejet : `utility += parameters.betaRejection_u * variables.rejectionProbability`. |
| [`ModeParameters.java`](file:///home/mmastio/popsynth/src/java/ModeParameters.java) | `core/src/main/java/org/eqasim/core/simulation/mode_choice/parameters/ModeParameters.java` | Ajout du champ `public double betaRejection_u = 0.0` dans la structure interne `DrtParameters`. |
| [`EqasimModeChoiceModule.java`](file:///home/mmastio/popsynth/src/java/EqasimModeChoiceModule.java) | `core/src/main/java/org/eqasim/core/simulation/mode_choice/EqasimModeChoiceModule.java` | Déclaration et liaison Guice du singleton `DrtFeedbackService`, enregistrement en tant que listener MATSim (`addControlerListenerBinding`, `addEventHandlerBinding`), et injection dans le provider de `DefaultDrtPredictor`. |
| [`IDFModeParameters.java`](file:///home/mmastio/popsynth/src/java/IDFModeParameters.java) | `ile_de_france/src/main/java/org/eqasim/ile_de_france/mode_choice/parameters/IDFModeParameters.java` | Initialisation des coefficients par défaut pour le DRT dans `buildDefault()` : `betaTravelTime_u_min = -0.06`, `betaWaitingTime_u_min = -0.09`, `betaRejection_u = -5.0`. |
| [`IDFModeChoiceModule.java`](file:///home/mmastio/popsynth/src/java/IDFModeChoiceModule.java) | `ile_de_france/src/main/java/org/eqasim/ile_de_france/mode_choice/IDFModeChoiceModule.java` | Configuration du pipeline DMC spécifique au scénario Île-de-France/Toulouse. |
| [`FlatDrtCostModel.java`](file:///home/mmastio/popsynth/src/java/FlatDrtCostModel.java) | `ile_de_france/src/main/java/org/eqasim/ile_de_france/mode_choice/costs/FlatDrtCostModel.java` | Implémentation d'un modèle de coût monétaire DRT configurable. |
| [`pom.xml`](file:///home/mmastio/popsynth/src/java/pom.xml) | `pom.xml` | Configuration racine Maven (compilation Java 25, dépendances MATSim et Eqasim). |

---

## 4. Modifications Côté Python (`src/add_drt.py`)

Le script de préparation du scénario [`src/add_drt.py`](file:///home/mmastio/popsynth/src/add_drt.py) génère désormais automatiquement dans chaque dossier d'exécution le fichier de configuration `mode_parameters.yml` :

```yaml
drt.alpha_u: -0.5
drt.betaTravelTime_u_min: -0.06
drt.betaWaitingTime_u_min: -0.09
drt.betaRejection_u: -5.0
```

Des arguments en ligne de commande ont été ajoutés pour permettre la variation paramétrique lors des plans d'expérience :
- `--beta-travel-time` (défaut : `-0.06`)
- `--beta-waiting-time` (défaut : `-0.09`)
- `--beta-rejection` (défaut : `-5.0`)
- `--drt-constant` (défaut : `-0.5`)

---

## 5. Procédure de Compilation

Pour reconstruire le JAR autonome intégrant ces modifications :

```bash
cd /home/mmastio/popsynth/eqasim-java
mvn clean package -pl ile_de_france -am -P standalone -DskipTests
```

L'artefact produit se situe dans :
`/home/mmastio/popsynth/eqasim-java/ile_de_france/target/ile_de_france-2.2.0.jar`
