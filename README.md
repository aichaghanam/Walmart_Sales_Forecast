# Prévision des Ventes Walmart 

> Modèle de Machine Learning pour prédire les ventes hebdomadaires par magasin et département, optimisé sur la métrique **WMAE** (Weighted Mean Absolute Error).

**Application déployée :** [sales-prediction-app-942r.onrender.com](https://sales-prediction-app-942r.onrender.com)

---

## Aperçu du projet

Ce projet utilise les données historiques de **45 magasins Walmart** (2010–2012) pour prédire les ventes hebdomadaires à l'aide de techniques avancées de feature engineering et d'un modèle **XGBoost**. L'objectif principal est de minimiser le WMAE, une métrique qui pénalise 5x plus fortement les erreurs sur les semaines de fêtes.

| Métrique | Résultat |
|----------|----------|
| **WMAE** | 1 576,86 $ |
| **MAE** | 1 516,88 $ |
| **RMSE** | 3 507,91 $ |
| **R²** | 0,9749 |

---

## Structure du projet

```
├── train.csv                   # Données d'entraînement (421 570 lignes)
├── stores.csv                  # Métadonnées des 45 magasins
├── features.csv                # Variables externes (CPI, MarkDowns, météo...)
├── Partie_1_Projet_Walmart.ipynb  # Prétraitement, EDA, Feature Engineering + Modélisation
├── results_xgb_sincos.csv      # Résultats sauvegardés du modèle XGBoost
└── README.md
```

---

## Pipeline complet

### Bloc A — Préparation des données

#### A1 · Importation & Fusion
- Chargement de `train.csv`, `stores.csv`, `features.csv`
- Fusion en un DataFrame unique de **421 570 lignes × 16 colonnes**
- Suppression de la colonne dupliquée `IsHoliday_y`

#### A2 · Analyse Exploratoire (EDA)

**Valeurs manquantes — MarkDowns (64–74% de NaN)**
- Les MarkDowns n'existent qu'à partir du 2011-11-11 : les NaN représentent l'*absence de promotion*
- Imputation par **0** (décision métier, pas statistique)

**Valeurs négatives**
- `MarkDown2` et `MarkDown3` : valeurs négatives isolées sur quelques magasins — probables erreurs de saisie → remplacement par 0 via `clip(lower=0)`
- `Weekly_Sales` : 1 285 valeurs négatives (0,30%) correspondant à des **retours clients** → suppression des lignes `Weekly_Sales <= 0`

**Analyse univariée**
- `Weekly_Sales` : distribution fortement asymétrique à droite → justifie la transformation log
- `Temperature` : distribution normale (~60–70°F)
- `Fuel_Price` : distribution bimodale (deux régimes de prix)
- `Size` : trois pics distincts correspondant aux types A, B, C
- `MarkDowns` : concentrés en 0 après imputation, asymétrie attendue

**Analyse bivariée**
- `Size` est la variable la plus corrélée avec `Weekly_Sales` (Pearson r = 0,24)
- `MarkDown1` × `MarkDown4` très corrélés (r = 0,84) — multicolinéarité surveillée
- Hiérarchie des types : **Type A > Type B > Type C** (ventes moyennes et médianes)
- Effet `IsHoliday` dilué sur l'ensemble : +6% en moyenne, mais concentré sur certains départements

#### A3 · Vérification qualité
- **0 doublon** détecté sur la clé `Store + Dept + Date`
- `Store` et `Dept` conservés en `int64` (identifiants, pas des valeurs continues — encodage one-hot contre-productif : 45 × 81 colonnes)

#### A4 · Analyse bivariée avancée
- Scatter plot `Weekly_Sales vs Size` par type de magasin
- `CPI` : deux groupes régionaux distincts, pas de relation linéaire directe
- `Unemployment` : aucune tendance sur 5–14%, impact indirect via localisation

#### A5 · Analyse temporelle

**Série complète sans trous** : 143 semaines consécutives du 2010-02-05 au 2012-10-26

**Autocorrélation (ACF)**

| Lag | r | Décision |
|-----|---|----------|
| 1 | 0,325 | ✅ Retenu — meilleur prédicteur immédiat |
| 2 | 0,212 | ✅ Retenu — signal significatif |
| 4 | 0,174 | ✅ Retenu — signal mensuel |
| 52 | 0,481 | ❌ Abandonné — perte 38% des données |

**Analyse des fêtes**

| Fête | Ventes moy/sem | IsHoliday | Décision |
|------|---------------|-----------|----------|
| Xmas_Week | 26 408 $ | **False** ❌ | Colonne créée |
| Thanksgiving | 22 223 $ | True ✅ | Signal capté |
| Black_Friday | 16 710 $ | **False** ❌ | Colonne créée |
| SuperBowl | 16 376 $ | True ✅ | Semaine ordinaire |
| LaborDay | 15 881 $ | True ✅ | Semaine ordinaire |
| NewYear | 14 535 $ | True ✅ | Chute post-Noël |

> Kaggle a encodé New Year (31 déc) comme fête alors que le vrai pic (24 déc) est invisible → 3 colonnes manuelles créées.

---

### Bloc B — Feature Engineering

```python
# Transformation de la cible
df["Weekly_Sales_Log"] = np.log(df["Weekly_Sales"])

# Variables temporelles
df["Year"]       = df["Date"].dt.year
df["Month"]      = df["Date"].dt.month
df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)

# Encodage cyclique (préserve la continuité semaine 52 → semaine 1)
df["Week_sin"] = np.sin(2 * np.pi * df["WeekOfYear"] / 52)
df["Week_cos"] = np.cos(2 * np.pi * df["WeekOfYear"] / 52)

# Features promotionnelles
df["Is_Promo"] = (df[markdown_cols].sum(axis=1) > 0).astype(int)

# Encodage du type de magasin
df = pd.get_dummies(df, columns=["Type"], dtype=int)

# Lags (par Store × Dept)
df["Lag_1"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(1)
df["Lag_2"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(2)
df["Lag_4"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(4)

# Rolling statistics (shift(1) pour éviter la fuite de données)
df["Rolling_Mean_4"] = df.groupby(["Store","Dept"])["Weekly_Sales"]\
                         .transform(lambda x: x.shift(1).rolling(4).mean())
df["Rolling_Std_4"]  = df.groupby(["Store","Dept"])["Weekly_Sales"]\
                         .transform(lambda x: x.shift(1).rolling(4).std())

# Colonnes fêtes manuelles
df["Thanksgiving"] = df["Date"].isin(pd.to_datetime(["2010-11-26","2011-11-25"])).astype(int)
df["Black_Friday"] = df["Date"].isin(pd.to_datetime(["2010-12-03","2011-12-02"])).astype(int)
df["Xmas_Week"]    = df["Date"].isin(pd.to_datetime(["2010-12-24","2011-12-23"])).astype(int)

# Target encoding (calculé sur le train uniquement pour éviter le data leakage)
store_mean = train_df.groupby("Store")["Weekly_Sales"].mean()
dept_mean  = train_df.groupby("Dept")["Weekly_Sales"].mean()
```

**Dimensions finales après feature engineering :** 407 117 lignes × 33 colonnes  
**Après suppression des NaN de lags :** 407 117 lignes × 33 colonnes

---

### Bloc C — Modélisation

#### Séparation temporelle Train / Test

```
Train : 2010-03-05 → 2011-12-30   (280 257 lignes)
Test  : 2012-01-06 → 2012-10-26   (126 860 lignes)
```

> Coupure temporelle stricte à 2012-01-01 — aucune fuite de données future.

#### Features finales (30 colonnes)

```
IsHoliday, Size, Temperature, Fuel_Price,
MarkDown1–5, CPI, Unemployment,
Year, Month, WeekOfYear, Week_sin, Week_cos,
Is_Promo, Type_A, Type_B, Type_C,
Lag_1, Lag_2, Lag_4, Rolling_Mean_4, Rolling_Std_4,
Thanksgiving, Black_Friday, Xmas_Week,
Store_enc, Dept_enc
```

#### Modèle XGBoost

```python
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# Prédiction en espace log → retransformation exponentielle
y_pred_log = model.predict(X_test)
y_pred     = np.exp(y_pred_log)
```

#### Calcul du WMAE

```python
weights = np.where(test_df["IsHoliday"] == 1, 5, 1)
wmae = np.sum(weights * np.abs(y_test_real - y_pred)) / np.sum(weights)
```

Les semaines de fêtes (`IsHoliday = True`) ont un **poids ×5** dans la métrique finale.

---

## Résultats

```
WMAE   : 1 576,86 $
MAE    : 1 516,88 $
RMSE   : 3 507,91 $
R²     : 0,9749
WMAE/RMSE : 0,4495
```

**Analyse de la courbe d'apprentissage**
- La courbe de validation converge et se stabilise à partir de ~100 000 lignes
- Aucun overfitting sévère : les deux courbes convergent sans se rejoindre
- La limite de performance provient du volume de données, pas de l'architecture du modèle

---

## Décisions clés

| Décision | Justification |
|----------|---------------|
| Imputation MarkDowns → 0 | Absence de promotion, pas une erreur |
| Suppression Weekly_Sales ≤ 0 | Retours clients — ne pas modéliser |
| Transformation log de la cible | Corrige l'asymétrie, stabilise la variance |
| Encodage cyclique semaine | Préserve la continuité temporelle (semaine 52 → 1) |
| Lags 1, 2, 4 (pas 52) | Lag 52 cause une perte de 38% des données |
| Target encoding Store/Dept | Évite l'explosion de dimensions (45×81 colonnes) |
| 3 colonnes fêtes manuelles | IsHoliday de Kaggle rate Noël et Black Friday |
| Cutoff temporel strict | Simule un vrai déploiement en production |

---

## Installation & Utilisation

```bash
# Cloner le dépôt
git clone <url-du-repo>
cd walmart-sales-forecast

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate

# Installer les dépendances
pip install pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels

# Lancer le notebook
jupyter notebook Partie_1_Projet_Walmart.ipynb
```

### Application web déployée

L'application est accessible directement sans installation : [sales-prediction-app-942r.onrender.com](https://sales-prediction-app-942r.onrender.com)

---

## Données sources

Les données proviennent du challenge Kaggle [Walmart Recruiting — Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting).

| Fichier | Lignes | Description |
|---------|--------|-------------|
| `train.csv` | 421 570 | Ventes hebdomadaires par Store × Dept |
| `stores.csv` | 45 | Type (A/B/C) et taille des magasins |
| `features.csv` | 8 190 | MarkDowns, CPI, Unemployment, Température, Prix carburant |

---

## Technologies utilisées

- **Python 3.x**
- **pandas** — manipulation des données
- **NumPy** — calculs numériques
- **Matplotlib / Seaborn** — visualisations
- **statsmodels** — analyse ACF
- **scikit-learn** — métriques et learning curves
- **XGBoost** — modèle de prédiction



Projet réalisé dans le cadre d'un cours de Machine Learning appliqué à la prévision des ventes retail.
