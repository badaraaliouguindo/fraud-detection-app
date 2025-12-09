# Fraud Detection App

## Description

Ce projet est une **application Streamlit** pour la **détection de fraudes bancaires**.  
Elle utilise un **modèle de Machine Learning Random Forest** entraîné sur le dataset **Credit Card Fraud Detection** de Kaggle.

L'application permet à l'utilisateur d'entrer les caractéristiques d'une transaction et d'obtenir :

- Une prédiction : transaction normale ou fraude
- La **probabilité** de fraude associée
- Un graphique indiquant visuellement le risque

---

## Création du modèle (détaillée)

### Dataset

- Dataset : [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Nombre de transactions : 284 807
- Nombre de fraudes : 492 (~0,17%) → très déséquilibré

---

### Prétraitement

1. Séparation des features (`Time`, `V1`…`V28`, `Amount`) et de la target (`Class`)
2. Standardisation des variables avec **StandardScaler**
3. Gestion du déséquilibre via **SMOTE** (Synthetic Minority Over-sampling Technique) pour générer artificiellement des exemples de fraudes dans l’ensemble d’entraînement
4. Split stratifié **train/test** pour conserver la proportion de fraudes dans le test

---

### Modèles testés

#### 1️) Régression Logistique (baseline)

- **Objectif :** avoir une première estimation et comprendre le comportement du dataset déséquilibré
- **Résultats :**
    - Precision fraude : 0.06
    - Recall fraude : 0.92
    - F1-score fraude : 0.11
    - Accuracy : 0.97

**Observation :** le modèle détectait la plupart des fraudes (high recall) mais produisait énormément de faux positifs (très faible précision).

#### 2️) Random Forest (modèle final)

- **Objectif :** améliorer la précision et l’équilibre global entre recall et precision
- **Hyperparamètres principaux :**
    - `n_estimators = 300`
    - `class_weight = "balanced_subsample"`
    - `random_state = 42`
    - `n_jobs = -1`

- **Résultats :**
    - Precision fraude : 0.95
    - Recall fraude : 0.77
    - F1-score fraude : 0.85
    - ROC-AUC : 0.95

**Observation :** le modèle final détecte efficacement les fraudes tout en limitant fortement les faux positifs, offrant un excellent compromis pour un usage réel.

---

## Fonctionnalités de l’application

- Modèle Random Forest pré-entraîné (`fraud_detection_model.pkl`)
- Interface interactive via **Streamlit**
- Couleurs et graphique pour visualiser la probabilité de fraude
- Gestion simple des entrées via la **barre latérale**

---

## Installation

1. Cloner le dépôt :

git clone <URL_DU_REPO>
cd fraud-detection-app

2. Installer les dépendances

```bash
pip install -r requirements.txt 
```
3. Lancer l'application

```bash
streamlit run app.py
```

### Structure du projet

```bash
fraud-detection-app/
│── app.py                     # Application Streamlit
│── fraud_detection_model.pkl   # Modèle ML pré-entraîné
│── requirements.txt           # Dépendances Python
│── README.md                  # Description du projet
│── model_training.ipynb       # Creation du modele
```

## Auteurs / Contact

Projet réalisé par : Guindo Badara Aliou

Contact : badaraaliouguindo@gmail.com