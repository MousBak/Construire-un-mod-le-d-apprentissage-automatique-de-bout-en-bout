# Modèle de Machine Learning de bout en bout - Immobilier

> Machine Learning / End-to-End Project

## Vue d'ensemble

Projet complet de prédiction de prix immobiliers combinant exploration des données, entraînement d'un modèle, sauvegarde du modèle et application Python. Le dépôt montre la logique d'un workflow ML au-delà du notebook seul.

## Objectifs du projet

- Explorer les facteurs qui influencent le prix immobilier.
- Entraîner un modèle de régression.
- Sauvegarder le modèle entraîné pour réutilisation.
- Préparer une base d’application pour servir la prédiction.

## Démarche

- Analyse exploratoire du fichier Real_Estate.csv.
- Préparation des variables et entraînement du modèle.
- Sérialisation dans real_estate_model.pkl.
- Organisation en scripts main.py et model.py.

## Stack technique

- Python
- Pandas
- Scikit-learn
- Pickle
- Jupyter Notebook
- Machine Learning

## Structure du dépôt

- `Prévision_des_prix_de_l'immobilier_avec_Python.ipynb`
- `Real_Estate.csv`
- `main.py`
- `model.py`
- `real_estate.zip`
- `real_estate_model.pkl`

## Lancer ou consulter le projet

```bash
pip install pandas scikit-learn matplotlib seaborn jupyter
jupyter notebook "Prévision_des_prix_de_l'immobilier_avec_Python.ipynb"
python main.py
```

## Compétences démontrées

- Workflow ML de bout en bout.
- Régression supervisée.
- Sauvegarde et réutilisation de modèle.
- Structuration notebook + scripts.

## Pistes d'amélioration

- Retirer venv/ et __pycache__/ du dépôt via .gitignore.
- Ajouter un requirements.txt.
- Ajouter une API ou interface Streamlit pour tester les prédictions.

## Auteur

**Bakayoko Moussa**  
Data Analyst / BI Analyst / Analytics Engineering Jr  
Portfolio : https://mousbak.github.io/  
GitHub : https://github.com/MousBak
