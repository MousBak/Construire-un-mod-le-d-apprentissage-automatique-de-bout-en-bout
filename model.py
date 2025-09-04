import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

# Charger les données
real_estate_data = pd.read_csv(
    r"C:\Users\moussa.bakayoko\Desktop\Projet perso\Prevision-des-prix-de-l-immobilier-avec-python-main\Real_Estate.csv"
)

# Colonnes features et target
features_columns = ['Distance to the nearest MRT station', 'Number of convenience stores', 'Latitude', 'Longitude']
target_column = 'House price of unit area'

X = real_estate_data[features_columns]
y = real_estate_data[target_column]

# Séparer train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle
model = LinearRegression()
model.fit(X_train, y_train)

# Sauvegarder le modèle
with open("real_estate_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")
