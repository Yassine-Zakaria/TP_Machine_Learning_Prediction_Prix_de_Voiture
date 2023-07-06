import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Importation des données
data = pd.read_csv('car_price_prediction.csv', encoding='latin1')

# Sélection des caractéristiques
X = data[['Prod. year', 'Cylinders', 'Gear box type']]

# Sélection de la variable à prédire
y = data['Prix']

# Encodage des variables catégorielles
X = pd.get_dummies(X)

# Conversion des données en nombres à virgule flottante
X = X.astype(float)
y = y.astype(float)

# Séparation des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer un transformateur d'imputation
imputer = SimpleImputer(strategy='mean')

# Ajuster le transformateur aux données d'entraînement
imputer.fit(X_train)

# Transformer les données d'entraînement et de test en remplaçant les valeurs NaN par la moyenne des colonnes
X_train_imputed = imputer.transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Supprimer les lignes contenant des valeurs NaN dans y_train
mask = ~np.isnan(y_train)
X_train_imputed = X_train_imputed[mask]
y_train = y_train[mask]

# Entraîner le modèle de prédiction sur les données prétraitées
#rf = RandomForestRegressor(n_estimators=100, random_state=42)
#rf.fit(X_train_imputed, y_train)
# Sélection des modèles disponibles
models = {
    'Régression linéaire': LinearRegression(),
    'Forêt aléatoire': RandomForestRegressor()
}

# Affichage de la page
st.title('Prédiction de prix de voitures')
model_name = st.selectbox('Choisissez le modèle à utiliser pour la prédiction', list(models.keys()))

# Entraînement du modèle sélectionné
model = models[model_name]
model.fit(X_train_imputed, y_train)

# Prédiction d'une voiture
Annee = st.slider('Année de la voiture', min_value=2000, max_value=2023)
Kilometrage = st.slider('Kilométrage de la voiture', min_value=80, max_value=400, step=10)
Puissance = st.slider('Puissance de la voiture', min_value=50, max_value=500, step=10)

input_data = np.zeros(X.shape[1])
input_data[0] = Annee
input_data[1] = Kilometrage
input_data[2] = Puissance

prediction = model.predict([input_data])

# Affichage de la prédiction
st.write(f'Le prix de cette voiture est estimé à : {prediction[0]:.2f} euros.')
# Calculer le score du modèle sur les données de test
score = model.score(X_test_imputed, y_test)

# Afficher le score
st.write(f'Le score du modèle sur les données de test est : {score:.2f}')
