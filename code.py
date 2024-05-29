import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import matplotlib.pyplot as plt

# Charger les données à partir du fichier CSV
data = pd.read_csv('age_vs_poids_vs_taille_vs_sexe.csv')

# Diviser les données en fonctionnalités (X) et étiquettes (Y)
X = data[['sexe', 'taille', 'age']]
Y = data['poids']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Créer un modèle de régression linéaire
reg = LinearRegression()

# Entraîner le modèle sur les données d'entraînement
reg.fit(X_train, Y_train)

# Utiliser le modèle pour faire des prédictions sur les données de test
Y_pred = reg.predict(X_test)

# Calculer les métriques d'évaluation
mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
mape = mean_absolute_percentage_error(Y_test, Y_pred)

# Afficher les résultats des métriques
print(f"Erreur quadratique moyenne (MSE) : {mse}")
print(f"Erreur absolue moyenne (MAE) : {mae}")
print(f"Erreur absolue moyenne en pourcentage (MAPE) : {mape}")

# Visualisation des prédictions par rapport aux vraies valeurs
plt.figure(figsize=(8, 6))
plt.scatter(Y_test, Y_pred, color='blue', label='Prédictions')
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], color='red', linestyle='--', label='Réelles')
plt.title('Comparaison entre les Valeurs Réelles et les Prédictions')
plt.xlabel('Valeurs Réelles')
plt.ylabel('Valeurs Prédites')
plt.legend()
plt.show()
