import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

data = pd.DataFrame({
    'Age': [22,25,30,35,42,50,23,28,33,48],
    'Income': [60,75,80,120,150,110,95,90,105,135],
    'CreditHistory': [1,2,3,5,8,10,1,2,4,9],
    'Repaid': ['No','No','Yes','Yes','Yes','Yes','No','No','Yes','Yes']
})

X = data[['Age','Income','CreditHistory']]
y = data['Repaid']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_scaled, y)

new_customer = np.array([[27,95,3]])
new_customer_scaled = scaler.transform(new_customer)
prediction = knn.predict(new_customer_scaled)
print("Prediction for new customer:", prediction[0])

param_grid = {
    'n_neighbors': list(range(1,32)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean','manhattan']
}

grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid.fit(X_scaled, y)

print("Best parameters:", grid.best_params_)
print("Best score:", grid.best_score_)