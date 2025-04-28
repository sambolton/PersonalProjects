from DataScraping import X, y
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=1)

#Random Forest Health Prediction
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forestmodel = RandomForestRegressor(random_state=1)
forestmodel.fit(train_X, train_y)
predictions = forestmodel.predict(val_X)
print("MAE: ", mean_absolute_error(val_y, predictions))

print("Feature Importance: ", forestmodel.feature_importances_)
import matplotlib.pyplot as plt
import seaborn as sns
importances = forestmodel.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), features[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.show()
# The code above is a complete implementation of a Random Forest model for health prediction using the UCI Machine Learning Repository dataset.

