#Data Download from UCI Machine Learning Repository
from ucimlrepo import fetch_ucirepo 
import pandas as pd
  
# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
# data (as pandas dataframes) 
X = heart_disease.data.features
y = heart_disease.data.targets

merged = pd.concat([X, y], axis=1)
merged = merged.dropna()

X = merged[heart_disease.data.features.columns]
y = merged[heart_disease.data.targets.columns]

# # metadata 
# print(heart_disease.metadata) 
  
# # variable information 
# print(heart_disease.variables) 
