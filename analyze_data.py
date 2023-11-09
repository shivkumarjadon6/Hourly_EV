# compare algorithms
import pandas as pd
import numpy as np
from pandas import read_csv
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "./archive/reports.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)

# Assuming X_train is a NumPy array
column_names = ['Garage_ID', 'User_ID', 'User_type', 'Shared_ID', 'Start_plugin', 'Start_plugin_hour', 'End_plugout', 'End_plugout_hour', 'El_kWh', 'Duration_hours', 'month_plugin', 'weekdays_plugin', 'Plugin_category', 'Duration_category']

X_train_df = pd.DataFrame(X_train, columns=column_names)

# Check for NaN values
print("NaN values in X_train:")
print(X_train_df.isnull().sum())

# Assuming X_train and Y_train are DataFrames
print("NaN values in X_train:")
print(X_train.isnull().sum())

print("NaN values in Y_train:")
print(Y_train.isnull().sum())

# Assuming X_train is a NumPy array
nan_count = np.isnan(X_train).sum()

print("NaN values in X_train:", nan_count)

# Create an imputer with a strategy (e.g., mean, median)
imputer = SimpleImputer(strategy='mean')

# Fit and transform X_train
X_train_imputed = imputer.fit_transform(X_train)

# Replace X_train with the imputed version
X_train = pd.DataFrame(X_train_imputed, columns=X_train.columns)

# Remove rows with NaN in any column
X_train = X_train.dropna()
Y_train = Y_train.loc[X_train.index]


# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# Compare Algorithms
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()
