#importing the dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

# loading the dataset to a Pandas DataFrame
wine_dataset = pd.read_csv('winequality-red.csv')


#checking rows and coloumns in our dataframe
print("h")
print(wine_dataset.shape)

# first 5 rows of the dataset
print(wine_dataset.head())

# checking for missing values
print(wine_dataset.isnull().sum())

#DATA ANALYSIS AND VISUALISATION

# statistical measures of the dataset
print(wine_dataset.describe())

# number of values for each quality
sns.catplot(x='quality', data = wine_dataset, kind = 'count')


correlation = wine_dataset.corr()

# constructing a heatmap to understand the correlation between the columns
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt = '.1f', annot = True, annot_kws={'size':8}, cmap = 'Blues')

# separate the data and Label
X = wine_dataset.drop('quality',axis=1)

Y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

print(Y.shape, Y_train.shape, Y_test.shape)

LR = LogisticRegression()
LR.fit(X_train, Y_train)

# accuracy on test data
X_test_prediction = LR.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)*100

print('Logisitic Reg Accuracy : ', test_data_accuracy)

print(LR.coef_.tolist())



DT = DecisionTreeClassifier()
DT.fit(X_train, Y_train)

# accuracy on test data
X_test_prediction = DT.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)*100

print('Decision Tree Accuracy : ', test_data_accuracy)
plt.figure(figsize=(80,40))
plot_tree(DT,filled=(True),feature_names=X_train.columns)
print(DT.feature_importances_)


model = RandomForestClassifier()

model.fit(X_train, Y_train)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)*100

print('Accuracy : ', test_data_accuracy)

input_data = (7.3,0.65,0,1.2,0.065,15,21,0.9946,3.39,0.47,10)

# changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the data as we are predicting the label for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==1):
  print('Good Quality Wine')
else:
  print('Bad Quality Wine')#importing the dependencies
