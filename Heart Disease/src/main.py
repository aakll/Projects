#Importing libraries
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('data/heart-disease.csv')

#Checking
print(dataset.head())

# Splitting the dataset into features and target variable
X = dataset.drop('target', axis=1)
y = dataset['target']

# Splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

#View the shapes of the datasets
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

#2. Preparing the model
# Importing the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

# Creating the model
model = RandomForestClassifier()

# 3. Fitting the model to the training data
model.fit(X_train, y_train)

# 4. Making predictions
predictions = model.predict(X_test)

# 5. Evaluating the model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print("Training set accuracy:", train_score*100)
print("Testing set accuracy:", test_score*100)