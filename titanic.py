import numpy as np
import pandas as pd
import os

#Read the Training Data and Testing Data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

#Import the model and metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

#Define the parameters of the model
y = train_data["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.fet_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

#Setting up number of trees in the forest
#Then create model and test for accuracy
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)
accuracy_score(y, predictions, normalize=False)
print(classification_report(y,predictions))

#Creating output and print statement when done
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
