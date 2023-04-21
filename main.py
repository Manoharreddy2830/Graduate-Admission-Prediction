import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('Admission_Predict_Ver1.1.csv')

# Convert the 'Chance of Admit' column to binary
data['Chance of Admit '] = data['Chance of Admit '].apply(lambda x: 1 if x>=0.5 else 0)

# Separate features and target variable
X = data.drop(['Serial No.', 'Chance of Admit '], axis=1)
y = data['Chance of Admit ']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create instances of the classifiers
lr = LogisticRegression()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
nb = GaussianNB()

# Train the classifiers on the training set
lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)
nb.fit(X_train, y_train)

# Predict the target variable on the testing set
y_pred_lr = lr.predict(X_test)
y_pred_dt = dt.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_nb = nb.predict(X_test)

# Calculate the accuracy of the classifiers
acc_lr = accuracy_score(y_test, y_pred_lr)
acc_dt = accuracy_score(y_test, y_pred_dt)
acc_rf = accuracy_score(y_test, y_pred_rf)
acc_nb = accuracy_score(y_test, y_pred_nb)

# Print the accuracy of the classifiers
print("Accuracy of Logistic Regression: {:.2f}%".format(acc_lr*100))
print("Accuracy of Decision Tree: {:.2f}%".format(acc_dt*100))
print("Accuracy of Random Forest: {:.2f}%".format(acc_rf*100))
print("Accuracy of Naive Bayes: {:.2f}%".format(acc_nb*100))
new_data = [[326,112,4,4.0,4.5,8.5,1]]
new_prediction = lr.predict(new_data)
print('the predicted target variable for the new data point is :',new_prediction[0])
