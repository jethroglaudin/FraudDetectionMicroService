# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def run_model():
    # Importing the dataset
    dataset = pd.read_csv('transactions.csv')
    X = dataset.iloc[:, [1, 2, 4, 5, 7, 8]].values
    y = dataset.iloc[:, 9].values

    # Encoding categorical data
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
    X = np.array(ct.fit_transform(X))

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # Feature Scaling the dataset
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    # Training the Logistic Regression model on the Training set
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(max_iter=100000)
    classifier.fit(X_train, y_train)

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
    y_pred = classifier.predict(X_test)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(y_test, y_pred).ravel()
    accuracy = round(accuracy_score(y_test, y_pred),4) * 100
    precision = round(precision_score(y_test, y_pred),4)
    recall = round(recall_score(y_test, y_pred),4)
    result = [
        {
            'confusion_matrix': {
                'Non-Fraudulent Transactions Predicted True': true_negative.item(),
                'Non-Fradulent Transaction Predicted False': false_negative.item(),
                'Fraudlent Transaction Predicted False': false_positive.item(),
                'Fraudulent Transaction Predicted True': true_positive.item()
            },

            'metrics': {
                'Total data size': y_train.size + y_test.size,
                'Test data size': y_test.size,
                'Accuacy': round(accuracy,4) * 100,
                'Precision': precision,
                'Recall': recall
            }
        }

    ]
    return result
