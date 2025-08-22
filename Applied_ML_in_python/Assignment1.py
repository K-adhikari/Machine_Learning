# Load breast cancer data from scikit-learn datasets

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

print(cancer.DESCR) # print data set description



# How many features does the breast cancer dataset have?

cancer.data.shape[1]

# Dataset has 30 features.



# Convert the dataset to a dataframe with 'target' column added (target is not on the initial list of features)

cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
cancer_df['target'] = cancer.target
cancer_df.head()



# Check the instances of malignant and benign in the dataframe

distribution = cancer_df['target'].value_counts().sort_index()
target = pd.Series(data = distribution.values, index = ['malignant', 'benign'], name = 'target', dtype = 'int')
print(target)



# Split the dataframe into X (the data) and y(the labels)

X = cancer_df.drop(columns= 'target')
y = cancer_df['target']
print(X.shape, y.shape)



# Using train_test_split, split X and y into training and test sets (X_train, X_test, y_train, and y_test).

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0) #fixed random state will give reproducible results
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)



# Using KNeighborsClassifier, fit a k-nearest neighbors (knn) classifier with X_train, y_train and using one nearest neighbor (n_neighbors = 1).

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 1)



# Using your knn classifier, predict the class label for the test set X-test

knn.fit(X_train, y_train)
knn.fit(X_train, y_train)
prediction = knn.predict(X_test)
print(prediction)



# Find the score (mean accuracy) of your knn classifier using X_test and y_test.

accuracy = knn.score(X_test, y_test)
print(accuracy)



# Plot the different prediction scores between train and test sets, as well as malignant and benign cells

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')

train_score = knn.score(X_train, y_train)
test_score = knn.score(X_test, y_test)
    
# Predict on test set to evaluate per-class accuracy
y_test_pred = prediction

# Boolean masks for class-wise test samples
malignant_mask = y_test == 0
benign_mask = y_test == 1

# Compute accuracy for malignant (0) and benign (1)
malignant_score = np.mean(y_test_pred[malignant_mask] == y_test[malignant_mask])
benign_score = np.mean(y_test_pred[benign_mask] == y_test[benign_mask])

# Bar chart
plt.figure(figsize=(7, 5))
bars = plt.bar(
        ['Train Accuracy', 'Test Accuracy', 'Malignant Accuracy', 'Benign Accuracy'],
        [train_score, test_score, malignant_score, benign_score],
        color=['blue', 'green', 'red', 'orange']
    )
plt.ylim(0, 1)
plt.title('KNN Accuracy Comparison')
plt.ylabel('Accuracy')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()


