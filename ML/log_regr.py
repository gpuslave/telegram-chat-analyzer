import pandas as pd
from sklearn import metrics
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, GridSearchCV
from sklearn.utils import shuffle

df = pd.read_csv('ML\\shuffled_rus_words.csv')
# df = df.drop_duplicates()
df = shuffle(df)

# Split the data into the texts and the labels
texts = df['word'].tolist()
print("debug:", len(texts))
labels = df['is_laugh'].tolist()

# Split the data into a training set and a test set
texts_train, texts_test, labels_train, labels_test = train_test_split(
    texts, labels, test_size=0.1, random_state=42)

# Convert the texts to a matrix of token counts
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(texts_train)
X_test = vectorizer.transform(texts_test)

# # Define the parameter values that should be searched
# param_dist = {'C': [0.1, 1, 10, 100, 1000],
#               'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
#               'kernel': ['rbf']}

# Define the parameter values that should be searched
param_dist = {
    'n_estimators': [200, 500, 1000, 2000, 3500],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Instantiate the RandomizedSearchCV object with the new scoring metric
# rand = RandomizedSearchCV(SVC(), param_dist, n_iter=10,
#                           random_state=42)

model = RandomForestClassifier(n_estimators=10000)
# Fit the RandomizedSearchCV object to the data
model.fit(X_train, labels_train)
# # Print the best parameters
# print(rand.best_params_)

# # Use the best parameters to fit the model
# model = rand.best_estimator_

# # Perform 5-fold cross validation on the training data
# scores = cross_val_score(model, X_train, labels_train, cv=5)

# print("Cross-validation scores: ", scores)
# print("Average cross-validation score: ", scores.mean())
# # Train a logistic regression model on the training data
# model = SVC()
# model.fit(X_train, labels_train)

# # Use the model to predict the labels of the test data
predictions = model.predict(X_test)

accuracy = metrics.accuracy_score(labels_test, predictions)
precision = metrics.precision_score(labels_test, predictions)
recall = metrics.recall_score(labels_test, predictions)
f1 = metrics.f1_score(labels_test, predictions)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("Predictions:", predictions)
print("Actual labels:", labels_test)


def is_laugh(plain_text):
    # Convert the text to a matrix of token counts
    X = vectorizer.transform([plain_text])
    # Use the model to predict the label of the text
    prediction = model.predict(X)
    # Return True if the predicted label is 1 (laughter), and False otherwise
    return prediction[0] == 1


while True:
    text = input("Enter some text: ")
    print(is_laugh(text))
