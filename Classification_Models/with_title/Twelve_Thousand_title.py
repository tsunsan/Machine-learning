import pandas as pd
import os
# Load the preprocessed CSV file
df = pd.read_csv("../../csv/with_title/12k_amazon_handmade_reviews_balanced.csv")

import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

# Initialize NLP tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Text Preprocessing Function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Convert to lowercase and tokenize
    tokens = [word for word in tokens if word.isalnum()]  # Remove punctuation
    tokens = [word for word in tokens if word not in stop_words]  # Remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    tokens = [stemmer.stem(word) for word in tokens]  # Stemming
    return " ".join(tokens)

# Apply preprocessing
df["processed_text"] = df["combined_text"].astype(str).apply(preprocess_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["processed_text"])
y = df["sentiment"].map({"Positive": 2, "Neutral": 1, "Negative": 0})  # CHANGE THIS

# Split the dataset to be used for the algorithms
X_train_test, X_unseen, y_train_test, y_unseen = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)
X_train, X_test, y_train, y_test = train_test_split(X_train_test, y_train_test, test_size=0.20, random_state=42, stratify=y_train_test)

from sklearn.metrics import mean_absolute_error  # library to validate predicted output from actualy output
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score

# function on using mean_absolute_error
def evaluate_mae(y_test, y_predicted):
    mae = mean_absolute_error(y_test, y_predicted)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")

def checking(y_test, y_pred,model,X_test):
  # Evaluation Metrics
  conf_matrix = confusion_matrix(y_test, y_pred)
  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
  recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
  roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr")

  # Evaluate using 10-Fold Cross-Validation
  cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
  cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")


  # Print Evaluation Metrics
  print("Confusion Matrix:\n", conf_matrix)
  print("Accuracy:", accuracy)
  print("Precision:", precision)
  print("Recall:", recall)
  print("ROC-AUC:", roc_auc)
  print("Cross-Validation Accuracy Scores:", cv_scores)
  print("Mean CV Accuracy:", np.mean(cv_scores))


#HANS ARAGONA
#SVM
from sklearn.svm import SVC

svm_classifier = SVC(probability=True)  # Enable probability estimates  # instatiating the model
svm_classifier.fit(X_train, y_train)  # traning/ fitting the model

svm_predictions = svm_classifier.predict(X_test)  # testing the model
svm_unseen = svm_classifier.predict(X_unseen)

print("SVM RESULTS ====================================")

print("SVM SEEN ****************************************")
evaluate_mae(y_test,svm_predictions)
checking(y_test,svm_predictions,svm_classifier, X_test)

print("SVM UNSEEN ****************************************")
evaluate_mae(y_unseen,svm_unseen)
checking(y_unseen,svm_unseen,svm_classifier, X_unseen)

#NAIVE BASE
from sklearn.naive_bayes import MultinomialNB

# Train Naïve Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Predictions on test set
nb_predictions = nb_classifier.predict(X_test)
nb_unseen = nb_classifier.predict(X_unseen)

print("NAIVE BASE RESULTS ====================================")

print("NAIVE BASE SEEN ****************************************")
evaluate_mae(y_test,nb_predictions)
checking(y_test,nb_predictions,nb_classifier, X_test)

print("NAIVE BASE UNSEEN ****************************************")
evaluate_mae(y_unseen,nb_unseen)
checking(y_unseen,nb_unseen,nb_classifier, X_unseen)

#XGBOOST
from xgboost import XGBClassifier
xgb_model = XGBClassifier(objective='multi:softprob', num_class=5, n_estimators=200)
xgb_model.fit(X_train, y_train)

# Predict and evaluate the model
xbg_pred = xgb_model.predict(X_test)
xbg_unseen = xgb_model.predict(X_unseen)

print("XGBOOST RESULTS ====================================")

print("XGBOOST SEEN ****************************************")
evaluate_mae(y_test,xbg_pred)
checking(y_test,xbg_pred,xgb_model, X_test)

print("XGBOOST UNSEEN ****************************************")
evaluate_mae(y_unseen,xbg_unseen )
checking(y_unseen,xbg_unseen ,xgb_model, X_unseen)

#Import models
import joblib

# Save the trained model
joblib.dump(svm_classifier, 'models/svm_model_12k.pkl')
joblib.dump(nb_classifier, 'models/nb_model_12k.pkl')
joblib.dump(xgb_model, 'models/xgb_model_12k.pkl')
