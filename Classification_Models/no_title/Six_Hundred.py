import pandas as pd
import os
# Load the preprocessed CSV file
df = pd.read_csv("../../csv/no_title/600_no_title_amazon_handmade_reviews_balanced.csv")

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
df["processed_text"] = df["text"].astype(str).apply(preprocess_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["processed_text"])
y = df["sentiment"].map({"Positive": 2, "Neutral": 1, "Negative": 0})  # CHANGE THIS

# Split the dataset to be used for the algorithms
X_train_test, X_unseen, y_train_test, y_unseen = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)
X_train, X_test, y_train, y_test = train_test_split(X_train_test, y_train_test, test_size=0.20, random_state=42, stratify=y_train_test)

from sklearn.metrics import mean_absolute_error  # library to validate predicted output from actualy output

# function on using mean_absolute_error
def evaluate_mae(y_test, y_predicted):
    mae = mean_absolute_error(y_test, y_predicted)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

def training_and_testing(model, X, y, filename, original_df):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results = []
    unsuccessful_results = []

    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_roc_auc = 0

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Fit the model on the training set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        conf_matrix = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr")

        # Accumulate metrics
        total_accuracy += accuracy
        total_precision += precision
        total_recall += recall
        total_roc_auc += roc_auc

        # Store results
        results.append({
            "Fold Index": fold_idx,
            "Confusion Matrix": conf_matrix,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "ROC AUC": roc_auc
        })

        # Collect unsuccessful predictions with original text
        for i, (true, pred) in enumerate(zip(y_test, y_pred)):
            if true != pred:
                original_text = original_df.iloc[test_idx[i]]["text"]
                unsuccessful_results.append(f"True: {true}, Predicted: {pred}, Text: {original_text}")

    # Calculate overall averages
    num_folds = cv.get_n_splits()
    avg_accuracy = total_accuracy / num_folds
    avg_precision = total_precision / num_folds
    avg_recall = total_recall / num_folds
    avg_roc_auc = total_roc_auc / num_folds

    print("Overall Averages:")
    print(f"Accuracy: {avg_accuracy:.4f}")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")
    print(f"ROC AUC: {avg_roc_auc:.4f}")

    # Write unsuccessful predictions to file
    with open(filename, "w", encoding="utf-8") as file:
        file.write("\n".join(unsuccessful_results))

    return results




#HANS ARAGONA
#SVM
from sklearn.svm import SVC

svm_classifier = SVC(probability=True)  # Enable probability estimates  # instatiating the model
svm_classifier.fit(X_train, y_train)  # traning/ fitting the model

svm_predictions = svm_classifier.predict(X_test)  # testing the model
svm_unseen = svm_classifier.predict(X_unseen)

print("SVM RESULTS ====================================")

print("SVM TESTING SEEN ****************************************")
evaluate_mae(y_test,svm_predictions)
svm_training_and_testing=training_and_testing(svm_classifier,X,y,"results/svm600_unsuccessful_results.txt",df)

#printing of training and testing results
with open('training_testing/svm600_training_and_testing.txt', 'w') as file:
    for res in svm_training_and_testing:
        file.write(f"{res}\n")


#NAIVE BASE
from sklearn.naive_bayes import MultinomialNB

# Train Naïve Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Predictions on test set
nb_predictions = nb_classifier.predict(X_test)
nb_unseen = nb_classifier.predict(X_unseen)

print("NAIVE BASE RESULTS ====================================")

print("NAIVE BASE TESTING SEEN ****************************************")
evaluate_mae(y_test,nb_predictions)


#XGBOOST
from xgboost import XGBClassifier
xgb_model = XGBClassifier(objective='multi:softprob', num_class=5, n_estimators=200)
xgb_model.fit(X_train, y_train)

# Predict and evaluate the model
xbg_pred = xgb_model.predict(X_test)
xbg_unseen = xgb_model.predict(X_unseen)


print("XGBOOST TESTING SEEN ****************************************")
evaluate_mae(y_test,xbg_pred)



#Import models
import joblib

# Save the trained model
joblib.dump(svm_classifier, 'models/svm_model_600.pkl')
joblib.dump(nb_classifier, 'models/nb_model_600.pkl')
joblib.dump(xgb_model, 'models/xgb_model_600.pkl')
