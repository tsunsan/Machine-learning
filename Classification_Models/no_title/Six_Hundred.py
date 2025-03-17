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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold


def labeled_confusion_matrix(conf_matrix, class_labels):
    return pd.DataFrame(conf_matrix, index=class_labels, columns=class_labels)


def training_and_testing(model, X_train, y_train, X_test, y_test, filename, filename_Two, original_df):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results = []
    unsuccessful_results = []
    successful_results = []

    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_roc_auc = 0
    total_conf_matrix = None
    total_mae = 0

    for fold_idx, (train_idx, _) in enumerate(cv.split(X_train, y_train), start=1):
        X_train_fold, y_train_fold = X_train[train_idx], y_train.iloc[train_idx]

        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_test)

        conf_matrix = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr")
        mae = mean_absolute_error(y_test, y_pred)

        total_accuracy += accuracy
        total_precision += precision
        total_recall += recall
        total_roc_auc += roc_auc
        total_mae += mae

        if total_conf_matrix is None:
            total_conf_matrix = conf_matrix
        else:
            total_conf_matrix += conf_matrix

        results.append({
            "Mean Absolute Error": mae,
            "Fold Index": fold_idx,
            "Confusion Matrix": conf_matrix,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "ROC AUC": roc_auc
        })

    for i, (true, pred) in enumerate(zip(y_test, y_pred)):
        original_text = original_df.iloc[i]["text"]
        if true != pred:
            unsuccessful_results.append(f"True: {true}, Predicted: {pred}, Text: {original_text}")
        else:
            successful_results.append(f"True: {true}, Predicted: {pred}, Text: {original_text}")

    num_folds = cv.get_n_splits()
    avg_accuracy = total_accuracy / num_folds
    avg_precision = total_precision / num_folds
    avg_recall = total_recall / num_folds
    avg_roc_auc = total_roc_auc / num_folds
    avg_mae = total_mae / num_folds

    class_labels = ["negative", "neutral", "positive"]
    labeled_conf_matrix = labeled_confusion_matrix(total_conf_matrix, class_labels)

    print("Overall Averages:")
    print(f"Accuracy: {avg_accuracy:.4f}")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")
    print(f"ROC AUC: {avg_roc_auc:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")
    print("Overall Confusion Matrix:")
    print(labeled_conf_matrix)

    overall_metrics = {
        "Overall Accuracy": avg_accuracy,
        "Overall Precision": avg_precision,
        "Overall Recall": avg_recall,
        "Overall ROC AUC": avg_roc_auc,
        "Overall Mean Average Error": avg_mae,
        "Overall Confusion Matrix": labeled_conf_matrix.to_dict()
    }

    results.append({"Overall Metrics": overall_metrics})

    with open(filename, "w", encoding="utf-8") as file:
        file.write("\n".join(unsuccessful_results))

    with open(filename_Two, "w", encoding="utf-8") as file:
        file.write("\n".join(successful_results))

    return results, model


#HANS ARAGONA
#SVM
from sklearn.svm import SVC

svm_classifier = SVC(probability=True)  # Enable probability estimates  # instatiating the model
print("SVM")
svm_training_and_testing, svm_classifier =training_and_testing(svm_classifier,X_train, y_train,X_test,y_test,
                                              "results/svm600_unsuccessful_results.txt","results/svm600_successful_results.txt",df)
#printing of training and testing results
with open('training_testing/svm600_training_and_testing.txt', 'w') as file:
    for res in svm_training_and_testing:
        file.write(f"{res}\n")

#NAIVE BASE
from sklearn.naive_bayes import MultinomialNB
nb_classifier = MultinomialNB()
print("NAIVE BASE")
nb_training_and_testing, nb_classifier =training_and_testing(nb_classifier,X_train, y_train,X_test,y_test,
                                              "results/nb600_unsuccessful_results.txt","results/nb600_successful_results.txt",df)
#printing of training and testing results
with open('training_testing/nb600_training_and_testing.txt', 'w') as file:
    for res in nb_training_and_testing:
        file.write(f"{res}\n")

#XGBOOST
from xgboost import XGBClassifier
xgb_classifier = XGBClassifier(objective='multi:softprob', num_class=5, n_estimators=200)
print("XGBOOST")
xgb_training_and_testing, xgb_classifier =training_and_testing(xgb_classifier,X_train, y_train,X_test,y_test,
                                              "results/xgb600_unsuccessful_results.txt","results/xgb600_successful_results.txt",df)
#printing of training and testing results
with open('training_testing/xgb600_training_and_testing.txt', 'w') as file:
    for res in xgb_training_and_testing:
        file.write(f"{res}\n")

 #saving the model
import joblib
# Save the trained model
joblib.dump(svm_classifier, 'models/svm_model_600.pkl')
joblib.dump(nb_classifier, 'models/nb_model_600.pkl')
joblib.dump(xgb_classifier, 'models/xgb_model_600.pkl')

#valididating the model
def validate (filename, X_unseen, y_unseen,path):
    # Load the model
    model = joblib.load(filename)

    # Make predictions on unseen data
    y_pred = model.predict(X_unseen)
    y_proba = model.predict_proba(X_unseen)

    # Calculate metrics
    conf_matrix = confusion_matrix(y_unseen, y_pred)
    accuracy = accuracy_score(y_unseen, y_pred)
    precision = precision_score(y_unseen, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_unseen, y_pred, average='weighted', zero_division=1)
    roc_auc = roc_auc_score(y_unseen, y_proba, multi_class="ovr")
    mae = mean_absolute_error(y_unseen, y_pred)

    # Create labeled confusion matrix
    class_labels = ['negative', 'neutral', 'positive']
    labeled_conf_matrix = pd.DataFrame(conf_matrix, index=class_labels, columns=class_labels)

    validated_metrics = {
        "Validated Accuracy": accuracy,
        "Validated Precision": precision,
        "Overall Recall": recall,
        "Overall ROC AUC": roc_auc,
        "Overall Mean Average Error": mae
    }
    with open(path, 'w') as file:
        for key, value in validated_metrics.items():
            file.write(f"{key}: {value}\n")
        file.write("Overall Confusion Matrix:\n")
        file.write(labeled_conf_matrix.to_string())
    # Print the metrics
    print("Overall Averages: VALIDATION")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Average MAE: {mae:.4f}")
    print("Overall Confusion Matrix:")
    print(labeled_conf_matrix)

validate('models/svm_model_600.pkl', X_unseen, y_unseen,'validated/svm600_validated.txt')
validate('models/nb_model_600.pkl', X_unseen, y_unseen,'validated/nb600_validated.txt')
validate('models/xgb_model_600.pkl', X_unseen, y_unseen, 'validated/xgb600_validated.txt')



