# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 22:49:02 2025

@author: dalys
"""

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import random

#pip install imbalanced-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import cohen_kappa_score
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Download nltk tokenizer data
nltk.download('punkt')

#Path to the CSV file with the captions to analyze
csv_path = r'DF_train'
df = pd.read_csv(csv_path)
#TODO replace with the path you would like to have assessed CSV saved to
output_csv = r'C:analyzed_transcript_RF.csv'
# Remove rows where the 'caption' column is empty or NaN
#df.dropna(subset=['caption'], inplace=True)
#df = df.dropna(subset=df.columns.values)
# display dataframe
print(df)


# Load dataset from DataFrame
def load_data_from_dataframe(df):
    return list(zip(df['Text'], df['question'], df['answer']))

# Sample dataset in DataFrame format
data_df = pd.DataFrame(df)

data = load_data_from_dataframe(data_df)

# Tokenize and build vocabulary
all_words = []
for text, question, _ in data:
    all_words.extend(word_tokenize(text.lower()))
    all_words.extend(word_tokenize(question.lower()))
word_counts = Counter(all_words)
vocab = {word: idx + 1 for idx, (word, _) in enumerate(word_counts.items())}
vocab["<UNK>"] = 0  # Unknown words

# Convert data to numerical format
def encode_text(text):
    return ' '.join(word_tokenize(text.lower()))

def encode_question(question):
    return ' '.join(word_tokenize(question.lower()))

answer_words = list(set(data_df["answer"].str.lower()))  # Get unique answers
answer_vocab = {word: idx for idx, word in enumerate(answer_words)}
answer_vocab["<UNK>"] = len(answer_vocab)  # Add unknown token

def encode_answer(answer):
    return answer_vocab.get(answer.lower(), answer_vocab["<UNK>"])

# Prepare data for training
text_data = [encode_text(t) for t, _, _ in data]
question_data = [encode_question(q) for _, q, _ in data]
labels = [encode_answer(a) for _, _, a in data]

# Define RF model
class QARandomForest:
    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators
        self.vectorizer = TfidfVectorizer()
        self.model = None  # Will be initialized in fit()

    def fit(self, text_data, question_data, labels):
        combined_data = [t + ' ' + q for t, q in zip(text_data, question_data)]
        X_tfidf = self.vectorizer.fit_transform(combined_data)

        min_class_count = min(Counter(labels).values())
        k = max(1, min(min_class_count - 1, 5))  # Safe k for SMOTE
        smote = SMOTE(random_state=42, k_neighbors=k)
        X_resampled, y_resampled = smote.fit_resample(X_tfidf, labels)

        # Class weights
        classes = np.array(list(set(y_resampled)))
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_resampled)
        class_weight_dict = dict(zip(classes, weights))

        # Train with weights
        self.model = RandomForestClassifier(n_estimators=self.n_estimators,
                                            random_state=42,
                                            class_weight=class_weight_dict)
        self.model.fit(X_resampled, y_resampled)

    def predict(self, text_data, question_data):  # <- Make sure this is inside the class
        combined_data = [t + ' ' + q for t, q in zip(text_data, question_data)]
        X_tfidf = self.vectorizer.transform(combined_data)
        return self.model.predict(X_tfidf)
# Train the model
model = QARandomForest(100)
model.fit(text_data, question_data, labels)

def predict_answer(text, question):
    prediction = model.predict([encode_text(text)], [encode_question(question)])[0]
    return list(answer_vocab.keys())[list(answer_vocab.values()).index(prediction)]

# Test prediction
print("STAGE!")
csv_path_test = r'C:\Users\dalys\OneDrive\Documents\My Documents\Cornell Postdoc Papers\Cronobacter\Social_Scrape_Paper\Python_AI\Compare2\Risk\DF_test.csv' #TODO: replace with the path to the CSV on your computer
df_test = pd.read_csv(csv_path_test)

#print(predict_answer(df_test['Text'][10], df_test['question'][i]))

# Process test dataset
data2 = []
for i in range(572):
    result = predict_answer(df_test['Text'][i], df_test['question'][i])
    words = df_test['Video.ID'][i]
    tt = df_test['Text'][i]
    real = df_test['answer'][i]
  
  # Append the result to the list as a dictionary
    data2.append({'code': words,'Text': tt,'Real': real, 'result': result})

data2 = pd.DataFrame(data2)
data2.to_csv('RF_output 03_SMOTEk1.csv', index=False)


##=======METRIC Calc===============
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

metrics = []

# Make sure your labels are consistent and in the same format
# Optionally convert to lowercase or consistent format:
data2['Real'] = data2['Real'].str.lower()
data2['result'] = data2['result'].str.lower()

# Use average='weighted' if you have multiple classes
kappa = cohen_kappa_score(data2['Real'], data2['result'])
precision = precision_score(data2['Real'], data2['result'], average='weighted', zero_division=0)
recall = recall_score(data2['Real'], data2['result'], average='weighted', zero_division=0)
f1 = f1_score(data2['Real'], data2['result'], average='weighted', zero_division=0)

metrics.append({
    'kappa': round(kappa, 4),
    'precision': round(precision, 4),
    'true_pos_rate/sensitivity': round(recall, 4),
    'F1': round(f1, 4)
})

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv('RF_smote_metrics.csv', index=False)
#filename = "kappa_svn.txt"

#with open(filename, 'w') as file:
  #  file.write(str(kappa))

