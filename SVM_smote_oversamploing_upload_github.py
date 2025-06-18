# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 05:32:40 2025

@author: dalys
"""
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from imblearn.over_sampling import SMOTE

#Path to the CSV file with the captions to analyze
csv_path = r'C:\DF_train.csv' #TODO: replace with the path to the CSV on your computer
df = pd.read_csv(csv_path)
#TODO replace with the path you would like to have assessed CSV saved to
output_csv = r'C:\analyzed_transcript_RF.csv'
# Remove rows where the 'caption' column is empty or NaN
#df.dropna(subset=['caption'], inplace=True)
#df = df.dropna(subset=df.columns.values)
# display dataframe
print(df)

nltk.download('punkt')

# Load and clean your dataset
df.dropna(subset=['Text', 'question', 'answer'], inplace=True)

# Encode text & question
def encode_text(text):
    return ' '.join(word_tokenize(text.lower()))

def encode_question(question):
    return ' '.join(word_tokenize(question.lower()))

# Prepare label encoding
answer_words = list(set(df["answer"].str.lower()))
answer_vocab = {word: idx for idx, word in enumerate(answer_words)}
answer_vocab["<UNK>"] = len(answer_vocab)

def encode_answer(answer):
    return answer_vocab.get(answer.lower(), answer_vocab["<UNK>"])

# Prepare data
text_data = [encode_text(t) for t in df['Text']]
question_data = [encode_question(q) for q in df['question']]
labels = [encode_answer(a) for a in df['answer']]
combined_data = [t + ' ' + q for t, q in zip(text_data, question_data)]

# === Split and vectorize ===
X_train, X_test, y_train, y_test = train_test_split(
    combined_data, labels, test_size=0.2, stratify=labels, random_state=42
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# === Apply SMOTE for oversampling ===
min_class_count = min(Counter(y_train).values())
k_neighbors = max(1, min(min_class_count - 1, 5))  # Prevent SMOTE crash
smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vec, y_train)

# === Predict and Evaluate ===
# === Train SVM model ===
model = SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# === Predict Answer Function ===
def predict_answer(text, question):
    input_text = encode_text(text) + ' ' + encode_question(question)
    input_vec = vectorizer.transform([input_text])
    prediction = model.predict(input_vec)[0]
    return list(answer_vocab.keys())[list(answer_vocab.values()).index(prediction)]
          
          
# Test prediction
print("STAGE!")
csv_path_test = r'C:\DF_test.csv' #TODO: replace with the path to the CSV on your computer
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
data2.to_csv('SVM_output_oversampling_smote.csv', index=False)

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
metrics_df.to_csv('SVM_smote_metrics.csv', index=False)
#filename = "kappa_svn.txt"

#with open(filename, 'w') as file:
  #  file.write(str(kappa))

