import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import random
from sklearn.metrics import cohen_kappa_score

# Download nltk tokenizer data
nltk.download('punkt')

#Path to the CSV file with the captions to analyze
csv_path = r'C:DF_train.csv' #TODO: replace with the path to the CSV on your computer
df = pd.read_csv(csv_path)
#TODO replace with the path you would like to have assessed CSV saved to
output_csv = r'C:\analyzed_transcript_NN.csv'
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
def encode_text(text, max_len=500):
    encoded = [vocab.get(word, vocab["<UNK>"]) for word in word_tokenize(text.lower())]
    if len(encoded) < max_len:
        encoded += [0] * (max_len - len(encoded))  # Padding
    else:
        encoded = encoded[:max_len]  # Truncate
    return encoded

def encode_question(question, max_len=500):
    encoded = [vocab.get(word, vocab["<UNK>"]) for word in word_tokenize(question.lower())]
    if len(encoded) < max_len:
        encoded += [0] * (max_len - len(encoded))  # Padding
    else:
        encoded = encoded[:max_len]  # Truncate
    return encoded

answer_words = list(set(data_df["answer"].str.lower()))  # Get unique answers
answer_vocab = {word: idx for idx, word in enumerate(answer_words)}
answer_vocab["<UNK>"] = len(answer_vocab)  # Add unknown token

def encode_answer(answer):
    return answer_vocab.get(answer.lower(), answer_vocab["<UNK>"])

max_text_length = max(len(word_tokenize(t.lower())) for t, _, _ in data)
max_question_length = max(len(word_tokenize(q.lower())) for _, q, _ in data)
encoded_data = [(encode_text(t, max_text_length), encode_question(q, max_question_length), encode_answer(a)) for t, q, a in data]

# Define dataset class
class QA_Dataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text, question, answer = self.data[idx]
        return torch.tensor(text, dtype=torch.long), torch.tensor(question, dtype=torch.long), torch.tensor(answer, dtype=torch.long)

# Create train and test loaders
random.shuffle(encoded_data)
split = int(0.8 * len(encoded_data))
train_data, test_data = encoded_data[:split], encoded_data[split:]
train_loader = DataLoader(QA_Dataset(train_data), batch_size=2, shuffle=True)
test_loader = DataLoader(QA_Dataset(test_data), batch_size=2, shuffle=False)

# Define neural network
class QAModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(QAModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, text, question):
        text_embed = self.embedding(text).mean(dim=1)
        question_embed = self.embedding(question).mean(dim=1)
        x = torch.cat((text_embed, question_embed), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Model initialization
vocab_size = len(vocab)
embed_dim = 10
hidden_dim = 16

model = QAModel(vocab_size, embed_dim, hidden_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for texts, questions, answers in train_loader:
        optimizer.zero_grad()
        outputs = model(texts, questions)
        loss = criterion(outputs, answers)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# Testing
def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, questions, answers in test_loader:
            outputs = model(texts, questions)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == answers).sum().item()
            total += answers.size(0)
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

    with open('out.txt', 'w') as output:
        output.write(f"Test Accuracy: {100 * correct / total:.2f}%")

evaluate()


#############################TEST DATA####################################
# Function for user input prediction
def predict_answer(text, question):
    model.eval()
    encoded_t = encode_text(text, max_text_length)
    encoded_q = encode_question(question, max_question_length)
    input_text = torch.tensor([encoded_t], dtype=torch.long)
    input_question = torch.tensor([encoded_q], dtype=torch.long)
    with torch.no_grad():
        output = model(input_text, input_question)
        _, predicted = torch.max(output, 1)
    return list(answer_vocab.keys())[list(answer_vocab.values()).index(predicted.item())]



csv_path_test = r'C:\DF_test.csv' #TODO: replace with the path to the CSV on your computer
df_test = pd.read_csv(csv_path_test)

# Apply the analysis functions to the 'caption' column
#print(predict_answer(df_test['Text'][10], df_test['question'][i]))

# Initialize an empty list to store data
data2= []

# Loop and process data
for i in range(572):
    # Simulate some data processing
    result = predict_answer(df_test['Text'][i], df_test['question'][i])
    words = df_test['Video.ID'][i]
    tt = df_test['Text'][i]
    real = df_test['answer'][i]
    
    # Append the result to the list as a dictionary
    data2.append({'code': words,'Text': tt,'Real': real, 'result': result})

print(data2)
data2 = pd.DataFrame(data2)
data2.to_csv('NN_output_02.csv', index=False)

# Create a DataFrame from the list of dictionaries
df = pd.DataFrame(data)

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
metrics_df.to_csv('NN_metrics.csv', index=False)
#filename = "kappa_svn.txt"

#with open(filename, 'w') as file:
  #  file.write(str(kappa))