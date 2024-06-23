# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import make_pipeline
# from sklearn.metrics import accuracy_score
# import re

# # Load stop words from english.txt
# with open('C:/Users/naman/OneDrive/Desktop/MINIPROJECT/MediEase/backend/data/english.txt', 'r',encoding='utf-8') as file:
#     stop_words = set(file.read().splitlines())

# def preprocess_text(text):
#     # Tokenize and remove unwanted words
#     words = re.findall(r'\b\w+\b', text)
#     filtered_words = [word for word in words if word.lower() not in stop_words]
#     return ' '.join(filtered_words)

# # Sample dataset
# data = {
#     'text': [
#         "This is a sample sentence",
#         "Another example text",
#         "Machine learning is fun",
#         "The quick brown fox jumps over the lazy dog",
#         "Text preprocessing is important"
#     ],
#     'label': [0, 1, 0, 1, 0]  # Sample labels (e.g., 0: Negative, 1: Positive)
# }

# # Create DataFrame
# df = pd.DataFrame(data)

# # Preprocess text data
# df['text'] = df['text'].apply(preprocess_text)

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# # Create a text classification pipeline
# model = make_pipeline(CountVectorizer(), MultinomialNB())

# # Train the model
# model.fit(X_train, y_train)

# # Predict on the test set
# y_pred = model.predict(X_test)

# Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy:.2f}')



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import re

# Load stop words from english.txt
with open('C:/Users/naman/OneDrive/Desktop/MINIPROJECT/MediEase/backend/data/english.txt', 'r', encoding='utf-8') as file:
    stop_words = set(file.read().splitlines())

def preprocess_text(text):
    # Tokenize and remove unwanted words
    words = re.findall(r'\b\w+\b', text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Load your actual dataset (replace with your data loading mechanism)
df = pd.read_csv('path_to_your_summary_data.csv')  # Replace with actual path

# Preprocess text data
df['clean_summary'] = df['summary'].apply(preprocess_text)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['clean_summary'], df['label'], test_size=0.2, random_state=42)

# Create a text classification pipeline
# Using CountVectorizer()
# model = make_pipeline(CountVectorizer(), MultinomialNB())

# Using TfidfVectorizer() (consider this for potentially better performance)
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
