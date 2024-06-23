import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

def preprocess_english(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stop_words = set(file.read().splitlines())
    return stop_words

def preprocess_text(text, unwanted_words):
    # Lowercase the text
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords and unwanted words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token.lower() not in unwanted_words]

    # Join tokens back into a single string
    clean_text = ' '.join(tokens)
    return clean_text

def preprocess_csv(file_path, unwanted_words):
    df = pd.read_csv(file_path)
    df['clean_summary'] = df['summary'].apply(lambda x: preprocess_text(x, unwanted_words))
    return df

# Example usage:
if __name__ == "__main__":
    # Preprocess english.txt to get unwanted words
    unwanted_words = preprocess_english('C:/Users/naman/OneDrive/Desktop/MINIPROJECT/MediEase/backend/data/english.txt')
    
    # Preprocess summary data using unwanted_words
    df_cleaned = preprocess_csv('C:/Users/naman/OneDrive/Desktop/MINIPROJECT/MediEase/backend/data/your_summary_data.csv', unwanted_words)
    
    # Print the cleaned summaries
    print(df_cleaned['clean_summary'].head())
