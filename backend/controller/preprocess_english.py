import re

def preprocess_english(file_path):
    # Load stop words from english.txt
    with open(file_path, 'r', encoding='utf-8') as file:
        stop_words = set(file.read().splitlines())

    def preprocess_text(text):
        # Tokenize and remove unwanted words
        words = re.findall(r'\b\w+\b', text)
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)

    # Test the preprocess function
    sample_text = "This is a sample text containing unwanted words like actually, basically, and so on."
    clean_text = preprocess_text(sample_text)
    print("Cleaned Text:", clean_text)

    return stop_words

# Example usage:
if __name__ == "__main__":
    unwanted_words = preprocess_english('C:/Users/naman/OneDrive/Desktop/MINIPROJECT/MediEase/backend/data/english.txt')
    print("Unwanted Words:", unwanted_words)
