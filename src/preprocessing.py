import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from sklearn.preprocessing import LabelEncoder
import yaml
import os

# Load config
def load_config():
    with open("config/config.yml", "r") as f:
        return yaml.safe_load(f)

# Initialize PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    # Transform the text to lowercase
    text = text.lower()

    # Tokenization using NLTK
    text = nltk.word_tokenize(text)

    # Removing special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # Removing stop words and punctuation
    text = y[:]
    y.clear()

    # Loop through the tokens and remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # Stemming using Porter Stemmer
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    # Join the processed tokens back into a single string
    return " ".join(y)

def preprocess():
    config = load_config()
    
    # NLTK Downloads
    print("Downloading NLTK data...")
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')

    # Load data
    raw_path = config['data']['raw_path']
    encoding = config['data']['encoding']
    print(f"Loading data from {raw_path}...")
    df = pd.read_csv(raw_path, encoding=encoding)

    # Drop columns
    drop_cols = config['preprocessing']['drop_columns']
    print(f"Dropping columns: {drop_cols}")
    df.drop(columns=drop_cols, inplace=True)

    # Rename columns
    rename_map = config['preprocessing']['rename_columns']
    print(f"Renaming columns: {rename_map}")
    df.rename(columns=rename_map, inplace=True)

    # Encode target
    print("Encoding target column...")
    encoder = LabelEncoder()
    df['target'] = encoder.fit_transform(df['target'])

    # Remove duplicates
    print("Removing duplicates...")
    df = df.drop_duplicates(keep='first')

    # Feature Engineering: num_characters
    print("Calculating num_characters...")
    df['num_characters'] = df['text'].apply(len)

    # Text Transformation
    print("Transforming text...")
    df['transformed_text'] = df['text'].apply(transform_text)

    # Save processed data
    processed_path = config['data']['processed_path']
    # Ensure directory exists
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    
    df.to_csv(processed_path, index=False)
    print(f"Processed data saved to {processed_path}")

if __name__ == "__main__":
    preprocess()
