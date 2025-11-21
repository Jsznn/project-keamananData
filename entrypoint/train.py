import pandas as pd
import yaml
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def load_config():
    with open("config/config.yml", "r") as f:
        return yaml.safe_load(f)

def train():
    config = load_config()
    
    # Load processed data
    processed_path = config['data']['processed_path']
    print(f"Loading data from {processed_path}...")
    df = pd.read_csv(processed_path)
    
    # Handle missing values in transformed_text (if any)
    df['transformed_text'] = df['transformed_text'].fillna('')

    X = df['transformed_text']
    y = df['target']

    # Split data
    test_size = config['data']['test_size']
    random_state = config['data']['random_state']
    print(f"Splitting data (test_size={test_size}, random_state={random_state})...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # TF-IDF Vectorization
    print("Vectorizing text...")
    tfidf_config = config['feature_extraction']['tfidf']
    tfidf = TfidfVectorizer(stop_words=tfidf_config['stop_words'], min_df=tfidf_config['min_df'])
    X_train_tfidf = tfidf.fit_transform(X_train)
    
    # Save vectorizer
    os.makedirs("models", exist_ok=True)
    joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
    print("Saved tfidf_vectorizer.pkl")

    # SMOTE
    print("Applying SMOTE...")
    smote_config = config['smote']
    smote = SMOTE(random_state=smote_config['random_state'], k_neighbors=smote_config['k_neighbors'])
    X_train_smote, y_train_smote = smote.fit_resample(X_train_tfidf, y_train)
    print(f"Training data shape after SMOTE: {X_train_smote.shape}")

    # Train Models
    models_config = config['models']
    
    # Naive Bayes
    if 'naive_bayes' in models_config:
        print("Training Naive Bayes...")
        nb = MultinomialNB()
        nb.fit(X_train_smote, y_train_smote)
        joblib.dump(nb, "models/naive_bayes.pkl")
        print("Saved naive_bayes.pkl")

    # Logistic Regression
    if 'logistic_regression' in models_config:
        print("Training Logistic Regression...")
        lr_config = models_config['logistic_regression']
        lr = LogisticRegression(max_iter=lr_config['max_iter'])
        lr.fit(X_train_smote, y_train_smote)
        joblib.dump(lr, "models/logistic_regression.pkl")
        print("Saved logistic_regression.pkl")

    # Linear SVC
    if 'linear_svc' in models_config:
        print("Training Linear SVC...")
        svc = LinearSVC()
        svc.fit(X_train_smote, y_train_smote)
        joblib.dump(svc, "models/linear_svc.pkl")
        print("Saved linear_svc.pkl")

    print("Training complete.")

if __name__ == "__main__":
    train()
