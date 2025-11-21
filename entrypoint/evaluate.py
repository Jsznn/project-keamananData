import pandas as pd
import yaml
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import json
import os

def load_config():
    with open("config/config.yml", "r") as f:
        return yaml.safe_load(f)

def evaluate():
    config = load_config()
    
    # Load processed data
    processed_path = config['data']['processed_path']
    print(f"Loading data from {processed_path}...")
    df = pd.read_csv(processed_path)
    
    # Handle missing values
    df['transformed_text'] = df['transformed_text'].fillna('')

    X = df['transformed_text']
    y = df['target']

    # Split data (must match training split)
    test_size = config['data']['test_size']
    random_state = config['data']['random_state']
    print(f"Splitting data (test_size={test_size}, random_state={random_state})...")
    _, X_test, _, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Load Vectorizer
    print("Loading vectorizer...")
    try:
        tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    except FileNotFoundError:
        print("Error: models/tfidf_vectorizer.pkl not found. Run train.py first.")
        return

    X_test_tfidf = tfidf.transform(X_test)

    # Evaluate Models
    models_to_eval = {
        "Naive Bayes": "models/naive_bayes.pkl",
        "Logistic Regression": "models/logistic_regression.pkl",
        "Linear SVM": "models/linear_svc.pkl"
    }

    results = []

    for name, path in models_to_eval.items():
        print(f"\nEvaluating {name}...")
        try:
            model = joblib.load(path)
        except FileNotFoundError:
            print(f"Warning: {path} not found. Skipping.")
            continue

        y_pred = model.predict(X_test_tfidf)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1
        })

        print(f"=== {name} Report ===")
        print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

    # Summary
    if results:
        print("\n=== Evaluation Summary ===")
        results_df = pd.DataFrame(results)
        results_df = pd.DataFrame(results)
        print(results_df)
        
        # Save metrics to JSON
        metrics_path = "models/metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nMetrics saved to {metrics_path}")

if __name__ == "__main__":
    evaluate()
