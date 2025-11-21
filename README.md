# Email Spam Detection Project

This project implements a machine learning pipeline for detecting spam emails. It includes data preprocessing, model training with handling for imbalanced data (SMOTE), model evaluation, and a Streamlit web application for real-time inference.

## 🚀 Features

- **Data Preprocessing**: Text cleaning, tokenization, stopword removal, and stemming.
- **Model Training**: Trains Naive Bayes, Logistic Regression, and Linear SVM models.
- **Imbalanced Data Handling**: Uses SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.
- **Evaluation**: Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1-Score) saved to JSON.
- **Web App**: Interactive Streamlit dashboard for spam detection and model performance visualization.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Jsznn/project-keamananData.git
    cd project-keamananData
    ```

2.  **Create and activate a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ Usage

### 1. Data Preprocessing
Clean and transform the raw dataset (`data/spam.csv`). This will generate `data/processed_spam.csv`.
```bash
python src/preprocessing.py
```

### 2. Model Training
Train the models and save them to the `models/` directory. This script also saves the TF-IDF vectorizer.
```bash
python entrypoint/train.py
```

### 3. Model Evaluation
Evaluate the trained models on the test set and generate a performance report. Metrics are saved to `models/metrics.json`.
```bash
python entrypoint/evaluate.py
```

### 4. Run Streamlit App
Launch the web application to test the models with your own text and view evaluation metrics.
```bash
streamlit run app.py
```

## 📂 Project Structure

- `config/`: Configuration files (e.g., `config.yml` for file paths and model hyperparameters).
- `data/`: Storage for raw (`spam.csv`) and processed (`processed_spam.csv`) datasets.
- `entrypoint/`: Scripts for training (`train.py`) and evaluation (`evaluate.py`).
- `models/`: Saved model artifacts (`.pkl`) and evaluation metrics (`metrics.json`).
- `notebooks/`: Jupyter notebooks for initial data exploration and experimentation.
- `src/`: Source code for core functionality (e.g., `preprocessing.py`).
- `app.py`: Streamlit application entry point.
- `requirements.txt`: List of Python dependencies.

## 📊 Models Implemented

- **Naive Bayes (MultinomialNB)**: A baseline probabilistic classifier suitable for text data.
- **Logistic Regression**: A robust linear model for binary classification.
- **Linear SVM**: A Support Vector Machine optimized for high-dimensional sparse data (like text).