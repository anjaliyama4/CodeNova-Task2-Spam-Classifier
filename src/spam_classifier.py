
from pathlib import Path
import re
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "SMSSpamCollection"
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = PROJECT_DIR / "models"

# Create output folders if needed
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Spam Classifier project setup loaded successfully!")

def load_dataset():
    """Load and validate the SMS Spam Collection dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}\n"
            "Please place the SMSSpamCollection file inside the data folder."
        )

    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8",
        on_bad_lines="error",
    )

    df["label"] = df["label"].str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()

    # Keep only valid labels and non-empty messages
    df = df[df["label"].isin(["ham", "spam"])]
    df = df[df["message"] != ""].copy()
    df = df.drop_duplicates(subset=["label", "message"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("The dataset is empty or has an unexpected format.")

    return df

def preprocess_text(text):
    """Lowercase, tokenize, and remove English stop words."""
    tokens = re.findall(r"\b[a-zA-Z]+\b", str(text).lower())

    filtered_tokens = [
        word for word in tokens
        if word not in ENGLISH_STOP_WORDS
    ]

    return filtered_tokens

def build_pipeline(model):
    """Create a text-processing and classification pipeline."""
    vectorizer = TfidfVectorizer(
        tokenizer=preprocess_text,
        token_pattern=None,
        lowercase=False,
        max_features=10000,
        ngram_range=(1, 2),
    )

    return Pipeline([
        ("tfidf", vectorizer),
        ("classifier", model),
    ])


# Step 2: Define the two models
MODELS = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight=None,
        random_state=42,
    ),
}

def compare_models(X_train, y_train, X_val, y_val):
    """Train both models and compare their spam detection metrics."""
    results = []
    trained_models = {}

    for name, model in MODELS.items():
        print(f"\nTraining {name}...")

        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_val)

        spam_precision = precision_score(
            y_val, predictions, pos_label="spam", zero_division=0
        )
        spam_recall = recall_score(
            y_val, predictions, pos_label="spam", zero_division=0
        )
        spam_f1 = f1_score(
            y_val, predictions, pos_label="spam", zero_division=0
        )
        accuracy = accuracy_score(y_val, predictions)

        results.append({
            "Model": name,
            "Spam Precision": spam_precision,
            "Spam Recall": spam_recall,
            "Spam F1-score": spam_f1,
            "Accuracy": accuracy,
        })

        trained_models[name] = pipeline

        print(f"Spam precision: {spam_precision:.4f}")
        print(f"Spam recall:    {spam_recall:.4f}")
        print(f"Spam F1-score:  {spam_f1:.4f}")
        print(f"Accuracy:       {accuracy:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "validation_model_comparison.csv", index=False)

    print("\nValidation model comparison:")
    print(results_df.to_string(index=False))

    return trained_models, results_df

def evaluate_model(model, X_test, y_test):
    """Evaluate the selected model on the untouched test set."""
    predictions = model.predict(X_test)

    precision = precision_score(
        y_test, predictions, pos_label="spam", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, pos_label="spam", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, pos_label="spam", zero_division=0
    )
    accuracy = accuracy_score(y_test, predictions)

    print("\nFINAL TEST RESULTS")
    print(f"Spam precision: {precision:.4f}")
    print(f"Spam recall:    {recall:.4f}")
    print(f"Spam F1-score:  {f1:.4f}")
    print(f"Accuracy:       {accuracy:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, labels=["ham", "spam"], zero_division=0
    ))

    # Save test metrics
    metrics = pd.DataFrame([{
        "Spam Precision": precision,
        "Spam Recall": recall,
        "Spam F1-score": f1,
        "Accuracy": accuracy,
    }])
    metrics.to_csv(OUTPUT_DIR / "test_metrics.csv", index=False)

    # Save the confusion matrix image
    cm = confusion_matrix(y_test, predictions, labels=["ham", "spam"])
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["HAM", "SPAM"],
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("Spam Classifier - Test Set Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

    print("\nSaved test metrics and confusion matrix in the outputs folder.")

def predict_message(model, message):
    """Predict whether a custom SMS is SPAM or HAM."""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Please enter a non-empty message.")

    prediction = model.predict([message])[0]
    return prediction.upper()



# Your existing code continues below
if __name__ == "__main__":
    df = load_dataset()
    # Keep the rest of your existing code here



if __name__ == "__main__":
    df = load_dataset()
    
    # Separate messages (X) from their labels (y)
    X = df["message"]
    y = df["label"]

    # Split off 20% for final testing
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Split the remaining data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.20,
        random_state=42,
        stratify=y_trainval,
    )

    print("\nDataset split:")
    print(f"Training messages: {len(X_train)}")
    print(f"Validation messages: {len(X_val)}")
    print(f"Testing messages: {len(X_test)}")
    
    # Train and compare both models using the validation set
    trained_models, results_df = compare_models(
        X_train, y_train, X_val, y_val
    )
    
    # Select the model with the highest validation spam precision.
    # Use spam F1-score as the tie-breaker.
    best_row = results_df.sort_values(
        by=["Spam Precision", "Spam F1-score"],
        ascending=False,
    ).iloc[0]

    best_name = best_row["Model"]
    best_model = trained_models[best_name]

    print(f"\nSelected model: {best_name}")

    # Evaluate only once on the untouched test set
    evaluate_model(best_model, X_test, y_test)

    # Save the trained model and its TF-IDF vectorizer together
    model_path = MODEL_DIR / "best_spam_pipeline.joblib"
    joblib.dump(best_model, model_path)
    print(f"Saved reusable model to: {model_path}")
    
    print("\nCustom SMS prediction")
    print("Enter a message to classify, or type 'quit' to exit.")

    while True:
        message = input("\nYour message: ").strip()

        if message.lower() == "quit":
            print("Prediction session ended.")
            break

        if not message:
            print("Please enter a message.")
            continue

        try:
            result = predict_message(best_model, message)
            print(f"Prediction: {result}")
        except ValueError as error:
            print(error)





    print("\nDataset loaded successfully!")
    print(f"Total messages: {len(df)}")
    print("\nLabel counts:")
    print(df["label"].value_counts())
    print("\nFirst five messages:")
    print(df.head().to_string(index=False))
    
    sample_message = "Congratulations! You have WON a FREE prize today."
    print("\nPreprocessing test:")
    print("Original:", sample_message)
    print("Cleaned tokens:", preprocess_text(sample_message))


