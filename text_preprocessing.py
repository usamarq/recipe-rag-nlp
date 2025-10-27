# --------------------------------------------------------------
# text_preprocessing.py
# Task 2: Text preprocessing for Project 22 – Recipe RAG
# --------------------------------------------------------------

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Uncomment the first time you run this
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

# -------------------------------------------------
# 1. Preprocessing utilities
# -------------------------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text: str):
    """
    Cleans and tokenizes recipe text while preserving meaningful numeric values.
    
    Steps:
        • Lowercase
        • Keep letters, digits, and spaces
        • Merge numbers with nearby units (e.g. "500 g" → "500g")
        • Tokenize
        • Remove stopwords
        • Lemmatize
    Returns a list of cleaned tokens.
    """
    if not isinstance(text, str):
        return []

    # Keep only letters, digits, and spaces (remove punctuation)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text).lower()

    # Merge numbers and their units (e.g. 500 g → 500g)
    text = re.sub(
        r'(\d+)\s*(g|gram|ml|kg|tbsp|tsp|cup|cups|teaspoon|tablespoon|minute|min|hour|hr|cal|kcal)',
        r'\1\2',
        text
    )

    # Normalize multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text)

    # Lemmatize & remove stopwords
    cleaned = [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in stop_words and len(tok) > 1
    ]
    return cleaned


# -------------------------------------------------
# 2. Main preprocessing pipeline
# -------------------------------------------------
def text_preprocess_dataset(input_path: str, output_path: str):
    print("=== Loading cleaned dataset ===")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"Initial dataset shape: {df.shape}")

    # --- Define columns to preprocess ---
    text_cols = ['title', 'ingredients', 'directions', 'tags']
    existing_cols = [col for col in text_cols if col in df.columns]
    print(f"Text columns detected: {existing_cols}")

    # --- Track original columns ---
    original_columns = set(df.columns)

    # --- Apply preprocessing ---
    print("\n=== Starting text preprocessing ===")
    for col in existing_cols:
        print(f" → Processing column: {col}")
        df[f'processed_{col}'] = df[col].astype(str).apply(preprocess_text)

    # --- Generate quick statistics ---
    print("\n=== Generating statistics ===")
    stat_cols = ['processed_title', 'processed_ingredients', 'processed_directions', 'processed_tags']
    for col in stat_cols:
        if col in df.columns:
            df[f'num_{col}_tokens'] = df[col].apply(lambda x: len(x) if isinstance(x, list) else 0)
            print(f"Average tokens in {col}: {df[f'num_{col}_tokens'].mean():.2f}")

    # --- Show sample rows for verification ---
    print("\nSample processed text:")
    sample_cols = ['title', 'processed_title', 'ingredients', 'processed_ingredients']
    print(df[sample_cols].head(3))

    # --- Summary report ---
    print("\n=== SUMMARY REPORT ===")
    new_columns = set(df.columns) - original_columns
    print(f"New columns added: {sorted(list(new_columns))}")
    print(f"Final dataset shape: {df.shape}")

    # --- Save results ---
    df.to_csv(output_path, index=False)
    print(f"\n✅ Text preprocessing complete.")
    print(f"Saved processed dataset → {output_path}")

    return df


# -------------------------------------------------
# 3. Run when executed directly
# -------------------------------------------------
if __name__ == "__main__":
    text_preprocess_dataset(
        input_path="data/hummus_recipes_cleaned_basic.csv",
        output_path="data/hummus_recipes_preprocessed.csv"
    )
