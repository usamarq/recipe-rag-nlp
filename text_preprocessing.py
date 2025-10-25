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
    Perform standard NLP cleaning steps:
    - Lowercasing
    - Keep only letters and spaces
    - Tokenization
    - Stopword removal
    - Lemmatization
    Returns a cleaned string (joined tokens).
    """
    if not isinstance(text, str):
        return ""

    # Lowercase + remove non-letter characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize
    cleaned_tokens = [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in stop_words and len(tok) > 1
    ]

    # Join back into a single string for easy TF-IDF or embedding use
    return " ".join(cleaned_tokens)


# -------------------------------------------------
# 2. Main preprocessing pipeline
# -------------------------------------------------
def text_preprocess_dataset(input_path: str, output_path: str):
    print("=== Loading cleaned dataset ===")
    df = pd.read_csv(input_path)
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
    df['num_ingredient_tokens'] = df['processed_ingredients'].apply(
        lambda x: len(x.split()) if isinstance(x, str) else 0
    )
    df['num_tag_tokens'] = df['processed_tags'].apply(
        lambda x: len(x.split()) if isinstance(x, str) else 0
    )

    # --- Summary report ---
    print("\n=== SUMMARY REPORT ===")
    new_columns = set(df.columns) - original_columns
    print(f"New columns added: {sorted(list(new_columns))}")
    print(f"Average ingredient tokens: {df['num_ingredient_tokens'].mean():.2f}")
    print(f"Average tag tokens: {df['num_tag_tokens'].mean():.2f}")
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
