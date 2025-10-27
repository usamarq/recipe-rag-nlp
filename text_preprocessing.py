import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import ast # Added ast here as well, though not used in preprocess_text

# # --- NLTK Resource Download Check/Setup ---
# # It's good practice to ensure resources are available
# try:
#     stopwords.words('english')
# except LookupError:
#     print("NLTK stopwords not found. Downloading...")
#     nltk.download('stopwords', quiet=True)
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     print("NLTK punkt tokenizer not found. Downloading...")
#     nltk.download('punkt', quiet=True)
# try:
#     nltk.data.find('corpora/wordnet')
# except LookupError:
#     print("NLTK wordnet not found. Downloading...")
#     nltk.download('wordnet', quiet=True)
# # ----------------------------------------------


# -------------------------------------------------
# 1. Preprocessing utilities
# -------------------------------------------------
# Initialize outside the function for efficiency
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
    """Loads a CSV, applies preprocess_text to specified columns, saves the result."""
    print("=== Loading cleaned dataset ===")
    try:
        df = pd.read_csv(input_path, low_memory=False)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return None
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
        # Apply preprocess_text. Convert NaN to empty string first.
        df[f'processed_{col}'] = df[col].fillna('').astype(str).apply(preprocess_text)
        # Convert list back to string representation for CSV saving
        # This is necessary because CSV can't store lists directly
        df[f'processed_{col}'] = df[f'processed_{col}'].apply(str)


    # --- Generate quick statistics ---
    print("\n=== Generating statistics ===")
    stat_cols = ['processed_title', 'processed_ingredients', 'processed_directions', 'processed_tags']
    for col in stat_cols:
         if col in df.columns:
            # Temporarily convert back to list to get length
            df[f'num_{col}_tokens'] = df[col].apply(lambda x: len(ast.literal_eval(x)) if isinstance(x, str) and x.startswith('[') else 0)
            print(f"Average tokens in {col}: {df[f'num_{col}_tokens'].mean():.2f}")


    # --- Show sample rows for verification ---
    print("\nSample processed text:")
    sample_cols = ['title', 'processed_title', 'ingredients', 'processed_ingredients']
    # Ensure columns exist before displaying
    display_sample_cols = [c for c in sample_cols if c in df.columns]
    print(df[display_sample_cols].head(3))

    # --- Summary report ---
    print("\n=== SUMMARY REPORT ===")
    new_columns = set(df.columns) - original_columns
    print(f"New columns added: {sorted(list(new_columns))}")
    print(f"Final dataset shape: {df.shape}")

    # --- Save results ---
    try:
        df.to_csv(output_path, index=False)
        print(f"\n✅ Text preprocessing complete.")
        print(f"Saved processed dataset → {output_path}")
    except Exception as e:
        print(f"Error saving processed file: {e}")

    return df


# -------------------------------------------------
# 3. Run when executed directly
# -------------------------------------------------
if __name__ == "__main__":
    text_preprocess_dataset(
        input_path="data/hummus_recipes_cleaned_basic.csv", # Make sure this input exists
        output_path="data/hummus_recipes_preprocessed.csv"
    )
