import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- NLTK Resource Download ---
# The first time you run this, you'll need to download these resources.
# You can uncomment these lines, run the script once, then comment them back.
# try:
#     stopwords.words('english')
# except LookupError:
#     print("Downloading NLTK stopwords...")
#     nltk.download('stopwords')
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     print("Downloading NLTK punkt tokenizer...")
#     nltk.download('punkt')
# try:
#     nltk.data.find('corpora/wordnet')
# except LookupError:
#     print("Downloading NLTK wordnet...")
#     nltk.download('wordnet')
# --------------------------------

# --- Text Preprocessing Function ---
def preprocess_text(text):
    """
    Cleans and preprocesses a single text string by:
    1. Lowercasing
    2. Removing non-alphabetic characters
    3. Tokenizing
    4. Removing stopwords
    5. Lemmatizing
    Returns a list of cleaned tokens.
    """
    if not isinstance(text, str):
        return []

    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # 1. Lowercasing and removing non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A).lower()

    # 2. Tokenizing
    tokens = word_tokenize(text)

    # 3. Removing stopwords and Lemmatizing
    cleaned_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words and len(token) > 1
    ]

    return cleaned_tokens

# --- Main Data Cleaning and Processing Function ---
def clean_and_process_dataset(input_path='data/hummus_recipes.csv'):
    """
    Loads the raw dataset, performs basic cleaning, and applies text
    preprocessing to relevant columns.
    Returns a cleaned and processed DataFrame.
    """
    print("Loading raw dataset...")
    df = pd.read_csv(input_path)

    # --- Basic Data Cleaning ---
    # Drop rows where essential fields like title or directions are missing
    df.dropna(subset=['title', 'directions', 'ingredients'], inplace=True)

    # Remove recipes with unrealistic values (e.g., 0 calories or duration)
    df = df[df['calories [cal]'] > 0]
    df = df[df['duration'] > 0]
    
    # Ensure numerical columns are of the correct type (if needed)
    # This dataset is fairly clean, but this is good practice
    # For example: df['calories [cal]'] = pd.to_numeric(df['calories [cal]'], errors='coerce')

    print("Basic cleaning complete. Starting text preprocessing...")

    # --- Text Preprocessing on Relevant Columns ---
    # We apply the function to each text column and store the result (a list of tokens)
    # in a new column.
    text_columns = ['title', 'directions', 'tags', 'ingredients']
    for col in text_columns:
        print(f"Processing column: {col}...")
        df[f'processed_{col}'] = df[col].apply(preprocess_text)

    print("✅ Preprocessing finished successfully!")
    return df


# --- Execution Block ---
# This part runs only when you execute the script directly (e.g., `python preprocess.py`)
if __name__ == '__main__':
    # Process the dataset
    processed_df = clean_and_process_dataset()

    # Define the output path for the processed file
    output_path = 'data/processed_recipes.csv'

    # Save the processed DataFrame to a new CSV file
    # We use index=False to avoid saving the DataFrame index as a column
    processed_df.to_csv(output_path, index=False)
    
    print(f"Processed data saved to '{output_path}'")
    print("\nSample of processed data:")
    # Display the new processed columns for verification
    print(processed_df[['title', 'processed_title', 'ingredients', 'processed_ingredients']].head())
