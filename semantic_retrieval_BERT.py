import os
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ast # Import ast for literal_eval
import time # Optional: for timing

# Import preprocess_text function for query processing
try:
    from text_preprocessing import preprocess_text
except ImportError:
    print("Warning: text_preprocessing.py not found. Using basic fallback for query processing.")
    # Define a basic fallback
    import re
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    def preprocess_text(text: str):
        if not isinstance(text, str): return []
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text).lower()
        text = re.sub(r'(\d+)\s*(g|gram|ml|kg|tbsp|tsp|cup|cups|teaspoon|tablespoon|minute|min|hour|hr|cal|kcal)', r'\1\2', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = word_tokenize(text)
        cleaned = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words and len(tok) > 1]
        return cleaned


# -------------------------------------------------
# 1. Device check and model setup
# -------------------------------------------------
def setup_device_and_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """ Checks GPU availability and loads the Sentence-BERT model."""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("⚠️ No GPU detected, using CPU")

    model = SentenceTransformer(model_name, device=device)
    print(f"Loaded model: {model_name} on {device.upper()}")
    return model, device


# -------------------------------------------------
# 2. Load dataset
# -------------------------------------------------
def load_dataset(input_path="data/hummus_recipes_preprocessed.csv"):
    """Loads the preprocessed dataset."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ Dataset not found at: {input_path}")
    # Specify columns needed to reduce memory usage if possible
    # Add 'recipe_id' and display columns if needed later, or load metadata separately
    cols_needed_for_embedding = ["processed_title", "processed_ingredients", "processed_tags", "processed_directions"]
    try:
         # Check which columns actually exist
         all_cols = pd.read_csv(input_path, nrows=0).columns
         load_cols = [c for c in cols_needed_for_embedding if c in all_cols]
         if not load_cols:
              raise ValueError("None of the required text columns found for embedding.")
         print(f"Loading columns for embedding: {load_cols}")
         df = pd.read_csv(input_path, usecols=load_cols, low_memory=True)
         print(f"✅ Dataset loaded for embedding: {df.shape}")
         return df, load_cols # Return available columns too
    except Exception as e:
         print(f"Error loading dataset: {e}")
         raise


# -------------------------------------------------
# 3. Build combined text field (Corrected)
# -------------------------------------------------
def build_combined_text_for_embedding(df, available_cols):
    """
    Parses string representations of lists and combines actual tokens
    into a single string per recipe for embedding.
    """
    print(f"Using columns for embeddings: {available_cols}")
    start_time = time.time()

    def combine_and_clean_row(row):
        all_tokens = []
        for col in available_cols:
            text_repr = row.get(col, "[]") # Default to empty list string
            # Check if it looks like a list string before trying to parse
            if isinstance(text_repr, str) and text_repr.startswith('[') and text_repr.endswith(']'):
                try:
                    # Parse the string back into a list
                    actual_tokens = ast.literal_eval(text_repr)
                    if isinstance(actual_tokens, list):
                        all_tokens.extend(actual_tokens) # Add the tokens
                except (ValueError, SyntaxError):
                     # Handle cases where parsing fails (e.g., malformed string)
                     # Optionally split the raw string as a fallback
                     # print(f"Warning: Could not parse list string in row {row.name}, col {col}. Splitting raw string.")
                     all_tokens.extend(text_repr.strip("[]").replace("'", "").replace('"', '').split(', '))
            elif isinstance(text_repr, str): # Fallback if it's not list-like string
                 all_tokens.extend(text_repr.split())
        # Join the collected *actual* tokens with spaces
        return " ".join(all_tokens)

    # Apply the function row-wise to create the text for embedding
    df["text_for_embedding"] = df.apply(combine_and_clean_row, axis=1)

    end_time = time.time()
    print(f"✅ Combined text field for embedding created in {end_time - start_time:.2f} seconds.")
    # Display a sample to verify
    print("\nSample of text prepared for embedding:")
    print(df["text_for_embedding"].head())
    return df


# -------------------------------------------------
# 4. Create embeddings and save
# -------------------------------------------------
def create_and_save_embeddings(df, model, device, output_dir="data", batch_size=96):
    """ Generates embeddings using Sentence-BERT and saves them as .npy. """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "recipe_embeddings.npy")

    # Check if embeddings file already exists
    if os.path.exists(output_path):
        print(f"⚠️ Embeddings file already exists at {output_path}. Skipping generation.")
        print("   Delete the file if you want to regenerate embeddings.")
        try:
             embeddings = np.load(output_path)
             print(f"   Loaded existing embeddings. Shape: {embeddings.shape}")
             # Optional: Check if shape matches current dataframe size
             if len(df) != embeddings.shape[0]:
                  print(f"   Warning: Existing embeddings count ({embeddings.shape[0]}) doesn't match DataFrame size ({len(df)}). Consider regenerating.")
             return embeddings
        except Exception as e:
             print(f"   Error loading existing embeddings: {e}. Proceeding to generate new ones.")


    print(f"\n⚙️ Generating embeddings on {device.upper()} (batch size={batch_size}) ...")
    # Use the correctly combined text
    texts = df["text_for_embedding"].tolist()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True # Important for cosine similarity
    )

    try:
        np.save(output_path, embeddings)
        print(f"✅ Embeddings created and saved → {output_path}")
        print(f"Shape: {embeddings.shape}")
    except Exception as e:
        print(f"Error saving embeddings: {e}")

    return embeddings


# -------------------------------------------------
# 5. Semantic search
# -------------------------------------------------
def semantic_search(query, model, df_metadata, embeddings, top_k=5):
    """
    Performs semantic retrieval using embeddings and displays results from metadata.
    Expects df_metadata containing 'recipe_id', 'title', etc. indexed same as embeddings.
    """
    if model is None or embeddings is None or df_metadata is None:
         print("Missing components for semantic search.")
         return pd.DataFrame() # Return empty DataFrame

    print(f"\n🔍 Query: {query}")
    # Preprocess query and join tokens (consistent with how embeddings were likely created)
    cleaned_query_text = " ".join(preprocess_text(query))
    if not cleaned_query_text:
         print("Query is empty after preprocessing.")
         return pd.DataFrame()
    print(f"   Processed query text: '{cleaned_query_text}'")

    # Encode query
    query_emb = model.encode([cleaned_query_text], normalize_embeddings=True, show_progress_bar=False)

    # Calculate scores
    scores = cosine_similarity(query_emb, embeddings)[0]
    top_indices = np.argsort(-scores)[:top_k]

    # Get results from metadata DataFrame
    # Ensure df_metadata index matches embedding indices (0 to N-1)
    results = df_metadata.iloc[top_indices][["recipe_id", "title", "calories_cal", "totalfat_g", "protein_g"]].copy()
    results["similarity"] = scores[top_indices]

    print("\nTop semantic results:\n")
    # Display results nicely
    for i, (_, row) in enumerate(results.iterrows()):
         print(f"{i+1}. [ID: {row.get('recipe_id','N/A')}] {row.get('title','N/A')} "
               f"({row.get('calories_cal','?')} cal) - Score: {row.get('similarity', 0):.4f}")

    return results


# -------------------------------------------------
# 6. Main execution flow
# -------------------------------------------------
def main():
    start_total_time = time.time()
    model, device = setup_device_and_model()

    try:
        # Load only text columns for embedding generation
        df_for_embed, available_cols = load_dataset()
        # Create the combined text suitable for the model
        df_for_embed = build_combined_text_for_embedding(df_for_embed, available_cols)
        # Generate or load embeddings
        embeddings = create_and_save_embeddings(df_for_embed, model, device, batch_size=96)
        # Clear the large text DataFrame from memory after embedding
        del df_for_embed

        # --- Load Metadata Separately for Searching ---
        # Assumes the full preprocessed CSV contains metadata columns
        metadata_cols = ["recipe_id", "title", "calories_cal", "totalfat_g", "protein_g"]
        print("\nLoading metadata for search results...")
        try:
             # Check columns exist
             input_path="data/hummus_recipes_preprocessed.csv"
             all_cols = pd.read_csv(input_path, nrows=0).columns
             load_meta_cols = [c for c in metadata_cols if c in all_cols]
             if not load_meta_cols: raise ValueError("No metadata columns found.")
             df_meta = pd.read_csv(input_path, usecols=load_meta_cols)
             print(f"Metadata loaded: {df_meta.shape}")
             # Ensure metadata index matches embedding order (usually default 0..N-1)
             df_meta = df_meta.reset_index(drop=True)
        except Exception as e:
             print(f"Error loading metadata: {e}")
             df_meta = None # Set to None if loading fails

        # --- Run Test Queries ---
        if embeddings is not None and df_meta is not None:
             # Check if embedding shape matches metadata length
             if embeddings.shape[0] != len(df_meta):
                  print("\n!!! WARNING: Embeddings shape does not match metadata length. Results might be incorrect. !!!")
                  print(f"Embeddings shape: {embeddings.shape[0]}, Metadata length: {len(df_meta)}")

             test_queries = [
                 "low carb chicken meal",
                 "high protein vegan breakfast",
                 "sugar free dessert",
                 "quick healthy pasta dinner"
             ]
             for q in test_queries:
                 semantic_search(q, model, df_meta, embeddings, top_k=5)
        else:
             print("\nCannot run test searches due to missing embeddings or metadata.")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    end_total_time = time.time()
    print(f"\nTotal script execution time: {end_total_time - start_total_time:.2f} seconds.")


if __name__ == "__main__":
    main()
