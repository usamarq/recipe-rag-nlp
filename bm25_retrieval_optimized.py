import os
import re
import pickle
import pandas as pd
from rank_bm25 import BM25Okapi
import numpy as np
import time
import ast # For literal_eval

# --- Import the preprocessing function ---
# Assumes text_preprocessing.py is in the same directory or accessible via PYTHONPATH
try:
    from text_preprocessing import preprocess_text
except ImportError:
    print("Error: Could not import preprocess_text from text_preprocessing.py.")
    print("Please ensure text_preprocessing.py is in the correct location.")
    # Define a basic fallback if import fails, though it won't match exactly
    def preprocess_text(text: str):
        if not isinstance(text, str): return []
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text).lower()
        return text.split()
# -----------------------------------------


# --- Stage 1: Index Building ---

def build_index_and_metadata(data_path="data/hummus_recipes_preprocessed.csv",
                             index_path="data/bm25_index.pkl",
                             metadata_path="data/recipe_metadata.pkl"):
    """Loads preprocessed data, builds BM25 index and metadata map, saves them."""
    print(f"--- Starting Index Building Stage ---")
    start_time = time.time()
    print(f"Loading full data for indexing from: {data_path}")
    # Define columns needed for indexing and metadata
    text_cols = ["processed_title", "processed_ingredients", "processed_tags", "processed_directions"]
    metadata_cols = ["recipe_id", "title", "calories_cal", "totalfat_g", "protein_g"] # Add more metadata if needed
    try:
        # Read columns first to identify available ones
        all_cols = pd.read_csv(data_path, nrows=0).columns
        available_text_cols = [c for c in text_cols if c in all_cols]
        cols_for_metadata = [c for c in metadata_cols if c in all_cols]
        cols_to_load = list(set(cols_for_metadata + available_text_cols)) # Ensure no duplicates

        # Check for essential columns
        if "recipe_id" not in cols_to_load or "title" not in cols_to_load:
             raise ValueError("Missing essential 'recipe_id' or 'title' column in source file.")

        df = pd.read_csv(data_path, usecols=cols_to_load, low_memory=True)
        print(f"Loaded {len(df)} recipes with columns: {list(df.columns)}")

    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return None, None
    except ValueError as ve:
         print(f"Error reading required columns: {ve}")
         return None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

    # --- Build Tokenized Corpus using ast.literal_eval ---
    print(f"Tokenizing documents using columns: {available_text_cols}...")
    tokenized_docs = []
    processed_count = 0
    error_count = 0
    for index, row in df.iterrows():
        combined_tokens = []
        for col in available_text_cols:
            text_repr = row.get(col, "[]") # Default to empty list string representation
            if isinstance(text_repr, str) and text_repr.startswith('[') and text_repr.endswith(']'):
                try:
                    actual_tokens = ast.literal_eval(text_repr)
                    if isinstance(actual_tokens, list):
                        combined_tokens.extend(actual_tokens)
                except (ValueError, SyntaxError):
                    # Handle cases where the string is not a valid list representation
                    error_count += 1
                    # Optionally add fallback logic, e.g., text_repr.split()
            elif isinstance(text_repr, str): # Handle plain strings (less likely)
                 combined_tokens.extend(text_repr.split())
        tokenized_docs.append(combined_tokens)
        processed_count += 1
        if processed_count % 50000 == 0: # Progress indicator
             print(f"   ...tokenized {processed_count} documents")

    if error_count > 0:
        print(f"Warning: Encountered {error_count} errors parsing list strings during tokenization.")
    if not tokenized_docs:
        print("Error: No documents found after tokenization.")
        return None, None
    print(f"Finished tokenizing {len(tokenized_docs)} documents.")

    # --- Build BM25 Index ---
    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_docs)
    del tokenized_docs # Free up memory

    # --- Create Metadata Map (Index -> Metadata) ---
    print("Creating metadata map...")
    metadata_map = {}
    for idx, row in df[cols_for_metadata].iterrows():
        metadata_map[idx] = row.to_dict()
    del df # Free up DataFrame memory

    # --- Save Index and Metadata ---
    try:
        print(f"Saving BM25 index to: {index_path}")
        with open(index_path, "wb") as f: pickle.dump(bm25, f)
        print(f"Saving metadata map to: {metadata_path}")
        with open(metadata_path, "wb") as f: pickle.dump(metadata_map, f)
        print("Index and metadata saved successfully.")
    except Exception as e:
        print(f"Error saving files: {e}")
        return None, None

    end_time = time.time()
    print(f"--- Index Building complete in {end_time - start_time:.2f} seconds ---")
    return bm25, metadata_map

# --- Stage 2: Loading and Searching ---

def load_index_and_metadata(index_path="data/bm25_index.pkl",
                             metadata_path="data/recipe_metadata.pkl"):
    """Loads the pre-built BM25 index and metadata map."""
    print(f"--- Starting Search Stage ---")
    start_time = time.time()
    bm25 = None
    metadata_map = None
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        print("Error: Index or metadata file not found.")
        return None, None # Return immediately if files are missing

    try:
        print(f"Loading BM25 index from: {index_path}")
        with open(index_path, "rb") as f: bm25 = pickle.load(f)
        print(f"Loading metadata map from: {metadata_path}")
        with open(metadata_path, "rb") as f: metadata_map = pickle.load(f)
        # Basic validation
        if not hasattr(bm25, 'get_scores') or not isinstance(metadata_map, dict):
            raise ValueError("Loaded files seem invalid.")
        print("Index and metadata loaded successfully.")
    except Exception as e:
        print(f"Error loading files: {e}")
        return None, None
    end_time = time.time()
    print(f"--- Loading complete in {end_time - start_time:.2f} seconds ---")
    return bm25, metadata_map

def preprocess_query(query: str):
    """Applies preprocessing using the imported function."""
    return preprocess_text(query) # Use imported function

def search_bm25_optimized(query: str, bm25, metadata_map, top_k=5):
    """Performs BM25 search using loaded index and metadata."""
    if bm25 is None or metadata_map is None:
        print("BM25 model or metadata map not available.")
        return []

    print(f"\nSearching for: '{query}'")
    start_time = time.time()
    tokens = preprocess_query(query)
    print(f"   Tokens: {tokens}")
    if not tokens:
        print("Query has no valid tokens.")
        return []

    try:
        # Check index validity
        if not hasattr(bm25, 'doc_freqs') or not bm25.doc_freqs:
             print("Error: BM25 index appears empty or corrupted. Please rebuild.")
             return []
        scores = bm25.get_scores(tokens)
    except ValueError as ve:
         # Handle potential ValueError if query token is not in corpus
         print(f"Warning: {ve}. Some query terms might not be in the index.")
         # Attempt to filter query tokens (simple approach)
         known_tokens = [tok for tok in tokens if tok in bm25.idf]
         if not known_tokens:
              print("No known query tokens found in the index.")
              return []
         print(f"   Searching with known tokens: {known_tokens}")
         try:
              scores = bm25.get_scores(known_tokens)
         except Exception as e_inner:
              print(f"Error getting scores even with known tokens: {e_inner}")
              return []
    except Exception as e:
        print(f"Unexpected error getting scores: {e}.")
        return []


    # Proceed if scores were successfully calculated
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        # Check if index is valid for metadata_map (it should be if index isn't corrupted)
        if idx in metadata_map:
            score = scores[idx]
            if score > 1e-6: # Use a small threshold instead of > 0 for floating point
                doc_metadata = metadata_map[idx]
                results.append({
                    'index': idx,
                    'recipe_id': doc_metadata.get('recipe_id', 'N/A'),
                    'title': doc_metadata.get('title', 'N/A'),
                    'calories_cal': doc_metadata.get('calories_cal', '?'),
                    'totalfat_g': doc_metadata.get('totalfat_g', '?'),
                    'protein_g': doc_metadata.get('protein_g', '?'),
                    'bm25_score': score
                })
        else:
            print(f"Warning: Index {idx} from BM25 not found in metadata_map.")


    end_time = time.time()
    print(f"\nTop {len(results)} results found in {end_time - start_time:.4f} seconds:")
    for rank, res in enumerate(results, 1):
        print(f"{rank}. [ID: {res['recipe_id']}] {res['title']} "
              f"({res['calories_cal']} cal) - Score: {res['bm25_score']:.4f}")

    return results # Return list of dictionaries

# --- Main Execution Block ---
if __name__ == "__main__":
    data_path = "data/hummus_recipes_preprocessed.csv"
    index_cache_path = "data/bm25_index.pkl"
    metadata_cache_path = "data/recipe_metadata.pkl"
    FORCE_REBUILD = False # Set to True to force rebuilding the index and metadata

    bm25_model = None
    metadata = None

    # Try loading first unless forcing rebuild
    if not FORCE_REBUILD and os.path.exists(index_cache_path) and os.path.exists(metadata_cache_path):
        bm25_model, metadata = load_index_and_metadata(index_cache_path, metadata_cache_path)

    # If loading failed or rebuild forced, build them
    if bm25_model is None or metadata is None:
        if not FORCE_REBUILD: print("\nCache files not found or invalid.")
        print("Attempting to build index and metadata...")
        # Delete potentially corrupted files before rebuilding
        if os.path.exists(index_cache_path): os.remove(index_cache_path)
        if os.path.exists(metadata_cache_path): os.remove(metadata_cache_path)

        bm25_model, metadata = build_index_and_metadata(data_path, index_cache_path, metadata_cache_path)

    # Perform searches if model and metadata are ready
    if bm25_model and metadata:
        print("\n--- Running Example Searches ---")
        print("-" * 30)
        search_results_1 = search_bm25_optimized(
            "low fat chicken under 500 calories",
            bm25_model, metadata, top_k=5
        )
        print("-" * 30)
        search_results_2 = search_bm25_optimized(
            "high protein vegan salad",
            bm25_model, metadata, top_k=5
        )
        print("-" * 30)
        search_results_3 = search_bm25_optimized(
            "gluten-free pasta with tomato",
            bm25_model, metadata, top_k=5
        )
        print("-" * 30)
    else:
        print("Failed to initialize BM25 model and metadata. Cannot perform search.")

