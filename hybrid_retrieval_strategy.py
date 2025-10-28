import pandas as pd
import numpy as np
import pickle
import time
import torch
import os
import re
import ast # For literal_eval in filter_metadata if needed
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Import Functions from other scripts ---
try:
    # Function to load BM25 index and metadata
    # NOTE: Assuming your bm25 script is named 'bm25_retrieval_optimized.py' based on your input
    from bm25_retrieval_optimized import load_index_and_metadata as load_bm25_components
    # The actual BM25 *search* function needs to be adapted or imported if it handles filtering
    # from bm25_retrieval_optimized import search_bm25_optimized
    # Function to load the Sentence Transformer model
    # NOTE: Assuming your semantic script is named 'semantic_retrieval_BERT.py'
    from semantic_retrieval_BERT import setup_device_and_model
    # Import the common text preprocessing function
    from text_preprocessing import preprocess_text
    # Import the main query parser
    from query_parser import parse_query
except ImportError as e:
    print(f"Error importing functions from helper scripts: {e}")
    print("Please ensure text_preprocessing.py, bm25_retrieval_optimized.py, ")
    print("semantic_retrieval_BERT.py, and query_parser.py are accessible.")
    exit() # Exit if imports fail

# --- Configuration ---
BM25_INDEX_PATH = "data/bm25_index.pkl"
METADATA_PATH = "data/recipe_metadata.pkl"
EMBEDDINGS_PATH = "data/recipe_embeddings.npy"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Helper Functions ---

def min_max_scale_scores(results_list, score_key='score'):
    """Applies Min-Max scaling (0-1) to scores in a list of result dicts."""
    if not results_list: return results_list, 0.0, 1.0 # Return range if empty
    scores = [res.get(score_key, 0) for res in results_list]
    min_score = min(scores) if scores else 0
    max_score = max(scores) if scores else 1
    score_range = max_score - min_score
    for res in results_list:
        original_score = res.get(score_key, 0)
        if score_range > 1e-9: # Avoid division by zero
            scaled_score = (original_score - min_score) / score_range
        else:
            scaled_score = 0.5 # All scores were identical
        res['scaled_' + score_key] = scaled_score
    # Return list and the min/max used for scaling (optional, could be useful)
    return results_list #, min_score, max_score


def filter_metadata(metadata_map, filters):
    """
    Filters the metadata_map based on extracted filter conditions.
    Returns: A set of indices that satisfy ALL filters.
    """
    if not filters:
        return set(metadata_map.keys()) # Return all indices if no filters

    print(f"Applying filters: {filters}")
    allowed_indices = set()
    start_time = time.time()

    for index, meta in metadata_map.items():
        match_all = True
        for attribute, operator, value in filters:
            match_current = False
            # Handle potential missing metadata or metadata needing parsing (like tags)
            meta_value = meta.get(attribute)

            if meta_value is None: # Attribute missing
                match_all = False; break

            try:
                if attribute == 'tags':
                    # Assumes metadata_map stores 'processed_tags' correctly (list or parsable string)
                    tags_list = meta.get('processed_tags', []) # Need to ensure this key exists in metadata_map
                    if isinstance(tags_list, str) and tags_list.startswith('['):
                         try: tags_list = ast.literal_eval(tags_list)
                         except: tags_list = []
                    if not isinstance(tags_list, list): tags_list = [] # Ensure it's a list

                    if operator == 'contains' and value.lower() in [tag.lower() for tag in tags_list]:
                        match_current = True
                else: # Numerical Filters
                    meta_value_num = float(meta_value)
                    filter_value_num = float(value)
                    if operator == '<' and meta_value_num < filter_value_num: match_current = True
                    elif operator == '>' and meta_value_num > filter_value_num: match_current = True
                    elif operator == '=' and meta_value_num == filter_value_num: match_current = True
                    # Add <=, >= if needed in parser and here

            except (ValueError, TypeError): # Handle conversion errors
                match_all = False; break

            if not match_current:
                match_all = False; break

        if match_all:
            allowed_indices.add(index)

    end_time = time.time()
    print(f"Filtering complete: {len(allowed_indices)} recipes match. (Time: {end_time - start_time:.4f}s)")
    return allowed_indices

# --- Search Functions Adapted for Filtering ---

def search_bm25_filtered(text_query: str, bm25, metadata_map, allowed_indices: set, k: int):
    """Performs BM25 search ONLY on allowed indices."""
    if not allowed_indices: return []
    tokens = preprocess_text(text_query) # Preprocess the text query PART
    if not tokens: return []
    try:
        all_scores = bm25.get_scores(tokens)
    except Exception: # Basic error handling for unknown tokens
        known_tokens = [tok for tok in tokens if tok in bm25.idf]
        if not known_tokens: return []
        all_scores = bm25.get_scores(known_tokens)

    # Consider only allowed indices with positive scores
    filtered_scores = {idx: all_scores[idx] for idx in allowed_indices
                       if idx < len(all_scores) and all_scores[idx] > 1e-6}
    sorted_indices = sorted(filtered_scores, key=filtered_scores.get, reverse=True)

    results = []
    for idx in sorted_indices[:k]:
        res = metadata_map[idx].copy()
        res['bm25_score'] = filtered_scores[idx]
        res['index'] = idx
        results.append(res)
    return results

def search_semantic_filtered(text_query: str, model, embeddings, metadata_map, allowed_indices: set, k: int):
    """Performs semantic search ONLY on allowed indices."""
    if not allowed_indices: return []
    query_text_for_embed = " ".join(preprocess_text(text_query)) # Preprocess and join
    if not query_text_for_embed: return []
    query_emb = model.encode([query_text_for_embed], normalize_embeddings=True, show_progress_bar=False)
    all_scores = cosine_similarity(query_emb, embeddings)[0]

    # Consider only allowed indices
    filtered_scores = {idx: all_scores[idx] for idx in allowed_indices if idx < len(all_scores)}
    sorted_indices = sorted(filtered_scores, key=filtered_scores.get, reverse=True)

    results = []
    for idx in sorted_indices[:k]:
        # You might add a score threshold here if needed, e.g. filtered_scores[idx] > 0.2
        res = metadata_map[idx].copy()
        res['semantic_score'] = filtered_scores[idx]
        res['index'] = idx
        results.append(res)
    return results

# --- Hybrid Search Function (Using Query Parsing and Filtering) ---

def hybrid_search_with_parsing(
        full_query: str, bm25_model, semantic_model, embeddings, metadata_map,
        alpha: float = 0.5, top_k_initial: int = 100, top_k_final: int = 10):
    """
    Performs hybrid search by parsing the query, filtering, searching, and combining.
    """
    print(f"\n--- Starting Hybrid Search w/ Parsing (alpha={alpha}) ---")
    print(f"Full Query: '{full_query}'")
    start_time = time.time()

    # 1. Parse Query using imported function
    try:
        filters, expanded_text_query = parse_query(full_query)
    except Exception as e:
        print(f"Error during query parsing: {e}")
        print("Falling back to using full query for text search without filters.")
        filters = []
        expanded_text_query = full_query # Use original query as fallback

    # 2. Filter Metadata based on parsed filters
    allowed_indices = filter_metadata(metadata_map, filters)

    if not allowed_indices:
        print("No recipes found matching the specified filters.")
        return [] # Return empty list if no recipes pass filters

    # 3. Run Filtered Searches using the expanded_text_query
    print(f"Searching with text: '{expanded_text_query}' on {len(allowed_indices)} candidates.")
    try:
        bm25_raw = search_bm25_filtered(expanded_text_query, bm25_model, metadata_map, allowed_indices, k=top_k_initial)
        for r in bm25_raw: r['score'] = r.pop('bm25_score') # Rename for scaling
    except Exception as e:
        print(f"Error during filtered BM25 search: {e}")
        bm25_raw = []

    try:
        semantic_raw = search_semantic_filtered(expanded_text_query, semantic_model, embeddings, metadata_map, allowed_indices, k=top_k_initial)
        for r in semantic_raw: r['score'] = r.pop('semantic_score') # Rename for scaling
    except Exception as e:
        print(f"Error during filtered Semantic search: {e}")
        semantic_raw = []

    # 4. Scale scores independently
    bm25_scaled = min_max_scale_scores(bm25_raw, score_key='score')
    semantic_scaled = min_max_scale_scores(semantic_raw, score_key='score')

    # 5. Combine results and calculate hybrid scores (using index as key)
    hybrid_results_dict = {}
    # Process BM25
    for res in bm25_scaled:
        key = res['index']
        hybrid_results_dict[key] = {'bm25_scaled': res['scaled_score'], 'semantic_scaled': 0, 'metadata': res}
    # Process Semantic
    for res in semantic_scaled:
        key = res['index']
        if key not in hybrid_results_dict:
            hybrid_results_dict[key] = {'bm25_scaled': 0, 'semantic_scaled': 0, 'metadata': res}
        hybrid_results_dict[key]['semantic_scaled'] = res['scaled_score']
        if 'metadata' not in hybrid_results_dict[key]: hybrid_results_dict[key]['metadata'] = res

    # Calculate hybrid score
    final_ranked_list = []
    for key, scores_data in hybrid_results_dict.items():
        hybrid_score = (alpha * scores_data['bm25_scaled']) + ((1 - alpha) * scores_data['semantic_scaled'])
        result_entry = scores_data['metadata']
        result_entry['hybrid_score'] = hybrid_score
        result_entry['bm25_scaled_score'] = scores_data['bm25_scaled']
        result_entry['semantic_scaled_score'] = scores_data['semantic_scaled']
        final_ranked_list.append(result_entry)

    # 6. Sort by hybrid score
    final_ranked_list.sort(key=lambda x: x['hybrid_score'], reverse=True)

    end_time = time.time()
    print(f"Hybrid search with parsing completed in {end_time - start_time:.4f} seconds.")

    return final_ranked_list[:top_k_final]


# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Loading all components for Hybrid Search with Query Parsing ---")
    components_loaded = False
    try:
        bm25_model, metadata_map = load_bm25_components(BM25_INDEX_PATH, METADATA_PATH)
        semantic_model, device = setup_device_and_model(MODEL_NAME)
        embeddings = np.load(EMBEDDINGS_PATH)
        # Basic validation
        if bm25_model and metadata_map and semantic_model and embeddings is not None:
             if len(metadata_map) == embeddings.shape[0]:
                  print("✅ All components loaded successfully.")
                  components_loaded = True
             else:
                  print(f"⚠️ Mismatch between metadata count ({len(metadata_map)}) and embeddings count ({embeddings.shape[0]}).")
        else:
             print("❌ Failed to load one or more components.")

    except Exception as e:
        print(f"❌ Error during component loading: {e}")

    if components_loaded:
        # --- Run Example Queries ---
        queries_to_test = [
            "low fat chicken under 500 calories",
            "high protein vegan salad",
            "gluten-free pasta with tomato",
            "quick healthy pasta dinner"
        ]
        alpha_value = 0.5 # Example weight

        for query in queries_to_test:
            # Use the new hybrid search function
            hybrid_top_results = hybrid_search_with_parsing(
                query,
                bm25_model,
                semantic_model,
                embeddings,
                metadata_map,
                alpha=alpha_value,
                top_k_initial=50, # Fetch more initially
                top_k_final=5     # Show top 5 hybrid
            )

            print(f"\nTop {len(hybrid_top_results)} Filtered Hybrid Results for: '{query}'")
            if hybrid_top_results:
                for rank, res in enumerate(hybrid_top_results, 1):
                    print(f"{rank}. [ID: {res.get('recipe_id','N/A')}] {res.get('title','N/A')} "
                          f"({res.get('calories_cal','?')} cal, {res.get('protein_g', '?')}g protein, {res.get('duration','?')} min)\n"
                          f"   Hybrid Score: {res.get('hybrid_score', 0):.4f} "
                          f"(BM25 Scaled: {res.get('bm25_scaled_score', 0):.4f}, "
                          f"Semantic Scaled: {res.get('semantic_scaled_score', 0):.4f})")
            else:
                print("No results found.")
            print("-" * 50)
    else:
        print("\nCannot perform searches as components failed to load.")
