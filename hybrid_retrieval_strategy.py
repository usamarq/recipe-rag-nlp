import pandas as pd
import numpy as np
import pickle
import time
import torch
import os
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Import Functions from other scripts ---
try:
    # Function to load BM25 index and metadata
    from bm25_retrieval_optimized import load_index_and_metadata as load_bm25_components
    # The actual BM25 search function using loaded components
    from bm25_retrieval_optimized import search_bm25_optimized
    # Function to load the Sentence Transformer model
    from semantic_retrieval_BERT import setup_device_and_model
    # Import the common text preprocessing function
    from text_preprocessing import preprocess_text
except ImportError as e:
    print(f"Error importing functions from helper scripts: {e}")
    print("Please ensure text_preprocessing.py, bm25_search_optimized_v2.py, ")
    print("and semantic_retrieval.py are in the Python path.")
    exit() # Exit if imports fail

# --- Configuration ---
BM25_INDEX_PATH = "data/bm25_index.pkl"
METADATA_PATH = "data/recipe_metadata.pkl"
EMBEDDINGS_PATH = "data/recipe_embeddings.npy"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Helper Functions ---

def min_max_scale_scores(results_list, score_key='score'):
    """Applies Min-Max scaling (0-1) to scores in a list of result dicts."""
    if not results_list: return results_list
    scores = [res.get(score_key, 0) for res in results_list]
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score
    for res in results_list:
        original_score = res.get(score_key, 0)
        if score_range > 1e-9: # Avoid division by zero
            scaled_score = (original_score - min_score) / score_range
        else:
            scaled_score = 0.5 # All scores were identical
        res['scaled_' + score_key] = scaled_score # Add scaled score with distinct key
    return results_list

def semantic_search_adapted(query: str, model, embeddings, metadata_map, top_k=10):
    """
    Performs semantic search using loaded components.
    Returns list of dicts including 'semantic_score'.
    """
    if model is None or embeddings is None or metadata_map is None: return []

    query_text = " ".join(preprocess_text(query))
    if not query_text: return []

    query_emb = model.encode([query_text], normalize_embeddings=True, show_progress_bar=False)
    scores = cosine_similarity(query_emb, embeddings)[0]
    top_indices = np.argsort(-scores)[:top_k]

    results = []
    for idx in top_indices:
         if idx in metadata_map:
             score = scores[idx]
             # Optionally add a threshold, e.g. if score > 0.1:
             res = metadata_map[idx].copy()
             res['semantic_score'] = score # Raw semantic score
             res['index'] = idx
             results.append(res)
    return results

# --- Hybrid Search Function ---

def hybrid_search(query: str, bm25_model, semantic_model, embeddings, metadata_map,
                  alpha: float = 0.7, top_k_initial: int = 50, top_k_final: int = 10):
    """
    Performs hybrid search combining BM25 and Semantic results.

    Args:
        query: User search query.
        bm25_model: Loaded BM25 model.
        semantic_model: Loaded Sentence Transformer model.
        embeddings: Loaded recipe embeddings numpy array.
        metadata_map: Dictionary mapping index to recipe metadata.
        alpha: Weight for BM25 score (0 to 1). Semantic weight = 1 - alpha.
        top_k_initial: How many results to retrieve initially from each search.
        top_k_final: How many final hybrid results to return.

    Returns:
        List of ranked result dictionaries, including hybrid scores.
    """
    print(f"\n--- Performing Hybrid Search (alpha={alpha}) ---")
    start_time = time.time()

    # 1. Get initial results from both systems
    try:
        bm25_raw = search_bm25_optimized(query, bm25_model, metadata_map, top_k=top_k_initial)
        # Rename score key for clarity before scaling
        for r in bm25_raw: r['score'] = r.pop('bm25_score')
    except Exception as e:
        print(f"Error during BM25 search: {e}")
        bm25_raw = []

    try:
        semantic_raw = semantic_search_adapted(query, semantic_model, embeddings, metadata_map, top_k=top_k_initial)
        # Rename score key
        for r in semantic_raw: r['score'] = r.pop('semantic_score')
    except Exception as e:
        print(f"Error during Semantic search: {e}")
        semantic_raw = []

    # 2. Scale scores independently (0-1 range)
    bm25_scaled = min_max_scale_scores(bm25_raw, score_key='score')
    semantic_scaled = min_max_scale_scores(semantic_raw, score_key='score')

    # 3. Combine results and calculate hybrid scores
    hybrid_results_dict = {} # Use recipe_id as key for combining

    # Process BM25 results
    for res in bm25_scaled:
        recipe_id = res.get('recipe_id')
        if recipe_id is None: continue # Skip if no ID
        if recipe_id not in hybrid_results_dict:
            hybrid_results_dict[recipe_id] = {'bm25_scaled': 0, 'semantic_scaled': 0, 'metadata': res}
        hybrid_results_dict[recipe_id]['bm25_scaled'] = res['scaled_score']
        # Ensure metadata is stored if not already present (should be)
        if 'metadata' not in hybrid_results_dict[recipe_id]: hybrid_results_dict[recipe_id]['metadata'] = res


    # Process Semantic results
    for res in semantic_scaled:
        recipe_id = res.get('recipe_id')
        if recipe_id is None: continue # Skip if no ID
        if recipe_id not in hybrid_results_dict:
            hybrid_results_dict[recipe_id] = {'bm25_scaled': 0, 'semantic_scaled': 0, 'metadata': res}
        hybrid_results_dict[recipe_id]['semantic_scaled'] = res['scaled_score']
        # Ensure metadata is stored if not already present
        if 'metadata' not in hybrid_results_dict[recipe_id]: hybrid_results_dict[recipe_id]['metadata'] = res

    # Calculate hybrid score for each unique recipe found
    final_ranked_list = []
    for recipe_id, scores in hybrid_results_dict.items():
        hybrid_score = (alpha * scores['bm25_scaled']) + ((1 - alpha) * scores['semantic_scaled'])
        result_entry = scores['metadata'] # Get metadata dict associated with this recipe_id
        result_entry['hybrid_score'] = hybrid_score
        # Optionally keep original scaled scores for inspection
        result_entry['bm25_scaled_score_contribution'] = scores['bm25_scaled']
        result_entry['semantic_scaled_score_contribution'] = scores['semantic_scaled']
        final_ranked_list.append(result_entry)

    # 4. Sort by hybrid score
    final_ranked_list.sort(key=lambda x: x['hybrid_score'], reverse=True)

    end_time = time.time()
    print(f"Hybrid search completed in {end_time - start_time:.4f} seconds.")

    return final_ranked_list[:top_k_final]


# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Loading all components for Hybrid Search ---")
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
        alpha_value = 0.5 # Example weight, adjust as needed HERE

        for query in queries_to_test:
            hybrid_top_results = hybrid_search(
                query,
                bm25_model,
                semantic_model,
                embeddings,
                metadata_map,
                alpha=alpha_value,
                top_k_initial=50, # Retrieve more initially
                top_k_final=5     # Show top 5 hybrid
            )

            print(f"\nTop {len(hybrid_top_results)} Hybrid Results for: '{query}'")
            if hybrid_top_results:
                for rank, res in enumerate(hybrid_top_results, 1):
                    print(f"{rank}. [ID: {res.get('recipe_id','N/A')}] {res.get('title','N/A')} "
                          f"({res.get('calories_cal','?')} cal)\n"
                          f"   Hybrid Score: {res.get('hybrid_score', 0):.4f} "
                          f"(BM25 Scaled: {res.get('bm25_scaled_score_contribution', 0):.4f}, "
                          f"Semantic Scaled: {res.get('semantic_scaled_score_contribution', 0):.4f})")
            else:
                print("No results found.")
            print("-" * 50)
    else:
        print("\nCannot perform searches as components failed to load.")
