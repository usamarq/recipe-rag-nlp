# --------------------------------------------------------------
# bm25_search.py
# Task 3 – Simple BM25-based retrieval for recipe search
# --------------------------------------------------------------

import pandas as pd
from rank_bm25 import BM25Okapi
from text_preprocessing import preprocess_text  # <-- reuse your function!
import re

# -------------------------------------------------
# 1. Load the preprocessed dataset
# -------------------------------------------------
def load_dataset(path="data/hummus_recipes_preprocessed.csv"):
    print(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    print(f"Dataset shape: {df.shape}")
    return df


# -------------------------------------------------
# 2. Build BM25 index
# -------------------------------------------------
def build_bm25_index(df):
    """
    Combine key text fields and tokenize for BM25.
    """
    print("Building BM25 index...")

    # Combine relevant processed fields
    text_cols = [
        "processed_title",
        "processed_ingredients",
        "processed_tags",
        "processed_directions",
    ]
    available_cols = [c for c in text_cols if c in df.columns]

    print(f"Using columns: {available_cols}")

    # Combine text into one document per recipe
    df["combined_text"] = df[available_cols].fillna("").agg(" ".join, axis=1)

    # Tokenize (split on spaces — already lemmatized/clean)
    tokenized_docs = [doc.split() for doc in df["combined_text"]]

    bm25 = BM25Okapi(tokenized_docs)
    print(f"✅ BM25 index built on {len(tokenized_docs)} recipes.")
    return bm25, df


# -------------------------------------------------
# 3. Preprocess and search
# -------------------------------------------------
def preprocess_query(query: str):
    """
    Apply the same preprocessing as dataset text.
    """
    return preprocess_text(query)


def search_bm25(query: str, bm25, df, top_k=5):
    """
    Search using BM25 and return top_k results.
    """
    print(f"\n🔍 Query: {query}")

    # Preprocess query using same pipeline
    tokens = preprocess_query(query)
    if not tokens:
        print("⚠️ Query resulted in no valid tokens after preprocessing.")
        return []

    scores = bm25.get_scores(tokens)
    top_indices = scores.argsort()[-top_k:][::-1]

    print(f"\nTop {top_k} results:")
    for rank, idx in enumerate(top_indices, 1):
        title = df.loc[idx, "title"] if "title" in df.columns else "(no title)"
        cal = df.loc[idx, "calories_cal"] if "calories_cal" in df.columns else "?"
        print(f"{rank}. {title}  ({cal} cal)")

    return df.iloc[top_indices][["title", "calories_cal", "totalfat_g", "protein_g"]]


# -------------------------------------------------
# 4. Run example
# -------------------------------------------------
if __name__ == "__main__":
    df = load_dataset()
    bm25, df = build_bm25_index(df)

    # Example queries
    search_bm25("low fat chicken under 500 calories", bm25, df, top_k=5)
    search_bm25("high protein vegan salad", bm25, df, top_k=5)
