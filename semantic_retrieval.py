# --------------------------------------------------------------
# semantic_retrieval.py
# Task 4: Semantic Embeddings & Retrieval (BERT-based)
# --------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from text_preprocessing import preprocess_text


# -------------------------------------------------
# 1. Device check and model setup
# -------------------------------------------------
def setup_device_and_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Checks for GPU availability and loads the Sentence-BERT model accordingly.
    Uses CUDA if available, otherwise falls back to CPU.
    """
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
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ Dataset not found at: {input_path}")
    df = pd.read_csv(input_path)
    print(f"✅ Dataset loaded: {df.shape}")
    return df


# -------------------------------------------------
# 3. Build combined text field
# -------------------------------------------------
def build_combined_text(df):
    """
    Combines relevant text columns into a single searchable document per recipe.
    """
    text_cols = [
        "processed_title",
        "processed_ingredients",
        "processed_tags",
        "processed_directions"
    ]
    available_cols = [c for c in text_cols if c in df.columns]
    print(f"Using columns for embeddings: {available_cols}")

    df["combined_text"] = df[available_cols].fillna("").agg(" ".join, axis=1)
    print("✅ Combined text field created.")
    return df


# -------------------------------------------------
# 4. Create embeddings and save
# -------------------------------------------------
def create_and_save_embeddings(df, model, device, output_dir="data", batch_size=96):
    """
    Generates embeddings using Sentence-BERT and saves them as .npy.
    Adjusts batch size automatically based on VRAM (~6GB default).
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "recipe_embeddings.npy")

    print(f"\n⚙️  Generating embeddings on {device.upper()} (batch size={batch_size}) ...")
    texts = df["combined_text"].tolist()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    np.save(output_path, embeddings)
    print(f"✅ Embeddings created and saved → {output_path}")
    print(f"Shape: {embeddings.shape}")

    return embeddings


# -------------------------------------------------
# 5. Semantic search
# -------------------------------------------------
def semantic_search(query, model, df, embeddings, top_k=5):
    """
    Perform semantic retrieval based on cosine similarity between
    the query embedding and recipe embeddings.
    """
    print(f"\n🔍 Query: {query}")
    cleaned_query = " ".join(preprocess_text(query))
    query_emb = model.encode([cleaned_query], normalize_embeddings=True)

    scores = cosine_similarity(query_emb, embeddings)[0]
    top_indices = np.argsort(-scores)[:top_k]

    results = df.iloc[top_indices][["title", "calories_cal", "totalfat_g", "protein_g"]].copy()
    results["similarity"] = scores[top_indices]
    print("\nTop results:\n")
    print(results)
    return results


# -------------------------------------------------
# 6. Main
# -------------------------------------------------
def main():
    model, device = setup_device_and_model()
    df = load_dataset()
    df = build_combined_text(df)

    embeddings = create_and_save_embeddings(df, model, device, batch_size=96)

    # Optional: Test queries
    test_queries = [
        "low carb chicken meal",
        "high protein vegan breakfast",
        "sugar free dessert",
        "quick healthy pasta dinner"
    ]

    for q in test_queries:
        semantic_search(q, model, df, embeddings, top_k=5)


if __name__ == "__main__":
    main()
