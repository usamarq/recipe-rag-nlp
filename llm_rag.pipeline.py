#!/usr/bin/env python3
"""
recipe_rag.py

Retrieval-Augmented Generation system for recipe recommendations using:
 - Ollama LLM: llama3.1:latest
 - Ollama embeddings: nomic-embed-text:latest
 - In-memory vector store: langchain_core.vectorstores.InMemoryVectorStore

Features:
 - Load recipes CSV (expects columns similar to the user's header row)
 - Create embeddings and index recipes into an in-memory vector store
 - Retrieve top-k recipes for a query with optional metadata filters (e.g., duration, tags)
 - Pass retrieved recipes as context to LLM and produce a natural language recommendation
"""

from __future__ import annotations

import os
import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# LangChain / Ollama related imports
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.prompts import PromptTemplate
    from langchain.schema import Document
except Exception as e:
    raise RuntimeError("Please install langchain (and dependencies).") from e

# Ollama LLM and embeddings from community wrappers — adjust import path if your environment differs
try:
    # Many wrappers have same names - this matches common community wrapper naming
    from langchain_community.llms import Ollama
    from langchain_community.embeddings import OllamaEmbeddings
except Exception:
    # fallback: try another import path or signal user to install langchain-community
    raise RuntimeError("Please install langchain-community (pip install langchain-community).")

# In-memory vector store requested by user
try:
    from langchain_core.vectorstores import InMemoryVectorStore
except Exception:
    # If import fails, provide informative error
    raise RuntimeError(
        "Could not import InMemoryVectorStore from langchain_core.vectorstores. "
        "Make sure you have the correct langchain_core package installed or adjust the import path."
    )

# Logging
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration via environment variables (can be overridden)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")  # default ollama local host/port
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.1:latest")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text:latest")
VECTOR_STORE_COLLECTION = os.environ.get("VECTOR_STORE_COLLECTION", "recipes")
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "16"))


@dataclass
class RecipeDoc:
    recipe_id: str
    title: str
    description: str
    duration: Optional[int]  # minutes if available
    tags: List[str]
    serves: Optional[int]
    calories_cal: Optional[float]
    raw_row: Dict[str, Any]


class RecipeRAG:
    """
    RAG system for recipe recommendation.
    - load CSV
    - index into InMemoryVectorStore
    - retrieve top-k by semantic similarity with optional metadata filters
    - generate a recommendation message with LLM
    """

    def __init__(self,
                 csv_path: str,
                 llm_model: str = LLM_MODEL,
                 embedding_model: str = EMBEDDING_MODEL,
                 ollama_host: str = OLLAMA_HOST,
                 vectorstore: Optional[InMemoryVectorStore] = None):
        self.csv_path = csv_path
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.ollama_host = ollama_host

        # Setup LLM & embeddings (wrap in retries for production robustness)
        logger.info("Initializing Ollama LLM and Embeddings...")
        self.llm = self._init_ollama_llm()
        self.embeddings = self._init_ollama_embeddings()

        # Text splitter for long fields
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)

        # Vector store (InMemoryVectorStore from langchain_core.vectorstores)
        if vectorstore is None:
            self.vectorstore = InMemoryVectorStore()  # instantiate empty and fill later
        else:
            self.vectorstore = vectorstore

        # Keep local map of id -> metadata for easy filtering
        self._metadata_index: Dict[str, Dict[str, Any]] = {}

    def _init_ollama_llm(self):
        # wrap in a retry decorator in case ollama server is still starting
        @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(5),
               retry=retry_if_exception_type(Exception))
        def _init():
            logger.info(f"Creating Ollama LLM client (model={self.llm_model})")
            # Ollama wrapper typically accepts model and optionally base_url
            return Ollama(model=self.llm_model, base_url=self.ollama_host, temperature=0.1)
        return _init()

    def _init_ollama_embeddings(self):
        @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(5),
               retry=retry_if_exception_type(Exception))
        def _init():
            logger.info(f"Creating Ollama embeddings client (model={self.embedding_model})")
            return OllamaEmbeddings(model=self.embedding_model, base_url=self.ollama_host)
        return _init()

    def _row_to_recipe_doc(self, row: pd.Series) -> RecipeDoc:
        # Handle potential column name variations and types
        recipe_id = str(row.get("recipe_id", row.get("new_recipe_id", row.name)))
        title = str(row.get("title", "")).strip()
        description = str(row.get("description", "")).strip()
        duration_raw = row.get("duration", None)
        try:
            duration = int(duration_raw) if pd.notnull(duration_raw) and str(duration_raw).strip() != "" else None
        except Exception:
            # sometimes duration might be "15 min" -> extract digits
            import re
            m = re.search(r"(\d+)", str(duration_raw or ""))
            duration = int(m.group(1)) if m else None

        tags_raw = row.get("tags", "")
        if pd.isnull(tags_raw):
            tags = []
        else:
            # tags may be comma separated
            if isinstance(tags_raw, list):
                tags = tags_raw
            else:
                tags = [t.strip().lower() for t in str(tags_raw).split(",") if t.strip()]

        serves = None
        try:
            sp = row.get("serves", row.get("servingsperrecipe", None))
            serves = int(sp) if pd.notnull(sp) and str(sp).strip() != "" else None
        except Exception:
            serves = None

        calories = None
        try:
            cal = row.get("calories_cal", None)
            calories = float(cal) if pd.notnull(cal) and str(cal).strip() != "" else None
        except Exception:
            calories = None

        return RecipeDoc(
            recipe_id=recipe_id,
            title=title,
            description=description,
            duration=duration,
            tags=tags,
            serves=serves,
            calories_cal=calories,
            raw_row=row.to_dict()
        )

    def load_csv(self) -> List[RecipeDoc]:
        """Load recipes CSV and convert each row to a RecipeDoc"""
        logger.info("Loading CSV: %s", self.csv_path)
        df = pd.read_csv(self.csv_path, dtype=str).fillna("")
        docs = []
        for idx, row in df.iterrows():
            rd = self._row_to_recipe_doc(row)
            docs.append(rd)
        logger.info("Loaded %d recipes", len(docs))
        return docs

    def _doc_to_langchain_document(self, recipe: RecipeDoc) -> Document:
        """Create a langchain Document combining title, description, ingredients, directions etc."""
        # Form body text for embedding & chunking
        # We include title, description, duration, tags, and key nutritional info
        meta = {
            "recipe_id": recipe.recipe_id,
            "title": recipe.title,
            "duration": recipe.duration,
            "tags": recipe.tags,
            "serves": recipe.serves,
            "calories": recipe.calories_cal
        }

        body_pieces = [f"Title: {recipe.title}"]
        if recipe.description:
            body_pieces.append(f"Description: {recipe.description}")
        if recipe.duration is not None:
            body_pieces.append(f"Duration: {recipe.duration} minutes")
        if recipe.tags:
            body_pieces.append(f"Tags: {', '.join(recipe.tags)}")
        # include serialized raw row for completeness
        body_pieces.append(f"RawRow: {recipe.raw_row}")

        text = "\n\n".join(body_pieces)
        return Document(page_content=text, metadata=meta)

    def index_recipes(self, recipes: List[RecipeDoc], overwrite: bool = True) -> int:
        """
        Create embeddings and index recipes into the InMemoryVectorStore.
        Returns number of indexed documents.
        """
        if overwrite:
            # Reinitialize empty vector store
            logger.info("Clearing and recreating InMemoryVectorStore")
            self.vectorstore = InMemoryVectorStore()

        # Convert to langchain Documents
        documents = [self._doc_to_langchain_document(r) for r in recipes]
        logger.info("Converting %d recipes to documents and creating embeddings", len(documents))

        # Create embeddings in batches and add to vector store
        added = 0
        batch = []
        for i, doc in enumerate(documents, start=1):
            batch.append(doc)
            if len(batch) >= EMBED_BATCH_SIZE or i == len(documents):
                # compute embeddings via the embeddings object
                try:
                    # Many LangChain vectorstores accept from_documents with embeddings parameter
                    if hasattr(self.vectorstore, "from_documents"):
                        # Some variants are classmethods; handle both
                        try:
                            # attempt to use instance method if exists
                            self.vectorstore = InMemoryVectorStore.from_documents(batch, embedding=self.embeddings)
                        except Exception:
                            # fallback to classmethod style
                            self.vectorstore = InMemoryVectorStore.from_documents(batch, embedding=self.embeddings)
                    else:
                        # Generic add_documents approach (some vectorstores implement add_documents)
                        if hasattr(self.vectorstore, "add_documents"):
                            self.vectorstore.add_documents(batch, embedding=self.embeddings)
                        else:
                            # As ultimate fallback: compute embeddings and store in a naive store
                            raise RuntimeError("InMemoryVectorStore API not compatible in this environment.")
                    added += len(batch)
                    logger.info("Indexed batch size=%d", len(batch))
                except Exception as e:
                    logger.exception("Failed to index batch: %s", e)
                    raise
                finally:
                    batch = []

        # build metadata index
        self._metadata_index = {}
        try:
            # if vectorstore exposes docs or similar
            if hasattr(self.vectorstore, "documents"):
                # If the store exposes docs, iterate
                for d in getattr(self.vectorstore, "documents", []):
                    rid = d.metadata.get("recipe_id", None)
                    if rid:
                        self._metadata_index[str(rid)] = d.metadata
            else:
                # fallback: iterate input recipes
                for r in recipes:
                    self._metadata_index[str(r.recipe_id)] = {
                        "title": r.title,
                        "duration": r.duration,
                        "tags": r.tags,
                        "serves": r.serves,
                        "calories": r.calories_cal
                    }
        except Exception:
            logger.debug("Could not extract docs from vectorstore; metadata index built from input recipes")

        logger.info("Indexing complete. Total indexed: %d", added)
        return added

    def _filter_candidates(self, candidates: List[Document], filters: Dict[str, Any]) -> List[Document]:
        """
        Apply simple metadata filter to candidate documents.
        Supported filters: duration_lt (minutes), tag_in (list or str), vegetarian_bool, max_calories
        """
        if not filters:
            return candidates

        out = []
        for doc in candidates:
            meta = doc.metadata or {}
            keep = True

            duration_lt = filters.get("duration_lt")
            if duration_lt is not None:
                d = meta.get("duration")
                if d is None or (isinstance(d, (int, float)) and d >= duration_lt) or (isinstance(d, str) and d and int(d) >= duration_lt):
                    keep = False

            tag_in = filters.get("tag_in")
            if tag_in and keep:
                tags = meta.get("tags", [])
                if isinstance(tag_in, str):
                    tag_in_vals = [tag_in.lower()]
                else:
                    tag_in_vals = [t.lower() for t in tag_in]
                # if none of the tag_in values are present, drop
                if not any(t.lower() in tags for t in tag_in_vals):
                    keep = False

            max_calories = filters.get("max_calories")
            if max_calories and keep:
                cal = meta.get("calories")
                if cal is None or (isinstance(cal, (int, float)) and cal > max_calories):
                    keep = False

            if keep:
                out.append(doc)
        return out

    def retrieve(self,
                 query: str,
                 top_k: int = 5,
                 filters: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """
        Retrieve top-k semantically similar recipes for query.
        Returns list of (Document, score) sorted by descending similarity score.
        """
        logger.info("Embedding query and searching top_k=%d", top_k)
        # Compute query embedding
        query_embedding = self.embeddings.embed_query(query)

        # Most langchain stores expose similarity_search_by_vector or similar
        # We'll try several common method names for compatibility
        search_methods = [
            "similarity_search_by_vector",
            "similarity_search_with_score",
            "similarity_search",
            "search"
        ]

        results: List[Tuple[Document, float]] = []
        for method_name in search_methods:
            if hasattr(self.vectorstore, method_name):
                method = getattr(self.vectorstore, method_name)
                try:
                    # Different signatures: try common ones
                    out = None
                    try:
                        out = method(query_embedding, k=top_k)
                    except TypeError:
                        # maybe expects (query, k) or (query, k, ...) -> fallback
                        out = method(query, k=top_k)
                    # normalize to list of (doc, score)
                    if out:
                        # many returns are list[Document] or list[(Document, score)]
                        if isinstance(out, list) and len(out) > 0 and isinstance(out[0], tuple):
                            results = out  # already (doc, score)
                        elif isinstance(out, list) and len(out) > 0 and hasattr(out[0], "page_content"):
                            # no scores returned -> compute similarity (approx) by re-embedding checks (expensive)
                            # we'll fake a descending order by returning in same order and set score to None
                            results = [(d, None) for d in out[:top_k]]
                        else:
                            results = []
                    break
                except Exception as e:
                    logger.debug("search method %s failed: %s", method_name, e)
                    continue

        if not results:
            # fallback: brute-force compute cosine similarity against stored document embeddings if accessible
            if hasattr(self.vectorstore, "get_embeddings"):
                try:
                    stored = self.vectorstore.get_embeddings()
                    # get_embeddings -> list of (id, vector, metadata?) - unknown shape; skip for now
                except Exception:
                    pass
            logger.warning("Vector store didn't return results via known APIs; returning empty list")
            return []

        # If filters provided, apply simple metadata filtering
        docs_and_scores = []
        for doc, score in results:
            docs_and_scores.append((doc, score))
        if filters:
            filtered = self._filter_candidates([d for d, s in docs_and_scores], filters)
            docs_and_scores = [(d, s) for d, s in docs_and_scores if d in filtered]

        # Keep only top_k
        docs_and_scores = docs_and_scores[:top_k]
        logger.info("Retrieved %d candidates after filtering", len(docs_and_scores))
        return docs_and_scores

    def _format_context_from_docs(self, docs_with_scores: List[Tuple[Document, float]]) -> str:
        """Build a concise context string from retrieved docs to pass to the LLM"""
        parts = []
        for i, (doc, score) in enumerate(docs_with_scores, start=1):
            meta = doc.metadata or {}
            title = meta.get("title", f"Recipe #{i}")
            duration = meta.get("duration", "unknown")
            tags = meta.get("tags", [])
            calories = meta.get("calories", None)
            snippet = doc.page_content
            header = f"{i}. {title} ({duration} min){', ' + str(calories) + ' cal' if calories else ''}"
            parts.append(header)
            # include a short snippet (first 200 chars)
            parts.append(snippet[:400])
            if score is not None:
                parts.append(f"[similarity_score: {score}]")
            parts.append("")  # blank line between entries
        return "\n".join(parts)

    def generate_answer(self,
                        query: str,
                        docs_with_scores: List[Tuple[Document, float]],
                        max_recipes_to_show: int = 5) -> str:
        """
        Generate a user-facing answer: LLM is given the retrieved recipes as context and asked to recommend.
        """
        context = self._format_context_from_docs(docs_with_scores[:max_recipes_to_show])

        prompt_template = PromptTemplate(
            input_variables=["query", "context"],
            template=(
                "You are a helpful culinary assistant. A user asks: \"{query}\".\n\n"
                "Here are candidate recipes that match the query (each recipe includes title, duration, tags, and a short snippet):\n\n"
                "{context}\n\n"
                "Task: produce a concise, friendly recommendation message for the user. Include:\n"
                "- A short summary sentence with 2-4 recommended recipes (title, duration, calories if available) matching the request.\n"
                "- Why each recommendation matches (1 short reason each).\n"
                "- If none match perfectly, provide the best alternatives and explain why.\n\n"
                "Output only the recommendation text (no JSON or extra commentary)."
            )
        )

        prompt = prompt_template.format(query=query, context=context)

        logger.info("Invoking LLM to generate answer")
        # Ollama wrapper's .invoke may return a string or dict-like; handle both
        response = self.llm.invoke(prompt)
        if isinstance(response, dict) and "text" in response:
            answer = response["text"].strip()
        else:
            answer = str(response).strip()
        return answer


# Example usage
def main_example(csv_path: str):
    """
    Demonstrate full pipeline:
     - load CSV
     - index recipes
     - run an example query with metadata filtering (vegetarian under 20 minutes)
    """
    rag = RecipeRAG(csv_path=csv_path)

    # Load and index
    recipes = rag.load_csv()
    logger.info("Indexing recipes... this may take a while depending on model speed")
    indexed_count = rag.index_recipes(recipes)
    logger.info("Indexed %d recipes into in-memory store", indexed_count)

    # Example query
    query = "Suggest a quick vegetarian dinner under 20 minutes."
    # Filters for our simple metadata filter: duration_lt and tag_in
    filters = {"duration_lt": 20, "tag_in": "vegetarian"}

    retrieved = rag.retrieve(query=query, top_k=8, filters=filters)
    if not retrieved:
        print("No matching recipes were retrieved for the given filters/query.")
        return

    answer = rag.generate_answer(query, retrieved, max_recipes_to_show=5)
    print("\n===== Recommendation Answer =====\n")
    print(answer)
    print("\n===== Retrieved Candidates (debug) =====\n")
    for doc, score in retrieved:
        print("Title:", doc.metadata.get("title"))
        print("Duration:", doc.metadata.get("duration"))
        print("Tags:", doc.metadata.get("tags"))
        print("Score:", score)
        print("---")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Recipe RAG demo")
    parser.add_argument("--csv", type=str, default="./data/hummus_recipes_preprocessed.csv", help="Path to recipes CSV")
    args = parser.parse_args()

    # Quick sanity check for CSV path
    if not os.path.exists(args.csv):
        logger.error("CSV file not found: %s. Please provide a valid CSV path.", args.csv)
        exit(1)

    main_example(args.csv)
