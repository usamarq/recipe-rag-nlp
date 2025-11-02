import pandas as pd
# --- MODIFIED: Import SentenceTransformer ---
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import warnings
import os
import time
import json
from tqdm import tqdm
import pickle
import numpy as np
import ast
import re
import nltk
import torch # For device check

# --- Import custom parser and preprocessor ---
try:
    from query_parser import parse_query
    from text_preprocessing import preprocess_text
except ImportError:
    print("="*50)
    print("FATAL ERROR: query_parser.py or text_preprocessing.py not found.")
    print("This script relies on your custom parser scripts.")
    print("="*50)
    # Define a fallback parser
    def parse_query(query):
        print("Warning: Using fallback parser. No filters or expansion will be applied.")
        return [], query
    def preprocess_text(text):
        if not isinstance(text, str): return []
        print("Warning: Using fallback preprocessor.")
        return text.lower().split()
# -----------------------------------------

warnings.filterwarnings('ignore')


class RecipeRAGSystem:
    def __init__(self, csv_path, 
                 embedding_path="./data/recipe_embeddings.npy", # Path to the .npy file
                 embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", 
                 llm_model="phi3:mini", 
                 vector_store_path="faiss_recipe_index_minilm", # New Path for MiniLM
                 force_rebuild=False, 
                 batch_size=96): # Not used for building, but kept for consistency
        """
        Initialize the Recipe RAG System using pre-computed .npy embeddings.
        """
        self.csv_path = csv_path
        self.embedding_path = embedding_path # Path to .npy file
        self.embedding_model_name = embedding_model_name # HF name
        self.llm_model = llm_model
        self.vector_store_path = vector_store_path
        self.metadata_path = f"{vector_store_path}_metadata.json"
        
        print("="*80)
        print("Initializing Recipe RAG System (MiniLM .npy + Query Parsing)")
        print("="*80)
        
        # --- MODIFIED: Load SentenceTransformer for QUERYING ---
        print(f"\n[1/3] Initializing embeddings model ({self.embedding_model_name})...")
        start_time = time.time()
        if torch.cuda.is_available():
            self.device = "cuda"
            print(f"   ✓ Embedding: Found CUDA GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            self.device = "mps"
            print("   ✓ Embedding: Found Apple Silicon GPU (MPS).")
        else:
            self.device = "cpu"
            print("   ⚠️ Embedding: No GPU detected, using CPU.")
        
        # This model is for embedding the QUERY at search time
        self.embeddings_model = SentenceTransformer(self.embedding_model_name, device=self.device)
        print(f"   ✓ Query Embeddings model loaded on {self.device} in {time.time() - start_time:.2f}s")
        # -----------------------------------------------------------------
        
        print(f"\n[2/3] Initializing LLM ({self.llm_model} via Ollama)...")
        start_time = time.time()
        self.llm = Ollama(model=self.llm_model, temperature=0.7)
        print(f"   ✓ LLM loaded in {time.time() - start_time:.2f}s")
        
        print("\n[3/3] Loading/Creating vector store...")
        self._load_or_create_vector_store(force_rebuild)
        
        print("\n" + "="*80)
        print("✓ RAG System initialized successfully!")
        print("="*80 + "\n")

    def _load_or_create_vector_store(self, force_rebuild):
        """Load existing vector store or create new one from .npy file."""
        
        index_exists = os.path.exists(f"{self.vector_store_path}/index.faiss")
        metadata_exists = os.path.exists(self.metadata_path)
        
        if index_exists and metadata_exists and not force_rebuild:
            print(f"   ✓ Found existing vector store at '{self.vector_store_path}'")
            start_time = time.time()
            
            self.vector_store = FAISS.load_local(
                self.vector_store_path, 
                self.embeddings_model, # Use the SentenceTransformer object
                allow_dangerous_deserialization=True
            )
            
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            
            print(f"   ✓ Loaded {metadata.get('num_documents', 'N/A')} documents in {time.time() - start_time:.2f}s")
            print(f"   ✓ Index created: {metadata.get('created_at', 'N/A')}")
            
            # --- Load metadata_map from FAISS for filtering ---
            print("   Loading metadata map from index for filtering...")
            try:
                self.metadata_map = {i: self.vector_store.docstore.search(self.vector_store.index_to_docstore_id[i]).metadata 
                                     for i in range(len(self.vector_store.index_to_docstore_id))}
                print(f"   ✓ Reconstructed metadata map for {len(self.metadata_map)} items.")
            except Exception as e:
                print(f"   ⚠️ Warning: Could not reconstruct metadata map: {e}. Filtering will fail.")
                self.metadata_map = {}
            return
        
        # --- Create new vector store FROM .NPY ---
        print("   ⚠ No existing vector store found or force rebuild requested")
        print(f"   → Building new index from CSV ({self.csv_path}) and pre-computed embeddings ({self.embedding_path})...")
        
        if not os.path.exists(self.embedding_path):
            raise FileNotFoundError(f"Required embedding file not found: {self.embedding_path}.")
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Required CSV data file not found: {self.csv_path}.")
            
        print("   Loading recipe data (for metadata)...")
        start_time = time.time()
        
        # Load all columns needed for filtering and display
        all_needed_cols = [
            'recipe_id', 'title', 'description', 'duration', 'tags', 
            'processed_tags', 'ingredients', 
            'serves', 'calories_cal', 'protein_g', 'directions',
            'totalfat_g', 'saturatedfat_g', 'cholesterol_mg', 'sodium_mg', 
            'totalcarbohydrate_g', 'dietaryfiber_g', 'sugars_g',
            'ingredients_sizes', 'direction_size'
        ]
        try:
            all_cols_df = pd.read_csv(self.csv_path, nrows=0)
            cols_to_load = [col for col in all_needed_cols if col in all_cols_df.columns]
            df_raw = pd.read_csv(self.csv_path, usecols=cols_to_load, low_memory=False)
        except Exception as e:
            print(f"Error loading CSV from '{self.csv_path}': {e}. Check path and column names.")
            raise
        print(f"   ✓ Loaded {len(df_raw)} recipes in {time.time() - start_time:.2f}s")
        
        print(f"   Loading pre-computed embeddings from {self.embedding_path}...")
        precomputed_embeddings = np.load(self.embedding_path)
        
        if len(df_raw) != len(precomputed_embeddings):
            raise ValueError(
                f"Embedding-Data Mismatch: CSV has {len(df_raw)} rows, "
                f"embedding file has {len(precomputed_embeddings)} vectors."
            )

        print("   Processing data (cleaning NaNs)...")
        start_time = time.time()
        self.df = df_raw.copy()
        self.process_data() # Cleans and filters self.df in place
        
        kept_indices = self.df.index # Get indices of rows that were *kept*
        print(f"   ✓ Processed {len(self.df)} valid recipes in {time.time() - start_time:.2f}s")
        
        # --- Filter embeddings to match the filtered dataframe ---
        print(f"   Filtering embeddings from {len(precomputed_embeddings)} to {len(kept_indices)}...")
        embeddings_processed = precomputed_embeddings[kept_indices]
        del precomputed_embeddings, df_raw # Free memory

        print("   Preparing (document, embedding) pairs and metadata...")
        text_embeddings_list = []
        metadatas_list = []
        self.metadata_map = {}
        
        doc_index_counter = 0
        # Iterate over the *filtered* dataframe (self.df)
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="   Creating Documents"):
            
            # Text for LLM context (pretty, raw text)
            doc_text_for_llm = self.create_llm_context_text(row) 
            
            # Get the pre-computed embedding
            emb = embeddings_processed[doc_index_counter]

            tags_for_meta = []
            tags_data = row.get('processed_tags', "[]")
            if isinstance(tags_data, str) and tags_data.startswith('[') and tags_data.endswith(']'):
                try: tags_for_meta = ast.literal_eval(tags_data)
                except: tags_for_meta = []
            elif isinstance(tags_data, list):
                tags_for_meta = tags_data
            
            # Metadata for FAISS docstore AND internal filtering
            metadata = {
                'recipe_id': str(row.get('recipe_id', str(doc_index_counter))),
                'title': str(row.get('title', 'N/A')),
                'llm_context_text': doc_text_for_llm, # Store the "pretty" text
                'duration': float(row.get('duration', 0)),
                'calories_cal': float(row.get('calories_cal', 0)),
                'protein_g': float(row.get('protein_g', 0)),
                'totalfat_g': float(row.get('totalfat_g', 0)),
                'saturatedfat_g': float(row.get('saturatedfat_g', 0)),
                'cholesterol_mg': float(row.get('cholesterol_mg', 0)),
                'sodium_mg': float(row.get('sodium_mg', 0)),
                'totalcarbohydrate_g': float(row.get('totalcarbohydrate_g', 0)),
                'dietaryfiber_g': float(row.get('dietaryfiber_g', 0)),
                'sugars_g': float(row.get('sugars_g', 0)),
                'ingredients_sizes': int(row.get('ingredients_sizes', 0)),
                'direction_size': int(row.get('direction_size', 0)),
                'tags_processed': [str(t).lower() for t in tags_for_meta], # Clean list for filtering
            }
            
            # --- Use the "pretty" text for page_content, FAISS.from_embeddings will use the vector ---
            text_embeddings_list.append((doc_text_for_llm, emb.tolist()))
            metadatas_list.append(metadata)
            
            self.metadata_map[doc_index_counter] = metadata
            doc_index_counter += 1
        
        del embeddings_processed # Free memory
        del self.df

        print(f"   Creating vector store from {len(text_embeddings_list)} pre-computed embeddings...")
        start_time = time.time()
        
        # --- Use FAISS.from_embeddings ---
        # This method takes pre-computed vectors and does not call the embedding model
        self.vector_store = FAISS.from_embeddings(
            text_embeddings=text_embeddings_list,
            embedding=self.embeddings_model, # Pass the *query* embedder
            metadatas=metadatas_list
        )
        print(f"   ✓ Vector store created in {time.time() - start_time:.2f}s")
        
        print(f"   Saving new vector store to {self.vector_store_path}...")
        self.vector_store.save_local(self.vector_store_path)

        # Save metadata JSON
        metadata_json = {
            'num_documents': len(metadatas_list),
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'csv_path': self.csv_path,
            'embedding_model': self.embedding_model_name
        }
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata_json, f, indent=2)
        
        print(f"\n   ✓ Vector store and metadata saved.")

    def process_data(self):
        """Clean and process the recipe data"""
        initial_count = len(self.df)
        
        text_cols_to_fill = ['description', 'tags', 'processed_tags', 'ingredients', 
                             'directions', 'processed_title', 'processed_ingredients',
                             'processed_directions']
        numeric_cols = ['duration', 'calories_cal', 'protein_g', 'totalfat_g', 
                        'saturatedfat_g', 'cholesterol_mg', 'sodium_mg', 
                        'totalcarbohydrate_g', 'dietaryfiber_g', 'sugars_g',
                        'ingredients_sizes', 'direction_size']
        
        for col in text_cols_to_fill:
            if col in self.df.columns:
                if col == 'processed_tags':
                    self.df[col] = self.df[col].fillna('[]')
                else:
                    self.df[col] = self.df[col].fillna('')
            
        self.df['serves'] = self.df['serves'].fillna('N/A')

        for col in numeric_cols:
            if col in self.df.columns:
                 self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Drop only if title is missing
        drop_subset_existing = ['title'] 
        self.df = self.df.dropna(subset=drop_subset_existing)
        
        # FillNa for numeric columns *after* dropping
        for col in numeric_cols:
             if col in self.df.columns:
                 self.df[col] = self.df[col].fillna(0) # Fill remaining NaNs with 0
                 if col in ['duration', 'ingredients_sizes', 'direction_size']:
                      self.df[col] = self.df[col].astype(int)
                 else:
                      self.df[col] = self.df[col].astype(float) 

        removed = initial_count - len(self.df)
        if removed > 0:
            print(f"   ⚠ Removed {removed} invalid recipes (e.g., NaN in title)")
    
    def create_llm_context_text(self, row):
        """Create a rich text representation of a recipe for the LLM context"""
        tags_display = row.get('tags', 'N/A')
        if pd.isna(tags_display) or tags_display in ['nan', '[]', '']:
             try: 
                 tags_list = row.get('tags_processed', [])
                 tags_display = ", ".join(tags_list) if tags_list else "N/A"
             except: tags_display = "N/A"
        
        doc_text = f"""
Title: {row.get('title', 'N/A')}
Description: {row.get('description', 'N/A')}
Duration: {row.get('duration', 'N/A')} minutes
Tags: {tags_display}
Ingredients: {row.get('ingredients', 'N/A')}
Calories: {row.get('calories_cal', 'N/A')} cal
Protein: {row.get('protein_g', 'N/A')}g
Serves: {row.get('serves', 'N/A')}
Directions: {str(row.get('directions', 'N/A'))[:200]}...
        """.strip()
        return doc_text

    # --- REMOVED: create_document_text_for_embedding ---
    # Not needed, we load from .npy file

    # --- REMOVED: _create_vector_store_with_progress ---
    # Replaced by logic in _load_or_create_vector_store

    # --- NEW: Filter function (using the map) ---
    def _filter_indices(self, filters_list: list) -> set:
        """
        Filters the internal metadata_map based on a list of filter tuples.
        Returns a set of allowed indices (integers).
        """
        if not self.metadata_map:
             print("   ⚠️ Metadata map not loaded. Cannot filter.")
             try:
                self.metadata_map = {i: self.vector_store.docstore.search(self.vector_store.index_to_docstore_id[i]).metadata 
                                     for i in range(len(self.vector_store.index_to_docstore_id))}
                print(f"   ✓ Reconstructed metadata map for {len(self.metadata_map)} items.")
             except Exception as e:
                 print(f"   ❌ Failed to reconstruct metadata map: {e}. Filtering will fail.")
                 return set()
             
        if not filters_list:
            # print("   No filters found, returning all indices.")
            return set(self.metadata_map.keys())

        print(f"   Applying {len(filters_list)} filter(s): {filters_list}")
        allowed_indices = set()
        
        for index, meta in self.metadata_map.items():
            match_all = True
            for attribute, operator, value in filters_list:
                meta_value = meta.get(attribute) 
                
                if attribute == 'tags':
                     meta_value = meta.get('tags_processed', [])
                
                if meta_value is None: 
                    match_all = False; break
                
                try:
                    if attribute == 'tags':
                        tags_list = meta_value # This is the clean list
                        filter_tag = str(value).lower()
                        if operator == 'contains':
                            if filter_tag not in [str(t).lower() for t in tags_list]:
                                match_all = False; break
                    
                    else: # Numerical filtering
                        meta_value_num = float(meta_value)
                        filter_value_num = float(value)
                        
                        if operator == '<' and not (meta_value_num < filter_value_num):
                            match_all = False; break
                        elif operator == '>' and not (meta_value_num > filter_value_num):
                            match_all = False; break
                        elif operator == '=' and not (np.isclose(meta_value_num, filter_value_num)): # Use isclose for float
                            match_all = False; break
                        
                except Exception: 
                    match_all = False; break
                
                if not match_all: 
                    break
            
            if match_all: 
                allowed_indices.add(index)
                
        print(f"   Found {len(allowed_indices)} recipes after filtering.")
        return allowed_indices

    # --- MODIFIED: retrieve_recipes ---
    def retrieve_recipes(self, text_query: str, filters_list: list, k: int):
        """Retrieve top-k recipes by filtering *after* semantic search."""
        
        # 1. Preprocess text query (NO prefix for MiniLM)
        processed_query_tokens = preprocess_text(text_query)
        processed_query_for_embedding = " ".join(processed_query_tokens)
        
        print(f"   Text query for embedding: '{processed_query_for_embedding}'")
        
        # 2. Perform semantic search (retrieve more candidates)
        k_retrieval = max(k * 10, 50) 
        print(f"   Retrieving {k_retrieval} candidates *before* filtering...")
        try:
            # --- MODIFIED: Use self.embeddings_model.encode to get query vector ---
            query_vector = self.embeddings_model.encode([processed_query_for_embedding], 
                                                        normalize_embeddings=True, # Match your .npy creation
                                                        show_progress_bar=False)
            
            # --- MODIFIED: Use FAISS.similarity_search_by_vector ---
            # This is the correct way to query FAISS with an external vector
            scores, indices = self.vector_store.index.search(query_vector.astype(np.float32), k_retrieval)
            
            # Manually reconstruct the (doc, score) list
            results_with_scores = []
            for i, idx in enumerate(indices[0]):
                 if idx == -1: continue # FAISS can return -1 for no more neighbors
                 doc = self.vector_store.docstore.search(self.vector_store.index_to_docstore_id[idx])
                 # FAISS returns distances (L2) by default, not similarity. 
                 # If embeddings were normalized, L2 distance can be converted to cosine similarity
                 # or we can just rank by distance (lower is better)
                 # For MiniLM (normalized), cosine_sim = 1 - (L2_dist^2 / 2)
                 # Let's use distance (lower is better) for now, or just rank by order.
                 # Let's stick to similarity_search_with_score for simplicity, which *should*
                 # use the embedder.
            
            # --- REVERTING to similarity_search_with_score (simpler) ---
            # This relies on FAISS.load_local correctly using self.embeddings_model
            results_with_scores = self.vector_store.similarity_search_with_score(
                processed_query_for_embedding, # Use query *without* prefix
                k=k_retrieval
            )
            # -----------------------------------------------------------

        except Exception as e:
            print(f"   ❌ Error during FAISS similarity search: {e}")
            return []
        
        if not results_with_scores:
             print("   Semantic search returned no initial results.")
             return []
        
        # 3. Apply filters to the retrieved results
        print(f"   Applying {len(filters_list)} filter(s) to {len(results_with_scores)} retrieved candidates...")
        filtered_docs = []
        
        for doc, score in results_with_scores:
            try:
                metadata = doc.metadata
                match = True 
                
                for attribute, operator, value in filters_list:
                    meta_value = metadata.get(attribute) 
                    if attribute == 'tags':
                         meta_value = metadata.get('tags_processed', [])
                    
                    if meta_value is None: 
                        match = False; break
                    
                    if attribute == 'tags':
                        tags_list = meta_value
                        filter_tag = str(value).lower()
                        if operator == 'contains' and (filter_tag not in [str(t).lower() for t in tags_list]):
                            match = False; break
                    else:
                        try:
                            meta_value_num = float(meta_value)
                            filter_value_num = float(value)
                            if operator == '<' and not (meta_value_num < filter_value_num): match = False; break
                            elif operator == '>' and not (meta_value_num > filter_value_num): match = False; break
                            elif operator == '=' and not (np.isclose(meta_value_num, filter_value_num)): match = False; break
                        except Exception:
                            match = False; break
                
                if match:
                    filtered_docs.append(doc) 
                    if len(filtered_docs) >= k:
                        break
            except Exception as e:
                print(f"Warning: Error applying filter on doc {doc.metadata.get('recipe_id')}: {e}")
                continue
                
        print(f"   Returning {len(filtered_docs)} docs after filtering.")
        return filtered_docs[:k]

    
    def format_context(self, docs):
        """Format retrieved documents into context string"""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_text = doc.metadata.get('llm_context_text', doc.page_content)
            context_parts.append(f"Recipe {i}:\n{context_text}\n") 
        return "\n".join(context_parts)
    
    def generate_response(self, query, context):
        """Generate natural language response using LLM"""
        prompt_template = ChatPromptTemplate.from_template("""
You are a helpful recipe recommendation assistant. Based on the user's query and the retrieved recipes below, provide a natural, conversational recommendation.

User Query: {query}

Retrieved Recipes:
{context}

Instructions:
- Recommend the most suitable recipes based on the query
- Mention key details like duration, calories, and why each recipe fits
- Be concise but informative
- Use a friendly, conversational tone
- If no recipes match well, explain why and suggest alternatives
- If recipes seem irrelevant, state that you found some options but they might not be a perfect match.
- Base your answer *only* on the text provided in "Retrieved Recipes".

Your Response:
""")
        chain = (
            {"query": RunnablePassthrough(), "context": RunnablePassthrough()}
            | prompt_template
            | self.llm
            | StrOutputParser()
        )
        response = chain.invoke({"query": query, "context": context})
        return response
    
    # --- MODIFIED: query function ---
    def query(self, user_query, k=5):
        """
        Complete RAG pipeline with query parsing.
        """
        print(f"\nProcessing query: '{user_query}'")
        
        # 1. Parse query into filters and text using imported parser
        start_time = time.time()
        try:
            filters_list, expanded_text_query = parse_query(user_query)
            print(f"   ✓ Query parsed in {time.time() - start_time:.2f}s")
            print(f"   Filters found: {filters_list}")
            print(f"   Text for search: '{expanded_text_query}'")
        except Exception as e:
            print(f"   ⚠️ Error during query parsing: {e}. Defaulting.")
            filters_list, expanded_text_query = [], user_query
        
        print(f"\nRetrieving top-{k} recipes...")
        start_time = time.time()
        
        # 2. Retrieve using parsed components
        retrieved_docs = self.retrieve_recipes(
            text_query=expanded_text_query, # Pass expanded text
            filters_list=filters_list,      # Pass filter list
            k=k
        )
        retrieval_time = time.time() - start_time
        
        if not retrieved_docs:
            print(f"✓ Retrieval complete in {retrieval_time:.2f}s. No recipes found matching criteria.")
            return "I couldn't find any recipes that matched your search and filter criteria. You could try removing a filter or rephrasing your query."
        
        print(f"✓ Retrieved {len(retrieved_docs)} recipes in {retrieval_time:.2f}s")
        
        # 3. Format context
        context = self.format_context(retrieved_docs)
        
        # 4. Generate response
        print("Generating response (using GPU via Ollama)...")
        start_time = time.time()
        response = self.generate_response(user_query, context) 
        generation_time = time.time() - start_time
        
        print(f"✓ Response generated in {generation_time:.2f}s")
        print(f"✓ Total query time: {retrieval_time + generation_time:.2f}s")
        
        return response


# Example usage
if __name__ == "__main__":
    # NLTK Download block
    nltk_resources = ['punkt', 'wordnet', 'stopwords']
    for resource in nltk_resources:
        try:
            if resource == 'punkt': nltk.data.find(f'tokenizers/{resource}')
            elif resource == 'wordnet': nltk.data.find(f'corpora/{resource}')
            elif resource == 'stopwords': nltk.data.find(f'corpora/{resource}')
        except LookupError:
            print(f"NLTK resource '{resource}' not found. Downloading...")
            nltk.download(resource, quiet=True)
    # ------------------------------------

    csv_path = "data/hummus_recipes_preprocessed.csv"
    # --- MODIFIED: Use the .npy file created by all-MiniLM-L6-v2 ---
    embedding_path = "data/recipe_embeddings.npy" 
    
    rag = RecipeRAGSystem(
        csv_path, 
        embedding_path=embedding_path,
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", # --- MODIFIED ---
        llm_model="phi3:mini", 
        vector_store_path="faiss_recipe_index_minilm", # --- MODIFIED: New Path ---
        force_rebuild=False,  # <-- SET TO TRUE ONCE TO REBUILD!
        batch_size=96 
    )
    
    # --- MODIFIED: Example Queries (no filter dicts) ---
    
    print("\n" + "="*80)
    print("EXAMPLE 1: Quick vegetarian dinner under 20 minutes")
    print("="*80)
    query1 = "Suggest a quick vegetarian dinner under 20 minutes."
    response1 = rag.query(query1, k=3)
    print("\n" + response1)
    
    print("\n" + "="*80)
    print("EXAMPLE 2: High protein breakfast")
    print("="*80)
    query2 = "I need a high protein breakfast recipe"
    response2 = rag.query(query2, k=3)
    print("\n" + response2) 
    
    print("\n" + "="*80)
    print("EXAMPLE 3: Low calorie lunch < 400 cal")
    print("="*80)
    query3 = "What's a healthy low calorie lunch option under 400 cal?"
    response3 = rag.query(query3, k=3)
    print("\n" + response3)
    
    print("\n" + "="*80)
    print("EXAMPLE 4: Gluten-free query")
    print("="*80)
    query5 = "gluten-free pasta with tomato"
    response5 = rag.query(query5, k=3)
    print("\n" + response5)

