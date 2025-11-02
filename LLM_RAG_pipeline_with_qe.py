import pandas as pd
# --- MODIFIED: Removed OllamaEmbeddings, as FAISS.from_documents will call the object ---
from langchain_community.embeddings import OllamaEmbeddings
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

# --- NEW: Import the full query parser ---
try:
    from query_parser import parse_query
    from text_preprocessing import preprocess_text # Also import this for query
except ImportError:
    print("="*50)
    print("FATAL ERROR: query_parser.py or text_preprocessing.py not found.")
    print("Please ensure structured_filters.py, query_expansion.py, and query_parser.py are present.")
    print("="*50)
    # Define a fallback parser that does nothing
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
                 embedding_model="all-minilm", 
                 llm_model="llama3.1:latest", 
                 vector_store_path="faiss_recipe_index_qe",
                 force_rebuild=False, 
                 batch_size=50):
        """
        Initialize the Recipe RAG System
        """
        self.csv_path = csv_path
        self.embedding_model_name = embedding_model # Store name
        self.llm_model = llm_model
        self.vector_store_path = vector_store_path
        self.batch_size = batch_size
        self.checkpoint_path = f"{vector_store_path}_checkpoint.pkl"
        self.metadata_path = f"{vector_store_path}_metadata.json"
        
        print("="*80)
        print("Initializing Recipe RAG System")
        print("="*80)
        
        print(f"\n[1/3] Initializing embeddings model ({self.embedding_model_name} via Ollama)...")
        start_time = time.time()
        self.embeddings = OllamaEmbeddings(model=self.embedding_model_name)
        print(f"   ✓ Embeddings model loaded in {time.time() - start_time:.2f}s")
        
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
        """Load existing vector store or create new one with progress tracking"""
        
        index_exists = os.path.exists(f"{self.vector_store_path}/index.faiss")
        metadata_exists = os.path.exists(self.metadata_path)
        
        if index_exists and metadata_exists and not force_rebuild:
            print(f"   ✓ Found existing vector store at '{self.vector_store_path}'")
            start_time = time.time()
            
            self.vector_store = FAISS.load_local(
                self.vector_store_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            
            print(f"   ✓ Loaded {metadata['num_documents']} documents in {time.time() - start_time:.2f}s")
            print(f"   ✓ Created: {metadata['created_at']}")
            
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
        
        # --- Create new vector store ---
        print("   ⚠ No existing vector store found or force rebuild requested")
        print("   → Starting vector store creation...\n")
        
        print("   Loading recipe data...")
        start_time = time.time()
        
        # --- MODIFIED: Load all columns needed for filtering/display ---
        metadata_cols = [
            'recipe_id', 'title', 'description', 'duration', 'tags', 
            'processed_tags', # *** ADDED PROCESSED_TAGS ***
            'ingredients', 'serves', 'calories_cal', 'protein_g', 'directions',
            'totalfat_g', 'saturatedfat_g', 'cholesterol_mg', 'sodium_mg', 
            'totalcarbohydrate_g', 'dietaryfiber_g', 'sugars_g',
            'ingredients_sizes', 'direction_size'
        ]
        try:
            all_cols_df = pd.read_csv(self.csv_path, nrows=0)
            cols_to_load = [col for col in metadata_cols if col in all_cols_df.columns]
            self.df = pd.read_csv(self.csv_path, usecols=cols_to_load, low_memory=False)
        except Exception as e:
            print(f"Error loading CSV: {e}. Check paths and column names.")
            raise
        print(f"   ✓ Loaded {len(self.df)} recipes in {time.time() - start_time:.2f}s")
        
        print("   Processing data...")
        start_time = time.time()
        self.process_data() # Cleans and filters self.df in place
        print(f"   ✓ Processed {len(self.df)} valid recipes in {time.time() - start_time:.2f}s\n")
        
        # --- MODIFIED: Build metadata_map *before* creating vector store ---
        print("   Preparing documents and metadata...")
        self.metadata_map = {}
        documents = []
        doc_index_counter = 0
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="   Creating Documents"):
            
            # --- MODIFIED: Create text for LLM (pretty) ---
            doc_text_for_llm = self.create_document_text(row) 
            
            # --- MODIFIED: Create text for EMBEDDING (Nomic prefix + processed) ---
            doc_text_for_embedding = self.create_document_text_for_embedding(row)

            tags_for_meta = []
            tags_data = row.get('processed_tags', "[]")
            if isinstance(tags_data, str) and tags_data.startswith('['):
                try: tags_for_meta = ast.literal_eval(tags_data)
                except: tags_for_meta = []
            elif isinstance(tags_data, list):
                tags_for_meta = tags_data
            
            metadata = {
                'recipe_id': str(row.get('recipe_id', str(doc_index_counter))),
                'title': str(row.get('title', 'N/A')),
                'duration': float(row.get('duration', 0)),
                'tags_raw': str(row.get('tags', '')), 
                'tags_processed': [str(t).lower() for t in tags_for_meta], # For filter
                'calories_cal': float(row.get('calories_cal', 0)),
                'protein_g': float(row.get('protein_g', 0)),
                'serves': str(row.get('serves', 'N/A')),
                'totalfat_g': float(row.get('totalfat_g', 0)),
                'saturatedfat_g': float(row.get('saturatedfat_g', 0)),
                'cholesterol_mg': float(row.get('cholesterol_mg', 0)),
                'sodium_mg': float(row.get('sodium_mg', 0)),
                'totalcarbohydrate_g': float(row.get('totalcarbohydrate_g', 0)),
                'dietaryfiber_g': float(row.get('dietaryfiber_g', 0)),
                'sugars_g': float(row.get('sugars_g', 0)),
                'ingredients_sizes': int(row.get('ingredients_sizes', 0)),
                'direction_size': int(row.get('direction_size', 0)),
            }
            
            # --- MODIFIED: Use doc_text_for_embedding for Document, doc_text_for_llm for metadata ---
            # FAISS.from_documents embeds 'page_content'. We want that to be the Nomic-prefixed string.
            # We store the *clean* text in the metadata for the LLM to read.
            metadata['llm_context_text'] = doc_text_for_llm # Store pretty text here
            
            doc = Document(page_content=doc_text_for_embedding, metadata=metadata)
            documents.append(doc)
            
            self.metadata_map[doc_index_counter] = metadata
            doc_index_counter += 1
        
        # --- Create vector store with progress tracking ---
        print(f"\n   Creating embeddings for {len(documents)} documents...")
        self.vector_store = self._create_vector_store_with_progress(documents)
        del self.df # Free memory

        # Save metadata
        metadata_json = {
            'num_documents': len(documents),
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'csv_path': self.csv_path,
            'embedding_model': self.embedding_model_name
        }
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata_json, f, indent=2)
        
        print(f"\n   ✓ Vector store saved to '{self.vector_store_path}'")

    def process_data(self):
        """Clean and process the recipe data"""
        initial_count = len(self.df)
        
        self.df['description'] = self.df['description'].fillna('')
        self.df['tags'] = self.df['tags'].fillna('')
        self.df['processed_tags'] = self.df['processed_tags'].fillna('[]') 
        self.df['ingredients'] = self.df['ingredients'].fillna('')
        self.df['directions'] = self.df['directions'].fillna('')
        self.df['serves'] = self.df['serves'].fillna('N/A')

        numeric_cols = ['duration', 'calories_cal', 'protein_g', 'totalfat_g', 
                        'saturatedfat_g', 'cholesterol_mg', 'sodium_mg', 
                        'totalcarbohydrate_g', 'dietaryfiber_g', 'sugars_g',
                        'ingredients_sizes', 'direction_size']
        for col in numeric_cols:
            if col in self.df.columns:
                 self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Drop only if truly essential data is missing
        drop_subset_existing = ['title', 'duration']
        
        self.df = self.df.dropna(subset=drop_subset_existing)
        
        # FillNa for numeric columns *after* dropping, so we keep more data
        for col in numeric_cols:
             if col in self.df.columns:
                 # Fill NaNs with 0 (or a suitable default)
                 self.df[col] = self.df[col].fillna(0)
                 if col in ['duration', 'ingredients_sizes', 'direction_size']:
                      self.df[col] = self.df[col].astype(int)
                 else:
                      self.df[col] = self.df[col].astype(float) 

        removed = initial_count - len(self.df)
        if removed > 0:
            print(f"   ⚠ Removed {removed} invalid recipes (NaN in title or duration)")
    
    def create_document_text(self, row):
        """Create a rich text representation of a recipe for the LLM context"""
        tags_display = row.get('tags', 'N/A')
        if pd.isna(tags_display) or tags_display in ['nan', '[]']:
             try: tags_display = ", ".join(ast.literal_eval(row.get('processed_tags', '[]')))
             except: tags_display = "N/A"
        
        # This is the "pretty" text the LLM will see
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

    # --- NEW: Function to create text for Nomic embedding ---
    def create_document_text_for_embedding(self, row):
        """
        Creates the specific text string to be embedded.
        Parses processed tokens from string-lists and joins them.
        """
        all_tokens = []
        # Use the *processed* columns for embedding
        text_cols = ["processed_title", "processed_ingredients", "processed_tags", "processed_directions"]
        
        for col in text_cols:
            text_repr = row.get(col, "[]")
            if isinstance(text_repr, str) and text_repr.startswith('[') and text_repr.endswith(']'):
                try:
                    actual_tokens = ast.literal_eval(text_repr)
                    if isinstance(actual_tokens, list):
                        all_tokens.extend(actual_tokens)
                except: pass
        
        text_to_embed = " ".join(all_tokens)
        # Add Nomic prefix
        return f"search_document: {text_to_embed}"

    def _create_vector_store_with_progress(self, documents: list):
        """Create vector store with progress tracking and incremental saving"""
        total_docs = len(documents)
        vector_store = None
        
        # --- Checkpoint logic removed for simplicity, as FAISS.from_documents is batch-aware ---
        # The OllamaEmbeddings class will handle batching internally
        
        print(f"\n   Embedding {total_docs} documents. This will take a long time...")
        print("   " + "="*76)
        start_time = time.time()
        
        try:
            vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings,
                # You can add show_progress=True if your langchain version supports it
            )
        except Exception as e:
            print(f"\n   ✗ Error during FAISS.from_documents: {e}")
            print("   This often happens if Ollama times out or errors during embedding.")
            raise

        total_time = time.time() - start_time
        print(f"\n   ✓ Embedded {total_docs} documents in {total_time/60:.2f} minutes")
        print(f"   ✓ Average speed: {total_docs/total_time:.2f} docs/second")
        
        # Save final vector store
        print("\n   Saving final vector store...")
        vector_store.save_local(self.vector_store_path)
        
        # Clean up (Checkpoint files already removed from logic)
        
        return vector_store

    # --- NEW: Filter function (using the map) ---
    def _filter_indices(self, filters_list: list) -> set:
        """
        Filters the internal metadata_map based on a list of filter tuples.
        Returns a set of allowed indices (integers).
        """
        if not filters_list:
            return set(self.metadata_map.keys()) # Return all indices

        print(f"   Applying {len(filters_list)} filter(s)...")
        allowed_indices = set()
        
        for index, meta in self.metadata_map.items():
            match_all = True
            for attribute, operator, value in filters_list:
                meta_value = meta.get(attribute)
                
                if meta_value is None: 
                    # Special check: 'tags' filter on 'processed_tags'
                    if attribute == 'tags':
                         meta_value = meta.get('tags_processed', []) # Check processed tags
                    else:
                         match_all = False; break # Attribute missing
                
                try:
                    if attribute == 'tags':
                        tags_list = meta_value # Should be meta.get('tags_processed', [])
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
                        elif operator == '=' and not (meta_value_num == filter_value_num):
                            match_all = False; break
                        
                except Exception: 
                    match_all = False; break
                
            if match_all:
                allowed_indices.add(index)
                
        print(f"   Found {len(allowed_indices)} recipes after filtering.")
        return allowed_indices

    # --- MODIFIED: retrieve_recipes ---
    def retrieve_recipes(self, text_query: str, filters_list: list, k: int):
        """Retrieve top-k recipes by filtering *after* semantic search."""
        
        # 1. Preprocess text query and add Nomic prefix
        processed_query_tokens = preprocess_text(text_query)
        processed_query_for_embedding = " ".join(processed_query_tokens)
        query_with_prefix = f"search_query: {processed_query_for_embedding}"
        
        print(f"   Text query for embedding: '{query_with_prefix}'")
        
        # 2. Perform semantic search (retrieve more candidates)
        # Note: This searches *all* docs, then filters. 
        # For large datasets, pre-filtering (if FAISS supports it) or
        # using a multi-stage retriever is better, but this matches the script's logic.
        print(f"   Retrieving {k*10} candidates *before* filtering...") # Fetch more
        results_with_scores = self.vector_store.similarity_search_with_score(
            query_with_prefix, 
            k=max(k*10, 50) # Get at least 50, or 10*k
        )
        
        if not results_with_scores:
             print("   Semantic search returned no results.")
             return []
        
        # 3. Apply filters to the retrieved results
        print(f"   Applying {len(filters_list)} filter(s) to retrieved candidates...")
        filtered_docs = []
        
        for doc, score in results_with_scores:
            try:
                metadata = doc.metadata
                match = True 
                
                for attribute, operator, value in filters_list:
                    meta_value = metadata.get(attribute) 
                    
                    # Special check: 'tags' filter on 'tags_processed'
                    if attribute == 'tags':
                         meta_value = metadata.get('tags_processed', [])
                    
                    if meta_value is None: match = False; break
                    
                    if attribute == 'tags':
                        tags_list = meta_value # This is now the clean list
                        filter_tag = str(value).lower()
                        if operator == 'contains' and (filter_tag not in tags_list):
                            match = False; break
                    
                    else: # Numerical filtering
                        meta_value_num = float(meta_value)
                        filter_value_num = float(value)
                        
                        if operator == '<' and not (meta_value_num < filter_value_num):
                            match = False; break
                        elif operator == '>' and not (meta_value_num > filter_value_num):
                            match = False; break
                        elif operator == '=' and not (meta_value_num == filter_value_num):
                            match = False; break
                
                if match:
                    filtered_docs.append(doc) 
                    if len(filtered_docs) >= k:
                        break # We found enough
            except Exception as e:
                print(f"Warning: Error applying filter on doc {doc.metadata.get('recipe_id')}: {e}")
                continue
                
        print(f"   Returning {len(filtered_docs)} docs after filtering.")
        return filtered_docs[:k]

    
    def format_context(self, docs):
        """Format retrieved documents into context string"""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            # --- MODIFIED: Use the 'llm_context_text' from metadata ---
            # This is the "pretty" text we stored during indexing
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
        # Use the *original* user query for the LLM prompt
        response = self.generate_response(user_query, context) 
        generation_time = time.time() - start_time
        
        print(f"✓ Response generated in {generation_time:.2f}s")
        print(f"✓ Total query time: {retrieval_time + generation_time:.2f}s")
        
        return response


# Example usage
if __name__ == "__main__":
    # --- MODIFIED: NLTK Download block ---
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
    
    rag = RecipeRAGSystem(
        csv_path, 
        embedding_model="minilm-l6-v2:latest", # BERT for embeddings
        llm_model="phi3:mini", # Phi-3 for generation
        vector_store_path="faiss_recipe_index_qe", # Use a new path for the bert and qe index
        force_rebuild=False,  # Change to True ONCE to build the Nomic index
        batch_size=96 # BERT minilm
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
    response5 = rag.query(query5, k= 3)
    print("\n" + response5)
