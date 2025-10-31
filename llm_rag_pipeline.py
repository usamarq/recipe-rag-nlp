import pandas as pd
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

warnings.filterwarnings('ignore')


class RecipeRAGSystem:
    def __init__(self, csv_path, embedding_model="nomic-embed-text:latest", 
                 llm_model="llama3.1:latest", vector_store_path="faiss_recipe_index",
                 force_rebuild=False, batch_size=50):
        """
        Initialize the Recipe RAG System
        
        Args:
            csv_path: Path to the CSV file containing recipes
            embedding_model: Ollama embedding model name
            llm_model: Ollama LLM model name
            vector_store_path: Path to save/load the FAISS index
            force_rebuild: If True, rebuild index even if it exists
            batch_size: Number of documents to process before saving checkpoint
        """
        self.csv_path = csv_path
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.vector_store_path = vector_store_path
        self.batch_size = batch_size
        self.checkpoint_path = f"{vector_store_path}_checkpoint.pkl"
        self.metadata_path = f"{vector_store_path}_metadata.json"
        
        # Initialize components
        print("="*80)
        print("Initializing Recipe RAG System")
        print("="*80)
        
        print("\n[1/3] Initializing embeddings model (using GPU via Ollama)...")
        start_time = time.time()
        self.embeddings = OllamaEmbeddings(model=self.embedding_model)
        print(f"   ✓ Embeddings model loaded in {time.time() - start_time:.2f}s")
        
        print("\n[2/3] Initializing LLM (using GPU via Ollama)...")
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
        
        # Check if complete vector store exists
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
            
            # Load metadata
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            
            print(f"   ✓ Loaded {metadata['num_documents']} documents in {time.time() - start_time:.2f}s")
            print(f"   ✓ Created: {metadata['created_at']}")
            return
        
        # Need to create vector store
        print("   ⚠ No existing vector store found or force rebuild requested")
        print("   → Starting vector store creation...\n")
        
        # Load data
        print("   Loading recipe data...")
        start_time = time.time()
        self.df = pd.read_csv(self.csv_path)
        print(f"   ✓ Loaded {len(self.df)} recipes in {time.time() - start_time:.2f}s")
        
        # Process data
        print("   Processing data...")
        start_time = time.time()
        self.process_data()
        print(f"   ✓ Processed {len(self.df)} valid recipes in {time.time() - start_time:.2f}s\n")
        
        # Create vector store with progress tracking
        self.vector_store = self._create_vector_store_with_progress()
        
        # Save metadata
        metadata = {
            'num_documents': len(self.df),
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'csv_path': self.csv_path,
            'embedding_model': self.embedding_model
        }
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n   ✓ Vector store saved to '{self.vector_store_path}'")

    def process_data(self):
        """Clean and process the recipe data"""
        initial_count = len(self.df)
        
        self.df['description'] = self.df['description'].fillna('')
        self.df['tags'] = self.df['tags'].fillna('')
        self.df['duration'] = pd.to_numeric(self.df['duration'], errors='coerce')
        self.df['calories_cal'] = pd.to_numeric(self.df['calories_cal'], errors='coerce')
        self.df['protein_g'] = pd.to_numeric(self.df['protein_g'], errors='coerce')
        self.df = self.df.dropna(subset=['title', 'duration'])
        
        removed = initial_count - len(self.df)
        if removed > 0:
            print(f"   ⚠ Removed {removed} invalid recipes")
    
    def create_document_text(self, row):
        """Create a rich text representation of a recipe for embedding"""
        doc_text = f"""
Title: {row['title']}
Description: {row['description']}
Duration: {row['duration']} minutes
Tags: {row['tags']}
Ingredients: {row['ingredients'] if pd.notna(row['ingredients']) else 'N/A'}
Calories: {row['calories_cal']} cal
Protein: {row['protein_g']}g
Serves: {row['serves']}
Directions: {row['directions'][:200] if pd.notna(row['directions']) else 'N/A'}...
        """.strip()
        return doc_text

    def _create_vector_store_with_progress(self):
        """Create vector store with progress tracking and incremental saving"""
        total_docs = len(self.df)
        
        # Check for checkpoint
        start_idx = 0
        documents = []
        vector_store = None
        
        if os.path.exists(self.checkpoint_path):
            print("   ✓ Found checkpoint file, resuming from last saved position...")
            try:
                with open(self.checkpoint_path, 'rb') as f:
                    checkpoint = pickle.load(f)
                    start_idx = checkpoint['last_index']
                    print(f"   → Resuming from document {start_idx}/{total_docs}")
                    
                    # Load partial vector store if it exists
                    if os.path.exists(f"{self.vector_store_path}_partial"):
                        vector_store = FAISS.load_local(
                            f"{self.vector_store_path}_partial",
                            self.embeddings,
                            allow_dangerous_deserialization=True
                        )
                        print(f"   ✓ Loaded partial vector store with {start_idx} documents")
            except Exception as e:
                print(f"   ⚠ Error loading checkpoint: {e}")
                print("   → Starting from beginning")
                start_idx = 0
                vector_store = None
        
        print(f"\n   Creating embeddings for documents {start_idx} to {total_docs}")
        print("   " + "="*76)
        
        overall_start_time = time.time()
        batch_start_time = time.time()
        
        # Progress bar
        with tqdm(total=total_docs-start_idx, desc="   Embedding", 
                  unit="docs", ncols=80, initial=0) as pbar:
            
            for idx in range(start_idx, total_docs):
                row = self.df.iloc[idx]
                
                # Create document
                doc_text = self.create_document_text(row)
                metadata = {
                    'recipe_id': str(row['recipe_id']) if pd.notna(row['recipe_id']) else str(idx),
                    'title': str(row['title']),
                    'duration': float(row['duration']) if pd.notna(row['duration']) else 0,
                    'tags': str(row['tags']),
                    'calories': float(row['calories_cal']) if pd.notna(row['calories_cal']) else 0,
                    'protein': float(row['protein_g']) if pd.notna(row['protein_g']) else 0,
                    'serves': str(row['serves']) if pd.notna(row['serves']) else 'N/A',
                }
                doc = Document(page_content=doc_text, metadata=metadata)
                documents.append(doc)
                
                # Process batch
                if len(documents) >= self.batch_size or idx == total_docs - 1:
                    try:
                        if vector_store is None:
                            # Create initial vector store
                            vector_store = FAISS.from_documents(
                                documents=documents,
                                embedding=self.embeddings
                            )
                        else:
                            # Add to existing vector store
                            new_store = FAISS.from_documents(
                                documents=documents,
                                embedding=self.embeddings
                            )
                            vector_store.merge_from(new_store)
                        
                        # Save checkpoint
                        vector_store.save_local(f"{self.vector_store_path}_partial")
                        with open(self.checkpoint_path, 'wb') as f:
                            pickle.dump({'last_index': idx + 1}, f)
                        
                        # Calculate statistics
                        batch_time = time.time() - batch_start_time
                        docs_per_sec = len(documents) / batch_time if batch_time > 0 else 0
                        elapsed_total = time.time() - overall_start_time
                        remaining_docs = total_docs - (idx + 1)
                        eta_seconds = remaining_docs / docs_per_sec if docs_per_sec > 0 else 0
                        
                        pbar.set_postfix({
                            'speed': f'{docs_per_sec:.1f} docs/s',
                            'ETA': f'{eta_seconds/60:.1f}m'
                        })
                        
                        pbar.update(len(documents))
                        documents = []
                        batch_start_time = time.time()
                        
                    except Exception as e:
                        print(f"\n   ✗ Error processing batch at index {idx}: {e}")
                        raise
        
        total_time = time.time() - overall_start_time
        print(f"\n   ✓ Embedded {total_docs} documents in {total_time/60:.2f} minutes")
        print(f"   ✓ Average speed: {total_docs/total_time:.2f} docs/second")
        
        # Save final vector store
        print("\n   Saving final vector store...")
        vector_store.save_local(self.vector_store_path)
        
        # Clean up checkpoint files
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
        if os.path.exists(f"{self.vector_store_path}_partial"):
            import shutil
            shutil.rmtree(f"{self.vector_store_path}_partial")
        
        return vector_store

    def retrieve_recipes(self, query, k=5, filters=None):
        """Retrieve top-k recipes based on query"""
        results = self.vector_store.similarity_search(query, k=k*3)
        
        if filters:
            filtered_results = []
            for doc in results:
                if 'max_duration' in filters:
                    if doc.metadata['duration'] > filters['max_duration']:
                        continue
                if 'min_duration' in filters:
                    if doc.metadata['duration'] < filters['min_duration']:
                        continue
                if 'tags' in filters:
                    tags_lower = doc.metadata['tags'].lower()
                    filter_tags = filters['tags'].lower()
                    if filter_tags not in tags_lower:
                        continue
                if 'max_calories' in filters:
                    if doc.metadata['calories'] > filters['max_calories']:
                        continue
                if 'min_protein' in filters:
                    if doc.metadata['protein'] < filters['min_protein']:
                        continue
                filtered_results.append(doc)
                if len(filtered_results) >= k:
                    break
            return filtered_results[:k]
        return results[:k]
    
    def format_context(self, docs):
        """Format retrieved documents into context string"""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"""
Recipe {i}:
Title: {doc.metadata['title']}
Duration: {doc.metadata['duration']} minutes
Calories: {doc.metadata['calories']} cal
Protein: {doc.metadata['protein']}g
Serves: {doc.metadata['serves']}
Tags: {doc.metadata['tags']}
Details: {doc.page_content[:300]}...
""")
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
    
    def query(self, user_query, k=5, filters=None):
        """Complete RAG pipeline"""
        print(f"\nProcessing query: '{user_query}'")
        if filters:
            print(f"Filters: {filters}")
        
        start_time = time.time()
        print(f"\nRetrieving top-{k} recipes...")
        retrieved_docs = self.retrieve_recipes(user_query, k=k, filters=filters)
        retrieval_time = time.time() - start_time
        
        if not retrieved_docs:
            return "I couldn't find any recipes matching your criteria. Try adjusting your filters or query."
        
        print(f"✓ Retrieved {len(retrieved_docs)} recipes in {retrieval_time:.2f}s")
        
        context = self.format_context(retrieved_docs)
        
        print("Generating response (using GPU via Ollama)...")
        start_time = time.time()
        response = self.generate_response(user_query, context)
        generation_time = time.time() - start_time
        
        print(f"✓ Response generated in {generation_time:.2f}s")
        print(f"✓ Total query time: {retrieval_time + generation_time:.2f}s")
        
        return response


# Example usage
if __name__ == "__main__":
    # Initialize the RAG system
    csv_path = "data/hummus_recipes_preprocessed.csv"
    
    # Set force_rebuild=True to rebuild from scratch
    # Set force_rebuild=False to use existing index (default)
    rag = RecipeRAGSystem(
        csv_path, 
        force_rebuild=False,  # Change to True to rebuild
        batch_size=50  # Save every 50 documents
    )
    
    # Example Query 1: Quick vegetarian dinner
    print("\n" + "="*80)
    print("EXAMPLE 1: Quick vegetarian dinner")
    print("="*80)
    
    query1 = "Suggest a quick vegetarian dinner under 20 minutes."
    filters1 = {
        'max_duration': 20,
        'tags': 'vegetarian'
    }
    
    response1 = rag.query(query1, k=3, filters=filters1)
    print("\n" + response1)
    
    # Example Query 2: High protein breakfast
    print("\n" + "="*80)
    print("EXAMPLE 2: High protein breakfast")
    print("="*80)
    
    query2 = "I need a high protein breakfast recipe"
    filters2 = {
        'min_protein': 15,
        'max_duration': 30
    }
    
    response2 = rag.query(query2, k=3, filters=filters2)
    print("\n" + response2)
    
    # Example Query 3: Low calorie lunch
    print("\n" + "="*80)
    print("EXAMPLE 3: Low calorie lunch")
    print("="*80)
    
    query3 = "What's a healthy low calorie lunch option?"
    filters3 = {
        'max_calories': 400
    }
    
    response3 = rag.query(query3, k=3, filters=filters3)
    print("\n" + response3)
    
    # Example Query 4: No filters
    print("\n" + "="*80)
    print("EXAMPLE 4: General pasta query")
    print("="*80)
    
    query4 = "Show me some pasta recipes"
    response4 = rag.query(query4, k=3)
    print("\n" + response4)