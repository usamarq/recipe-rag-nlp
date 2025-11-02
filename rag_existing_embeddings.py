import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
import warnings
import os
import numpy as np
import re
import nltk

# Custom Embeddings wrapper for Sentence Transformers
class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper to make SentenceTransformer compatible with LangChain"""
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        # Check device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        print(f"Loading SentenceTransformer on {self.device.upper()}...")
        self.model = SentenceTransformer(model_name, device=self.device)
    
    def embed_documents(self, texts):
        """Embed a list of documents"""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings.tolist()
    
    def embed_query(self, text):
        """Embed a single query"""
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embedding.tolist()

# Preprocessing function
try:
    from text_preprocessing import preprocess_text
except ImportError:
    print("Warning: text_preprocessing.py not found. Using basic fallback for query processing.")
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    def preprocess_text(text: str):
        if not isinstance(text, str):
            return []
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text).lower()
        text = re.sub(r'(\d+)\s*(g|gram|ml|kg|tbsp|tsp|cup|cups|teaspoon|tablespoon|minute|min|hour|hr|cal|kcal)', r'\1\2', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = word_tokenize(text)
        cleaned = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words and len(tok) > 1]
        return cleaned

warnings.filterwarnings('ignore')


class RecipeRAGSystem:
    def __init__(self, csv_path, 
                 embedding_path="./data/recipe_embeddings.npy",
                 llm_model="phi3:mini", 
                 vector_store_path="faiss_recipe_index_all-MiniLM-L6-v2"):
        """
        Initialize the Recipe RAG System
        """
        self.csv_path = csv_path
        self.embedding_path = embedding_path
        self.llm_model = llm_model
        self.vector_store_path = vector_store_path
        
        # Use SentenceTransformer embeddings (matching your embedding creation)
        print("Initializing SentenceTransformer embeddings...")
        self.embeddings = SentenceTransformerEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        print("Initializing LLM (this will use the GPU via Ollama)...")
        self.llm = Ollama(model=self.llm_model, temperature=0.7)
        
        # Load or create vector store
        if os.path.exists(self.vector_store_path):
            print(f"Loading existing vector store from {self.vector_store_path}...")
            self.vector_store = FAISS.load_local(
                self.vector_store_path, 
                self.embeddings,
                allow_dangerous_deserialization=True 
            )
        else:
            print(f"No existing FAISS index found at {self.vector_store_path}.")
            print(f"Building new index from CSV ({self.csv_path}) and pre-computed embeddings ({self.embedding_path})...")
            
            if not os.path.exists(self.embedding_path):
                raise FileNotFoundError(f"Required embedding file not found: {self.embedding_path}.")
            if not os.path.exists(self.csv_path):
                raise FileNotFoundError(f"Required CSV data file not found: {self.csv_path}.")

            # Load RAW recipe data
            print("Loading raw recipe data...")
            df_raw = pd.read_csv(self.csv_path, low_memory=False)
            
            # Load pre-computed embeddings
            print(f"Loading pre-computed embeddings from {self.embedding_path}...")
            precomputed_embeddings = np.load(self.embedding_path)
            
            # Sanity check
            if len(df_raw) != len(precomputed_embeddings):
                raise ValueError(
                    f"Embedding-Data Mismatch: The RAW CSV file has {len(df_raw)} rows, "
                    f"but the embedding file has {len(precomputed_embeddings)} vectors. "
                    "These files must correspond one-to-one."
                )

            # Process the data to find out which rows to KEEP
            print("Processing data to filter valid recipes...")
            self.df = df_raw.copy()
            self.process_data()
            
            # Get the indices of the rows that were KEPT
            kept_indices = self.df.index
            
            print(f"Original data: {len(df_raw)} rows. Processed data: {len(self.df)} rows.")
            
            # Filter the embeddings to match
            embeddings_processed = precomputed_embeddings[kept_indices]

            print("Preparing (document, embedding) pairs and metadata...")
            text_embeddings_list = []
            metadatas_list = []
            
            # Combine "Pretty" Text with "Processed" Embeddings
            for (idx, row), emb in zip(self.df.iterrows(), embeddings_processed):
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
                
                text_embeddings_list.append((doc_text, emb.tolist()))
                metadatas_list.append(metadata)
            
            print(f"Creating vector store from {len(text_embeddings_list)} pre-computed embeddings...")
            self.vector_store = FAISS.from_embeddings(
                text_embeddings=text_embeddings_list,
                embedding=self.embeddings,
                metadatas=metadatas_list
            )
            
            print(f"Saving new vector store to {self.vector_store_path}...")
            self.vector_store.save_local(self.vector_store_path)
        
        print("RAG System initialized successfully!")

    def process_data(self):
        """Clean and process the recipe data"""
        self.df['description'] = self.df['description'].fillna('')
        self.df['tags'] = self.df['tags'].fillna('')
        self.df['duration'] = pd.to_numeric(self.df['duration'], errors='coerce')
        self.df['calories_cal'] = pd.to_numeric(self.df['calories_cal'], errors='coerce')
        self.df['protein_g'] = pd.to_numeric(self.df['protein_g'], errors='coerce')
        self.df = self.df.dropna(subset=['title', 'duration'])
    
    def create_document_text(self, row):
        """Create a rich text representation of a recipe for the LLM context"""
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

    def retrieve_recipes(self, query, k=3, filters=None):
        """
        Retrieve top-k recipes based on query
        """
        # Preprocess the query to match the document embeddings
        processed_query_tokens = preprocess_text(query)
        processed_query = " ".join(processed_query_tokens)
        print(f"   Original query: '{query}'")
        print(f"   Processed query for embedding: '{processed_query}'")
        
        if not processed_query:
            print("   Query is empty after processing.")
            return []
        
        # Search using the *processed* query
        results = self.vector_store.similarity_search(processed_query, k=k*3)
        
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
            context_parts.append(f"Recipe {i}:\n{doc.page_content}\n")
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
        """
        Complete RAG pipeline
        """
        print(f"\nProcessing query: '{user_query}'")
        print(f"Filters: {filters}")
        
        print(f"\nRetrieving top-{k} recipes...")
        retrieved_docs = self.retrieve_recipes(user_query, k=k, filters=filters)
        
        if not retrieved_docs:
            return "I couldn't find any recipes matching your criteria. Try adjusting your filters or query."
        
        print(f"Retrieved {len(retrieved_docs)} recipes")
        
        context = self.format_context(retrieved_docs)
        
        print("Generating response (using M1 GPU via Ollama)...")
        response = self.generate_response(user_query, context)
        
        return response


# Example usage
if __name__ == "__main__":
    # Ensure NLTK data is available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("NLTK 'punkt' not found. Downloading...")
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("NLTK 'wordnet' not found. Downloading...")
        nltk.download('wordnet')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("NLTK 'stopwords' not found. Downloading...")
        nltk.download('stopwords')

    csv_path = "data/hummus_recipes_preprocessed.csv" 
    embedding_path = "data/recipe_embeddings.npy"
    
    # Initialize with SentenceTransformer embeddings
    rag = RecipeRAGSystem(
        csv_path, 
        embedding_path=embedding_path
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
    
    # Example Query 4: General pasta query
    print("\n" + "="*80)
    print("EXAMPLE 4: General pasta query")
    print("="*80)
    
    query4 = "Show me some pasta recipes"
    response4 = rag.query(query4, k=3)
    print("\n" + response4)