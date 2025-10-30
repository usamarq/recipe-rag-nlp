import pandas as pd
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import warnings
warnings.filterwarnings('ignore')


class RecipeRAGSystem:
    def __init__(self, csv_path, embedding_model="nomic-embed-text:latest", llm_model="llama3.1:latest"):
        """
        Initialize the Recipe RAG System
        
        Args:
            csv_path: Path to the CSV file containing recipes
            embedding_model: Ollama embedding model name
            llm_model: Ollama LLM model name
        """
        self.csv_path = csv_path
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        
        # Initialize components
        print("Initializing embeddings...")
        self.embeddings = OllamaEmbeddings(model=self.embedding_model)
        
        print("Initializing LLM...")
        self.llm = Ollama(model=self.llm_model, temperature=0.7)
        
        # Load and process data
        print("Loading recipe data...")
        self.df = pd.read_csv(csv_path)
        self.process_data()
        
        # Create vector store
        print("Creating vector store...")
        self.vector_store = self.create_vector_store()
        
        print("RAG System initialized successfully!")
    
    def process_data(self):
        """Clean and process the recipe data"""
        # Handle missing values
        self.df['description'] = self.df['description'].fillna('')
        self.df['tags'] = self.df['tags'].fillna('')
        self.df['duration'] = pd.to_numeric(self.df['duration'], errors='coerce')
        self.df['calories_cal'] = pd.to_numeric(self.df['calories_cal'], errors='coerce')
        self.df['protein_g'] = pd.to_numeric(self.df['protein_g'], errors='coerce')
        
        # Drop rows with missing critical data
        self.df = self.df.dropna(subset=['title', 'duration'])
    
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
    
    def create_vector_store(self):
        """Create vector store from recipe data"""
        documents = []
        
        for idx, row in self.df.iterrows():
            # Create document text
            doc_text = self.create_document_text(row)
            
            # Create metadata
            metadata = {
                'recipe_id': str(row['recipe_id']) if pd.notna(row['recipe_id']) else str(idx),
                'title': str(row['title']),
                'duration': float(row['duration']) if pd.notna(row['duration']) else 0,
                'tags': str(row['tags']),
                'calories': float(row['calories_cal']) if pd.notna(row['calories_cal']) else 0,
                'protein': float(row['protein_g']) if pd.notna(row['protein_g']) else 0,
                'serves': str(row['serves']) if pd.notna(row['serves']) else 'N/A',
            }
            
            # Create document
            doc = Document(page_content=doc_text, metadata=metadata)
            documents.append(doc)
        
        # Create vector store
        vector_store = InMemoryVectorStore.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        return vector_store
    
    def retrieve_recipes(self, query, k=5, filters=None):
        """
        Retrieve top-k recipes based on query
        
        Args:
            query: User query string
            k: Number of recipes to retrieve
            filters: Dictionary of filters (e.g., {'max_duration': 20, 'tags': 'vegetarian'})
        
        Returns:
            List of retrieved documents
        """
        # Perform similarity search
        results = self.vector_store.similarity_search(query, k=k*3)  # Get more to filter
        
        # Apply filters if provided
        if filters:
            filtered_results = []
            for doc in results:
                # Duration filter
                if 'max_duration' in filters:
                    if doc.metadata['duration'] > filters['max_duration']:
                        continue
                
                if 'min_duration' in filters:
                    if doc.metadata['duration'] < filters['min_duration']:
                        continue
                
                # Tags filter
                if 'tags' in filters:
                    tags_lower = doc.metadata['tags'].lower()
                    filter_tags = filters['tags'].lower()
                    if filter_tags not in tags_lower:
                        continue
                
                # Calories filter
                if 'max_calories' in filters:
                    if doc.metadata['calories'] > filters['max_calories']:
                        continue
                
                # Protein filter
                if 'min_protein' in filters:
                    if doc.metadata['protein'] < filters['min_protein']:
                        continue
                
                filtered_results.append(doc)
                
                # Stop when we have enough results
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
        
        # Create chain
        chain = (
            {"query": RunnablePassthrough(), "context": RunnablePassthrough()}
            | prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        # Generate response
        response = chain.invoke({"query": query, "context": context})
        return response
    
    def query(self, user_query, k=5, filters=None):
        """
        Complete RAG pipeline
        
        Args:
            user_query: Natural language query from user
            k: Number of recipes to retrieve
            filters: Optional filters dictionary
        
        Returns:
            Natural language response
        """
        print(f"\nProcessing query: '{user_query}'")
        print(f"Filters: {filters}")
        
        # Step 1: Retrieve top-k recipes
        print(f"\nRetrieving top-{k} recipes...")
        retrieved_docs = self.retrieve_recipes(user_query, k=k, filters=filters)
        
        if not retrieved_docs:
            return "I couldn't find any recipes matching your criteria. Try adjusting your filters or query."
        
        print(f"Retrieved {len(retrieved_docs)} recipes")
        
        # Step 2: Format context
        context = self.format_context(retrieved_docs)
        
        # Step 3: Generate response using LLM
        print("Generating response...")
        response = self.generate_response(user_query, context)
        
        return response


# Example usage
if __name__ == "__main__":
    # Initialize the RAG system
    csv_path = "data/hummus_recipes_preprocessed.csv" 
    rag = RecipeRAGSystem(csv_path)
    
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