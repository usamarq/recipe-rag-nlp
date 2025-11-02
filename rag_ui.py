import streamlit as st
import pandas as pd
import os
from pathlib import Path

# Import your RAG system
# Assuming the previous code is in a file called 'recipe_rag.py'
from rag_existing_embeddings import RecipeRAGSystem

# Page configuration
st.set_page_config(
    page_title="Recipe Recommendation System",
    page_icon="🍳",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* I added a body rule to set the main page background to dark */
    body {
        background-color: #121212; 
    }

    .recipe-card {
        background-color: #1e1e1e; /* Changed from light grey to dark grey */
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 5px solid #4CAF50;
    }
    .recipe-title {
        color: #e0e0e0; /* Changed from dark to light grey for readability */
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .recipe-meta {
        color: #a0a0a0; /* Changed from dark grey to lighter grey */
        font-size: 14px;
        margin-bottom: 10px;
    }
    .nutrition-info {
        display: inline-block;
        background-color: #2c3a2c; /* Changed from light green to dark green */
        color: #b2f3b2; /* Added light green text color for contrast */
        padding: 5px 10px;
        border-radius: 5px;
        margin-right: 10px;
        margin-bottom: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'results' not in st.session_state:
    st.session_state.results = None

# Initialize RAG system
@st.cache_resource
def initialize_rag():
    """Initialize the RAG system (cached for performance)"""
    csv_path = "data/hummus_recipes_preprocessed.csv"
    embedding_path = "data/recipe_embeddings.npy"
    
    with st.spinner("Loading RAG system... This may take a moment."):
        rag = RecipeRAGSystem(csv_path, embedding_path=embedding_path)
    return rag

# Header
st.title("🍳 Recipe Recommendation System")
st.markdown("### Find the perfect recipe with AI-powered search")

# Sidebar for filters
st.sidebar.header("🔍 Search Filters")

# Query input
query = st.text_input(
    "What are you looking for?",
    placeholder="e.g., Quick vegetarian dinner, high protein breakfast..."
)

# Number of results
k = st.sidebar.slider("Number of results", min_value=1, max_value=10, value=3)

# Filter options
st.sidebar.subheader("Filter Options")

# Duration filter
use_duration = st.sidebar.checkbox("Filter by cooking time")
if use_duration:
    duration_range = st.sidebar.slider(
        "Maximum cooking time (minutes)",
        min_value=5,
        max_value=120,
        value=30,
        step=5
    )
else:
    duration_range = None

# Calorie filter
use_calories = st.sidebar.checkbox("Filter by calories")
if use_calories:
    max_calories = st.sidebar.number_input(
        "Maximum calories",
        min_value=0,
        max_value=2000,
        value=500,
        step=50
    )
else:
    max_calories = None

# Protein filter
use_protein = st.sidebar.checkbox("Filter by protein")
if use_protein:
    min_protein = st.sidebar.number_input(
        "Minimum protein (g)",
        min_value=0,
        max_value=100,
        value=10,
        step=5
    )
else:
    min_protein = None

# Tags filter
use_tags = st.sidebar.checkbox("Filter by tags")
if use_tags:
    tags_input = st.sidebar.text_input(
        "Tags (e.g., vegetarian, vegan, gluten-free)",
        placeholder="Enter tags..."
    )
else:
    tags_input = None

# Search button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    search_button = st.button("🔍 Search Recipes", use_container_width=True)

# Main search logic
if search_button and query:
    # Initialize RAG system if not already done
    if st.session_state.rag_system is None:
        st.session_state.rag_system = initialize_rag()
    
    # Build filters dictionary
    filters = {}
    if duration_range:
        filters['max_duration'] = duration_range
    if max_calories:
        filters['max_calories'] = max_calories
    if min_protein:
        filters['min_protein'] = min_protein
    if tags_input:
        filters['tags'] = tags_input
    
    # Execute search
    with st.spinner("Searching for recipes..."):
        try:
            # Get retrieved documents
            retrieved_docs = st.session_state.rag_system.retrieve_recipes(
                query, k=k, filters=filters if filters else None
            )
            
            # Get LLM response
            if retrieved_docs:
                context = st.session_state.rag_system.format_context(retrieved_docs)
                llm_response = st.session_state.rag_system.generate_response(query, context)
            else:
                llm_response = "No recipes found matching your criteria."
            
            st.session_state.results = {
                'docs': retrieved_docs,
                'response': llm_response
            }
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Display results
if st.session_state.results:
    results = st.session_state.results
    
    # Display AI recommendation
    st.markdown("---")
    st.subheader("🤖 AI Recommendation")
    st.info(results['response'])
    
    # Display recipe cards
    st.markdown("---")
    st.subheader(f"📋 Top {len(results['docs'])} Recipe(s)")
    
    for i, doc in enumerate(results['docs'], 1):
        metadata = doc.metadata
        
        # Create columns for layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Display image if available
            if 'image_url' in metadata and metadata['image_url']:
                try:
                    st.image(metadata['image_url'], use_container_width=True)
                except:
                    st.image("https://via.placeholder.com/300x200?text=No+Image", use_container_width=True)
            else:
                st.image("https://via.placeholder.com/300x200?text=No+Image", use_container_width=True)
        
        with col2:
            # Recipe title
            st.markdown(f"<div class='recipe-title'>{i}. {metadata['title']}</div>", unsafe_allow_html=True)
            
            # Recipe metadata
            duration = metadata.get('duration', 'N/A')
            serves = metadata.get('serves', 'N/A')
            st.markdown(f"<div class='recipe-meta'>⏱️ {duration} min | 🍽️ Serves {serves}</div>", unsafe_allow_html=True)
            
            # Nutrition information
            st.markdown("**Nutrition:**")
            col_cal, col_prot, col_tags = st.columns(3)
            
            with col_cal:
                calories = metadata.get('calories', 0)
                st.markdown(f"<div class='nutrition-info'>🔥 {calories:.0f} cal</div>", unsafe_allow_html=True)
            
            with col_prot:
                protein = metadata.get('protein', 0)
                st.markdown(f"<div class='nutrition-info'>💪 {protein:.1f}g protein</div>", unsafe_allow_html=True)
            
            with col_tags:
                tags = metadata.get('tags', 'N/A')
                if tags and tags != 'N/A':
                    # Display first few tags
                    tag_list = tags.split(',')[:3]
                    for tag in tag_list:
                        st.markdown(f"<span style='background-color: #000000; padding: 3px 8px; border-radius: 3px; margin-right: 5px; font-size: 12px;'>🏷️ {tag.strip()}</span>", unsafe_allow_html=True)
            
            # Expandable section for full details
            with st.expander("View Full Recipe Details"):
                st.markdown(doc.page_content)
        
        st.markdown("---")

# Footer with instructions
if not query:
    st.markdown("---")
    st.markdown("""
    ### How to use:
    1. **Enter your search query** in the text box above
    2. **Apply filters** (optional) using the sidebar
    3. **Click Search** to find recipes
    4. **View results** with AI recommendations and detailed recipe cards
    
    **Example queries:**
    - "Quick vegetarian dinner under 30 minutes"
    - "High protein breakfast recipes"
    - "Low calorie lunch options"
    - "Easy pasta recipes"
    """)

# Initialize RAG on first load
if st.session_state.rag_system is None and not search_button:
    with st.spinner("Initializing system..."):
        st.session_state.rag_system = initialize_rag()
    st.success("System ready! Enter a query to get started.")