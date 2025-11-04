# Project 22 — Retrieval-Augmented Recipe Search and Information Retrieval

Build a recipe retrieval and QA system over the HUMMUS dataset using keyword IR (BM25), semantic embeddings, hybrid retrieval, structured filtering, query understanding/expansion, and RAG with an LLM. Includes EDA, evaluation, and an optional Streamlit UI.

## Repo structure (key files)
- `download_data.py` — Download HUMMUS dataset to `data/`.
- `basic_cleaning.py` — Basic data cleaning (numeric conversions, filtering).
- `text_preprocessing.py` — Normalization/tokenization/lemmatization for recipes; dataset-level preprocessing.
- `bm25_retrieval_optimized.py` — Build and use a BM25 inverted index; save/load index + metadata.
- `semantic_retrieval_BERT.py` — Create Sentence-BERT embeddings and run semantic retrieval.
- `hybrid_retrieval_strategy.py` — Hybrid search (BM25 + semantic) with query parsing + structured filters.
- `structured_filters.py` — Map vague terms (e.g., “healthy”, “quick”) to numeric filters.
- `query_expansion.py` — Query expansion via WordNet synonyms + Lesk-based refinement.
- `query_parser.py` — Extract explicit numeric/tag filters; apply vague-term filters; expand query.
- `llm_rag_pipeline.py` — RAG pipeline using Ollama embeddings + LLM, FAISS vector store.
- `LLM_RAG_pipeline_with_qe.py` — RAG pipeline integrated with the query parser and processed columns.
- `rag_existing_embeddings.py` — RAG pipeline using precomputed Sentence-BERT embeddings.
- `rag_ui.py` — Streamlit UI for interactive search + AI explanation.
- `task9_comparision.py` — Automated evaluation (Precision@k, Recall@k, MRR, NDCG, MAP, F1) across systems.
- `result_visuals.py` — Visualizations over evaluation outputs.
- `data_exploration.ipynb` — EDA notebook.
- `data/` — Data, preprocessed CSV, indices, embeddings.
- `evaluation_results_*` — Saved evaluation outputs.
- `environment.yml`, `pyproject.toml` — Dependency versions.

## How to run

### 1) Environment
You can use Conda (recommended) or `pip`:

- Conda:
  - Create env: `conda env create -f environment.yml`
  - Activate: `conda activate nlp310`

- Pip:
  - Python: 3.10
  - Install dependencies listed below (see “Dependencies and versions”). You can also derive a `requirements.txt` from `pyproject.toml` if preferred.

Note: Some RAG scripts require [Ollama](https://ollama.com) with models (e.g., `llama3.1`, `phi3`, `all-minilm`); ensure `ollama` is installed and models pulled.

### 2) Data
- Download HUMMUS dataset:
  - `python download_data.py`  → saves to `data/hummus_recipes.csv`

### 3) Cleaning and preprocessing
- Basic cleaning:
  - `python basic_cleaning.py` → outputs `data/hummus_recipes_cleaned_basic.csv`
- Text preprocessing (tokenization, lemmatization, numeric-unit merges):
  - `python text_preprocessing.py` → outputs `data/hummus_recipes_preprocessed.csv`

### 4) Keyword search (BM25)
- Build/load BM25 index + metadata and run example queries:
  - `python bm25_retrieval_optimized.py`
  - Artifacts:
    - `data/bm25_index.pkl`
    - `data/recipe_metadata.pkl`

### 5) Semantic embeddings retrieval
- Create Sentence-BERT embeddings and run semantic search:
  - `python semantic_retrieval_BERT.py`
  - Artifacts:
    - `data/recipe_embeddings.npy`

### 6) Hybrid search (BM25 + semantic)
- With query parsing and structured filters:
  - `python hybrid_retrieval_strategy.py`
  - Adjust `alpha` (BM25 weight) as needed.

### 7) RAG pipelines
Option A — Ollama embeddings + LLM:
- `python llm_rag_pipeline.py`
  - Uses LangChain FAISS; requires Ollama running and models available (see script defaults).

Option B — RAG with query parsing + processed fields:
- `python LLM_RAG_pipeline_with_qe.py`

Option C — RAG over precomputed Sentence-BERT embeddings:
- `python rag_existing_embeddings.py`

### 8) Streamlit UI (optional)
- `streamlit run rag_ui.py`
- Enter queries, set filters, view ranked results and AI explanations.

### 9) Evaluation
- Automated evaluation over multiple strategies and systems:
  - `python task9_comparision.py`
  - Outputs CSVs under:
    - `evaluation_results_hybrid/`
    - `evaluation_results_majority/`
    - `evaluation_results_rrf/`

- Visualize results:
  - `python result_visuals.py`

## Mapping project specifications to files

1. EDA
   - `data_exploration.ipynb`
   - Visual outputs saved under `Visualizations/` (e.g., `Figure_*.png`).

2. Text preprocessing (recipes + queries)
   - `text_preprocessing.py` (recipes: `title`, `ingredients`, `directions`, `tags`)
   - Queries consistently preprocessed via `preprocess_text` (imported where needed).

3. Indexing for search (inverted index, TF‑IDF/BM25)
   - `bm25_retrieval_optimized.py` (BM25 index build/load/search)

4. Semantic embeddings for retrieval
   - `semantic_retrieval_BERT.py` (Sentence-BERT embedding pipeline + semantic search)

5. Hybrid search (weighted combo)
   - `hybrid_retrieval_strategy.py` (BM25 + semantic with min-max scaling and `alpha`)

6. Structured attribute filtering
   - `query_parser.py` (regex-based extraction of `<`, `>`, `=` for `calories_cal`, `protein_g`, `totalfat_g`, `duration`, and tags)
   - `structured_filters.py` (vague-term → numeric thresholds)
   - Integrated in `hybrid_retrieval_strategy.py` and `LLM_RAG_pipeline_with_qe.py`

7. Query understanding & expansion
   - `query_expansion.py` (WordNet synonyms, Lesk refinement)
   - `structured_filters.py` (vague terms: “healthy”, “quick”, etc.)
   - Wiring in `query_parser.py`

8. RAG (Retrieval-Augmented Generation)
   - `llm_rag_pipeline.py` (Ollama embeddings + LLM + FAISS)
   - `LLM_RAG_pipeline_with_qe.py` (RAG + query parsing + processed fields)
   - `rag_existing_embeddings.py` (RAG using precomputed Sentence-BERT embeddings)

9. Evaluation of retrieval models
   - `task9_comparision.py` (Precision@k, Recall@k, MRR, NDCG, MAP, F1 across BM25, Semantic, Hybrid, RAG)
   - Results saved under `evaluation_results_*` directories
   - `result_visuals.py` for analysis plots

10. Explainability and user experience
   - LLM prompts in RAG scripts (`llm_rag_pipeline.py`, `LLM_RAG_pipeline_with_qe.py`, `rag_existing_embeddings.py`) generate natural language explanations (“matches because ...”).
   - UI highlights metadata snippets in `rag_ui.py`.

11. Visualization & Interface (optional)
   - UI: `rag_ui.py` (Streamlit)
   - Evaluation visuals: `result_visuals.py`
   - EDA plots: `data_exploration.ipynb` and `Visualizations/`

12. Extensions (optional)
   - Multi-turn and personalization can be added by extending `query_parser.py` and RAG scripts; the current codebase lays the groundwork (filters, parsing, and LLM prompting).

If a single file implements multiple tasks, comments inside code highlight purpose and stages. See e.g. headers in:
- `basic_cleaning.py` (Stage 1: Basic cleaning)
- `text_preprocessing.py` (Preprocessing utilities + dataset pipeline)
- `bm25_retrieval_optimized.py` (Index building + search stages)
- `hybrid_retrieval_strategy.py` (filtering, BM25/semantic search, hybrid combine)

## Example commands

- BM25 quick test:
  - `python bm25_retrieval_optimized.py` (runs example queries)

- Semantic quick test:
  - `python semantic_retrieval_BERT.py` (runs example queries after embeddings creation)

- Hybrid quick test:
  - `python hybrid_retrieval_strategy.py` (uses parsed query + filters)

- RAG examples:
  - `python llm_rag_pipeline.py`
  - `python LLM_RAG_pipeline_with_qe.py`
  - Ensure Ollama is running and required models are available.

- UI:
  - `streamlit run rag_ui.py`

## Dependencies and versions (pinned)

Python: 3.10.18

Core libraries (from `environment.yml` / `pyproject.toml`):
- pandas==2.3.3
- numpy==1.26.4
- nltk==3.9.1
- scikit-learn==1.7.2
- scipy==1.13.1
- seaborn==0.13.2
- matplotlib==3.10.6
- rank-bm25==0.2.2
- sentence-transformers==5.1.2
- torch==2.5.1 (use CUDA build if applicable)
- transformers==4.57.1
- tqdm==4.67.1
- langchain (version as available)
- langchain-community (version as available)
- ollama>=0.0.5 (for local LLMs/embeddings)
- gdown==5.2.0 (for dataset download)

Also see:
- `environment.yml` (Conda env, includes more pins and CUDA variants)
- `pyproject.toml` (pip dependencies list)

Note:
- FAISS vector store is via `langchain_community.vectorstores.FAISS`. Some environments may require installing `faiss-cpu` explicitly (e.g., `pip install faiss-cpu`).

## Data artifacts produced
- `data/hummus_recipes_cleaned_basic.csv` — after basic cleaning
- `data/hummus_recipes_preprocessed.csv` — after text preprocessing
- `data/bm25_index.pkl`, `data/recipe_metadata.pkl` — BM25
- `data/recipe_embeddings.npy` — Sentence-BERT embeddings
- `faiss_recipe_index*/` — FAISS indices created by RAG pipelines

## Troubleshooting
- NLTK resources: scripts auto-download `punkt`, `stopwords`, `wordnet` if missing.
- Ollama: ensure daemon is running and models pulled (e.g., `ollama pull llama3.1`, `ollama pull all-minilm`).
- GPU: scripts auto-detect CUDA/MPS; otherwise run on CPU.

## License
Academic use for NLP/Text Mining course project.
