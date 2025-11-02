import pandas as pd
import numpy as np
import json
import os
import time
from typing import List, Dict, Tuple, Set
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

# Import your retrieval systems
try:
    from bm25_retrieval_optimized import load_index_and_metadata, search_bm25_optimized
    from semantic_retrieval_BERT import setup_device_and_model, semantic_search
    from hybrid_retrieval_strategy import hybrid_search_with_parsing
    from rag_existing_embeddings import RecipeRAGSystem
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")


# ============================================================================
# AUTOMATED GROUND TRUTH GENERATION STRATEGIES
# ============================================================================

class AutoGroundTruth:
    """Generate pseudo ground-truth using multiple automated strategies."""
    
    @staticmethod
    def majority_voting(all_system_results: Dict[str, List[str]], 
                       top_k: int = 10, threshold: int = 2) -> Set[str]:
        """
        Strategy 1: Majority Voting
        A recipe is considered relevant if it appears in top-k of at least 'threshold' systems.
        """
        vote_counts = defaultdict(int)
        
        for system_name, results in all_system_results.items():
            for recipe_id in results[:top_k]:
                vote_counts[recipe_id] += 1
        
        relevant = {rid for rid, count in vote_counts.items() if count >= threshold}
        return relevant
    
    @staticmethod
    def consensus_intersection(all_system_results: Dict[str, List[str]], 
                              top_k: int = 5) -> Set[str]:
        """
        Strategy 2: Consensus Intersection
        Only recipes that appear in ALL systems' top-k are considered relevant.
        Very strict but high precision.
        """
        if not all_system_results:
            return set()
        
        result_sets = [set(results[:top_k]) for results in all_system_results.values()]
        return set.intersection(*result_sets) if result_sets else set()
    
    @staticmethod
    def query_specific_filtering(query: str, df: pd.DataFrame, 
                                 filters: Dict = None) -> Set[str]:
        """
        Strategy 3: Query-Specific Filtering
        Use metadata filters to identify objectively relevant recipes.
        """
        relevant_ids = set()
        
        # Apply filters if provided
        filtered_df = df.copy()
        
        if filters:
            if 'max_calories' in filters:
                filtered_df = filtered_df[filtered_df['calories_cal'] <= filters['max_calories']]
            if 'min_protein' in filters:
                filtered_df = filtered_df[filtered_df['protein_g'] >= filters['min_protein']]
            if 'max_duration' in filters:
                filtered_df = filtered_df[filtered_df['duration'] <= filters['max_duration']]
            if 'tags' in filters:
                tag_filter = filters['tags'].lower()
                filtered_df = filtered_df[
                    filtered_df['tags'].str.lower().str.contains(tag_filter, na=False)
                ]
        
        # Add keyword matching from query
        query_tokens = set(query.lower().split())
        # Remove common words
        stop_words = {'a', 'an', 'the', 'with', 'under', 'over', 'recipe', 'recipes'}
        query_tokens = query_tokens - stop_words
        
        if query_tokens:
            for _, row in filtered_df.iterrows():
                title_tokens = set(str(row['title']).lower().split())
                tags_tokens = set(str(row.get('tags', '')).lower().split())
                
                # If query tokens appear in title or tags
                if query_tokens & (title_tokens | tags_tokens):
                    relevant_ids.add(str(row['recipe_id']))
        
        return relevant_ids
    
    @staticmethod
    def reciprocal_rank_fusion(all_system_results: Dict[str, List[str]], 
                               k: int = 60) -> List[Tuple[str, float]]:
        """
        Strategy 4: Reciprocal Rank Fusion (RRF)
        Combines rankings from multiple systems. Higher RRF score = more relevant.
        Returns ranked list with scores.
        """
        rrf_scores = defaultdict(float)
        
        for system_name, results in all_system_results.items():
            for rank, recipe_id in enumerate(results, 1):
                rrf_scores[recipe_id] += 1.0 / (k + rank)
        
        # Sort by RRF score
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked
    
    @staticmethod
    def hybrid_ground_truth(all_system_results: Dict[str, List[str]], 
                           query: str, df: pd.DataFrame, filters: Dict = None,
                           top_n: int = 10) -> Set[str]:
        """
        Strategy 5: Hybrid Approach
        Combines multiple strategies for robust ground truth.
        """
        # Get candidates from RRF
        rrf_ranked = AutoGroundTruth.reciprocal_rank_fusion(all_system_results)
        rrf_top = set([rid for rid, score in rrf_ranked[:top_n * 2]])
        
        # Get candidates from majority voting
        majority = AutoGroundTruth.majority_voting(all_system_results, top_k=10, threshold=2)
        
        # Get candidates from query-specific filtering
        filter_matches = AutoGroundTruth.query_specific_filtering(query, df, filters)
        
        # Combine: Take intersection of RRF top results and 
        # union of majority voting and filter matches
        relevant = (rrf_top & (majority | filter_matches)) | \
                   (majority & filter_matches)
        
        # If still empty, use majority voting with lower threshold
        if not relevant:
            relevant = AutoGroundTruth.majority_voting(all_system_results, top_k=15, threshold=1)
        
        return relevant


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Precision@K"""
    if k == 0 or not retrieved:
        return 0.0
    retrieved_at_k = retrieved[:k]
    relevant_retrieved = len([doc for doc in retrieved_at_k if doc in relevant])
    return relevant_retrieved / k


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Recall@K"""
    if not relevant or k == 0:
        return 0.0
    retrieved_at_k = retrieved[:k]
    relevant_retrieved = len([doc for doc in retrieved_at_k if doc in relevant])
    return relevant_retrieved / len(relevant)


def mean_reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """MRR"""
    for i, doc in enumerate(retrieved, 1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """NDCG@K"""
    def dcg(retrieved_list, relevant_set, k):
        dcg_val = 0.0
        for i, doc in enumerate(retrieved_list[:k], 1):
            rel = 1 if doc in relevant_set else 0
            dcg_val += rel / np.log2(i + 1)
        return dcg_val
    
    dcg_val = dcg(retrieved, relevant, k)
    # Ideal: all relevant docs at top
    ideal_retrieved = list(relevant)[:k]
    idcg = dcg(ideal_retrieved, relevant, k)
    
    return dcg_val / idcg if idcg > 0 else 0.0


def map_score(retrieved: List[str], relevant: Set[str], k: int = 10) -> float:
    """Mean Average Precision"""
    if not relevant:
        return 0.0
    
    precisions = []
    num_relevant = 0
    
    for i, doc in enumerate(retrieved[:k], 1):
        if doc in relevant:
            num_relevant += 1
            precisions.append(num_relevant / i)
    
    return sum(precisions) / len(relevant) if precisions else 0.0


# ============================================================================
# ADDITIONAL METRICS FOR AUTOMATED EVALUATION
# ============================================================================

def overlap_score(system_results: Dict[str, List[str]], k: int = 10) -> Dict:
    """
    Measure overlap between different systems.
    High overlap = systems agree on what's relevant.
    """
    systems = list(system_results.keys())
    overlaps = {}
    
    for i, sys1 in enumerate(systems):
        for sys2 in systems[i+1:]:
            set1 = set(system_results[sys1][:k])
            set2 = set(system_results[sys2][:k])
            
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            
            jaccard = intersection / union if union > 0 else 0
            overlap_pct = (intersection / k) * 100
            
            overlaps[f"{sys1}_vs_{sys2}"] = {
                'jaccard': jaccard,
                'overlap_percent': overlap_pct,
                'common_items': intersection
            }
    
    return overlaps


def rank_correlation(results1: List[str], results2: List[str], k: int = 10) -> float:
    """
    Spearman's rank correlation between two result lists.
    Measures if systems rank items similarly.
    """
    from scipy.stats import spearmanr
    
    # Get common items
    set1 = set(results1[:k])
    set2 = set(results2[:k])
    common = list(set1 & set2)
    
    if len(common) < 2:
        return 0.0
    
    # Get ranks for common items
    ranks1 = [results1.index(item) if item in results1 else k for item in common]
    ranks2 = [results2.index(item) if item in results2 else k for item in common]
    
    corr, _ = spearmanr(ranks1, ranks2)
    return corr if not np.isnan(corr) else 0.0


# ============================================================================
# TEST QUERIES
# ============================================================================

def create_automated_test_queries() -> List[Dict]:
    """Create test queries with filters for automated ground truth."""
    
    queries = [
        {
            'query_id': 'Q1',
            'query': 'low calorie chicken',
            'filters': {'max_calories': 400},
            'description': 'Low calorie chicken recipes'
        },
        {
            'query_id': 'Q2',
            'query': 'high protein vegan breakfast',
            'filters': {'min_protein': 15, 'tags': 'vegan'},
            'description': 'High protein vegan breakfast'
        },
        {
            'query_id': 'Q3',
            'query': 'quick pasta dinner',
            'filters': {'max_duration': 30},
            'description': 'Quick pasta recipes'
        },
        {
            'query_id': 'Q4',
            'query': 'gluten free dessert',
            'filters': {'tags': 'gluten-free'},
            'description': 'Gluten-free desserts'
        },
        {
            'query_id': 'Q5',
            'query': 'healthy salad',
            'filters': {'max_calories': 350},
            'description': 'Healthy salads'
        },
        {
            'query_id': 'Q6',
            'query': 'vegetarian soup',
            'filters': {'tags': 'vegetarian'},
            'description': 'Vegetarian soups'
        },
        {
            'query_id': 'Q7',
            'query': 'low carb meal',
            'filters': {},
            'description': 'Low carb meals'
        },
        {
            'query_id': 'Q8',
            'query': 'breakfast smoothie',
            'filters': {},
            'description': 'Breakfast smoothies'
        },
        {
            'query_id': 'Q9',
            'query': 'spicy curry',
            'filters': {},
            'description': 'Spicy curry recipes'
        },
        {
            'query_id': 'Q10',
            'query': 'chocolate dessert',
            'filters': {'max_calories': 400},
            'description': 'Chocolate desserts'
        },
        {
            'query_id': 'Q11',
            'query': 'fish dinner',
            'filters': {'max_duration': 45},
            'description': 'Fish dinner recipes'
        },
        {
            'query_id': 'Q12',
            'query': 'protein shake',
            'filters': {'min_protein': 20},
            'description': 'High protein shakes'
        }
    ]
    
    return queries


# ============================================================================
# RETRIEVAL WRAPPERS
# ============================================================================

def run_bm25_retrieval(query: str, bm25_model, metadata_map, k: int = 20) -> List[str]:
    """BM25 retrieval"""
    try:
        results = search_bm25_optimized(query, bm25_model, metadata_map, top_k=k)
        return [str(res['recipe_id']) for res in results if res.get('recipe_id')]
    except Exception as e:
        print(f"  BM25 error: {e}")
        return []


def run_semantic_retrieval(query: str, model, df_metadata, embeddings, k: int = 20) -> List[str]:
    """Semantic retrieval"""
    try:
        results = semantic_search(query, model, df_metadata, embeddings, top_k=k)
        if isinstance(results, pd.DataFrame):
            return [str(rid) for rid in results['recipe_id'].tolist()]
        return []
    except Exception as e:
        print(f"  Semantic error: {e}")
        return []


def run_hybrid_retrieval(query: str, bm25_model, semantic_model, embeddings, 
                        metadata_map, alpha: float = 0.5, k: int = 20) -> List[str]:
    """Hybrid retrieval"""
    try:
        results = hybrid_search_with_parsing(
            query, bm25_model, semantic_model, embeddings, metadata_map,
            alpha=alpha, top_k_initial=50, top_k_final=k
        )
        return [str(res['recipe_id']) for res in results if res.get('recipe_id')]
    except Exception as e:
        print(f"  Hybrid error: {e}")
        return []


def run_rag_retrieval(query: str, rag_system, filters: Dict, k: int = 20) -> List[str]:
    """RAG retrieval"""
    try:
        retrieved_docs = rag_system.retrieve_recipes(query, k=k, filters=filters)
        return [str(doc.metadata['recipe_id']) for doc in retrieved_docs 
                if doc.metadata.get('recipe_id')]
    except Exception as e:
        print(f"  RAG error: {e}")
        return []


# ============================================================================
# AUTOMATED EVALUATION
# ============================================================================

def evaluate_with_auto_ground_truth(
    bm25_model, semantic_model, embeddings, metadata_map, df_metadata, 
    rag_system, df_full, test_queries, k_values=[1, 3, 5, 10],
    ground_truth_strategy='hybrid', alpha=0.5
):
    """
    Evaluate all systems using automated ground truth generation.
    
    Args:
        ground_truth_strategy: 'majority', 'consensus', 'filter', 'rrf', or 'hybrid'
    """
    
    all_results = []
    
    print("\n" + "="*80)
    print("AUTOMATED EVALUATION (No Manual Labels Required)")
    print(f"Ground Truth Strategy: {ground_truth_strategy.upper()}")
    print("="*80 + "\n")
    
    for query_data in test_queries:
        query_id = query_data['query_id']
        query = query_data['query']
        filters = query_data.get('filters', {})
        
        print(f"\n{'='*60}")
        print(f"Processing {query_id}: {query}")
        print(f"{'='*60}")
        
        # Step 1: Get results from all systems
        print("  Retrieving from all systems...")
        
        bm25_results = run_bm25_retrieval(query, bm25_model, metadata_map, k=20)
        semantic_results = run_semantic_retrieval(query, semantic_model, df_metadata, embeddings, k=20)
        hybrid_results = run_hybrid_retrieval(query, bm25_model, semantic_model, 
                                             embeddings, metadata_map, alpha=alpha, k=20)
        rag_results = run_rag_retrieval(query, rag_system, filters, k=20)
        
        all_system_results = {
            'BM25': bm25_results,
            'Semantic': semantic_results,
            'Hybrid': hybrid_results,
            'RAG': rag_results
        }
        
        # Print retrieval counts
        for sys_name, results in all_system_results.items():
            print(f"    {sys_name}: {len(results)} results")
        
        # Step 2: Generate automated ground truth
        print(f"  Generating ground truth using {ground_truth_strategy} strategy...")
        
        if ground_truth_strategy == 'majority':
            relevant_ids = AutoGroundTruth.majority_voting(all_system_results, top_k=10, threshold=2)
        elif ground_truth_strategy == 'consensus':
            relevant_ids = AutoGroundTruth.consensus_intersection(all_system_results, top_k=5)
        elif ground_truth_strategy == 'filter':
            relevant_ids = AutoGroundTruth.query_specific_filtering(query, df_full, filters)
        elif ground_truth_strategy == 'rrf':
            rrf_results = AutoGroundTruth.reciprocal_rank_fusion(all_system_results)
            relevant_ids = set([rid for rid, score in rrf_results[:10]])
        else:  # hybrid
            relevant_ids = AutoGroundTruth.hybrid_ground_truth(all_system_results, query, df_full, filters)
        
        print(f"    Generated {len(relevant_ids)} relevant recipes")
        
        if not relevant_ids:
            print("    Warning: No relevant recipes identified. Skipping query.")
            continue
        
        # Step 3: Calculate metrics for each system
        print("  Calculating metrics...")
        
        for system_name, results in all_system_results.items():
            if not results:
                continue
                
            for k in k_values:
                metrics = {
                    'Query_ID': query_id,
                    'Query': query,
                    'System': system_name,
                    'k': k,
                    'Precision@k': precision_at_k(results, relevant_ids, k),
                    'Recall@k': recall_at_k(results, relevant_ids, k),
                    'NDCG@k': ndcg_at_k(results, relevant_ids, k),
                    'MAP': map_score(results, relevant_ids, k),
                    'MRR': mean_reciprocal_rank(results, relevant_ids) if k == max(k_values) else None,
                    'Num_Retrieved': len(results),
                    'Num_Relevant_GT': len(relevant_ids)
                }
                
                # F1 Score
                prec = metrics['Precision@k']
                rec = metrics['Recall@k']
                metrics['F1@k'] = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
                
                all_results.append(metrics)
        
        # Step 4: Calculate system overlap for this query
        overlap_stats = overlap_score(all_system_results, k=10)
        print(f"\n  System Overlap Analysis:")
        for comparison, stats in overlap_stats.items():
            print(f"    {comparison}: {stats['common_items']} common items ({stats['overlap_percent']:.1f}% overlap)")
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Calculate aggregated statistics
    aggregated = results_df.groupby(['System', 'k']).agg({
        'Precision@k': 'mean',
        'Recall@k': 'mean',
        'F1@k': 'mean',
        'NDCG@k': 'mean',
        'MAP': 'mean',
        'MRR': 'mean',
        'Query_ID': 'count'
    }).reset_index()
    
    aggregated = aggregated.rename(columns={'Query_ID': 'Num_Queries'})
    
    return results_df, aggregated


# ============================================================================
# REPORTING
# ============================================================================

def print_automated_evaluation_report(detailed_df: pd.DataFrame, aggregated_df: pd.DataFrame):
    """Print comprehensive evaluation report."""
    
    print("\n" + "="*80)
    print("AUTOMATED EVALUATION RESULTS")
    print("="*80 + "\n")
    
    # Overall performance
    print("Overall Performance (Averaged Across All K):")
    print("-" * 80)
    overall = aggregated_df.groupby('System').agg({
        'Precision@k': 'mean',
        'Recall@k': 'mean',
        'F1@k': 'mean',
        'NDCG@k': 'mean',
        'MAP': 'mean',
        'MRR': 'mean'
    }).round(4)
    print(overall.to_string())
    
    # Performance at each k
    for k in sorted(aggregated_df['k'].unique()):
        print(f"\n\nPerformance at K={k}:")
        print("-" * 80)
        k_results = aggregated_df[aggregated_df['k'] == k].copy()
        k_results = k_results.sort_values('NDCG@k', ascending=False)
        print(k_results[['System', 'Precision@k', 'Recall@k', 'F1@k', 'NDCG@k', 'MAP']].to_string(index=False))
    
    # Best performers
    print("\n\nBest System by Metric:")
    print("-" * 80)
    metrics = ['Precision@k', 'Recall@k', 'F1@k', 'NDCG@k', 'MAP', 'MRR']
    for metric in metrics:
        best_system = aggregated_df.groupby('System')[metric].mean().idxmax()
        best_value = aggregated_df.groupby('System')[metric].mean().max()
        print(f"{metric:15s}: {best_system:15s} ({best_value:.4f})")
    
    # Statistical summary
    print("\n\nStatistical Summary:")
    print("-" * 80)
    for metric in ['NDCG@k', 'MAP', 'F1@k']:
        print(f"\n{metric}:")
        summary = detailed_df.groupby('System')[metric].describe()[['mean', 'std', 'min', 'max']].round(4)
        print(summary.to_string())


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    
    # Paths
    CSV_PATH = "data/hummus_recipes_preprocessed.csv"
    BM25_INDEX_PATH = "data/bm25_index.pkl"
    METADATA_PATH = "data/recipe_metadata.pkl"
    EMBEDDINGS_PATH = "data/recipe_embeddings.npy"
    
    print("="*80)
    print("AUTOMATED RECIPE RETRIEVAL EVALUATION")
    print("No manual labeling required!")
    print("="*80)
    
    # Load components
    print("\n📦 Loading components...")
    
    try:
        bm25_model, metadata_map = load_index_and_metadata(BM25_INDEX_PATH, METADATA_PATH)
        semantic_model, device = setup_device_and_model()
        embeddings = np.load(EMBEDDINGS_PATH)
        
        df_metadata = pd.read_csv(CSV_PATH, usecols=['recipe_id', 'title', 'calories_cal', 
                                                      'totalfat_g', 'protein_g'])
        df_metadata = df_metadata.reset_index(drop=True)
        
        df_full = pd.read_csv(CSV_PATH)
        
        rag_system = RecipeRAGSystem(csv_path=CSV_PATH, embedding_path=EMBEDDINGS_PATH)
        
        print("✅ All components loaded!\n")
        
    except Exception as e:
        print(f"❌ Error loading components: {e}")
        return
    
    # Get test queries
    test_queries = create_automated_test_queries()
    
    # Run evaluation with different strategies
    strategies = ['hybrid', 'majority', 'rrf']
    
    for strategy in strategies:
        print(f"\n{'#'*80}")
        print(f"EVALUATING WITH {strategy.upper()} GROUND TRUTH STRATEGY")
        print(f"{'#'*80}")
        
        detailed_results, aggregated_results = evaluate_with_auto_ground_truth(
            bm25_model, semantic_model, embeddings, metadata_map, 
            df_metadata, rag_system, df_full, test_queries,
            k_values=[1, 3, 5, 10],
            ground_truth_strategy=strategy,
            alpha=0.5
        )
        
        print_automated_evaluation_report(detailed_results, aggregated_results)
        
        # Save results
        output_dir = f'evaluation_results_{strategy}'
        os.makedirs(output_dir, exist_ok=True)
        detailed_results.to_csv(f'{output_dir}/detailed_results.csv', index=False)
        aggregated_results.to_csv(f'{output_dir}/aggregated_results.csv', index=False)
        
        print(f"\n✅ Results saved to {output_dir}/")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()