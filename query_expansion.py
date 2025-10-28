import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.wsd import lesk
from typing import List, Set, Optional

# # --- NLTK Resource Downloads (Run once if needed) ---
# # Ensure necessary resources are available
# try:
#     wordnet.ensure_loaded()
# except LookupError:
#     print("Downloading NLTK WordNet...")
#     nltk.download('wordnet')
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     print("Downloading NLTK Punkt...")
#     nltk.download('punkt')
# # Lesk might implicitly use stopwords/other corpora, ensure they are available if errors occur
# # nltk.download('stopwords')
# # -----------------------------------------------------

def get_wordnet_synonyms(token: str, pos: Optional[str] = None, max_synonyms: int = 3) -> Set[str]:
    """
    Finds synonyms for a single token using WordNet, optionally filtered by part-of-speech.

    Args:
        token: The word to find synonyms for.
        pos: Optional NLTK part-of-speech tag (e.g., wordnet.NOUN, wordnet.VERB).
        max_synonyms: Maximum number of unique synonyms to return.

    Returns:
        A set of unique synonym strings (lowercase, underscores replaced).
    """
    synonyms = set()
    # wordnet.synsets can take a pos argument (n, v, a, r for noun, verb, adj, adv)
    wordnet_pos = pos if pos in [wordnet.NOUN, wordnet.VERB, wordnet.ADJ, wordnet.ADV] else None
    
    for syn in wordnet.synsets(token, pos=wordnet_pos):
        for lemma in syn.lemmas():
            synonym = lemma.name().lower().replace('_', ' ')
            # Basic filtering: not the original token, > 1 char, not already added
            if synonym != token.lower() and len(synonym) > 1 and synonym not in synonyms:
                 # Check if it contains only letters and spaces (simple filter)
                 if all(c.isalpha() or c.isspace() for c in synonym):
                     synonyms.add(synonym)
                     if len(synonyms) >= max_synonyms:
                         return synonyms # Return early if max reached
    return synonyms

def expand_query_with_synonyms(query_tokens: List[str], max_synonyms_per_word: int = 2) -> List[str]:
    """
    Expands a list of query tokens by adding WordNet synonyms.

    Args:
        query_tokens: The original list of tokens from the query.
        max_synonyms_per_word: Max synonyms to add for each original token.

    Returns:
        A list containing original tokens and added unique synonyms.
    """
    expanded_tokens = list(query_tokens) # Start with original
    added_synonyms_global = set() # Track all added synonyms

    for token in query_tokens:
        # We can optionally try to determine POS tag here for better synonyms,
        # but for simplicity, we'll search all POS or focus on nouns/verbs potentially.
        # Let's try getting synonyms for nouns and verbs primarily.
        noun_syns = get_wordnet_synonyms(token, pos=wordnet.NOUN, max_synonyms=max_synonyms_per_word)
        verb_syns = get_wordnet_synonyms(token, pos=wordnet.VERB, max_synonyms=max_synonyms_per_word)
        
        all_syns = noun_syns.union(verb_syns)

        for syn in all_syns:
            if syn not in expanded_tokens and syn not in added_synonyms_global:
                expanded_tokens.append(syn)
                added_synonyms_global.add(syn)

    return expanded_tokens


def refine_synonyms_with_lesk(original_query_text: str,
                              original_tokens: List[str],
                              added_synonyms: List[str]) -> List[str]:
    """
    Uses Lesk algorithm to check if added synonyms fit the original query context.

    Args:
        original_query_text: The full, original query text (before synonym expansion).
        original_tokens: The tokens from the query BEFORE expansion.
        added_synonyms: A list of synonyms that were added during expansion.

    Returns:
        A list of synonyms deemed relevant by Lesk in the context.
    """
    relevant_synonyms = []
    original_context_tokens = word_tokenize(original_query_text.lower()) # Tokenize context

    for synonym in added_synonyms:
        # Use Lesk to find the best sense of the *synonym* in the original context
        best_sense = lesk(original_context_tokens, synonym)

        if best_sense:
            # Simple relevance check: Does the best sense's definition or lemmas
            # overlap significantly with the original query tokens (excluding stopwords)?
            # This is a basic heuristic. More advanced checks could compare sense definitions.
            sense_words = set(word_tokenize(best_sense.definition().lower()))
            for lemma in best_sense.lemmas():
                 sense_words.add(lemma.name().lower().replace('_', ' '))

            # Check for overlap with original non-stopword tokens
            original_set = set(t.lower() for t in original_tokens if t.lower() not in nltk.corpus.stopwords.words('english'))
            overlap = original_set.intersection(sense_words)

            # Keep synonym if there's *some* overlap (adjust threshold as needed)
            if len(overlap) > 0:
                 # print(f"  -> Keeping synonym '{synonym}' (Lesk sense: {best_sense.name()}, Overlap: {overlap})")
                 relevant_synonyms.append(synonym)
            # else:
                 # print(f"  -> Discarding synonym '{synonym}' (Lesk sense: {best_sense.name()}, No overlap)")

        # else:
             # print(f"  -> Discarding synonym '{synonym}' (Lesk could not find sense in context)")


    return relevant_synonyms


# --- Example Usage ---
if __name__ == '__main__':
    query = "healthy aubergine bake"
    tokens = word_tokenize(query.lower())
    print(f"Original Tokens: {tokens}")

    expanded = expand_query_with_synonyms(tokens, max_synonyms_per_word=2)
    print(f"Expanded Tokens (Raw): {expanded}")

    original_for_lesk = "healthy aubergine bake" # The text query part *before* expansion
    original_tokens_for_lesk = word_tokenize(original_for_lesk.lower())
    added_only = [t for t in expanded if t not in original_tokens_for_lesk]
    print(f"Added Synonyms Only: {added_only}")

    print("\nRefining with Lesk...")
    relevant_added_synonyms = refine_synonyms_with_lesk(original_for_lesk, original_tokens_for_lesk, added_only)
    print(f"Relevant Added Synonyms: {relevant_added_synonyms}")

    final_tokens = original_tokens_for_lesk + relevant_added_synonyms
    print(f"Final Expanded & Refined Tokens: {final_tokens}")
    print(f"Final Query String: {' '.join(final_tokens)}")
