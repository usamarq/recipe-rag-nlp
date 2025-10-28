import re
import nltk
from nltk.tokenize import word_tokenize

# --- NLTK Resource Downloads (Run once if needed) ---
# try: wordnet.ensure_loaded()
# except LookupError: nltk.download('wordnet')
# try: nltk.data.find('tokenizers/punkt')
# except LookupError: nltk.download('punkt')
# -----------------------------------------------------


# --- Import Vague Term Logic ---
try:
    from structured_filters import apply_vague_term_filters
except ImportError:
    print("Warning: Could not import apply_vague_term_filters from structured_filters.py.")
    print("Vague term filtering will be skipped.")
    def apply_vague_term_filters(query, filters):
        return query, filters

# --- Import Expansion Logic ---
try:
    from query_expansion import expand_query_with_synonyms, refine_synonyms_with_lesk
except ImportError:
    print("Warning: Could not import from query_expansion.py. Expansion skipped.")
    def expand_query_with_synonyms(tokens, max_synonyms_per_word=2): return tokens
    def refine_synonyms_with_lesk(text, orig_tokens, added_synonyms): return []


# --- Configuration for Explicit Filters (Removed Lookbehind) ---
PATTERNS = {
    'calories_cal': [
        (re.compile(r'(?:under|less than|<)\s*(\d+)\s*(?:cal(?:ories)?|kcal)', re.I), '<'),
        (re.compile(r'(?:over|more than|>)\s*(\d+)\s*(?:cal(?:ories)?|kcal)', re.I), '>'),
        # --- Removed Lookbehind from Equality Pattern ---
        (re.compile(r'\b(\d+)\s*(?:cal(?:ories)?|kcal)\b', re.I), '=') # Added \b at end
    ],
    'protein_g': [
        (re.compile(r'(?:under|less than|<)\s*(\d+)\s*g(?:rams)?\s*protein', re.I), '<'),
        (re.compile(r'(?:over|more than|>)\s*(\d+)\s*g(?:rams)?\s*protein', re.I), '>'),
         # --- Optional: Add equality pattern if needed (NO lookbehind) ---
         (re.compile(r'\b(\d+)\s*g(?:rams)?\s*protein\b', re.I), '=')
    ],
     'totalfat_g': [
        (re.compile(r'(?:under|less than|<)\s*(\d+)\s*g(?:rams)?\s*(?:total\s*)?fat', re.I), '<'),
        (re.compile(r'(?:over|more than|>)\s*(\d+)\s*g(?:rams)?\s*(?:total\s*)?fat', re.I), '>'),
         # --- Optional: Add equality pattern if needed (NO lookbehind) ---
         (re.compile(r'\b(\d+)\s*g(?:rams)?\s*(?:total\s*)?fat\b', re.I), '=')
    ],
    'duration': [
        (re.compile(r'(?:under|less than|<)\s*(\d+)\s*(?:min(?:ute)?|hr|hour)s?', re.I), '<'),
        (re.compile(r'(?:over|more than|>)\s*(\d+)\s*(?:min(?:ute)?|hr|hour)s?', re.I), '>'),
        # --- Removed Lookbehind from Equality Pattern ---
        (re.compile(r'\b(\d+)\s*(?:min(?:ute)?|hr|hour)s?\b', re.I), '=') # Added \b at end
    ],
    'tags': [
        (re.compile(r'\b(vegetarian)\b', re.I), 'contains'),
        (re.compile(r'\b(vegan)\b', re.I), 'contains'),
        (re.compile(r'\b(gluten[-\s]?free)\b', re.I), 'contains'),
    ]
}

# --- Main Parsing Function (Relies on two-pass logic) ---

def parse_query(query: str):
    """
    Parses query for filters (vague & explicit) and performs synonym expansion + Lesk refinement.
    Returns: tuple: (list_of_filters, expanded_and_refined_text_query)
    """
    filters = []
    text_query = query
    # --- Step 1: Handle Vague Terms ---
    print("Applying vague term filters...")
    text_query, filters = apply_vague_term_filters(text_query, filters)
    query_after_vague = text_query
    print(f"  Query after vague terms: '{text_query}'")
    print(f"  Filters after vague terms: {filters}")

    # --- Step 2: Handle Explicit Filters ---
    print("Applying explicit filters...")
    phrases_to_remove_explicit = []
    query_to_parse_explicit = text_query
    for attribute, pattern_list in PATTERNS.items():
        # Process comparison patterns first (<, >)
        comparison_patterns = [pt for pt in pattern_list if pt[1] in ['<', '>']]
        for pattern, operator in comparison_patterns:
            for match in pattern.finditer(query_to_parse_explicit):
                value = None; phrase = match.group(0)
                if match.groups() and match.group(1): # Check if group 1 exists
                    try: value = float(match.group(1))
                    except ValueError: continue
                if value is not None:
                    filter_candidate = (attribute, operator, value)
                    if filter_candidate not in filters:
                         print(f"  -> Found comparison filter: {filter_candidate} from '{phrase}'")
                         filters.append(filter_candidate)
                    phrases_to_remove_explicit.append(phrase) # Mark phrase from comparison match

        # Process equality patterns (=) or tag patterns
        other_patterns = [pt for pt in pattern_list if pt[1] not in ['<', '>']]
        for pattern, operator in other_patterns:
             for match in pattern.finditer(query_to_parse_explicit):
                phrase = match.group(0) # The whole matched text
                # *** Crucial Check: Skip if this match is part of an already identified comparison phrase ***
                already_covered = any(phrase in comp_phrase or comp_phrase in phrase for comp_phrase in phrases_to_remove_explicit)
                if already_covered:
                    # print(f"  Skipping match '{phrase}' as it's covered by a comparison filter.")
                    continue

                value = None
                if operator == 'contains':
                     # Ensure group 1 exists for tags
                     if match.groups() and match.group(1):
                          tag_value = match.group(1).lower().replace('-', '').replace(' ', '')
                          value = 'gluten_free' if tag_value == 'glutenfree' else tag_value
                     else: continue # Skip if tag pattern didn't capture group 1
                elif match.groups() and match.group(1): # Captured a number for equality
                    try: value = float(match.group(1))
                    except ValueError: continue
                else: continue # Skip if equality pattern didn't capture group 1

                if value is not None:
                    filter_candidate = (attribute, operator, value)
                    if filter_candidate not in filters:
                         print(f"  -> Found other filter: {filter_candidate} from '{phrase}'")
                         filters.append(filter_candidate)
                    phrases_to_remove_explicit.append(phrase) # Mark phrase for removal

    # Remove all matched explicit phrases after finding them
    unique_phrases_to_remove = sorted(list(set(phrases_to_remove_explicit)), key=len, reverse=True)
    # print(f"  Phrases marked for removal: {unique_phrases_to_remove}") # Debugging
    for phrase in unique_phrases_to_remove:
         text_query = text_query.replace(phrase, '') # Remove all occurrences? Check logic if only one needed.
    print(f"  Query after explicit filters: '{text_query}'")
    print(f"  Filters after explicit: {filters}")


    # --- Step 3 & 4: Expand Synonyms & Refine with Lesk ---
    print("Expanding query with synonyms...")
    original_tokens_for_expansion = word_tokenize(text_query.lower())
    print(f"  Tokens before expansion: {original_tokens_for_expansion}")
    if not original_tokens_for_expansion:
         final_text_query = ""
         print("  No text left for expansion.")
    else:
         broadly_expanded_tokens = expand_query_with_synonyms(original_tokens_for_expansion)
         added_synonyms = [t for t in broadly_expanded_tokens if t not in original_tokens_for_expansion]
         print(f"  Synonyms added (before Lesk): {added_synonyms}")
         print(f"  Refining added synonyms using Lesk context: '{query_after_vague}'")
         relevant_added_synonyms = refine_synonyms_with_lesk(query_after_vague, original_tokens_for_expansion, added_synonyms)
         print(f"  Relevant synonyms after Lesk: {relevant_added_synonyms}")
         final_tokens = original_tokens_for_expansion + relevant_added_synonyms
         final_tokens_unique = list(dict.fromkeys(final_tokens))
         final_text_query = " ".join(final_tokens_unique)

    # --- Step 5: Final Cleanup ---
    final_text_query = re.sub(r'\s+', ' ', final_text_query).strip()

    # --- Step 6: Remove Duplicate Filters ---
    unique_filters = []; seen_filters = set()
    for f in filters:
         try: f_value_str = str(float(f[2])) if f[0] != 'tags' else str(f[2])
         except ValueError: f_value_str = str(f[2])
         f_key = (f[0], f[1], f_value_str)
         if f_key not in seen_filters:
              unique_filters.append(f); seen_filters.add(f_key)

    print(f"\nParser Final Output:")
    print(f"  Filters: {unique_filters}")
    print(f"  Text Query: '{final_text_query}'")
    return unique_filters, final_text_query

# --- Example Usage ---
if __name__ == '__main__':
    # # ... (NLTK check) ...
    # try: nltk.data.find('corpora/wordnet'); nltk.data.find('tokenizers/punkt')
    # except LookupError: print("Run nltk downloads first."); exit()

    test_queries = [
        "low fat chicken under 500 calories",
        "high protein vegan salad with aubergine",
        "quick vegetarian pasta < 30 min",
        "gluten-free soup over 10g protein",
        "healthy courgette bake dish 200 kcal" # Test equality separate from vague
    ]
    for q in test_queries:
        print(f"\n===== Processing Query: '{q}' =====")
        extracted_filters, expanded_text = parse_query(q)
        print("=" * (20 + len(q) + 10))

