import re
# --- Mapping Vague Terms to Structured Filters ---

# Define reasonable thresholds (ADJUST THESE AS NEEDED)
# Units are implied by the column names (g, mg, cal, minutes)
THRESHOLDS_MAP = {
    # --- Health Related ---
    'healthy': [
        ('calories_cal', '<', 600),       # Example: Max 600 calories per serving for 'healthy'
        ('saturatedfat_g', '<', 5),       # Example: Max 5g saturated fat
        ('sugars_g', '<', 15),          # Example: Max 15g sugar
        ('sodium_mg', '<', 750),         # Example: Max 750mg sodium
        ('dietaryfiber_g', '>', 3),    # Optional: Min 3g fiber
    ],
    'low_calorie': [
        ('calories_cal', '<', 400),       # Stricter calorie limit
    ],
    'low_fat': [
        ('totalfat_g', '<', 10),          # Example: Max 10g total fat
        ('saturatedfat_g', '<', 3),       # Stricter saturated fat
    ],
     'low_saturated_fat': [
        ('saturatedfat_g', '<', 3),
    ],
    'low_sugar': [
        ('sugars_g', '<', 5),            # Example: Max 5g sugar
    ],
    'low_sodium': [
        ('sodium_mg', '<', 500),         # Example: Max 500mg sodium
    ],
     'low_cholesterol': [
        ('cholesterol_mg', '<', 50),     # Example: Max 50mg cholesterol
    ],
     'low_carb': [
        ('totalcarbohydrate_g', '<', 20), # Example: Max 20g carbs
    ],
    'high_fiber': [
        ('dietaryfiber_g', '>', 5),       # Example: Min 5g fiber
    ],
    'high_protein': [
        ('protein_g', '>', 20),           # Example: Min 20g protein
    ],

    # --- Time/Effort Related ---
    'quick': [
        ('duration', '<', 30),           # Example: Max 30 minutes total duration
    ],
    'fast': [                             # Synonym for quick
        ('duration', '<', 30),
    ],
    'easy': [
        ('ingredients_sizes', '<', 10),    # Example: Max 10 ingredients
        ('direction_size', '<', 8),        # Example: Max 8 steps
    ],
    'simple': [                           # Synonym for easy
        ('ingredients_sizes', '<', 10),
        ('direction_size', '<', 8),
    ],
     'few_ingredients': [
        ('ingredients_sizes', '<', 8),     # Stricter limit for 'few'
    ],
     'short_steps': [ # Less common query term, but possible
        ('direction_size', '<', 5),
     ]
}

# --- Keywords/Phrases that trigger the above mappings ---
# This allows handling synonyms or related phrases easily
# Keys are lowercase phrases/words to look for in the query
# Values are the keys from THRESHOLDS_MAP
TRIGGER_WORDS_TO_CONCEPT = {
    'healthy': 'healthy',
    'wholesome': 'healthy',
    'nutritious': 'healthy',
    'low calorie': 'low_calorie',
    'low-calorie': 'low_calorie',
    'light': 'low_calorie', # Could also imply low fat
    'low fat': 'low_fat',
    'low-fat': 'low_fat',
    'reduced fat': 'low_fat',
    'low saturated fat': 'low_saturated_fat',
    'low sugar': 'low_sugar',
    'low-sugar': 'low_sugar',
    'sugar free': 'low_sugar', # Approx.
    'low sodium': 'low_sodium',
    'low-sodium': 'low_sodium',
    'low salt': 'low_sodium',
    'low cholesterol': 'low_cholesterol',
    'low-cholesterol': 'low_cholesterol',
    'low carb': 'low_carb',
    'low-carb': 'low_carb',
    'keto': 'low_carb', # Approximation
    'high fiber': 'high_fiber',
    'high-fiber': 'high_fiber',
    'high protein': 'high_protein',
    'high-protein': 'high_protein',
    'protein packed': 'high_protein',
    'quick': 'quick',
    'fast': 'fast',
    'speedy': 'quick',
    'easy': 'easy',
    'simple': 'simple',
    'beginner': 'easy',
    'few ingredients': 'few_ingredients',
    # Add more trigger words/phrases as needed
}

# --- Example of how to use it in the parser ---
def apply_vague_term_filters(query: str, current_filters: list):
    """
    Identifies vague terms and adds corresponding filters.
    Returns the updated query string after removing matched terms and the updated filter list.
    (This logic would go inside or be called by your main query parser)
    """
    updated_query = query
    query_lower = query.lower() # For matching

    # Iterate through trigger phrases (longer ones first to avoid partial matches)
    # Sort triggers by length, descending
    sorted_triggers = sorted(TRIGGER_WORDS_TO_CONCEPT.keys(), key=len, reverse=True)

    for trigger in sorted_triggers:
        # Use regex to find whole word/phrase matches
        pattern = re.compile(r'\b' + re.escape(trigger) + r'\b', re.IGNORECASE)
        match = pattern.search(updated_query) # Search in the potentially modified query

        if match:
            concept_key = TRIGGER_WORDS_TO_CONCEPT[trigger]
            if concept_key in THRESHOLDS_MAP:
                print(f"  -> Matched vague term: '{trigger}' -> Concept: '{concept_key}'")
                # Add the filters defined for this concept
                for filter_tuple in THRESHOLDS_MAP[concept_key]:
                     if filter_tuple not in current_filters: # Avoid duplicate filters
                          current_filters.append(filter_tuple)
                          print(f"     Added filter: {filter_tuple}")

                # Remove the matched phrase from the query string (replace only once)
                # Need to find the match in the *original case* query for accurate replacement
                original_match = re.search(pattern.pattern, query, re.IGNORECASE) # Search original query
                if original_match:
                     updated_query = updated_query[:match.start()] + updated_query[match.end():] # Remove based on indices found in updated_query

    # Clean up extra whitespace
    updated_query = re.sub(r'\s+', ' ', updated_query).strip()
    return updated_query, current_filters

# --- Test the vague term application ---
if __name__ == '__main__':
    test_q = "Show me a quick healthy low fat chicken recipe"
    print(f"Original Query: '{test_q}'")
    remaining_q, applied_filters = apply_vague_term_filters(test_q, [])
    print(f"Remaining Text: '{remaining_q}'")
    print(f"Applied Filters: {applied_filters}")