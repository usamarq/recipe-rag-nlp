# --------------------------------------------------------------
# basic_cleaning.py
# Stage 1: Basic cleaning before EDA (Project 22 - Recipe RAG)
# --------------------------------------------------------------

import pandas as pd
import re

def clean_column_name(name: str) -> str:
    """
    Converts raw column names like 'protein [g]' -> 'protein_g'
    and 'calories [cal]' -> 'calories_cal' using snake_case.
    """
    name = name.strip().lower()
    # Replace square brackets, spaces, and slashes with underscores
    name = re.sub(r'[\[\]/]', '', name)
    name = name.replace(' ', '_')
    return name


def basic_cleaning(input_path: str, output_path: str):
    print("=== Loading raw dataset ===")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"Initial shape: {df.shape}")

    # --------------------------------------------------
    # 1. Rename first unnamed column to row_number
    # --------------------------------------------------
    if df.columns[0].startswith('Unnamed'):
        df = df.rename(columns={df.columns[0]: 'row_number'})
        print("Renamed first unnamed column → 'row_number'")

    # --------------------------------------------------
    # 2. Drop recipes with '**' normalization comments
    # --------------------------------------------------
    if "normalization_comment" in df.columns:
        before = len(df)
        df = df[~df["normalization_comment"].str.contains(r"\*\*", na=False)]
        print(f"Dropped {before - len(df)} rows with '**' normalization issues")
        print(f"Shape after normalization filter: {df.shape}")

    # --------------------------------------------------
    # 3. Rename all columns to snake_case
    # --------------------------------------------------
    df.columns = [clean_column_name(c) for c in df.columns]
    print("Converted all column names to snake_case")

    # --------------------------------------------------
    # 4. Convert numeric columns safely
    # --------------------------------------------------
    numeric_cols = [
        "calories_cal", "protein_g", "sodium_mg",
        "totalfat_g", "saturatedfat_g", "cholesterol_mg",
        "totalcarbohydrate_g", "dietaryfiber_g", "sugars_g"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print("Converted numeric columns safely (invalid values → NaN)")

    # --------------------------------------------------
    # 5. Drop rows with zero or missing nutritional values
    # --------------------------------------------------
    before = len(df)
    if all(col in df.columns for col in ["calories_cal", "protein_g"]):
        df = df[(df["calories_cal"] > 0) & (df["protein_g"] > 0)]
        print(f"Removed {before - len(df)} rows with zero or missing nutrition values")

    # --------------------------------------------------
    # 6. Remove non-food or decorative entries
    # --------------------------------------------------
    if "title" in df.columns:
        before = len(df)
        non_food_keywords = [
            "napkin", "serviette", "folding", "table", "decor", "arrangement",
            "centerpiece", "placemat", "wrapping", "basket", "flower", "bouquet"
        ]
        pattern = "|".join(non_food_keywords)
        df = df[~df["title"].str.lower().str.contains(pattern, na=False)]
        print(f"Removed {before - len(df)} non-food or decorative entries")

    # --------------------------------------------------
    # 7. Apply simple thresholds to remove absurd values
    # --------------------------------------------------
    before = len(df)
    if all(col in df.columns for col in ["calories_cal", "protein_g", "sodium_mg"]):
        df = df[
            (df["calories_cal"].between(0, 900)) &
            (df["protein_g"].between(0, 60)) &
            (df["sodium_mg"].between(0, 3000))
        ]
        print(f"Removed {before - len(df)} rows with absurd numeric values")

    print(f"Shape after numeric filtering: {df.shape}")

    # --------------------------------------------------
    # 8. Drop empty text entries (title, ingredients)
    # --------------------------------------------------
    text_cols = ["title", "ingredients"]
    before = len(df)
    df.dropna(subset=[c for c in text_cols if c in df.columns], inplace=True)
    df = df[(df["title"].str.strip() != "") & (df["ingredients"].str.strip() != "")]
    print(f"Removed {before - len(df)} rows with missing text fields")

    # --------------------------------------------------
    # 9. Save cleaned dataset
    # --------------------------------------------------
    df.to_csv(output_path, index=False)
    print("\n✅ Basic cleaning complete.")
    print(f"Cleaned dataset saved → {output_path}")
    print(f"Final shape: {df.shape}")

    return df


if __name__ == "__main__":
    clean_and_processed = basic_cleaning(
        input_path="data/hummus_recipes.csv",
        output_path="data/hummus_recipes_cleaned_basic.csv"
    )
