import pandas as pd

# configuration 
INPUT_CSV = r"C:\Users\Bote\Desktop\pml\project2\statistics\player_data.csv"
FEATURES_OUT = r"C:\Users\Bote\Desktop\pml\project2\statistics\player_data_features.csv"
METADATA_OUT = r"C:\Users\Bote\Desktop\pml\project2\statistics\player_data_remains.csv"
MIN_THRESHOLD = 15

# Columns to REMOVE from feature set
NON_FEATURE_COLUMNS = [
    "_id",
    "GAME_ID",
    "GAME_DATE",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "TEAM_CITY",
    "PLAYER_ID",
    "PLAYER_NAME",
    "NICKNAME",
    "START_POSITION"   # label
]

def main():
    # Read dataset
    df = pd.read_csv(INPUT_CSV)

    # Filter by minutes played
    df = df[df["MIN"] >= MIN_THRESHOLD].copy()
    df.reset_index(drop=True, inplace=True)

    # Keep only rows with a known position
    VALID_POSITIONS = {"G", "F", "C"}

    df = df[df["START_POSITION"].isin(VALID_POSITIONS)].copy()
    df.reset_index(drop=True, inplace=True)

    # metadata for evaluation 
    metadata_cols = [c for c in NON_FEATURE_COLUMNS if c in df.columns]
    df_metadata = df[metadata_cols].copy()

    # features for models
    df_features = df.drop(columns=metadata_cols)

    # keep only numeric features
    df_features = df_features.select_dtypes(include=["number"])

    # Save outputs
    df_features.to_csv(FEATURES_OUT, index=False)
    df_metadata.to_csv(METADATA_OUT, index=False)

    print(f"Original samples: {len(pd.read_csv(INPUT_CSV))}")
    print(f"Filtered samples (MIN >= {MIN_THRESHOLD}): {len(df)}")
    print(f"Feature columns: {df_features.shape[1]}")
    print(f"Saved features to: {FEATURES_OUT}")
    print(f"Saved metadata to: {METADATA_OUT}")

if __name__ == "__main__":
    main()
