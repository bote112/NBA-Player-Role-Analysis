# %%
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier


# %%
FEATURE_PATH = r"C:\Users\Bote\Desktop\pml\project2\statistics\player_data_features.csv"
META_PATH = r"C:\Users\Bote\Desktop\pml\project2\statistics\player_data_remains.csv"

df_features = pd.read_csv(FEATURE_PATH)
df_meta = pd.read_csv(META_PATH)

print("Features shape:", df_features.shape)
print("Metadata shape:", df_meta.shape)


# %%
df_meta["START_POSITION"] = (
    df_meta["START_POSITION"]
    .astype(str)
    .str.strip()
    .str.upper()
)

VALID_POSITIONS = ["G", "F", "C"]
mask = df_meta["START_POSITION"].isin(VALID_POSITIONS)

df_features = df_features[mask].reset_index(drop=True)
df_meta = df_meta[mask].reset_index(drop=True)

print("Filtered samples:", len(df_meta))
print(df_meta["START_POSITION"].value_counts())


# %%
COUNT_STATS = [
    "OREB","AST","REB","DFGA","SAST","PFD","TO",
    "FG3A","STL","PASS","UFGM","FG3M","PTS","UFGA",
    "DRBC","FTM","ORBC","BLKA","PTS_FB","CFGA",
    "PTS_PAINT","TCHS","CFGM","DFGM","PTS_OFF_TOV",
    "FGA","FTA","PTS_2ND_CHANCE","FGM","PF","DREB",
    "BLK","RBC","FTAST"
]

df_features_norm = df_features.copy()

for col in COUNT_STATS:
    if col in df_features_norm.columns:
        df_features_norm[col] = df_features_norm[col] / df_features_norm["MIN"]

print("Per-minute normalization applied.")


# %%
# raw feature

X_full = df_features_norm.copy()

y = df_meta["START_POSITION"]

print("Full feature shape:", X_full.shape)


# %%
scaler_full = StandardScaler()
X_full_scaled = scaler_full.fit_transform(X_full)

print("Scaled full feature shape:", X_full_scaled.shape)


# %%
Xf_train, Xf_test, yf_train, yf_test = train_test_split(
    X_full_scaled,
    y,
    test_size=0.2,
    random_state=1,
    stratify=y
)

print("Train size (full):", len(Xf_train))
print("Test size (full):", len(Xf_test))


# %%
rf_full = RandomForestClassifier(
    n_estimators=400,
    random_state=1,
    n_jobs=-1
)

rf_full.fit(Xf_train, yf_train)


# %%
yf_pred = rf_full.predict(Xf_test)

rf_full_acc = accuracy_score(yf_test, yf_pred)
rf_full_ari = adjusted_rand_score(yf_test, yf_pred)

print(f"Random Forest (full features) accuracy: {rf_full_acc:.4f}")
print(f"Random Forest (full features) ARI: {rf_full_ari:.4f}")


# %%
VALID_POSITIONS = ["G", "F", "C"]

cm_rf_full = confusion_matrix(
    yf_test,
    yf_pred,
    labels=VALID_POSITIONS
)

cm_rf_full_df = pd.DataFrame(
    cm_rf_full,
    index=[f"true_{p}" for p in VALID_POSITIONS],
    columns=[f"pred_{p}" for p in VALID_POSITIONS]
)

cm_rf_full_df

print(classification_report(yf_test, yf_pred))

# %%
# Role-based aggregated feature representation

df_role = pd.DataFrame(index=df_features_norm.index)

def mean_existing(cols):
    return df_features_norm[[c for c in cols if c in df_features_norm.columns]].mean(axis=1)

# 1. Interior presence
df_role["INTERIOR_PRESENCE"] = mean_existing([
    "OREB","DREB","REB","ORBC","DRBC","RBC",
    "BLK","BLKA","PTS_PAINT",
    "OREB_PCT","DREB_PCT","REB_PCT",
    "PCT_PTS_PAINT","PCT_BLK","PCT_BLKA"
])

# 2. Perimeter activity
df_role["PERIMETER_ACTIVITY"] = mean_existing([
    "FG3A","FG3M","PCT_FGA_3PT","PCT_FG3A","FG3_PCT",
    "STL","PCT_STL","PCT_PTS_3PT"
])

# 3. Playmaking
df_role["PLAYMAKING"] = mean_existing([
    "AST","SAST","PASS",
    "AST_PCT","PCT_AST","AST_RATIO",
    "PCT_AST_FGM","PCT_AST_2PM","PCT_AST_3PM",
    "PCT_UAST_FGM","PCT_UAST_2PM","PCT_UAST_3PM"
])

# 4. Usage / offense
df_role["USAGE_OFFENSE"] = mean_existing([
    "USG_PCT","E_USG_PCT","FGA","UFGA","CFGA",
    "PTS","PCT_PTS","TCHS","POSS","PCT_FGA"
])

# 5. Defensive impact
df_role["DEFENSIVE_IMPACT"] = mean_existing([
    "STL","BLK","DFGA","DFGM","DFG_PCT",
    "TO","PCT_TOV","PF","PCT_PF"
])

# 6. Efficiency
df_role["EFFICIENCY"] = mean_existing([
    "FG_PCT","EFG_PCT","TS_PCT",
    "FT_PCT","FTM","FTA","FTAST",
    "CFG_PCT","UFG_PCT","PCT_FTM","FTA_RATE"
])

# 7. Activity / pace
df_role["ACTIVITY_PACE"] = mean_existing([
    "MIN","PACE","PACE_PER40","E_PACE","DIST","PIE"
])

print("Role-based feature shape:", df_role.shape)
df_role.head()


# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_role)

y = df_meta["START_POSITION"]

print("Scaled X shape:", X_scaled.shape)


# %%
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))


# %%
np.random.seed(1)

y_random = np.random.choice(VALID_POSITIONS, size=len(y_test))

rand_acc = accuracy_score(y_test, y_random)

label_to_int = {"G": 0, "F": 1, "C": 2}
y_test_int = y_test.map(label_to_int)
y_random_int = pd.Series(y_random).map(label_to_int)

rand_ari = adjusted_rand_score(y_test_int, y_random_int)

print(f"Random baseline accuracy: {rand_acc:.4f}")
print(f"Random baseline ARI: {rand_ari:.4f}")


# %%
cm_rand = confusion_matrix(y_test, y_random, labels=VALID_POSITIONS)

cm_rand_df = pd.DataFrame(
    cm_rand,
    index=[f"true_{p}" for p in VALID_POSITIONS],
    columns=[f"pred_{p}" for p in VALID_POSITIONS]
)

cm_rand_df


# %%
rf = RandomForestClassifier(
    n_estimators=400,
    random_state=1,
    n_jobs=-1
)

rf.fit(X_train, y_train)


# %%
y_rf_pred = rf.predict(X_test)

rf_acc = accuracy_score(y_test, y_rf_pred)
rf_ari = adjusted_rand_score(y_test, y_rf_pred)

print(f"Random Forest accuracy: {rf_acc:.4f}")
print(f"Random Forest ARI: {rf_ari:.4f}")


# %%
cm_rf = confusion_matrix(y_test, y_rf_pred, labels=VALID_POSITIONS)

cm_rf_df = pd.DataFrame(
    cm_rf,
    index=[f"true_{p}" for p in VALID_POSITIONS],
    columns=[f"pred_{p}" for p in VALID_POSITIONS]
)

cm_rf_df

print(classification_report(y_test, y_rf_pred))

# %%
# BEHAVIORAL 
# This set focuses on efficiency, shot selection, and tracking data

BEHAVIORAL_COLS = [
    # 1. Shot Profile
    "PCT_FGA_2PT", "PCT_FGA_3PT", "PCT_PTS_PAINT", 
    "PCT_PTS_2PT_MR", "PCT_PTS_FB", "PCT_PTS_OFF_TOV", 
    "PCT_PTS_FT", "FTA_RATE",
    
    # 2. Efficiency
    "TS_PCT", "EFG_PCT", "FG_PCT", "FG3_PCT", "FT_PCT",
    
    # 3. Ball Handling & Passing Style
    "AST_PCT", "AST_TOV", "AST_RATIO", "TM_TOV_PCT",
    "PCT_AST", "PCT_UAST_2PM", "PCT_UAST_3PM", "PCT_TOV",
    "USG_PCT",
    
    # 4. Tracking & Effort
    "DIST", "PACE", "PIE", "OFF_RATING", "DEF_RATING", 
    "OREB_PCT", "DREB_PCT", "DFG_PCT", "PCT_BLKA"
]

df_behavior = df_features[BEHAVIORAL_COLS].copy()

scaler_beh = StandardScaler()
X_beh_scaled = scaler_beh.fit_transform(df_behavior)

y = df_meta["START_POSITION"]

print(f"Behavioral Feature Shape: {X_beh_scaled.shape}")

# %%
X_beh_train, X_beh_test, y_beh_train, y_beh_test = train_test_split(
    X_beh_scaled,
    y,
    test_size=0.2,
    random_state=1,
    stratify=y
)

# %%
rf_beh = RandomForestClassifier(
    n_estimators=400,
    random_state=1,
    n_jobs=-1
)

rf_beh.fit(X_beh_train, y_beh_train)

# %%
y_beh_pred = rf_beh.predict(X_beh_test)

rf_beh_acc = accuracy_score(y_beh_test, y_beh_pred)
rf_beh_ari = adjusted_rand_score(y_beh_test, y_beh_pred)

print(f"Random Forest (Behavioral) Accuracy: {rf_beh_acc:.4f}")
print(f"Random Forest (Behavioral) ARI:      {rf_beh_ari:.4f}")

# %%
VALID_POSITIONS = ["G", "F", "C"]

cm_beh = confusion_matrix(
    y_beh_test,
    y_beh_pred,
    labels=VALID_POSITIONS
)

cm_beh_df = pd.DataFrame(
    cm_beh,
    index=[f"true_{p}" for p in VALID_POSITIONS],
    columns=[f"pred_{p}" for p in VALID_POSITIONS]
)

cm_beh_df

print(classification_report(y_beh_test, y_beh_pred))

# %%



