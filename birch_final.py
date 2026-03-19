# %%
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import adjusted_rand_score

import matplotlib.pyplot as plt


# %%
DATA_PATH = r"C:\Users\Bote\Desktop\pml\project2\statistics\player_data_features.csv"
META_PATH = r"C:\Users\Bote\Desktop\pml\project2\statistics\player_data_remains.csv"

df_features = pd.read_csv(DATA_PATH)
df_meta = pd.read_csv(META_PATH)

print("Feature shape:", df_features.shape)
print("Metadata shape:", df_meta.shape)

assert len(df_features) == len(df_meta), "Features and metadata not aligned!"


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
# Per-minute normalization for count-based statistics 

# Columns that represent raw counts
COUNT_STATS = [
    "OREB", "AST", "REB", "DFGA", "SAST", "PFD", "TO",
    "FG3A", "STL", "PASS", "UFGM", "FG3M", "PTS", "UFGA",
    "DRBC", "FTM", "ORBC", "BLKA", "PTS_FB", "CFGA",
    "PTS_PAINT", "TCHS", "CFGM", "DFGM", "PTS_OFF_TOV",
    "FGA", "FTA", "PTS_2ND_CHANCE", "FGM", "PF", "DREB",
    "BLK", "RBC", "FTAST"
]

# Work on a copy to keep the original data intact
df_features_norm = df_features.copy()

# Normalize counts per minute
for col in COUNT_STATS:
    if col in df_features_norm.columns:
        df_features_norm[col] = df_features_norm[col] / df_features_norm["MIN"]

print("Per-minute normalization applied to count-based stats.")


# %%
df_features_norm["INTERIOR_ACTIVITY"] = (
    df_features_norm["OREB"] +
    df_features_norm["DREB"] +
    df_features_norm["BLK"]
)

df_features_norm["PERIMETER_ACTIVITY"] = (
    df_features_norm["AST"] +
    df_features_norm["FG3A"] +
    df_features_norm["STL"]
)


df_features_norm["BALL_DOMINANCE"] = (
    df_features_norm["AST"] / (df_features_norm["TCHS"] + 1e-6)
)

df_features_norm["SHOT_PROFILE"] = (
    df_features_norm["FG3A"] - df_features_norm["PTS_PAINT"]
)


# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features_norm)

print("Scaled shape:", X_scaled.shape)


# %%
# pca = PCA(n_components=50, random_state=1)
# X_pca = pca.fit_transform(X_scaled)

# print("PCA shape:", X_pca.shape)
# print("Explained variance:", pca.explained_variance_ratio_.sum())


# %%
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    df_meta["START_POSITION"],
    test_size=0.2,
    random_state=1,
    stratify=df_meta["START_POSITION"]
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))


# %%
# Hyperparameter exploration: RAW FEATURES
thresholds = np.linspace(8.0, 12.0, 10)
branching_factors = [25, 50, 75, 100]

results_raw = []

for bf in branching_factors:
    for th in thresholds:
        birch = Birch(
            n_clusters=None,
            threshold=th,
            branching_factor=bf
        )
        birch.fit(X_train)   # RAW feature training set
        n_clusters = len(np.unique(birch.predict(X_train)))
        
        results_raw.append({
            "branching_factor": bf,
            "threshold": th,
            "n_clusters": n_clusters
        })

df_hp_raw = pd.DataFrame(results_raw)
df_hp_raw


# %%
plt.figure(figsize=(8,6))

for bf in branching_factors:
    subset = df_hp_raw[df_hp_raw["branching_factor"] == bf]
    plt.plot(
        subset["threshold"],
        subset["n_clusters"],
        marker="o",
        label=f"branching_factor={bf}"
    )

plt.axhline(3, color="gray", linestyle="--", label="target = 3 clusters")
plt.xlabel("threshold")
plt.ylabel("number of clusters")
plt.title("BIRCH hyperparameter sweep (RAW features)")
plt.legend()
plt.grid(True)
plt.show()


# %%
birch = Birch(
    n_clusters=None,
    threshold=10.65,
    branching_factor=50
)

birch.fit(X_train)

train_clusters = birch.predict(X_train)
test_clusters = birch.predict(X_test)

print("Clusters (train):", np.unique(train_clusters))


# %%
unique, counts = np.unique(train_clusters, return_counts=True)

for c, n in zip(unique, counts):
    print(f"Cluster {c}: {n} samples")


# %%
df_train_map = pd.DataFrame({
    "cluster": train_clusters,
    "position": y_train.values
})

cluster_to_position = (
    df_train_map
    .groupby("cluster")["position"]
    .agg(lambda x: x.value_counts().idxmax())
)

cluster_to_position


# %%
y_pred = pd.Series(test_clusters).map(cluster_to_position)

valid_idx = y_pred.notna()
y_pred = y_pred[valid_idx]
y_test_eval = y_test.iloc[valid_idx.index]

accuracy = (y_pred.values == y_test_eval.values).mean()
print(f"BIRCH test accuracy: {accuracy:.4f}")

ari = adjusted_rand_score(
    y_test_eval,
    test_clusters[valid_idx]
)

print(f"BIRCH ARI (test set): {ari:.4f}")

# %%
cm = confusion_matrix(
    y_test_eval,
    y_pred,
    labels=VALID_POSITIONS
)

cm_df = pd.DataFrame(
    cm,
    index=[f"true_{p}" for p in VALID_POSITIONS],
    columns=[f"pred_{p}" for p in VALID_POSITIONS]
)

cm_df


# %%
pca_2d = PCA(n_components=2, random_state=1)
Xr_2d = pca_2d.fit_transform(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(Xr_2d[:, 0], Xr_2d[:, 1], c=test_clusters, s=8, cmap="tab10")
plt.title("BIRCH clustering(RAW features, 2D PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()


# %%
# ROLE-BASED FEATURE REPRESENTATION

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
scaler_role = StandardScaler()
X_role_scaled = scaler_role.fit_transform(df_role)

print("Scaled role-based feature shape:", X_role_scaled.shape)


# %%
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_role_scaled,
    df_meta["START_POSITION"],
    test_size=0.2,
    random_state=2,
    stratify=df_meta["START_POSITION"]
)

print("Role-based train size:", len(Xr_train))
print("Role-based test size:", len(Xr_test))

# %%
# Hyperparameter exploration: ROLE-BASED FEATURES

thresholds_role = np.linspace(1.5, 4.0, 10)
branching_factors = [25, 50, 75, 100]

results_role = []

for bf in branching_factors:
    for th in thresholds_role:
        birch = Birch(
            n_clusters=None,
            threshold=th,
            branching_factor=bf
        )
        birch.fit(Xr_train)   # ROLE-based training set
        n_clusters = len(np.unique(birch.predict(Xr_train)))
        
        results_role.append({
            "branching_factor": bf,
            "threshold": th,
            "n_clusters": n_clusters
        })

df_hp_role = pd.DataFrame(results_role)
df_hp_role


# %%
plt.figure(figsize=(8,6))

for bf in branching_factors:
    subset = df_hp_role[df_hp_role["branching_factor"] == bf]
    plt.plot(
        subset["threshold"],
        subset["n_clusters"],
        marker="o",
        label=f"branching_factor={bf}"
    )

plt.axhline(3, color="gray", linestyle="--", label="target = 3 clusters")
plt.xlabel("threshold")
plt.ylabel("number of clusters")
plt.title("BIRCH hyperparameter sweep (ROLE-BASED features)")
plt.legend()
plt.grid(True)
plt.show()


# %%
birch_role = Birch(
    n_clusters=None,
    threshold=2.6,
    branching_factor=50
)

birch_role.fit(Xr_train)

train_clusters_role = birch_role.predict(Xr_train)
test_clusters_role = birch_role.predict(Xr_test)

print("Role-based clusters (train):", np.unique(train_clusters_role))

# %%
df_train_map_role = pd.DataFrame({
    "cluster": train_clusters_role,
    "position": yr_train.values
})

cluster_to_position_role = (
    df_train_map_role
    .groupby("cluster")["position"]
    .agg(lambda x: x.value_counts().idxmax())
)

cluster_to_position_role


# %%
y_pred_role = pd.Series(test_clusters_role).map(cluster_to_position_role)

valid_idx = y_pred_role.notna()
y_pred_role = y_pred_role[valid_idx]
yr_test_eval = yr_test.iloc[valid_idx.index]

accuracy_role = (y_pred_role.values == yr_test_eval.values).mean()
print(f"BIRCH accuracy (role-based features): {accuracy_role:.4f}")

ari_role = adjusted_rand_score(yr_test_eval, test_clusters_role[valid_idx])
print(f"BIRCH ARI (role-based features): {ari_role:.4f}")

# %%
VALID_POSITIONS = ["G", "F", "C"]

cm_role = confusion_matrix(
    yr_test_eval,
    y_pred_role,
    labels=VALID_POSITIONS
)

cm_role_df = pd.DataFrame(
    cm_role,
    index=[f"true_{p}" for p in VALID_POSITIONS],
    columns=[f"pred_{p}" for p in VALID_POSITIONS]
)

cm_role_df


# %%
pca_2d = PCA(n_components=2, random_state=1)
Xr_2d = pca_2d.fit_transform(Xr_test)

plt.figure(figsize=(8, 6))
plt.scatter(Xr_2d[:, 0], Xr_2d[:, 1], c=test_clusters_role, s=8, cmap="tab10")
plt.title("BIRCH clustering (role-based features, 2D PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()


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

print(f"Behavioral Feature Set Shape: {df_behavior.shape}")

# %%
# Split the behavioral features
X_beh_train, X_beh_test, y_beh_train, y_beh_test = train_test_split(
    X_beh_scaled,
    df_meta["START_POSITION"],
    test_size=0.2,
    random_state=2,
    stratify=df_meta["START_POSITION"]
)

print("Behavioral Train shape:", X_beh_train.shape)
print("Behavioral Test shape:", X_beh_test.shape)

# %%
# Hyperparameter exploration: BEHAVIORAL FEATURES

thresholds_beh = np.linspace(4.5, 6.5, 10)
branching_factors = [25, 50, 75, 100]

results_beh = []

for bf in branching_factors:
    for th in thresholds_beh:
        birch = Birch(
            n_clusters=None,
            threshold=th,
            branching_factor=bf
        )
        birch.fit(X_beh_train)
        
        n_clusters = len(np.unique(birch.predict(X_beh_train)))
        
        results_beh.append({
            "branching_factor": bf,
            "threshold": th,
            "n_clusters": n_clusters
        })

df_hp_beh = pd.DataFrame(results_beh)
df_hp_beh

# %%
plt.figure(figsize=(8,6))

for bf in branching_factors:
    subset = df_hp_beh[df_hp_beh["branching_factor"] == bf]
    plt.plot(
        subset["threshold"],
        subset["n_clusters"],
        marker="o",
        label=f"branching_factor={bf}"
    )

plt.axhline(3, color="gray", linestyle="--", label="target = 3 clusters")

plt.xlabel("Threshold")
plt.ylabel("Number of Clusters")
plt.title("BIRCH Hyperparameter Sweep (BEHAVIORAL Features)")
plt.legend()
plt.grid(True)
plt.show()

# %%
birch_role = Birch(
    n_clusters=None,
    threshold=5.65,
    branching_factor=50
)

birch_role.fit(X_beh_train)

train_clusters_beh = birch_role.predict(X_beh_train)
test_clusters_beh = birch_role.predict(X_beh_test)

print("Role-based clusters (train):", np.unique(train_clusters_beh))

# %%
df_train_map_role = pd.DataFrame({
    "cluster": train_clusters_beh,
    "position": y_beh_train.values
})

cluster_to_position_role = (
    df_train_map_role
    .groupby("cluster")["position"]
    .agg(lambda x: x.value_counts().idxmax())
)

cluster_to_position_role


# %%
y_pred_role = pd.Series(test_clusters_beh).map(cluster_to_position_role)

valid_idx = y_pred_role.notna()
y_pred_role = y_pred_role[valid_idx]
yr_test_eval = y_beh_test.iloc[valid_idx.index]

accuracy_role = (y_pred_role.values == yr_test_eval.values).mean()
print(f"BIRCH accuracy (Behavioral features): {accuracy_role:.4f}")

ari_role = adjusted_rand_score(yr_test_eval, test_clusters_beh[valid_idx])
print(f"BIRCH ARI (Behavioral features): {ari_role:.4f}")



# %%
VALID_POSITIONS = ["G", "F", "C"]

cm_role = confusion_matrix(
    yr_test_eval,
    y_pred_role,
    labels=VALID_POSITIONS
)

cm_role_df = pd.DataFrame(
    cm_role,
    index=[f"true_{p}" for p in VALID_POSITIONS],
    columns=[f"pred_{p}" for p in VALID_POSITIONS]
)

cm_role_df


# %%
pca_2d = PCA(n_components=2, random_state=1)
Xr_2d = pca_2d.fit_transform(X_beh_test)

plt.figure(figsize=(8, 6))
plt.scatter(Xr_2d[:, 0], Xr_2d[:, 1], c=test_clusters_beh, s=8, cmap="tab10")
plt.title("BIRCH clustering (Behavioral Features, 2D PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()



