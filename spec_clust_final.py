# %%
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, adjusted_rand_score

import matplotlib.pyplot as plt
import itertools
from sklearn.decomposition import PCA


# %%
DATA_PATH = r"C:\Users\Bote\Desktop\pml\project2\statistics\player_data_features.csv"
META_PATH = r"C:\Users\Bote\Desktop\pml\project2\statistics\player_data_remains.csv"

df_features = pd.read_csv(DATA_PATH)
df_meta = pd.read_csv(META_PATH)

assert len(df_features) == len(df_meta)

print("Feature shape:", df_features.shape)
print("Metadata shape:", df_meta.shape)


# %%
df_meta["START_POSITION"] = (
    df_meta["START_POSITION"]
    .astype(str)
    .str.strip()
    .str.upper()
)

df_meta["GAME_DATE"] = pd.to_datetime(df_meta["GAME_DATE"])
df_meta["SEASON"] = df_meta["GAME_DATE"].dt.year

VALID_POSITIONS = ["G", "F", "C"]
mask = df_meta["START_POSITION"].isin(VALID_POSITIONS)

df_features = df_features[mask].reset_index(drop=True)
df_meta = df_meta[mask].reset_index(drop=True)

print(df_meta["START_POSITION"].value_counts())


# %%
df = df_features.copy()
df["PLAYER_ID"] = df_meta["PLAYER_ID"]
df["SEASON"] = df_meta["SEASON"]
df["START_POSITION"] = df_meta["START_POSITION"]


# %%
COUNT_STATS = [
    "OREB", "AST", "REB", "DFGA", "SAST", "PFD", "TO",
    "FG3A", "STL", "PASS", "UFGM", "FG3M", "PTS", "UFGA",
    "DRBC", "FTM", "ORBC", "BLKA", "PTS_FB", "CFGA",
    "PTS_PAINT", "TCHS", "CFGM", "DFGM", "PTS_OFF_TOV",
    "FGA", "FTA", "PTS_2ND_CHANCE", "FGM", "PF", "DREB",
    "BLK", "RBC", "FTAST"
]

feature_cols = df_features.columns.tolist()
NON_COUNT_STATS = [c for c in feature_cols if c not in COUNT_STATS]

df_ps_features = (
    df
    .groupby(["PLAYER_ID", "SEASON"])
    .agg(
        {**{c: "sum" for c in COUNT_STATS},
         **{c: "mean" for c in NON_COUNT_STATS}}
    )
    .reset_index()
)

df_ps_labels = (
    df
    .groupby(["PLAYER_ID", "SEASON"])["START_POSITION"]
    .agg(lambda x: x.value_counts().idxmax())
    .reset_index()
)

print("Player–season samples:", len(df_ps_features))
print(df_ps_labels["START_POSITION"].value_counts())


# %%
df_ps_features_norm = df_ps_features.copy()

for col in COUNT_STATS:
    if col in df_ps_features_norm.columns and col != "MIN":
        df_ps_features_norm[col] /= df_ps_features_norm["MIN"]

print("Per-minute normalization applied (player–season).")


# %%
X = df_ps_features_norm.drop(columns=["PLAYER_ID", "SEASON"])
y = df_ps_labels["START_POSITION"]

# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Scaled shape:", X_scaled.shape)

# %%
X_train, _, y_train, _ = train_test_split(
    X_scaled,
    y,
    train_size=0.8,
    stratify=y,
    random_state=1
)

print("Train samples:", len(X_train))


# %%
MAX_SAMPLES = 10000

if len(X_train) > MAX_SAMPLES:
    X_spec, _, y_spec, _ = train_test_split(
        X_train,
        y_train,
        train_size=MAX_SAMPLES,
        stratify=y_train,
        random_state=1
    )
else:
    X_spec, y_spec = X_train, y_train

print("Spectral subset:", len(X_spec))


# %%
X_spec_norm = normalize(X_spec, norm="l2")

# %%
# Hyperparameter exploration: RAW FEATURES

CLUSTERS = [3, 4]
NEIGHBORS = [5, 15, 25, 50, 100, 250, 500, 1000, 1500]
ASSIGN_LABELS = ["kmeans", "discretize"]

param_grid = list(itertools.product(CLUSTERS, NEIGHBORS, ASSIGN_LABELS))

results = []

for n_clusters, n_neighbors, assign in param_grid:
    
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        n_neighbors=n_neighbors,
        assign_labels=assign,
        random_state=1
    )

    clusters = spectral.fit_predict(X_spec_norm)

    df_map = pd.DataFrame({
        "cluster": clusters,
        "position": y_spec.values
    })

    cluster_to_position = (
        df_map
        .groupby("cluster")["position"]
        .agg(lambda x: x.value_counts().idxmax())
    )

    y_pred = pd.Series(clusters).map(cluster_to_position)

    accuracy = (y_pred.values == y_spec.values).mean()
    ari = adjusted_rand_score(y_spec, clusters)

    results.append({
        "n_clusters": n_clusters,
        "n_neighbors": n_neighbors,
        "assign_labels": assign,
        "accuracy": accuracy,
        "ARI": ari
    })

    print(f"OK | k={n_clusters}, nn={n_neighbors}, assign={assign}")

results_df = pd.DataFrame(results)

results_df.sort_values("ARI", ascending=False).head(10)


# %%
# Plot ARI vs neighbors
plt.figure(figsize=(10, 6))
for assign in ASSIGN_LABELS:
    for k in CLUSTERS:
        subset = results_df[
            (results_df["assign_labels"] == assign) &
            (results_df["n_clusters"] == k)
        ]
        plt.plot(
            subset["n_neighbors"],
            subset["ARI"],
            marker="o",
            label=f"ARI | clusters={k}, assign={assign}"
        )

plt.xscale("log")
plt.xlabel("Number of neighbors (log scale)")
plt.ylabel("Adjusted Rand Index (ARI)")
plt.title("Spectral Clustering ARI vs neighbors (player–season)")
plt.legend()
plt.grid(True)
plt.show()

# Plot Accuracy vs neighbors
plt.figure(figsize=(10, 6))
for assign in ASSIGN_LABELS:
    for k in CLUSTERS:
        subset = results_df[
            (results_df["assign_labels"] == assign) &
            (results_df["n_clusters"] == k)
        ]
        plt.plot(
            subset["n_neighbors"],
            subset["accuracy"],
            marker="o",
            label=f"Acc | clusters={k}, assign={assign}"
        )

plt.xscale("log")
plt.xlabel("Number of neighbors (log scale)")
plt.ylabel("Accuracy")
plt.title("Spectral Clustering accuracy vs neighbors (player–season)")
plt.legend()
plt.grid(True)
plt.show()


# %%
spectral = SpectralClustering(
    n_clusters=3,
    affinity="nearest_neighbors",
    n_neighbors=1000,
    assign_labels="discretize",
    random_state=1
)

train_clusters = spectral.fit_predict(X_spec_norm)

print("Clusters:", np.unique(train_clusters))


# %%
df_train_map = pd.DataFrame({
    "cluster": train_clusters,
    "position": y_spec.values
})

cluster_to_position = (
    df_train_map
    .groupby("cluster")["position"]
    .agg(lambda x: x.value_counts().idxmax())
)

cluster_to_position


# %%
y_pred = pd.Series(train_clusters).map(cluster_to_position)

accuracy = (y_pred.values == y_spec.values).mean()
ari = adjusted_rand_score(y_spec, train_clusters)

print(f"Spectral accuracy (player–season): {accuracy:.4f}")
print(f"Spectral ARI (player–season): {ari:.4f}")


# %%
cm = confusion_matrix(
    y_spec,
    y_pred,
    labels=VALID_POSITIONS
)

pd.DataFrame(
    cm,
    index=[f"true_{p}" for p in VALID_POSITIONS],
    columns=[f"pred_{p}" for p in VALID_POSITIONS]
)


# %%
pca_2d = PCA(n_components=2, random_state=1)
X_2d = pca_2d.fit_transform(X_spec_norm)

plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=train_clusters, s=8, cmap="tab10")
plt.title("Spectral Clustering (player–season, 2D PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()


# %%
# Role-based aggregated feature representation

df_role = pd.DataFrame(index=df_features.index)

def mean_existing(cols):
    return df_features[[c for c in cols if c in df_features.columns]].mean(axis=1)

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

print("Scaled role-based shape:", X_role_scaled.shape)

# %%
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_role_scaled,
    df_meta["START_POSITION"],
    test_size=0.2,
    random_state=1,
    stratify=df_meta["START_POSITION"]
)

print("Train size:", len(Xr_train))
print("Test size:", len(Xr_test))


# %%
MAX_SAMPLES = 10000

if len(Xr_train) > MAX_SAMPLES:
    Xr_spec, _, yr_spec, _ = train_test_split(
        Xr_train,
        yr_train,
        train_size=MAX_SAMPLES,
        stratify=yr_train,
        random_state=1
    )
else:
    Xr_spec, yr_spec = Xr_train, yr_train

print("Spectral role-based subset size:", len(Xr_spec))

# %%
# Hyperparameter exploration: ROLE-BASED FEATURES

CLUSTERS = [3, 4]
NEIGHBORS = [5, 15, 25, 50, 100, 250, 500, 1000]
ASSIGN_LABELS = ["kmeans", "discretize"]

param_grid = list(itertools.product(CLUSTERS, NEIGHBORS, ASSIGN_LABELS))

results_role = []

for n_clusters, n_neighbors, assign in param_grid:
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        n_neighbors=n_neighbors,
        assign_labels=assign,
        random_state=1
    )

    clusters = spectral.fit_predict(Xr_spec)

    df_map = pd.DataFrame({
        "cluster": clusters,
        "position": yr_spec.values
    })

    cluster_to_position = (
        df_map
        .groupby("cluster")["position"]
        .agg(lambda x: x.value_counts().idxmax())
    )

    y_pred = pd.Series(clusters).map(cluster_to_position)

    accuracy = (y_pred.values == yr_spec.values).mean()
    ari = adjusted_rand_score(yr_spec, clusters)

    results_role.append({
        "n_clusters": n_clusters,
        "n_neighbors": n_neighbors,
        "assign_labels": assign,
        "accuracy": accuracy,
        "ARI": ari
    })

    print(f"OK | k={n_clusters}, nn={n_neighbors}, assign={assign}")

results_role_df = pd.DataFrame(results_role)

results_role_df.sort_values("ARI", ascending=False).head(10)




# %%
plt.figure(figsize=(10, 6))

for assign in ASSIGN_LABELS:
    for k in CLUSTERS:
        subset = results_role_df[
            (results_role_df["assign_labels"] == assign) &
            (results_role_df["n_clusters"] == k)
        ]
        plt.plot(
            subset["n_neighbors"],
            subset["ARI"],
            marker="o",
            label=f"ARI | clusters={k}, assign={assign}"
        )

plt.xscale("log")
plt.xlabel("Number of neighbors (log scale)")
plt.ylabel("Adjusted Rand Index (ARI)")
plt.title("Spectral Clustering ARI vs neighbors (ROLE-BASED)")
plt.legend()
plt.grid(True)
plt.show()

# Plot Accuracy vs neighbors
plt.figure(figsize=(10, 6))

for assign in ASSIGN_LABELS:
    for k in CLUSTERS:
        subset = results_role_df[
            (results_role_df["assign_labels"] == assign) &
            (results_role_df["n_clusters"] == k)
        ]
        plt.plot(
            subset["n_neighbors"],
            subset["accuracy"],
            marker="o",
            label=f"Acc | clusters={k}, assign={assign}"
        )

plt.xscale("log")
plt.xlabel("Number of neighbors (log scale)")
plt.ylabel("Accuracy")
plt.title("Spectral Clustering accuracy vs neighbors (ROLE-BASED)")
plt.legend()
plt.grid(True)
plt.show()

# %%
spectral_role = SpectralClustering(
    n_clusters=3,
    affinity="nearest_neighbors",
    n_neighbors=100,
    assign_labels="discretize",
    random_state=1
)

role_clusters = spectral_role.fit_predict(Xr_spec)

print("Spectral clusters (role-based):", np.unique(role_clusters))

# %%
df_role_map = pd.DataFrame({
    "cluster": role_clusters,
    "position": yr_spec.values
})

cluster_to_position_role = (
    df_role_map
    .groupby("cluster")["position"]
    .agg(lambda x: x.value_counts().idxmax())
)

cluster_to_position_role


# %%
y_pred_role = pd.Series(role_clusters).map(cluster_to_position_role)

accuracy_role = (y_pred_role.values == yr_spec.values).mean()
print(f"Spectral accuracy (role-based, train subset): {accuracy_role:.4f}")

ari_role = adjusted_rand_score(yr_spec, role_clusters)
print(f"Spectral ARI (role-based, train subset): {ari_role:.4f}")


# %%
VALID_POSITIONS = ["G","F","C"]

cm_role = confusion_matrix(
    yr_spec,
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
Xr_2d = pca_2d.fit_transform(Xr_spec)

plt.figure(figsize=(8,6))
plt.scatter(Xr_2d[:,0], Xr_2d[:,1], c=role_clusters, s=8, cmap="tab10")
plt.title("Spectral clustering (role-based features, 2D PCA)")
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

print(f"Behavioral Feature Shape: {X_beh_scaled.shape}")

# %%
X_beh_train, X_beh_test, y_beh_train, y_beh_test = train_test_split(
    X_beh_scaled,
    df_meta["START_POSITION"],
    test_size=0.2,
    random_state=1,
    stratify=df_meta["START_POSITION"]
)

# %%
MAX_SAMPLES = 10000

if len(X_beh_train) > MAX_SAMPLES:
    X_beh_spec, _, y_beh_spec, _ = train_test_split(
        X_beh_train,
        y_beh_train,
        train_size=MAX_SAMPLES,
        stratify=y_beh_train,
        random_state=1
    )
else:
    X_beh_spec, y_beh_spec = X_beh_train, y_beh_train

print(f"Spectral Behavioral Subset: {len(X_beh_spec)} samples")

# %%
# Hyperparameter exploration: BEHAVIORAL FEATURES

CLUSTERS = [3, 4]
NEIGHBORS = [5, 15, 25, 50, 100, 250, 500]
ASSIGN_LABELS = ["kmeans", "discretize"]

param_grid = list(itertools.product(CLUSTERS, NEIGHBORS, ASSIGN_LABELS))

results_beh = []

print("Starting Behavioral Sweep...")

for n_clusters, n_neighbors, assign in param_grid:
    
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        n_neighbors=n_neighbors,
        assign_labels=assign,
        random_state=1,
        n_jobs=-1
    )

    clusters = spectral.fit_predict(X_beh_spec)

    df_map = pd.DataFrame({
        "cluster": clusters,
        "position": y_beh_spec.values
    })

    cluster_to_position = (
        df_map
        .groupby("cluster")["position"]
        .agg(lambda x: x.value_counts().idxmax())
    )

    y_pred = pd.Series(clusters).map(cluster_to_position)

    accuracy = (y_pred.values == y_beh_spec.values).mean()
    ari = adjusted_rand_score(y_beh_spec, clusters)

    results_beh.append({
        "n_clusters": n_clusters,
        "n_neighbors": n_neighbors,
        "assign_labels": assign,
        "accuracy": accuracy,
        "ARI": ari
    })
    
    print(f"OK | k={n_clusters}, nn={n_neighbors}, assign={assign}")

results_beh_df = pd.DataFrame(results_beh)
print("\nTop 5 Results (sorted by ARI):")
results_beh_df.sort_values("ARI", ascending=False).head(5)

# %%
# Plot ARI vs neighbors
plt.figure(figsize=(10, 6))
for assign in ASSIGN_LABELS:
    for k in CLUSTERS:
        subset = results_beh_df[
            (results_beh_df["assign_labels"] == assign) &
            (results_beh_df["n_clusters"] == k)
        ]
        plt.plot(
            subset["n_neighbors"],
            subset["ARI"],
            marker="o",
            label=f"ARI | k={k}, {assign}"
        )

plt.xscale("log")
plt.xlabel("Number of neighbors (log scale)")
plt.ylabel("Adjusted Rand Index (ARI)")
plt.title("Spectral Clustering ARI vs neighbors(Behavioral Features)")
plt.legend()
plt.grid(True)
plt.show()

# --- Plot Accuracy vs Neighbors ---
plt.figure(figsize=(10, 6))

for assign in ASSIGN_LABELS:
    for k in CLUSTERS:
        subset = results_beh_df[
            (results_beh_df["assign_labels"] == assign) &
            (results_beh_df["n_clusters"] == k)
        ]
        plt.plot(
            subset["n_neighbors"],
            subset["accuracy"],
            marker="o",
            label=f"Acc | k={k}, {assign}"
        )

plt.xscale("log")
plt.xlabel("Number of neighbors (log scale)")
plt.ylabel("Accuracy")
plt.title("Spectral Clustering accuracy vs neighbors (Behavioral Features)")
plt.legend()
plt.grid(True)
plt.show()

# %%
spectral_beh = SpectralClustering(
    n_clusters=4,
    affinity="nearest_neighbors",
    n_neighbors=25,       
    assign_labels="kmeans",
    random_state=1,
    n_jobs=-1
)

beh_clusters = spectral_beh.fit_predict(X_beh_spec)

# %%
df_beh_map = pd.DataFrame({
    "cluster": beh_clusters,
    "position": y_beh_spec.values
})

cluster_to_pos_beh = (
    df_beh_map
    .groupby("cluster")["position"]
    .agg(lambda x: x.value_counts().idxmax())
)

print("\nCluster to Position Mapping:")
print(cluster_to_pos_beh)

# %%
y_pred_beh = pd.Series(beh_clusters).map(cluster_to_pos_beh)

acc_beh = (y_pred_beh.values == y_beh_spec.values).mean()
print(f"Spectral Accuracy (Behavioral): {acc_beh:.4f}")

ari_beh = adjusted_rand_score(y_beh_spec, beh_clusters)
print(f"Spectral ARI (Behavioral):      {ari_beh:.4f}")

# %%
VALID_POSITIONS = ["G", "F", "C"]

cm_beh = confusion_matrix(
    y_beh_spec,
    y_pred_beh,
    labels=VALID_POSITIONS
)

cm_beh_df = pd.DataFrame(
    cm_beh,
    index=[f"True_{p}" for p in VALID_POSITIONS],
    columns=[f"Pred_{p}" for p in VALID_POSITIONS]
)

cm_beh_df

# %%
pca_2d = PCA(n_components=2, random_state=1)
X_beh_2d = pca_2d.fit_transform(X_beh_spec)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_beh_2d[:, 0], X_beh_2d[:, 1], c=beh_clusters, s=8, cmap="tab10")
plt.title("Spectral Clustering (Behavioral Features, 2D PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()


