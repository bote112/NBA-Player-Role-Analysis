## Overview

This project explores whether unsupervised learning methods can recover meaningful basketball player roles from game-level statistics.

Modern NBA roles are increasingly fluid, making traditional position labels (Guard, Forward, Center) less representative of actual on-court behavior. This work investigates whether clustering algorithms can discover these roles directly from data, without using labels during training.


## Objective

- Learn player roles using **unsupervised clustering**
- Compare results against:
  - A **random baseline**
  - A **supervised Random Forest classifier**
- Analyze how **feature representation impacts clustering quality**


## Dataset

- ~255,000 player-game samples
- 115 total features (statistical + metadata)
- Filtered to:
  - Players with ≥ 15 minutes played
  - Valid positions: Guard (G), Forward (F), Center (C)

Final dataset:
- ~95,000 samples  
- 105 numerical features used for modeling  

Preprocessing pipeline:
- Per-minute normalization of count statistics  
- Standardization using `StandardScaler`  
- Removal of metadata and non-numeric features  


## Feature Representations

Three feature spaces were evaluated:

### 1. Raw Statistical Features
- Full set of normalized game statistics  
- Highest dimensionality and detail  

### 2. Role-Based Aggregated Features
- 7 interpretable dimensions:
  - Interior presence  
  - Perimeter activity  
  - Playmaking  
  - Usage / offense  
  - Defensive impact  
  - Efficiency  
  - Activity / pace  

### 3. Behavioral Features
- Focus on *how* players contribute:
  - Shot selection  
  - Efficiency metrics  
  - Usage patterns  
  - Tracking data (pace, distance, ratings)  


## Models

### BIRCH Clustering
- Scalable clustering for large datasets  
- Incrementally builds clustering feature trees  
- Hyperparameters tuned to produce ~3 clusters  


### Spectral Clustering
- Graph-based clustering using nearest-neighbor similarity  
- Applied on subsampled datasets due to computational cost  
- Tuned over:
  - Number of neighbors  
  - Cluster count  
  - Label assignment strategy  


### Supervised Baseline (Random Forest)
- Trained using true position labels  
- Provides an upper-bound benchmark  


## Evaluation

Since clustering does not produce labels directly:

- Clusters are mapped to positions using **majority voting**
- Metrics:
  - Accuracy  
  - Adjusted Rand Index (ARI)  
  - Confusion matrices  



## Results

### Baselines
- Random baseline:
  - Accuracy: 0.3325  
  - ARI: ~0  

### Supervised Model
- Random Forest (raw features):
  - Accuracy: 0.7275  
  - ARI: 0.3450  

### Unsupervised Models

#### BIRCH
- Accuracy: ~0.53 – 0.55  
- ARI: ~0.10 – 0.14  

#### Spectral Clustering
- Accuracy: up to ~0.58  
- ARI: up to ~0.21  


## Key Findings

### 1. Centers are Easily Identifiable
- Consistently form a distinct cluster  
- Strong statistical signature (rebounds, blocks, paint scoring)  

### 2. Guards and Forwards Overlap
- Frequently merged or inconsistently separated  
- Reflects modern position fluidity  

### 3. Feature Representation Matters
- Role-based features improve interpretability  
- Behavioral features help supervised models  
- Raw features perform best for clustering  

### 4. Unsupervised Limitations
- Clustering captures coarse structure  
- Cannot fully separate overlapping roles  


## Visualization

PCA projections (see report):
- Clear separation for Centers  
- Significant overlap between Guards and Forwards  


## Limitations

Both clustering approaches faced practical and methodological limitations:

### BIRCH
* **Memory Constraints:** While highly scalable, BIRCH could not fully exploit its flexibility due to memory limitations.
* **Intensive Global Clustering:** Enforcing a fixed number of clusters requires an additional global clustering step over the CF tree, which proved too memory-intensive on the full dataset.
* **Indirect Parameter Control:** Consequently, the number of clusters had to be controlled indirectly through the threshold parameter, limiting fine-grained control over the clustering structure.

### Spectral Clustering
* **Poor Scalability:** The method relies on pairwise similarity computations that grow quadratically (or worse) with the number of samples.
* **Sparsification Trade-offs:** To make the method tractable, a k-nearest-neighbors (k-NN) graph was used to sparsify the similarity matrix and reduce computational costs.
* **Geometric Distortion:** This workaround introduces sensitivity to the choice of neighborhood size and can distort the underlying data geometry, particularly if the graph becomes too sparse or too dense.
