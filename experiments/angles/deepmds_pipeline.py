"""DeepMDS pipeline aligned to Gong et al. (2019)."""
#setup
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse.csgraph import connected_components # to check if the knn graph is fully connected, if not keep only the largest component
import sys
sys.path.append("../dim estimation")
from dimestimation_pipeline import build_knn_graph, compute_geodesic_distances

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Hyperparams ------------------------------------------------------------ 
# Paper Sec. 3.1 + Sec. 4.4: estimate the intrinsic dimension of the dataset
# using the ID estimation method (dimestimation_pipeline.py: estimate_intrinsic_dimension).
# Paper Table 1 examples: SphereFace IJB-C: ID=16 (k=15*)
# embed_dim is set from the global intrinsic dimension computed in the notebook
# and passed via init_globals when running this script with runpy.run_path()
embed_dim = math.ceil(global_id) if 'global_id' in dir() else 16  # round up: 11.12 -> 12 

# Paper Sec. 3.2: "For example, a direct mapping from R^512 → R^15 is instead decomposed
# into multiple mapping functions R^512 → R^256 → R^128 → R^64 → R^32 → R^15"
# Curriculum learning: progressively reduce dimensionality by 2x at each stage.
# AMBIENT SPACE DIMENSIONS: ArcFace/SphereFace/CosFace: 512-dim (paper uses SphereFace 512-dim)

stage_dims = [512, 256, 128, 64, 32, embed_dim]  #OJO CAMBIAR EMBED DIM

# Paper Sec. 4.4, Table 1: k=15 is selected for SphereFace (marked with *).
# "the choice of k is constrained by three factors: (1) k should be small enough to avoid shortcuts between points that are close to each other in the Euclidean space, but are potentially far away in the correspondind intrinsic manifold due to highly complicated local curvatures,
# (2) On the other hand, k should also be large enough to resullt in a connected graph  i.e., there are no isolated data samples., and (3) k that best matches the geodesic distance distribution of a hyper-sphere of the same ID i.e., k that minimizes the RMSE."
# Sec. 4.4: "we select the k-nearest neighbors using cosine similarity for SphereFace"
knn_k = 15 #OJO CAMBIAR SEGUN K SE OBTUVO EN ID ESTIMATION

# Batch size for stochastic training (Sec. 3.2: "trained in a stochastic fashion").
batch_size = 256 #reasonable choice for scalability.

# Sec. 4.5 mentions training but doesn't specify exact epoch counts.
epochs = 10

# Number of batches per epoch (for stochastic sampling from dataset).
steps_per_epoch = 50 # Not specified in paper

# Paper Sec. 4.5: "The parameters of the network are learned using the Adam [22] optimizer with a learning rate of 3 × 10^-4"
lr = 3e-4

# Paper Sec. 4.5: "and the regularization parameter λ = 3 × 10^-4"
# This corresponds to the λ||θ||_2^2 term in the MDS loss function (Sec. 3.2):
#"min_θ Σ[dH(xi,xj) - dL(f(xi;θ), f(xj;θ))]^2 + λ||θ||_2^2"
weight_decay = 3e-4

# Random seed for reproducibility: ensures experiments are repeatable 
# (same dataset, same code → same result). The value 0 is arbitrary; any integer (0, 42, 123, etc.) works. Not specified in paper.
rng = np.random.default_rng(0)

# -------------------- DeepMDS model ----------------------------------------------------------
class ResidualBlock(nn.Module):
    #skip connection laden residual units”
    #Paper Fig. 3, Sec. 4.5: Shows architecture with BatchNorm, PReLU, and (+) skip connection

    def __init__(self, dim):
        super().__init__()
        # Fig. 3 order: BatchNorm → PReLU → Linear 
        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.PReLU()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        #Fig. 3: x → BN → PReLU → Linear → (+x)
        #The + symbol shows skip connection adds x. NO activation after add.
        y = self.bn(x) 
        y = self.act(y)
        y = self.fc(y)
        return y + x  # Skip connection, NO PReLU after (per Fig. 3)   

class DeepMDSNet(nn.Module):
    #Paper Fig. 3: "The architecture of the proposed DeepMDS network, which consists of multiple stages of residual blocks and dimensionality reduction layers."
    #Paper Sec. 3.2: "progressively reduce dimensionality in multiple stages"
    #Paper Sec. 4.5: "each stage comprising of two residual units"
    #Paper Table 4: Stagewise (92.33%) >> Direct (80.25%)

    def __init__(self, in_dim, stage_dims):
        super().__init__()
        stages = []
        prev = in_dim
        for d in stage_dims:
            stages.append(
                nn.ModuleDict({
                    # Sec. 4.5: "two residual units" per stage
                    "blocks": nn.ModuleList([
                        ResidualBlock(prev), ResidualBlock(prev)]),
                    # Dimensionality reduction: prev → d
                    "reduction": nn.Sequential(
                        nn.Linear(prev, d),
                        nn.BatchNorm1d(d),
                        nn.PReLU(),
                    ),
                })
            )
            prev = d
        self.stages = nn.ModuleList(stages)

    def forward(self, x):
        #Forward through ALL stages, returning intermediate outputs.
        #Paper Sec. 3.2: Multi-stage loss uses outputs at each stage.
        outs = []
        h = x
        for stage in self.stages:
            # Apply 2 residual blocks
            for block in stage["blocks"]:
                h = block(h)
            # Reduce dimension
            h = stage["reduction"](h)
             # Store for multi-stage loss
            outs.append(h)
        return outs

    def forward_to_stage(self, x, stage_idx):
        #Forward up to specific stage (for curriculum/stagewise training).
        #Paper Sec. 3.2: Curriculum learning trains stages progressively.
        h = x
        for i, stage in enumerate(self.stages):
            if i > stage_idx:
                print(f"Reached stage {i}, stopping forward at stage {stage_idx}")
                break
            for block in stage["blocks"]:
                h = block(h)
            h = stage["reduction"](h)
        return h

#Geodesic distance computation: builds knn graph and computes geodesic distances using shortest paths.
def geodesic_distances_np(X, k=15):
    #   Paper Sec. 3.1 + Fig. 2(a): Graph induced geodesic distances capture manifold topology.

    graph = build_knn_graph(X, k=k, metric="cosine")
    n_comp, labels = connected_components(graph, directed=False, return_labels=True)
    if n_comp > 1:
        counts = np.bincount(labels)# Keep only largest connected component
        keep = np.where(labels == np.argmax(counts))[0]
        graph = graph[keep][:, keep]
        X = X[keep]
    else:
        keep = np.arange(X.shape[0])

    D = compute_geodesic_distances(graph)
    return D, keep

#MDS loss function: pairwise loss between geodesic distances (dH) and learned distances (dL).
def pairwise_loss(y, dH, mask):
    #MDS objective: preserve geodesic distances.
    #Paper Sec. 3.2, Eq. (4): min Σ(dH - dL)^2
    dL = torch.cdist(y, y, p=2)# Euclidean in intrinsic space
    diff = dL - dH
    diff = diff[mask]
    return (diff * diff).mean()

#Batch sampling function: samples random indices for stochastic training
def sample_batch_indices(n, batch_size):
    #Stochastic batch sampling for scalability.
    #Paper Sec. 3.2: "trained in a stochastic fashion"
    if n < batch_size:
        return rng.choice(n, size=n, replace=False)
    return rng.choice(n, size=batch_size, replace=False)

# -------------------- Train ----------------------------------------------------------
X = E.astype(np.float32)  # E normalized embeddings before in the notebook
n = X.shape[0]

model = DeepMDSNet(X.shape[1], stage_dims).to(device)
# Paper Sec. 3.2: Stage weights α_l for curriculum
alphas = np.linspace(0.2, 1.0, len(stage_dims))
alphas = alphas / alphas.sum()

# -------------------- Stage-wise Training - curriculum learning ---------------------------------------------------
#"We adopt a curriculum learning [3] approach... progressively reduce the dimensionality of the mapping in 
# multiple stages... For example, a direct mapping from R^512 → R^15 is instead decomposed into multiple
#  mapping functions R^512 → R^256 → R^128 → R^64 → R^32 → R^15"

model.train()
for stage_idx in range(len(stage_dims)):
    print(f"Training stage {stage_idx+1}/{len(stage_dims)} with output dim {stage_dims[stage_idx]}")
    #"The parameters of the network are learned using the Adam [22] optimizer with a learning rate of 3 × 10^-4"
    #"and the regularization parameter λ = 3 × 10^-4"
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    #"We observed that using the cosine-annealing scheduler [27] was critical to learning an effective mapping."
    #[27] = Loshchilov & Hutter, 2016: "SGDR: stochastic gradient descent with warm restarts"
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        losses = []
        for _ in range(steps_per_epoch):
            #"Specifically, DeepMDS is trained in a stochastic fashion, which allows it to scale."
            idx = sample_batch_indices(n, batch_size)

            Xb = X[idx]
            #"using graph induced geodesic distances to estimate the correlation dimension... 
            # the distance r between any pair of points in the manifold is computed as the shortest path 
            # between the points as induced by the graph connecting all the points in the representation."
            D, keep = geodesic_distances_np(Xb, k=knn_k)
            if keep.size < 3:
                continue

            Xb = Xb[keep]
            D = D.astype(np.float32)

            xb = torch.from_numpy(Xb).to(device)
            dH = torch.from_numpy(D).to(device)

            mask = torch.isfinite(dH) & (dH > 0)

            y = model.forward_to_stage(xb, stage_idx)#"We start with easier sub-tasks and progressively increase the difficulty of the tasks."
            loss = pairwise_loss(y, dH, mask)#"where dH(·) and dL(·) are distance (similarity) metrics in the ambient and intrinsic space, respectively.""min Σ_{i<j} (dH(xi, xj) - dL(yi, yj))^2"

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        print(
            f"Epoch {epoch+1}/{epochs} - loss: {np.mean(losses):.6f}"
        )

# -------------------- Joint Fine-tuning - multi stage loss-----------------------------------------------------
print("JOINT FINE-TUNING") #sec 3.2: "After training all stages, we fine-tune the entire network jointly using a multi-stage loss that combines the outputs of all stages."
#"After training all stages, we fine-tune the entire network jointly using a multi-stage loss that combines the outputs of all stages."
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

for epoch in range(epochs):
    losses = []
    for _ in range(steps_per_epoch):
        idx = sample_batch_indices(n, batch_size)

        Xb = X[idx]
        D, keep = geodesic_distances_np(Xb, k=knn_k)
        if keep.size < 3:
            continue

        Xb = Xb[keep]
        D = D.astype(np.float32)

        xb = torch.from_numpy(Xb).to(device)
        dH = torch.from_numpy(D).to(device)

        mask = torch.isfinite(dH) & (dH > 0)

        outs = model(xb)
        loss = 0.0
        for a, y in zip(alphas, outs):
            loss = loss + a * pairwise_loss(y, dH, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    scheduler.step()
    print(f"Fine-tune Epoch {epoch+1}/{epochs} - loss: {np.mean(losses):.6f}")

print("TRAINING COMPLETE!")
print(f"Model projects {X.shape[1]}-dim → {embed_dim}-dim")