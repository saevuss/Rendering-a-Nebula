import trimesh
import numpy as np

mesh = trimesh.load("objs/crabNebula.obj", force='mesh')
verts = mesh.vertices.astype(np.float64)

# PCA sui vertici — il primo autovettore è l'asse principale del mesh
center = verts.mean(axis=0)
verts_centered = verts - center
cov = np.cov(verts_centered.T)
eigenvalues, eigenvectors = np.linalg.eigh(cov)

# ordinati dal più grande al più piccolo
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("Asse principale (jet candidate):", eigenvectors[:, 0])
print("Asse secondario:", eigenvectors[:, 1])
print("Asse minore:", eigenvectors[:, 2])