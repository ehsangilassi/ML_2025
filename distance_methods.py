import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances


A = np.array([[1, 2]])
B = np.array([[4, 6]])

metrics = {
    "Euclidean (p=2)": ('minkowski', 2),
    "Manhattan (p=1)": ('minkowski', 1),
    "Chebyshev (p=∞)": ('chebyshev', None),
    "Minkowski (p=3)": ('minkowski', 3),
    "Cosine": ('cosine', None)
}

dist = pairwise_distances(A, B, metric='hamming')


for name, (metric, p) in metrics.items():
    if p is not None:
        dist = pairwise_distances(A, B, metric=metric, p=p)
    else:
        dist = pairwise_distances(A, B, metric=metric)
    print(f"{name} distance: {dist[0][0]:.4f}")



plt.figure(figsize=(6,6))
plt.grid(True)
plt.scatter(*A[0], color='blue', label='Point A (1,2)')
plt.scatter(*B[0], color='red', label='Point B (4,6)')
plt.plot([A[0][0], B[0][0]], [A[0][1], B[0][1]], 'k--', label='Euclidean Line')
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.axis('equal')
plt.show()
