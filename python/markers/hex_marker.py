import numpy as np
from scipy.spatial import Delaunay
import cv2
import os

def generate_hex_lattice(n_rings: int, ring_radius: float = 60.0):
    points = []
    ring_ids = []

    for k in reversed(range(1, n_rings)):
        r = k * ring_radius

        vertices = np.array([[r * np.cos(np.pi / 3 * v), r * np.sin(np.pi / 3 * v)] for v in range(6)])

        for v in range(6):
            p0 = vertices[v]
            p1 = vertices[(v + 1) % 6]
            for i in range(k):
                t = i / k
                pt = (1 - t) * p0 + t * p1
                points.append(pt.tolist())
                ring_ids.append(k)

    points.append([0.0, 0.0])
    ring_ids.append(0)

    return np.array(points, dtype=np.float32), np.array(ring_ids, dtype=np.int32)


def delaunay_edge_index(points: np.ndarray):
    tri = Delaunay(points)

    edges = set()
    for simplex in tri.simplices:
        a, b, c = simplex
        for u, v in [(a, b), (b, c), (a, c)]:
            if u > v:
                u, v = v, u
            edges.add((u, v))

    src, dst = [], []
    for u, v in edges:
        src += [u, v]
        dst += [v, u]

    edge_index = np.array([src, dst], dtype=np.int64)
    return edge_index, tri


def visualise(
    points: np.ndarray,
    ring_ids: np.ndarray,
    tri: Delaunay,
    edge_index: np.ndarray,
    n_rings: int,
    img_size: int = 800,
    margin: int = 60,
):
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

    pts = points.copy()
    max_extent = np.abs(pts).max() + 1e-6
    scale = (img_size / 2 - margin) / max_extent
    pts_img = pts * scale + img_size / 2

    def to_pt(p):
        return (int(round(p[0])), int(round(p[1])))

    ring_colors = [
        (200, 80, 80),
        (60, 140, 220),
        (60, 200, 120),
        (220, 160, 60),
        (160, 60, 220),
        (60, 200, 210),
        (220, 80, 160),
        (100, 180, 80),
        (60, 140, 220),
        (60, 200, 120),
        (220, 160, 60),
        (160, 60, 220),
        (220, 80, 160),
        (100, 180, 80),
    ]

    def ring_color(k):
        return ring_colors[min(k, len(ring_colors) - 1)]

    for simplex in tri.simplices:
        for i in range(3):
            a = to_pt(pts_img[simplex[i]])
            b = to_pt(pts_img[simplex[(i + 1) % 3]])
            cv2.line(img, a, b, (190, 190, 190), 1, cv2.LINE_AA)

    for idx, (p, k) in enumerate(zip(pts_img, ring_ids)):
        color = ring_color(k)
        radius = 9 if k == 0 else 6
        cv2.circle(img, to_pt(p), radius + 2, (255, 255, 255), -1)
        cv2.circle(img, to_pt(p), radius, color, -1)

    for idx, p in enumerate(pts_img):
        cv2.putText(img, str(idx), (int(p[0]) + 7, int(p[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 80), 1, cv2.LINE_AA)

    n_nodes = len(points)
    n_edges = edge_index.shape[1] // 2
    cv2.putText(img, f"Circular Lattice  n_rings={n_rings}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(
        img,
        f"nodes={n_nodes}  undirected edges={n_edges}  edge_index shape={edge_index.shape}",
        (12, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (100, 100, 100),
        1,
        cv2.LINE_AA,
    )

    for k in range(n_rings):
        color = ring_color(k)
        cx, cy = 14, img_size - 20 - k * 22
        cv2.circle(img, (cx, cy), 7, color, -1)
        label = f"Ring {k}" + (" (center)" if k == 0 else f"  {k*6} pts")
        cv2.putText(img, label, (cx + 12, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1, cv2.LINE_AA)

    cv2.imshow("output", img)
    cv2.waitKey(0)
    return img


if __name__ == "__main__":
    N_RINGS = 7  # Include centre
    RING_RADIUS = 70.0  # independent param
    
    dir = './outputs/markers'
    edge_index_dir = os.path.join(dir, "edge_index")
    keypoints_dir = os.path.join(dir, "keypoints")
    os.makedirs(edge_index_dir, exist_ok=True)
    os.makedirs(keypoints_dir, exist_ok=True)

    points, ring_ids = generate_hex_lattice(N_RINGS, RING_RADIUS)

    edge_index, tri = delaunay_edge_index(points)
    n_undirected = edge_index.shape[1] // 2
    print(f"  edge_index shape: {edge_index.shape}")

    visualise(points, ring_ids, tri, edge_index, N_RINGS, img_size=800)

    np.save(os.path.join(edge_index_dir, f"edge_index_hex_n{N_RINGS}.npy"), edge_index)
    np.save(os.path.join(keypoints_dir, f"keypoints_hex_n{N_RINGS}.npy"), points)
