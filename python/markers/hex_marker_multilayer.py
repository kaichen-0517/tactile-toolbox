"""
Multi-layer hexagonal marker lattice.

Each layer is an independent hexagonal lattice with the same topology.
Intra-layer edges are computed via Delaunay triangulation.
Inter-layer edges connect every node in layer l to its counterpart in layer l+1
(i.e. node i in layer l  <-->  node i + N in layer l+1, where N = nodes per layer).

Node numbering:
    layer 0 : [0,       N-1]
    layer 1 : [N,     2N-1]
    layer k : [k*N, (k+1)*N - 1]
"""

import numpy as np
from scipy.spatial import Delaunay
import cv2
import os

# ---------------------------------------------------------------------------
# Lattice generation
# ---------------------------------------------------------------------------


def generate_hex_lattice(n_rings: int, ring_radius: float = 60.0):
    """Return (points [M,2], ring_ids [M]) for a single hex lattice layer."""
    points: list[list[float]] = []
    ring_ids: list[int] = []

    for k in reversed(range(1, n_rings)):
        r = k * ring_radius
        vertices = np.array([[r * np.cos(np.pi / 3 * v), r * np.sin(np.pi / 3 * v)] for v in range(6)])
        for v in range(6):
            p0, p1 = vertices[v], vertices[(v + 1) % 6]
            for i in range(k):
                t = i / k
                points.append(((1 - t) * p0 + t * p1).tolist())
                ring_ids.append(k)

    points.append([0.0, 0.0])
    ring_ids.append(0)

    return np.array(points, dtype=np.float32), np.array(ring_ids, dtype=np.int32)


def generate_multilayer_lattice(
    n_rings: int,
    n_layers: int,
    ring_radius: float = 60.0,
    layer_z_gap: float = 1.0,
):
    """
    Stack n_layers copies of a hex lattice along the z-axis.

    Returns
    -------
    points_3d : ndarray, shape (N*n_layers, 3)
        (x, y, z) coordinates; z = layer_index * layer_z_gap.
    ring_ids  : ndarray, shape (N*n_layers,)
    layer_ids : ndarray, shape (N*n_layers,)
    edge_index : ndarray, shape (2, E)  — bidirectional, includes intra + inter edges.
    nodes_per_layer : int  (N)
    """
    base_pts, base_ring_ids = generate_hex_lattice(n_rings, ring_radius)
    N = len(base_pts)

    # ------------------------------------------------------------------
    # Build 3-D point array
    # ------------------------------------------------------------------
    all_pts_3d = []
    all_ring_ids = []
    all_layer_ids = []

    for l in range(n_layers):
        z = l * layer_z_gap
        layer_pts = np.column_stack([base_pts, np.full(N, z, dtype=np.float32)])
        all_pts_3d.append(layer_pts)
        all_ring_ids.append(base_ring_ids)
        all_layer_ids.append(np.full(N, l, dtype=np.int32))

    points_3d = np.vstack(all_pts_3d)
    ring_ids = np.concatenate(all_ring_ids)
    layer_ids = np.concatenate(all_layer_ids)

    # ------------------------------------------------------------------
    # Intra-layer edges (Delaunay on 2-D base points, replicated per layer)
    # ------------------------------------------------------------------
    tri = Delaunay(base_pts)
    intra_edges: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        a, b, c = simplex
        for u, v in [(a, b), (b, c), (a, c)]:
            if u > v:
                u, v = v, u
            intra_edges.add((u, v))

    src, dst = [], []
    for l in range(n_layers):
        offset = l * N
        for u, v in intra_edges:
            src += [u + offset, v + offset]
            dst += [v + offset, u + offset]

    # ------------------------------------------------------------------
    # Inter-layer edges (node i in layer l  <-->  node i in layer l+1)
    # ------------------------------------------------------------------
    for l in range(n_layers - 1):
        offset_l = l * N
        offset_l1 = (l + 1) * N
        for i in range(N):
            src += [i + offset_l, i + offset_l1]
            dst += [i + offset_l1, i + offset_l]

    edge_index = np.array([src, dst], dtype=np.int64)
    return points_3d, ring_ids, layer_ids, edge_index, N, tri


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

_RING_COLORS = [
    (200, 80, 80),
    (60, 140, 220),
    (60, 200, 120),
    (220, 160, 60),
    (160, 60, 220),
    (60, 200, 210),
    (220, 80, 160),
    (100, 180, 80),
]

_LAYER_COLORS = [
    (30, 30, 200),
    (200, 30, 30),
    (30, 160, 30),
    (180, 90, 10),
    (130, 10, 180),
    (10, 160, 160),
]

_INTER_LAYER_COLOR = (160, 160, 160)


def _ring_color(k: int):
    return _RING_COLORS[min(k, len(_RING_COLORS) - 1)]


def _layer_color(l: int):
    return _LAYER_COLORS[l % len(_LAYER_COLORS)]


def visualise_multilayer(
    points_3d: np.ndarray,
    ring_ids: np.ndarray,
    layer_ids: np.ndarray,
    edge_index: np.ndarray,
    n_rings: int,
    n_layers: int,
    nodes_per_layer: int,
    tri: Delaunay,
    img_size: int = 1000,
    margin: int = 60,
):
    """
    Draw all layers side-by-side.
    Intra-layer edges are shown in grey; inter-layer edges as dashed lines
    connecting corresponding nodes across panels.
    """
    N = nodes_per_layer
    panel_w = (img_size - 2 * margin) // n_layers
    panel_h = img_size - 2 * margin

    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

    base_pts_2d = points_3d[:N, :2]
    max_extent = np.abs(base_pts_2d).max() + 1e-6
    scale = min(panel_w, panel_h) / 2 / max_extent * 0.88

    def layer_origin(l: int):
        cx = margin + panel_w * l + panel_w // 2
        cy = img_size // 2
        return np.array([cx, cy], dtype=np.float32)

    def node_img(idx: int):
        l = layer_ids[idx]
        local_idx = idx - l * N
        p2d = points_3d[local_idx, :2]
        origin = layer_origin(l)
        px = origin[0] + p2d[0] * scale
        py = origin[1] - p2d[1] * scale  # y-flip for image coords
        return (int(round(px)), int(round(py)))

    # ------------------------------------------------------------------
    # Draw intra-layer Delaunay triangulation (light grey)
    # ------------------------------------------------------------------
    for l in range(n_layers):
        offset = l * N
        for simplex in tri.simplices:
            for i in range(3):
                a = node_img(simplex[i] + offset)
                b = node_img(simplex[(i + 1) % 3] + offset)
                cv2.line(img, a, b, (200, 200, 200), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Draw inter-layer edges (dashed, connecting corresponding nodes)
    # ------------------------------------------------------------------
    seen_inter: set[tuple[int, int]] = set()
    E = edge_index.shape[1]
    for e in range(E):
        u, v = int(edge_index[0, e]), int(edge_index[1, e])
        if layer_ids[u] != layer_ids[v]:
            key = (min(u, v), max(u, v))
            if key in seen_inter:
                continue
            seen_inter.add(key)
            pa, pb = node_img(u), node_img(v)
            _draw_dashed_line(img, pa, pb, _INTER_LAYER_COLOR, thickness=1, dash_len=6)

    # ------------------------------------------------------------------
    # Draw nodes
    # ------------------------------------------------------------------
    for idx in range(len(points_3d)):
        p = node_img(idx)
        l = int(layer_ids[idx])
        k = int(ring_ids[idx])
        color = _layer_color(l)
        r = 9 if k == 0 else 6
        cv2.circle(img, p, r + 2, (255, 255, 255), -1)
        cv2.circle(img, p, r, color, -1)
        cv2.putText(
            img,
            str(idx % N),
            (p[0] + 7, p[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )

    # ------------------------------------------------------------------
    # Layer labels
    # ------------------------------------------------------------------
    for l in range(n_layers):
        origin = layer_origin(l)
        lx, ly = int(origin[0]), margin // 2
        cv2.putText(
            img,
            f"Layer {l}",
            (lx - 25, ly),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _layer_color(l),
            2,
            cv2.LINE_AA,
        )

    # ------------------------------------------------------------------
    # Title / stats
    # ------------------------------------------------------------------
    total_nodes = len(points_3d)
    total_undirected = edge_index.shape[1] // 2
    intra_undirected = len(
        set(
            (min(int(edge_index[0, e]), int(edge_index[1, e])), max(int(edge_index[0, e]), int(edge_index[1, e])))
            for e in range(edge_index.shape[1])
            if layer_ids[int(edge_index[0, e])] == layer_ids[int(edge_index[1, e])]
        )
    )
    inter_undirected = total_undirected - intra_undirected

    cv2.putText(
        img, f"Multi-layer Hex  n_rings={n_rings}  n_layers={n_layers}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2, cv2.LINE_AA
    )
    cv2.putText(
        img,
        f"nodes={total_nodes} ({N}/layer)  undirected: intra={intra_undirected}  inter={inter_undirected}  total={total_undirected}",
        (12, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (100, 100, 100),
        1,
        cv2.LINE_AA,
    )

    cv2.imshow("Multi-layer Hex Lattice", img)
    cv2.waitKey(0)
    return img


def _draw_dashed_line(img, p1, p2, color, thickness=1, dash_len=8):
    """Draw a dashed line between p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dist = np.hypot(x2 - x1, y2 - y1)
    if dist < 1:
        return
    dx, dy = (x2 - x1) / dist, (y2 - y1) / dist
    drawn = 0.0
    on = True
    while drawn < dist:
        seg = min(dash_len, dist - drawn)
        if on:
            xa = int(round(x1 + dx * drawn))
            ya = int(round(y1 + dy * drawn))
            xb = int(round(x1 + dx * (drawn + seg)))
            yb = int(round(y1 + dy * (drawn + seg)))
            cv2.line(img, (xa, ya), (xb, yb), color, thickness, cv2.LINE_AA)
        drawn += seg
        on = not on


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    N_RINGS = 7  # rings per layer (includes centre)
    N_LAYERS = 2  # number of stacked layers
    RING_RADIUS = 70.0
    LAYER_Z_GAP = 1.0  # physical z-spacing (saved in the 3-D point array)

    out_dir = "./outputs/markers"
    edge_dir = os.path.join(out_dir, "edge_index")
    kp_dir = os.path.join(out_dir, "keypoints")
    os.makedirs(edge_dir, exist_ok=True)
    os.makedirs(kp_dir, exist_ok=True)

    points_3d, ring_ids, layer_ids, edge_index, N, tri = generate_multilayer_lattice(N_RINGS, N_LAYERS, RING_RADIUS, LAYER_Z_GAP)

    print(f"Nodes per layer : {N}")
    print(f"Total nodes     : {len(points_3d)}")
    print(f"edge_index shape: {edge_index.shape}  ({edge_index.shape[1]//2} undirected edges)")

    tag = f"hex_n{N_RINGS}_l{N_LAYERS}"
    np.save(os.path.join(edge_dir, f"edge_index_{tag}.npy"), edge_index)
    # np.save(os.path.join(kp_dir,   f"keypoints_{tag}.npy"),   points_3d)
    # np.save(os.path.join(kp_dir,   f"layer_ids_{tag}.npy"),   layer_ids)
    # np.save(os.path.join(kp_dir,   f"ring_ids_{tag}.npy"),    ring_ids)
    print(f"Saved to {out_dir}")

    visualise_multilayer(
        points_3d,
        ring_ids,
        layer_ids,
        edge_index,
        N_RINGS,
        N_LAYERS,
        N,
        tri,
    )
