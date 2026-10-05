import numpy as np

EPS = 1e-9
CORNERS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)


def point_in_polygon(p, poly):
    # convex polygon, either winding; points on the edge count as inside
    e = np.roll(poly, -1, axis=0) - poly
    cross = e[:, 0] * (p[1] - poly[:, 1]) - e[:, 1] * (p[0] - poly[:, 0])
    return bool(np.all(cross >= -1e-12) or np.all(cross <= 1e-12))


def boxes_vs_plate(h, c, A, poly, thickness, cp, Ap):
    """
    Overlap of N boxes with one flat convex polygon plate, by the separating axis test.
    h, c: (N, 3) half sizes and centers; A: (N, 3, 3) DCMs (columns are the box axes).
    poly: (n, 2) plate vertices in its local x-y; cp, Ap: plate center and DCM.
    Touching counts as not overlapping, like separating_axis_test.
    """
    h, c, A = np.atleast_2d(h), np.atleast_2d(c), np.asarray(A, float).reshape(-1, 3, 3)
    N, n = len(h), len(poly)
    box_v = c[:, None, :] + np.einsum('nij,nkj->nki', A, CORNERS[None] * h[:, None, :])    # (N, 8, 3)
    local = np.concatenate([np.c_[poly, np.full(n, -thickness / 2)], np.c_[poly, np.full(n, thickness / 2)]])
    plate_v = np.asarray(cp, float) + local @ np.asarray(Ap, float).T                       # (2n, 3)

    edges2 = np.roll(poly, -1, axis=0) - poly
    plate_edges = np.c_[edges2, np.zeros(n)] @ np.asarray(Ap, float).T                       # (n, 3)
    plate_normals = np.c_[edges2[:, 1], -edges2[:, 0], np.zeros(n)] @ np.asarray(Ap, float).T
    plate_dirs = np.vstack([plate_edges, np.asarray(Ap, float)[:, 2]])                      # (n+1, 3)

    box_axes = np.transpose(A, (0, 2, 1))                                                   # (N, 3, 3) rows
    cross = np.cross(box_axes[:, :, None, :], plate_dirs[None, None, :, :]).reshape(N, -1, 3)
    fixed = np.vstack([np.asarray(Ap, float)[:, 2], plate_normals])
    axes = np.concatenate([box_axes, np.broadcast_to(fixed, (N,) + fixed.shape), cross], axis=1)
    norm = np.linalg.norm(axes, axis=2, keepdims=True)
    valid = norm[..., 0] > 1e-9
    axes = axes / np.where(norm > 1e-9, norm, 1.0)

    pb = np.einsum('nvi,nai->nva', box_v, axes)                                             # (N, 8, K)
    pp = np.einsum('vi,nai->nva', plate_v, axes)                                            # (N, 2n, K)
    gap = np.minimum(pb.max(1), pp.max(1)) - np.maximum(pb.min(1), pp.min(1))               # (N, K)
    separated = (gap <= EPS) & valid
    return ~separated.any(axis=1)


def polygon_second_moments(poly):
    # area integrals of y^2, x^2 and xy over the polygon, about the origin
    x, y = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    a = x * y1 - x1 * y
    sign = np.sign(a.sum())
    iyy_int = sign * np.sum(a * (x * x + x * x1 + x1 * x1)) / 12   # integral of x^2
    ixx_int = sign * np.sum(a * (y * y + y * y1 + y1 * y1)) / 12   # integral of y^2
    ixy_int = sign * np.sum(a * (x * y1 + 2 * x * y + 2 * x1 * y1 + x1 * y)) / 24
    return ixx_int, iyy_int, ixy_int


def plate_inertia(poly, thickness, mass):
    # inertia of a thin polygon plate about its own origin, in its local frame
    area = 0.5 * abs(np.dot(poly[:, 0], np.roll(poly[:, 1], -1)) - np.dot(poly[:, 1], np.roll(poly[:, 0], -1)))
    sigma = mass / area
    ixx_int, iyy_int, ixy_int = polygon_second_moments(poly)
    t2 = mass * thickness ** 2 / 12
    return np.array([[sigma * ixx_int + t2, -sigma * ixy_int, 0],
                     [-sigma * ixy_int, sigma * iyy_int + t2, 0],
                     [0, 0, sigma * (ixx_int + iyy_int)]])


def box_inertia(dims, mass):
    return mass / 12 * np.diag([dims[1] ** 2 + dims[2] ** 2, dims[0] ** 2 + dims[2] ** 2, dims[0] ** 2 + dims[1] ** 2])
