#!/usr/bin/env python3
"""ASCII PLY -> OBJ, with smooth vertex normals and placeholder UVs.

This renderer's OBJ parser requires the full v/vt/vn face form, so every
corner gets a uv index even when the mesh has no texture coordinates.
"""
import math
import sys


def read_ply(path):
    with open(path, 'rb') as f:
        raw = f.read()
    end = raw.index(b'end_header\n') + len(b'end_header\n')
    header = raw[:end].decode('ascii').splitlines()
    body = raw[end:].decode('ascii').split('\n')

    nvert = nface = 0
    vprops = []
    element = None
    for line in header:
        p = line.split()
        if not p:
            continue
        if p[0] == 'element':
            element = p[1]
            if element == 'vertex':
                nvert = int(p[2])
            elif element == 'face':
                nface = int(p[2])
        elif p[0] == 'property' and element == 'vertex':
            vprops.append(p[-1])

    xi, yi, zi = vprops.index('x'), vprops.index('y'), vprops.index('z')
    verts, faces = [], []
    for line in body[:nvert]:
        p = line.split()
        verts.append((float(p[xi]), float(p[yi]), float(p[zi])))
    for line in body[nvert:nvert + nface]:
        p = [int(v) for v in line.split()]
        for k in range(2, p[0]):                  # fan-triangulate
            faces.append((p[1], p[k], p[k + 1]))
    return verts, faces


def smooth_normals(verts, faces):
    acc = [[0.0, 0.0, 0.0] for _ in verts]
    for a, b, c in faces:
        va, vb, vc = verts[a], verts[b], verts[c]
        e1 = (vb[0] - va[0], vb[1] - va[1], vb[2] - va[2])
        e2 = (vc[0] - va[0], vc[1] - va[1], vc[2] - va[2])
        n = (e1[1] * e2[2] - e1[2] * e2[1],
             e1[2] * e2[0] - e1[0] * e2[2],
             e1[0] * e2[1] - e1[1] * e2[0])
        for i in (a, b, c):                       # area-weighted by not normalising
            acc[i][0] += n[0]
            acc[i][1] += n[1]
            acc[i][2] += n[2]
    out = []
    for n in acc:
        l = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
        out.append((n[0] / l, n[1] / l, n[2] / l) if l > 1e-20 else (0.0, 1.0, 0.0))
    return out


def main(src, dst, scale_to=1.0, recentre=True):
    verts, faces = read_ply(src)
    norms = smooth_normals(verts, faces)

    lo = [min(v[i] for v in verts) for i in range(3)]
    hi = [max(v[i] for v in verts) for i in range(3)]
    ctr = [(lo[i] + hi[i]) / 2 for i in range(3)]
    radius = max(hi[i] - lo[i] for i in range(3)) / 2
    s = scale_to / radius if radius > 0 else 1.0

    with open(dst, 'w') as f:
        f.write(f'# converted from {src.split("/")[-1]}\n')
        f.write(f'# {len(verts)} vertices, {len(faces)} triangles\n')
        for v in verts:
            if recentre:
                f.write(f'v {(v[0]-ctr[0])*s:.6f} {(v[1]-ctr[1])*s:.6f} {(v[2]-ctr[2])*s:.6f}\n')
            else:
                f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        f.write('vt 0.000000 0.000000\n')          # placeholder, see docstring
        for n in norms:
            f.write(f'vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n')
        for a, b, c in faces:
            f.write(f'f {a+1}/1/{a+1} {b+1}/1/{b+1} {c+1}/1/{c+1}\n')
    print(f'{dst}: {len(verts)} verts, {len(faces)} tris')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2],
         float(sys.argv[3]) if len(sys.argv) > 3 else 1.0)
