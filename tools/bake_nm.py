#!/usr/bin/env python3
"""Bake glTF tangent-space normal maps into the object-space form this renderer reads.

src/model.cpp samples _nm.tga as an object-space normal (x<-R, y<-G, z<-B) and
the shader carries it to eye space with the inverse-transpose. A tangent-space
map cannot be used that way, so each texel is resolved here: rasterise the mesh
in UV space, interpolate the vertex frame (N, T, and B = cross(N,T)*w), and
rotate the sampled tangent-space normal into object space.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gltf2obj import load_accessor, write_tga, resize, world_transforms, safe_name


def derive_tangents(pos, nrm, uv, idx):
    """Per-vertex tangents from UV derivatives, for models that ship none.

    glTF only requires TANGENT when the author supplied it, and ToyCar does not. Accumulate the per-triangle tangent and bitangent,
    then orthogonalise against the normal and recover the handedness from
    whether the accumulated bitangent agrees with cross(N, T). Returns the VEC4
    the rest of this file expects, w carrying that handedness."""
    tan = np.zeros((len(pos), 3))
    bit = np.zeros((len(pos), 3))
    for a, b, c in idx:
        e1, e2 = pos[b] - pos[a], pos[c] - pos[a]
        d1, d2 = uv[b] - uv[a], uv[c] - uv[a]
        det = d1[0] * d2[1] - d2[0] * d1[1]
        if abs(det) < 1e-20:
            continue
        r = 1.0 / det
        t = (e1 * d2[1] - e2 * d1[1]) * r
        bt = (e2 * d1[0] - e1 * d2[0]) * r
        for i in (a, b, c):
            tan[i] += t
            bit[i] += bt

    t = tan - nrm * (nrm * tan).sum(1, keepdims=True)      # Gram-Schmidt
    l = np.linalg.norm(t, axis=1, keepdims=True)
    # a vertex with degenerate UVs gets an arbitrary but valid frame
    fallback = np.tile([1.0, 0.0, 0.0], (len(pos), 1))
    t = np.where(l > 1e-12, t / np.maximum(l, 1e-12), fallback)
    w = np.where((np.cross(nrm, t) * bit).sum(1) < 0.0, -1.0, 1.0)
    return np.concatenate([t, w[:, None]], axis=1)


def bake(pos, nrm, tan, uv, idx, ts_map, size):
    """Returns (HxWx3 float32 object-space normals, HxW bool coverage)."""
    out = np.zeros((size, size, 3), np.float32)
    seen = np.zeros((size, size), bool)

    # UV origin is bottom-left here (gltf2obj already flipped v), and the
    # texture rows are written bottom-up by write_tga, so texel row r maps to
    # v = (r + 0.5) / size with no further flip.
    px = uv * size - 0.5

    for tri in idx:
        p = px[tri]
        x0 = max(int(np.floor(p[:, 0].min())), 0)
        x1 = min(int(np.ceil(p[:, 0].max())) + 1, size)
        y0 = max(int(np.floor(p[:, 1].min())), 0)
        y1 = min(int(np.ceil(p[:, 1].max())) + 1, size)
        if x0 >= x1 or y0 >= y1:
            continue

        ys, xs = np.mgrid[y0:y1, x0:x1]
        d = ((p[1, 1] - p[2, 1]) * (p[0, 0] - p[2, 0]) +
             (p[2, 0] - p[1, 0]) * (p[0, 1] - p[2, 1]))
        if abs(d) < 1e-12:
            continue
        w0 = ((p[1, 1] - p[2, 1]) * (xs - p[2, 0]) +
              (p[2, 0] - p[1, 0]) * (ys - p[2, 1])) / d
        w1 = ((p[2, 1] - p[0, 1]) * (xs - p[2, 0]) +
              (p[0, 0] - p[2, 0]) * (ys - p[2, 1])) / d
        w2 = 1.0 - w0 - w1
        # a small negative tolerance closes the cracks between adjacent charts
        m = (w0 >= -0.5) & (w1 >= -0.5) & (w2 >= -0.5)
        if not m.any():
            continue

        wa, wb, wc = w0[m, None], w1[m, None], w2[m, None]
        n = wa * nrm[tri[0]] + wb * nrm[tri[1]] + wc * nrm[tri[2]]
        t = wa * tan[tri[0], :3] + wb * tan[tri[1], :3] + wc * tan[tri[2], :3]
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        t -= n * (n * t).sum(1, keepdims=True)          # Gram-Schmidt
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
        b = np.cross(n, t) * tan[tri[0], 3]

        ts = ts_map[ys[m], xs[m]]
        obj = t * ts[:, 0:1] + b * ts[:, 1:2] + n * ts[:, 2:3]
        out[ys[m], xs[m]] = obj
        seen[ys[m], xs[m]] = True
    return out, seen


def dilate(img, seen, rounds=4):
    """Bleed the baked values outward so bilinear taps across a UV seam do not
    read the empty gutter."""
    for _ in range(rounds):
        holes = ~seen
        if not holes.any():
            break
        acc = np.zeros_like(img)
        cnt = np.zeros(seen.shape, np.float32)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            s = np.roll(np.roll(seen, dy, 0), dx, 1)
            v = np.roll(np.roll(img, dy, 0), dx, 1)
            acc += v * s[:, :, None]
            cnt += s
        fill = holes & (cnt > 0)
        img[fill] = acc[fill] / cnt[fill, None]
        seen = seen | fill
    return img


def main(gltf_path, outdir, size=512):
    g = json.load(open(gltf_path))
    root = os.path.dirname(gltf_path)
    buf = open(os.path.join(root, g['buffers'][0]['uri']), 'rb').read()

    # Grouped by material, matching gltf2obj: one normal map per material, and
    # every primitive that uses it contributes to the same UV chart.
    world = world_transforms(g)
    groups = {}
    for mi, mesh in enumerate(g['meshes']):
        M = world.get(mi, np.eye(4))
        NM = np.linalg.inv(M[:3, :3]).T
        for prim in mesh['primitives']:
            groups.setdefault(prim['material'], []).append(
                (mesh.get('name', '').replace('_low', '').replace('_uv', ''), prim, M, NM))

    for mat_id, parts in groups.items():
        mat = g['materials'][mat_id]
        if 'normalTexture' not in mat:
            continue
        name = safe_name(mat.get('name') or parts[0][0] or f'material{mat_id}')
        uri = g['images'][g['textures'][mat['normalTexture']['index']]['source']]['uri']
        ts = np.asarray(resize(Image.open(os.path.join(root, uri)).convert('RGB'),
                               size)).astype(np.float32) / 255.0 * 2.0 - 1.0
        ts = ts[::-1]                              # texel row 0 is v=0

        acc_pos, acc_nrm, acc_uv, acc_tan, acc_idx, base = [], [], [], [], [], 0
        for _, prim, M, NM in parts:
            a = prim['attributes']
            pos = load_accessor(g, buf, a['POSITION']).astype(np.float64)
            nrm = load_accessor(g, buf, a['NORMAL']).astype(np.float64)
            uv = load_accessor(g, buf, a['TEXCOORD_0']).astype(np.float64)
            uv[:, 1] = 1.0 - uv[:, 1]                  # match gltf2obj
            idx = load_accessor(g, buf, prim['indices']).reshape(-1, 3).astype(np.int64)

            # the bake writes object-space normals, so the frame has to be
            # built in the same space gltf2obj wrote the vertices in
            pos = pos @ M[:3, :3].T + M[:3, 3]
            nrm = nrm @ NM.T
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)

            if 'TANGENT' in a:
                tan = load_accessor(g, buf, a['TANGENT']).astype(np.float64)
                tan[:, :3] = tan[:, :3] @ M[:3, :3].T
            else:
                tan = derive_tangents(pos, nrm, uv, idx)

            acc_pos.append(pos); acc_nrm.append(nrm); acc_uv.append(uv)
            acc_tan.append(tan); acc_idx.append(idx + base)
            base += len(pos)

        pos = np.concatenate(acc_pos); nrm = np.concatenate(acc_nrm)
        uv = np.concatenate(acc_uv); tan = np.concatenate(acc_tan)
        idx = np.concatenate(acc_idx)

        obj, seen = bake(pos, nrm, tan, uv, idx, ts, size)
        obj = dilate(obj, seen)
        l = np.maximum(np.linalg.norm(obj, axis=2, keepdims=True), 1e-12)
        rgb = np.clip((obj / l * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)

        # bake() indexes rows by v, so row 0 is v=0; write_tga takes a
        # top-down array like PIL hands out.
        path = os.path.join(outdir, name + '_nm.tga')
        n = write_tga(path, rgb[::-1])
        print(f'  {name:24} {seen.mean()*100:5.1f}% covered  {n//1024:>5} KB')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 512)
