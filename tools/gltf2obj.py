#!/usr/bin/env python3
"""glTF (separate .bin + PNG) -> the OBJ/TGA pair this renderer loads.

One .obj per primitive, because the renderer binds exactly one diffuse /
normal / specular map per Model. Every mesh is normalised by a single shared
transform so the parts stay assembled.

Faces are written in the full v/vt/vn form: the OBJ parser in src/model.cpp
requires all three indices.
"""
import json
import os
import struct
import sys

import numpy as np
from PIL import Image

COMPONENT = {5120: 'b', 5121: 'B', 5122: 'h', 5123: 'H', 5125: 'I', 5126: 'f'}
NCOMP = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}


def safe_name(s):
    """A material name that is safe as a filename and as a scene node name."""
    out = ''.join(c if (c.isalnum() or c in '-_') else '_' for c in s)
    return out.strip('_') or 'material'


def node_matrix(n):
    """A glTF node's local transform, as a 4x4. Either an explicit matrix or
    T * R * S built from the components, in that order per the spec."""
    if 'matrix' in n:
        return np.array(n['matrix'], dtype=np.float64).reshape(4, 4).T
    m = np.eye(4)
    if 'rotation' in n:
        x, y, z, w = n['rotation']
        m[:3, :3] = [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    if 'scale' in n:
        m[:3, :3] = m[:3, :3] @ np.diag(n['scale'])
    if 'translation' in n:
        m[:3, 3] = n['translation']
    return m


def world_transforms(g):
    """mesh index -> world matrix, walking the scene graph from its roots.

    Models routinely carry their placement on the node rather than baking it
    into the vertices -- AntiqueCamera rotates and scales both of its parts --
    and ignoring that drops parts on top of each other at the wrong size."""
    out = {}
    roots = g['scenes'][g.get('scene', 0)]['nodes']
    stack = [(i, np.eye(4)) for i in roots]
    while stack:
        i, parent = stack.pop()
        n = g['nodes'][i]
        world = parent @ node_matrix(n)
        if 'mesh' in n:
            out[n['mesh']] = world
        for c in n.get('children', []):
            stack.append((c, world))
    return out


def load_accessor(g, buf, idx):
    acc = g['accessors'][idx]
    bv = g['bufferViews'][acc['bufferView']]
    fmt = COMPONENT[acc['componentType']]
    n = NCOMP[acc['type']]
    itemsize = np.dtype(fmt).itemsize * n
    stride = bv.get('byteStride') or itemsize
    start = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
    out = np.empty((acc['count'], n), dtype=np.dtype(fmt))
    for i in range(acc['count']):
        off = start + i * stride
        out[i] = np.frombuffer(buf, dtype=np.dtype(fmt), count=n, offset=off)
    return out


def write_tga(path, arr, rle=True):
    """arr: HxWx3 uint8 RGB, or HxW uint8 grayscale. Written bottom-up BGR."""
    if arr.ndim == 2:
        bpp, data = 1, arr[::-1]
    else:
        bpp, data = 3, arr[::-1, :, ::-1]          # flip rows, RGB -> BGR
    h, w = data.shape[:2]
    flat = np.ascontiguousarray(data).reshape(-1, bpp)
    header = struct.pack('<BBBHHBHHHHBB', 0, 0,
                         (11 if bpp == 1 else 10) if rle else (3 if bpp == 1 else 2),
                         0, 0, 0, 0, 0, w, h, bpp * 8, 0)
    body = bytearray()
    if not rle:
        body += flat.tobytes()
    else:
        i, n = 0, len(flat)
        while i < n:
            run = 1
            while (run < 128 and i + run < n
                   and (flat[i + run] == flat[i]).all()):
                run += 1
            if run > 1:
                body.append(0x80 | (run - 1))
                body += flat[i].tobytes()
                i += run
            else:
                j = i + 1
                while (j < n and j - i < 128
                       and not (flat[j] == flat[j - 1]).all()):
                    j += 1
                if j < n and j - i > 1:
                    j -= 1                          # leave the pair for a run
                body.append(j - i - 1)
                body += flat[i:j].tobytes()
                i = j
    with open(path, 'wb') as f:
        f.write(header)
        f.write(bytes(body))
        f.write(b'\0' * 8 + b'TRUEVISION-XFILE.\0')
    return os.path.getsize(path)


def resize(im, size):
    return im.resize((size, size), Image.LANCZOS) if im.size != (size, size) else im


def main(gltf_path, outdir, tex_size=512, target_radius=1.0):
    g = json.load(open(gltf_path))
    root = os.path.dirname(gltf_path)
    buf = open(os.path.join(root, g['buffers'][0]['uri']), 'rb').read()
    os.makedirs(outdir, exist_ok=True)

    world = world_transforms(g)
    prims = []
    for mi, mesh in enumerate(g['meshes']):
        M = world.get(mi, np.eye(4))
        # normals transform by the inverse-transpose, or a non-uniform scale
        # tilts them off the surface
        NM = np.linalg.inv(M[:3, :3]).T
        for p in mesh['primitives']:
            pos = load_accessor(g, buf, p['attributes']['POSITION']).astype(np.float64)
            nrm = load_accessor(g, buf, p['attributes']['NORMAL']).astype(np.float64)
            uv = load_accessor(g, buf, p['attributes']['TEXCOORD_0']).astype(np.float64)
            idx = load_accessor(g, buf, p['indices']).reshape(-1, 3).astype(np.int64)
            pos = pos @ M[:3, :3].T + M[:3, 3]
            nrm = nrm @ NM.T
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
            prims.append((p['material'],
                          mesh.get('name', '').replace('_low', '').replace('_uv', ''),
                          pos, nrm, uv, idx))

    allpos = np.concatenate([p[2] for p in prims])
    ctr = (allpos.min(0) + allpos.max(0)) / 2
    scale = target_radius / (allpos.max(0) - allpos.min(0)).max() * 2

    # One output mesh per material, not per primitive. The renderer binds a
    # single diffuse/normal/specular set per Model, so the material is what
    # decides the split -- and a scene like Sponza carries 103 primitives
    # across 25 materials under one mesh name, which would otherwise write 103
    # files all called the same thing.
    groups = {}
    for mat_id, mesh_name, pos, nrm, uv, idx in prims:
        groups.setdefault(mat_id, []).append((mesh_name, pos, nrm, uv, idx))

    merged = []
    for mat_id, parts in groups.items():
        mat = g['materials'][mat_id]
        name = safe_name(mat.get('name') or parts[0][0] or f'material{mat_id}')
        pos = np.concatenate([p[1] for p in parts])
        nrm = np.concatenate([p[2] for p in parts])
        uv = np.concatenate([p[3] for p in parts])
        idx, base = [], 0
        for _, p_pos, _, _, p_idx in parts:
            idx.append(p_idx + base)
            base += len(p_pos)
        merged.append((name, pos, nrm, uv, np.concatenate(idx), mat))

    written = []
    for name, pos, nrm, uv, idx, mat in merged:
        base = os.path.join(outdir, name)
        v = (pos - ctr) * scale
        with open(base + '.obj', 'w') as f:
            f.write(f'# {name}: {len(v)} vertices, {len(idx)} triangles\n')
            for a in v:
                f.write('v %.5f %.5f %.5f\n' % tuple(a))
            for a in uv:
                f.write('vt %.5f %.5f\n' % (a[0], 1.0 - a[1]))   # glTF v is top-down
            for a in nrm:
                f.write('vn %.5f %.5f %.5f\n' % tuple(a))
            for t in idx + 1:
                f.write('f %d/%d/%d %d/%d/%d %d/%d/%d\n' %
                        (t[0], t[0], t[0], t[1], t[1], t[1], t[2], t[2], t[2]))

        pbr = mat.get('pbrMetallicRoughness', {})
        img = lambda i: Image.open(os.path.join(
            root, g['images'][g['textures'][i]['source']]['uri']))

        # A material may carry only a factor and no map -- ToyCar's glass and
        # fabric do. The renderer binds a texture per slot or nothing at all,
        # and "nothing" means it falls back to the flat per-mesh colour the
        # engine passes in, losing the material entirely. A 4x4 constant
        # texture is the cheapest way to keep the distinction.
        if 'baseColorTexture' in pbr:
            albedo = np.asarray(resize(
                img(pbr['baseColorTexture']['index']).convert('RGB'), tex_size))
        else:
            f = pbr.get('baseColorFactor', [1, 1, 1, 1])[:3]
            albedo = np.tile(np.clip(np.array(f) * 255, 0, 255).astype(np.uint8),
                             (4, 4, 1))
        d = write_tga(base + '_diffuse.tga', albedo)

        # OcclusionRoughMetal packs occlusion/roughness/metallic in R/G/B. The
        # shader reads this map as a Phong exponent, so gloss = 1 - roughness
        # maps onto a modest exponent range rather than being used directly.
        if 'metallicRoughnessTexture' in pbr:
            orm = np.asarray(resize(img(pbr['metallicRoughnessTexture']['index'])
                                    .convert('RGB'), tex_size)).astype(np.float32)
            gloss = 1.0 - orm[:, :, 1] / 255.0
        else:
            gloss = np.full((4, 4), 1.0 - pbr.get('roughnessFactor', 1.0), np.float32)
        s = write_tga(base + '_spec.tga',
                      np.clip(1.0 + gloss * 30.0, 1, 255).astype(np.uint8))
        written.append((name, len(v), len(idx), d, s))

    for name, nv, nf, d, s in written:
        print(f'  {name:16} {nv:>7} v {nf:>7} f   diffuse {d//1024:>5} KB  spec {s//1024:>4} KB')
    print(f'  total tracked: {sum(d + s for *_, d, s in written)//1024} KB of texture')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2],
         int(sys.argv[3]) if len(sys.argv) > 3 else 512)
