#!/usr/bin/env python3
"""Fetch the optional models the repo does not ship.

assets/ carries enough geometry to build, run and reproduce most of the README.
These are the extras, and none of them is tracked here, in each case because the
licence is somebody else's to grant rather than an open one:

  african_head, diablo3_pose   the two models the original tinyrenderer tutorial
                               uses, and what most of the numbers in the README
                               were measured against. Their permission is a
                               personal grant to Dmitry Sokolov for that project
                               ("feel free to use the model as an example in your
                               renderer"), so they are fetched from the original
                               repository, each with the readme.txt carrying that
                               permission and the author's name.

  sponza                       the Crytek Sponza atrium, the standard test scene
                               for this kind of renderer. Khronos ships it under
                               the CRYENGINE Limited License Agreement, so it is
                               fetched rather than redistributed. 50 MB of glTF,
                               converted on arrival into the 25 .obj files this
                               renderer reads -- a couple of minutes in total.

    python tools/fetch_models.py                 # everything
    python tools/fetch_models.py african_head
    python tools/fetch_models.py sponza
"""
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import urllib.request

BASE = 'https://raw.githubusercontent.com/ssloy/tinyrenderer/master/obj'
DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'assets', 'external')

# name -> (source directory in ssloy/tinyrenderer, files)
MODELS = {
    'african_head': ('african_head', [
        'african_head.obj', 'african_head_diffuse.tga',
        'african_head_nm.tga', 'african_head_spec.tga', 'readme.txt']),
    'diablo3_pose': ('diablo3_pose', [
        'diablo3_pose.obj', 'diablo3_pose_diffuse.tga',
        'diablo3_pose_nm.tga', 'diablo3_pose_spec.tga', 'readme.txt']),
}


GLTF_MODELS = {
    # name -> (source directory under glTF-Sample-Assets/Models, texture size)
    'sponza': ('Sponza', 512),
}
GLTF_BASE = ('https://raw.githubusercontent.com/KhronosGroup/'
             'glTF-Sample-Assets/main/Models')
TOOLS = os.path.dirname(os.path.abspath(__file__))


def fetch_gltf(name):
    """Download a glTF model and convert it into the .obj/.tga pair the
    renderer reads. The conversion is the same one that produced assets/, so
    see assets/README.md for what it does and why."""
    src, tex = GLTF_MODELS[name]
    out = os.path.join(DEST, name)
    os.makedirs(out, exist_ok=True)
    raw = os.path.join(out, '_gltf')
    os.makedirs(raw, exist_ok=True)

    def one(uri):
        dst = os.path.join(raw, uri)
        if os.path.exists(dst) and os.path.getsize(dst):
            return os.path.getsize(dst)
        urllib.request.urlretrieve(f'{GLTF_BASE}/{src}/glTF/{uri}', dst)
        return os.path.getsize(dst)

    total = one(f'{src}.gltf')
    g = json.load(open(os.path.join(raw, f'{src}.gltf')))
    files = [b['uri'] for b in g['buffers'] if 'uri' in b]
    files += [i['uri'] for i in g.get('images', []) if 'uri' in i]
    print(f'  downloading {len(files)} more files...')
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        total += sum(ex.map(one, files))
    print(f'  {total // 1048576} MB downloaded')

    gltf = os.path.join(raw, f'{src}.gltf')
    for script in ('gltf2obj.py', 'bake_nm.py'):
        print(f'  {script} (this takes a minute)...')
        r = subprocess.run([sys.executable, os.path.join(TOOLS, script),
                            gltf, out, str(tex)], capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout + r.stderr)
            return 0
    n = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)
            if f.endswith(('.obj', '.tga')))
    print(f'  {len([f for f in os.listdir(out) if f.endswith(".obj")])} meshes, '
          f'{n // 1048576} MB converted into {out}')
    return n


def fetch(name):
    src, files = MODELS[name]
    out = os.path.join(DEST, name)
    os.makedirs(out, exist_ok=True)
    total = 0
    for f in files:
        dst = os.path.join(out, f)
        if os.path.exists(dst):
            print(f'  {f:32} already there')
            total += os.path.getsize(dst)
            continue
        url = f'{BASE}/{src}/{f}'
        try:
            with urllib.request.urlopen(url) as r, open(dst, 'wb') as w:
                w.write(r.read())
        except Exception as e:
            print(f'  {f:32} FAILED: {e}')
            continue
        n = os.path.getsize(dst)
        total += n
        print(f'  {f:32} {n // 1024:>6} KB')
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    everything = list(MODELS) + list(GLTF_MODELS)
    ap.add_argument('models', nargs='*', choices=everything + [[]],
                    default=everything)
    args = ap.parse_args()
    names = args.models or everything

    total = 0
    for n in names:
        print(f'{n}:')
        total += fetch_gltf(n) if n in GLTF_MODELS else fetch(n)
    print(f'\n{total // 1024 // 1024} MB into {DEST}')
    print('Attribution is in each model\'s readme.txt. Render one with:')
    print('  build/windows/bin/capture_scene out.tga '
          'assets/external/african_head/african_head.obj')


if __name__ == '__main__':
    sys.exit(main())
