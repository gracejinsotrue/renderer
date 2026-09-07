#!/usr/bin/env python3
"""Regenerate every image in the README.

Each figure is one `capture_scene` run with its settings spelled out, so the
pictures in the README are reproducible rather than hand-collected screenshots.
capture_scene is an optional tool target:

    cmake --build build/windows --target capture_scene
    python tools/make_figures.py [--build-dir build/windows]

Needs the assets in assets/ and a working CUDA device. Writes PNGs to
docs/images/.
"""
import argparse
import glob
import os
import struct
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'docs', 'images')

TOYCAR = ['assets/toycar/ToyCar.obj', 'assets/toycar/Fabric.obj',
          'assets/toycar/Glass.obj']
CAMERA = ['assets/camera/camera.obj', 'assets/camera/tripod.obj']
# Not tracked: fetched and converted by tools/fetch_models.py sponza. The figure
# using it is skipped when it is absent rather than failing the run.
SPONZA = sorted(glob.glob(os.path.join(REPO, 'assets/external/sponza/*.obj')))
SPONZA = [os.path.relpath(p, REPO).replace(os.sep, '/') for p in SPONZA]
BUNNY = ['assets/bunny/bunny.obj']
GROUND = ['assets/scene/ground.obj']

# The engine renders into an 800x800 buffer but Engine::renderScene sets the
# viewport to the central three quarters, so NDC +/-1 lands on [100, 700] and
# everything outside that rect is beyond the projection. It is also exactly
# where the missing clip shows: setup.cu culls whole triangles against the
# frustum planes without cutting them, so silhouettes that cross a plane come
# out as a sawtooth in that border. Geometry straddling the boundary still
# rasterises correctly inside it, so the viewport rect is both the honest frame
# and a clean one.
VIEWPORT = (100, 100, 700, 700)

# name -> (models, environment, (crop box in full-frame pixels, nearest zoom))
FIGURES = {
    'hero': (TOYCAR + GROUND,
             dict(SCALE='0.95', ZOOM='-0.7', ORBIT='-0.9,0.25',
                  LIGHT='-0.7,1.0,0.8', INTENSITY='1.45'), None),
    'hero-camera': (CAMERA + GROUND,
                    dict(SCALE='0.72', ZOOM='-0.5', ORBIT='0.5,0.1',
                         LIGHT='0.6,0.9,1.0', INTENSITY='1.7'), None),
    # Three models on one ground plane, each placed and scaled through its own
    # scene node. Cropped to a band because the projection puts the horizon low
    # and there is nothing above it: the ground plane has to stay small enough
    # for the shadow map to cover, so it cannot reach the top of the frame.
    'scene': (BUNNY + CAMERA + TOYCAR + GROUND,
              dict(ZOOM='-0.85', ORBIT='0.22,0.06',
                   LIGHT='0.45,1.0,0.85', INTENSITY='1.5',
                   PLACE='bunny:-0.62,-0.62,0.25,0.38;'
                         'camera:0.02,-0.62,-0.2,0.38;'
                         'tripod:0.02,-0.62,-0.2,0.38;'
                         'ToyCar:0.62,-0.62,0.25,0.38;'
                         'Fabric:0.62,-0.62,0.25,0.38;'
                         'Glass:0.62,-0.62,0.25,0.38'),
              ((100, 420, 700, 700), 1)),
    'bunny': (BUNNY + GROUND,
              dict(SCALE='0.8', ZOOM='1.0', ORBIT='0.35,0.3',
                   LIGHT='0.6,1.0,0.7', INTENSITY='1.2'), None),
    # Cropped onto the junction where the ears meet the skull, the deepest
    # concavity in the scene and where this pass does most of its work. Over
    # the whole frame the two differ by a mean of 0.39/255: occlusion here
    # reads as creases and contact, not as a global dimming.
    'ssao-off': (BUNNY + GROUND,
                 dict(SCALE='0.8', ZOOM='1.0', ORBIT='0.35,0.3',
                      LIGHT='0.6,1.0,0.7', INTENSITY='1.2', NO_SSAO='1'),
                 ((225, 150, 385, 270), 3)),
    'ssao-on': (BUNNY + GROUND,
                dict(SCALE='0.8', ZOOM='1.0', ORBIT='0.35,0.3',
                     LIGHT='0.6,1.0,0.7', INTENSITY='1.2', SSAO_INTENSITY='1.0'),
                ((225, 150, 385, 270), 3)),
    # Crytek Sponza, the standard scene for this kind of renderer: 262,267
    # triangles across 25 materials, standing in the nave and looking down the
    # atrium. The camera has to be placed by hand: the orbit controls always
    # point at the origin from outside, and Sponza is a closed box from every
    # exterior angle.
    #
    # EYE is inside the open nave, which the geometry puts at z in (-0.11, 0.11)
    # with the end walls at x ~ +/-0.72. An earlier version of this figure stood
    # at x = -0.75, i.e. inside the west wall, and the near-plane straddling that
    # produced is what turned up the missing clip in mesh_setup_kernel. Standing
    # in the wall is no longer a hole in the frame, but it is still a wall in
    # front of the lens.
    #
    # SHADOW_BIAS is 50 here, not the default 2, and that is a finding rather
    # than a workaround. The bias is a fraction of a depth range that is fixed
    # at 0..255 whatever the world scale, and 2 is calibrated for a single model
    # viewed from outside. Stand the camera inside a scene and the depths pile
    # up against the top of that range -- 73% of this frame's depth buffer is
    # saturated at 255 -- so the comparison rejects far more than it should.
    # Sweeping the bias on this exact frame: mean luminance 48.1 at the default,
    # 69.3 at 50, and 81.0 once the test can no longer reject anything. Two
    # fifths of the darkness was the shadow map rather than the lighting, and 50
    # is the least aggressive value that recovers most of it while still casting
    # real shadows.
    #
    # What is left is the lighting: a bit over a third of the frame sits at the
    # flat ambient 20/255, because there is one directional lamp and everything
    # facing away from it gets nothing. The black sky through the open roof is
    # the cleared framebuffer; there is no environment map.
    'sponza': (SPONZA,
               dict(EYE='-0.45,-0.15,0.0', LOOK='0.60,-0.20,0.06',
                    LIGHT='0.9,0.35,0.4', INTENSITY='2.0',
                    SHADOW_BIAS='50'), None),
    'ssao-term': (BUNNY + GROUND,
                  dict(SCALE='0.8', ZOOM='1.0', ORBIT='0.35,0.3',
                       LIGHT='0.6,1.0,0.7', SSAO_DEBUG='1'), None),
    'gbuffer-normals': (TOYCAR + GROUND,
                        dict(SCALE='0.95', ZOOM='-0.7', ORBIT='-0.9,0.25',
                             SSAO_DEBUG='2'), None),
    # Same view at both sampling rates, cropped onto the silhouette where the
    # two differ most and magnified with nearest neighbour, so the figure shows
    # the pixels the renderer produced rather than a resampling of them.
    'ssaa-1x': (BUNNY, dict(SCALE='0.8', ZOOM='1.0', ORBIT='0.35,0.3',
                            LIGHT='0.6,1.0,0.7', SSAA='1'),
                ((280, 570, 400, 690), 4)),
    'ssaa-2x': (BUNNY, dict(SCALE='0.8', ZOOM='1.0', ORBIT='0.35,0.3',
                            LIGHT='0.6,1.0,0.7', SSAA='2'),
                ((280, 570, 400, 690), 4)),
}


def decode_tga(path):
    """The renderer writes RLE BGR with a top-left origin; Pillow mis-reads it."""
    from PIL import Image
    raw = open(path, 'rb').read()
    idlen, _, dtype, _, _, _, _, _, w, h, bpp, desc = struct.unpack('<BBBHHBHHHHBB', raw[:18])
    off, npx, n = 18 + idlen, w * h, bpp // 8
    px = bytearray()
    if dtype in (2, 3):
        px = bytearray(raw[off:off + npx * n])
    elif dtype in (10, 11):
        got = 0
        while got < npx:
            head = raw[off]
            off += 1
            count = (head & 0x7F) + 1
            if head & 0x80:
                px += raw[off:off + n] * count
                off += n
            else:
                px += raw[off:off + count * n]
                off += count * n
            got += count
        del px[npx * n:]
    else:
        raise ValueError(f'unsupported TGA type {dtype}')

    if n == 1:
        im = Image.frombytes('L', (w, h), bytes(px)).convert('RGB')
    else:
        mode = 'RGB' if n == 3 else 'RGBA'
        im = Image.frombytes(mode, (w, h), bytes(px))
        b = im.split()
        im = Image.merge(mode, (b[2], b[1], b[0]) + tuple(b[3:])).convert('RGB')
    return im if desc & 0x20 else im.transpose(Image.FLIP_TOP_BOTTOM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build-dir', default='build/windows')
    ap.add_argument('--only', nargs='*', help='regenerate just these figures')
    args = ap.parse_args()

    exe = os.path.join(REPO, args.build_dir, 'bin', 'capture_scene')
    if not os.path.exists(exe):
        exe += '.exe'
    if not os.path.exists(exe):
        sys.exit(f'capture_scene not built at {exe}\n'
                 f'  cmake --build {args.build_dir} --target capture_scene')

    os.makedirs(OUT, exist_ok=True)
    tmp = os.path.join(OUT, '_tmp.tga')

    for name, (models, env, crop) in FIGURES.items():
        if args.only and name not in args.only:
            continue
        if not models:
            print(f'  {name:18} SKIPPED, models not present '
                  f'(python tools/fetch_models.py {name})')
            continue
        e = dict(os.environ, **env)
        r = subprocess.run([exe, tmp] + models, cwd=REPO, env=e,
                           capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f'{name}: capture_scene failed\n{r.stdout}{r.stderr}')

        from PIL import Image
        im = decode_tga(tmp)
        box, zoom = crop if crop else (VIEWPORT, 1)
        im = im.crop(box)
        if zoom != 1:
            im = im.resize((im.width * zoom, im.height * zoom), Image.NEAREST)
        dst = os.path.join(OUT, name + '.png')
        im.save(dst, optimize=True)
        stats = next((l for l in r.stdout.splitlines() if l.startswith('STATS')), '')
        print(f'  {name:18} {im.size[0]}x{im.size[1]}  {os.path.getsize(dst)//1024:>4} KB  {stats}')

    if os.path.exists(tmp):
        os.remove(tmp)


if __name__ == '__main__':
    main()
