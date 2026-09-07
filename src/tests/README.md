# GPU rasterizer tests

Small standalone programs that link straight against the built `.o` files. No
window, no SDL, no interaction. Each one asserts and exits non-zero on failure.

    make tests          # regression suite -> tests/bin/
    make test_engine    # end-to-end Engine test (needs SDL2 + the scene graph)
    make test_ssao      # ambient occlusion, same requirements
    make test_ssaa      # supersampling, same requirements
    make test_background # background compositing, same requirements
    make test_meshcache # device geometry cache lifetime, same requirements
    make test_environment # HDR environment map, same requirements

Run them from `src/` so the relative model paths resolve.

## Regression suite

`reference_raster.cpp` is the CPU rasterizer. It is not part of the engine --
nothing in `src/` links it -- and exists only so the differential tests have an
independent implementation to disagree with.

| program | what it answers |
|---|---|
| `test_cull` | does the CUDA path draw the same pixels the CPU rasterizer does, on a real model? |
| `test_blit` | is the frame the right way up, and is R at byte 0 for SDL? |
| `test_mesh` | does GPU-resident geometry match the host staging path exactly? |
| `test_shaded` | do the CPU and CUDA shaded outputs stay numerically close on a textured model? |
| `test_frustum` | does the GPU mesh path reject geometry that sits fully behind the camera? |
| `test_engine` | does the real `Engine` render correctly on the CUDA path: scene graph, node transforms, two-pass shadows, light controls? |
| `test_ssao` | does ambient occlusion darken cavities without touching exposed surfaces, the background, or the geometry? |
| `test_ssaa` | does supersampling anti-alias the silhouette without moving, rescaling or blurring the image? |
| `test_background` | does the background composite on the GPU, under the geometry, the right way up? |
| `test_meshcache` | does the Model -> device mesh cache get dropped when Scene::clear() frees the Models? |
| `test_environment` | does the HDR environment decode, and is it sampled along the view ray rather than pasted on? |

`test_engine` runs headless via SDL's dummy video driver and steps `render()`
directly instead of calling `run()`. It is the only test that covers the
integration seams; everything else drives the kernels in `src/cuda/` on
its own.

`test_ssao` compares a concave landmark against a convex one rather than
just checking that the frame changed. "SSAO changed something" is nearly
free to satisfy and would pass on a pass that merely dimmed the image; the
ratio between the two landmarks is what actually separates occlusion from a
brightness slider.

`test_meshcache` counts live device meshes rather than comparing pixels for
its main assertion. The failure it guards against is a cache keyed by `Model*`
outliving the Models: whether that draws the wrong geometry depends on the
allocator handing a new Model the address of a freed one, so a pixel check
would fail only sometimes. The leak is there on every run.

Backface culling is the one thing here no differential test can check. Both
rasterizers shared an inverted sign, so they agreed exactly on a wrong image:
from the front, culling the front faces of a closed mesh leaves the inside of
its far side, which keeps a plausible silhouette while lighting it by normals
that point away. What catches it is that culling is only ever an optimization,
so for a closed mesh the render has to be identical with it disabled. Comparing
against a no-cull render is the check; `capture_scene` is there to make that
comparison easy.

`test_environment` moves the camera to four known directions and demands the
colour belonging to each, rather than checking that the frame changed or that
some expected colour is present. Both of those pass on a wallpaper, which is
the thing an environment map has to not be. It caught the bug it was written
for on the first run: the view ray was built from the wrong end of the depth
range, so it pointed back at the camera and every direction sampled the
opposite side of the sphere. Getting both axes wrong at once is why a
single-direction check would have been no use -- looking one way still
produced a plausible colour, just the wrong one.

It also covers the tone map, which nothing else does: the five differential
tests all call `cudaSetToneMapping(0)`, because they compare the shading path
against the CPU rasterizer and want the kernel's own numbers rather than a
display transform of them. The check that matters there is that a radiance of
1.5 does not come out at 255. Anything above white would clip to 255 under a
plain clamp, so that one threshold is what separates a tone curve from no tone
curve at all; a monotonicity check on its own would pass either way.

The exposure sweep sums the three channels instead of reading the brightest.
Green is already at 241 with exposure 1, near where the curve flattens, so it
moves 14 levels over an eightfold change while the dim channels move 135.
Probing one channel measures where that channel sits on the curve, not whether
exposure works.

The diffuse IBL checks hang on one number that can be worked out on paper.
The integral of cosine over a hemisphere is pi, so a constant environment L
convolves to exactly L and a surface renders at albedo * L and nothing else.
Drop the sin(theta) weighting, the solid angle, or the divide by pi and the
result lands somewhere different; with L = 0.5 and the untextured mesh colour
the frame has to read 100, 85, 75, and it does, exactly. The direct light is
set to zero and the tone map turned off for those checks, so what is in the
frame is the irradiance term alone rather than a display of it.

The directional checks that follow assert ordering only. What the centre pixel
shows is whichever surface the bunny presents there, which is not
axis-aligned, so the absolute values belong to the model rather than to the
convolution. What they catch is a dropped eye-to-world rotation: the map is
built in world space and normals arrive in eye space, and without the
transpose the centre normal reads as +Z from every camera and all three views
come back equal.

The convolution was also checked once against a CPU implementation of the same
integral over a real sky map, outside the suite. It agreed: E/pi of 5.9
upward against 0.15 downward, and a predicted 211/255 for an up-facing surface
where the sky behind it sits at 94. That is worth recording because the render
looks wrong at a glance -- the object comes out brighter than the sky lighting
it -- and it is not. A sun contributes about 6 units of irradiance where the
blue sky radiance is under 3, which is a white object in sunlight.

Its .hdr files are written by the test rather than committed. A map small
enough to commit is still a binary nobody can read, and generating it means the
Radiance decoder is checked against known values: the same pixels are written
twice, once flat and once run-length encoded, and the two have to decode
bit-identically. The RLE path is the one real files use and the only one with
enough logic to be wrong.

`test_background` uses a two-band image rather than a flat colour. A flat
background cannot tell a correct composite from a vertically flipped one,
which is the mistake the first version of the kernel actually made.

`test_ssaa` checks two things that have to hold together. Anti-aliasing
shows up as partial coverage at the silhouette, so it counts edge pixels
that land between background and surface. But supersampling changes the
viewport the whole pipeline renders through, so the easy way to get it
wrong is an image that is smooth but scaled or shifted; the bounding box
and coverage checks are what catch that, and a smoothness check on its own
would not. The interior-patch check separates it from a blur.

`test_shaded` is a differential test, so its CPU reference -- the `struct S`
shader inside the test -- has to be derived from the shading model, not from
the kernel it is checking. An earlier version copied the kernel's normal
handling into the reference, which made the two agree by construction and hid
a real shading bug. If you change how the kernel shades, work out
independently what the reference should be; never paste the kernel's version
into it.

## Local tools

`make tools` builds the benchmarking and inspection programs. These measure
or render for eyeballing rather than asserting, so they are gitignored and
the target skips whichever are absent. `bench_meshes` is the exception: it
needs SDL and the scene graph, so it has its own target and is tracked, being
the measurement that justifies batching the setup launch.

| program | what it does |
|---|---|
| `test_wind` | prints fill counts per winding order, to check the backface cull side |
| `bench` | cost of getting a finished frame to the screen |
| `bench_cull` | what backface culling is worth, across three models |
| `bench_tile` | tile size sweep |
| `profile_frame` | per-stage frame breakdown (setup / bin / raster / DMA) |
| `bench_meshes` | per-frame cost against mesh count, through the real Engine |
| `capture_scene` | renders any set of models headless to a TGA, for eyeballing |
| `render_png` | renders a model to TGA so the shading can be looked at |

`render_png` takes: `model out [gpu|cpu] [texmask dns] [S|N shadows] [bias]
[l|p filter] [d|g camera]`, where `l`/`p` select bilinear or point filtering and
`d`/`g` select the default or glancing-angle view.

    tests/bin/render_png ../obj/african_head.obj /tmp/linear.tga g dns x 0 l
    tests/bin/render_png ../obj/african_head.obj /tmp/linear_glancing.tga g dns x 0 l g
