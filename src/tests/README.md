# GPU rasterizer tests

Small standalone programs that link straight against the built `.o` files. No
window, no SDL, no interaction. Each one asserts and exits non-zero on failure.

    make tests          # regression suite -> tests/bin/
    make test_engine    # end-to-end Engine test (needs SDL2 + the scene graph)
    make test_ssao      # ambient occlusion, same requirements
    make test_ssaa      # supersampling, same requirements
    make test_background # background compositing, same requirements

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

`test_engine` runs headless via SDL's dummy video driver and steps `render()`
directly instead of calling `run()`. It is the only test that covers the
integration seams; everything else drives `cuda_triangle.cu` on its own.

`test_ssao` compares a concave landmark against a convex one rather than
just checking that the frame changed. "SSAO changed something" is nearly
free to satisfy and would pass on a pass that merely dimmed the image; the
ratio between the two landmarks is what actually separates occlusion from a
brightness slider.

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
| `render_png` | renders a model to TGA so the shading can be looked at |

`render_png` takes: `model out [gpu|cpu] [texmask dns] [S|N shadows] [bias]
[l|p filter] [d|g camera]`, where `l`/`p` select bilinear or point filtering and
`d`/`g` select the default or glancing-angle view.

    tests/bin/render_png ../obj/african_head.obj /tmp/linear.tga g dns x 0 l
    tests/bin/render_png ../obj/african_head.obj /tmp/linear_glancing.tga g dns x 0 l g
