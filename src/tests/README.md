# GPU rasterizer tests

Small standalone programs that link straight against the built `.o` files. No
window, no SDL, no interaction. Each one asserts and exits non-zero on failure.

    make tests          # regression suite -> tests/bin/
    make test_engine    # end-to-end Engine test (needs SDL2 + the scene graph)
    make test_ssao      # ambient occlusion, same requirements

Run them from `src/` so the relative model paths resolve.

## Regression suite

| program | what it answers |
|---|---|
| `test_cull` | does the CUDA path draw the same pixels the CPU rasterizer does, on a real model? |
| `test_blit` | is the frame the right way up, and is R at byte 0 for SDL? |
| `test_mesh` | does GPU-resident geometry match the host staging path exactly? |
| `test_shaded` | do the CPU and CUDA shaded outputs stay numerically close on a textured model? |
| `test_frustum` | does the GPU mesh path reject geometry that sits fully behind the camera? |
| `test_engine` | does the real `Engine` render correctly on the CUDA path: scene graph, node transforms, two-pass shadows, light controls? |
| `test_ssao` | does ambient occlusion darken cavities without touching exposed surfaces, the background, or the geometry? |

`test_engine` runs headless via SDL's dummy video driver and steps `render()`
directly instead of calling `run()`. It is the only test that covers the
integration seams; everything else drives `cuda_triangle.cu` on its own.

`test_ssao` compares a concave landmark against a convex one rather than
just checking that the frame changed. "SSAO changed something" is nearly
free to satisfy and would pass on a pass that merely dimmed the image; the
ratio between the two landmarks is what actually separates occlusion from a
brightness slider.

`test_shaded` is a differential test, so its CPU reference has to be derived
from `ShadowMappingShader::fragment`, not from the kernel it is checking. An
earlier version copied the kernel's normal handling into the reference, which
made the two agree by construction and hid a real shading bug. If you change
how the kernel shades, change the reference to match the CPU shader, never to
match the kernel.

## Local tools

`make tools` builds the benchmarking and inspection programs. These measure
or render for eyeballing rather than asserting, so they are gitignored and
the target skips whichever are absent.

| program | what it does |
|---|---|
| `test_wind` | prints fill counts per winding order, to check the backface cull side |
| `bench` | cost of getting a finished frame to the screen |
| `bench_cull` | what backface culling is worth, across three models |
| `bench_tile` | tile size sweep |
| `profile_frame` | per-stage frame breakdown (setup / bin / raster / DMA) |
| `render_png` | renders a model to TGA so the shading can be looked at |

`render_png` takes: `model out [gpu|cpu] [texmask dns] [S|N shadows] [bias]
[l|p filter] [d|g camera]`, where `l`/`p` select bilinear or point filtering and
`d`/`g` select the default or glancing-angle view.

    tests/bin/render_png ../obj/african_head.obj /tmp/linear.tga g dns x 0 l
    tests/bin/render_png ../obj/african_head.obj /tmp/linear_glancing.tga g dns x 0 l g
