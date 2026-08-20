# GPU rasterizer tests

Small standalone programs that link straight against the built `.o` files. No
window, no SDL, no interaction. Each one asserts and exits non-zero on failure.

    make tests          # regression suite -> tests/bin/
    make test_engine    # end-to-end Engine test (needs SDL2 + the scene graph)

Run them from `src/` so the relative model paths resolve.

## Regression suite

| program | what it answers |
|---|---|
| `test_cull` | does the CUDA path draw the same pixels the CPU rasterizer does, on a real model? |
| `test_blit` | is the frame the right way up, and is R at byte 0 for SDL? |
| `test_mesh` | does GPU-resident geometry match the host staging path exactly? |
| `test_deform` | does a sculpt or blend shape reach the GPU, and does an updated mesh match a freshly created one? |
| `test_shaded` | do the CPU and CUDA shaded outputs stay numerically close on a textured model? |
| `test_frustum` | does the GPU mesh path reject geometry that sits fully behind the camera? |
| `test_engine` | does the real `Engine` render correctly on the CUDA path: scene graph, node transforms, two-pass shadows, light controls? |

`test_engine` runs headless via SDL's dummy video driver and steps `render()`
directly instead of calling `run()`. It is the only test that covers the
integration seams; everything else drives `cuda_triangle.cu` on its own.

## Local tools

`make tools` builds the benchmarking and inspection programs. These measure or
render for eyeballing rather than asserting, so they are gitignored and the
target skips whichever are absent:

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
