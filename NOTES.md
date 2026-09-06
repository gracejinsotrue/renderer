# Engineering notes

Working notes from porting this renderer from a CPU rasterizer to a CUDA one.
Kept because the measurements are the interesting part: most of what follows is
a number that contradicted an assumption.

## Build environment

Builds under WSL, not Windows natively. The Makefile's `/usr/include/SDL2` and
`/usr/local/cuda/lib64` are Linux paths, and the Windows-side toolchain is a
dead end: nvcc there wants MSVC `cl.exe`, and MinGW g++ is not a host compiler
nvcc supports.

In WSL: CUDA 12.0 at `/usr/local/cuda` (nvcc not on PATH), SDL2 headers in
`/usr/include/SDL2`. CUDA 12.0's nvcc rejects host gcc > 12 and the default here
is 14, so the Makefile pins `-ccbin g++-11`. Plain `make` from inside WSL.

Model sizes, for reading the numbers below:

| model | faces |
|---|---|
| `mita/body.obj` | 1,744 |
| `obj/african_head.obj` | 2,492 |
| `rumi/head.obj` | 10,874 |
| `rumi/hair.obj` | 41,963 |
| `rumi/body.obj` | 136,725 |

## Where this started

Single-threaded CPU end to end. No OpenMP, no threads. A CUDA path existed and
was toggled with `K`, but it had two showstoppers:

- **No shading.** `TGAColor color(150, 100, 100); // todo: fix`. No textures, no
  normal map, no specular, no shadows. The toggle produced flat pink.
- **O(pixels x triangles).** One thread per pixel looping over every triangle.
  800x800 with `rumi/body.obj` is ~8.7e10 bounding-box tests per frame.

It also did no backface culling, which the CPU path did, so it processed roughly
twice the geometry. The Makefile had no nvcc rule at all: `cuda_triangle.o` was
linked but never built by it.

## What the GPU path does now

Twelve kernels. Geometry is uploaded once per `Model` and lives on the device;
only the matrices change per frame. The finished frame is written top-down in
R,G,B so it can go straight into an SDL texture with no host per-pixel work.

| kernel | job |
|---|---|
| `mesh_setup_kernel` | transforms every queued mesh, backface + 5-plane frustum cull, appends survivors |
| `count_kernel` / `scatter_kernel` | bins triangles into 16x16 tiles via a cub exclusive scan |
| `tiled_raster_kernel` | one block per tile, one thread per pixel, full shading and shadow lookup |
| `shadow_raster_kernel` | depth-only, shares the same bins |
| `ssao_kernel` + blur + apply | hemisphere occlusion over the finished frame |
| `background_kernel` | composites the background image under the geometry |
| `downsample_kernel` | supersample resolve |
| `zbuffer_fill_kernel` | depth clear |

`ssao_debug`, which renders the occlusion term, normals or depth on their own,
is the twelfth.

## Measurements

### Where a frame went, after tiling but before device-resident geometry

800x800, `rumi/body.obj`, 136,725 faces / 37,655 drawn, 8.07 ms/frame:

| stage | ms | % |
|---|---|---|
| host submit + cull (CPU loop over every face) | 3.43 | 43% |
| triangle upload H2D (~2.7 MB/frame) | 1.72 | 21% |
| bin: count + scan + scatter | 0.59 | 7% |
| raster kernel | 1.12 | 14% |
| frame DMA back | 1.21 | 15% |

Only 21% of the frame was actual rasterization. 64% was the CPU submit loop plus
shipping triangles across the bus every frame. That is what motivated moving
geometry onto the device permanently, and it is the single most useful thing
measuring produced: the obvious optimization target (the raster kernel) was 14%
of the heaviest frame and 4% of a light one.

Same model today, after that change: 3.66 ms/frame.

### Getting the frame to the screen

| path | ms/frame |
|---|---|
| device -> host TGAImage, per-pixel swizzle (old) | 7.10 |
| device -> locked SDL texture, one DMA (current) | 1.60 |

### Batching the setup launch

A kernel launch measures ~11 us here, and setup used to run once per mesh per
pass: 80 launches a frame for a 40-mesh scene. Through the engine at 800x800
with 2x SSAA (`tests/bin/bench_meshes`):

| meshes | per-mesh launch | one launch |
|---|---|---|
| 1 | 1.915 ms | 1.935 ms |
| 8 | 4.774 ms | 4.202 ms |
| 20 | 7.966 ms | 6.757 ms |
| 40 | 11.591 ms | 10.485 ms |

One mesh is marginally slower, which is the draw-table upload that a
single-mesh scene gains nothing back from.

### CPU and GPU agreement

Mean byte difference 0.17 with identical pixel coverage, on a textured model
with shadows. `tests/test_shaded` is what holds this.

## Limits that are the environment, not the code

Two things worth knowing before trying to optimize the frame further here.

**CUDA/OpenGL interop does not work under WSL.** WSLg's OpenGL is llvmpipe, a
CPU software rasterizer:

    GL_VENDOR   : Mesa
    GL_RENDERER : llvmpipe (LLVM 20.1.2, 256 bits)
    cudaGLGetDevices: OS call failed or operation not supported (count=0)

So `cudaGraphicsGLRegisterBuffer` cannot succeed, and even if it did, the
fullscreen quad would be drawn on the CPU. The frame has to come back over the
bus in this environment.

**Device-to-host bandwidth is capped at ~1.8 GB/s.** Measured across four orders
of magnitude of transfer size, so it is a ceiling rather than per-call overhead:

| transfer | ms | GB/s |
|---|---|---|
| 0.06 MB | 0.083 | 0.79 |
| 0.50 MB | 0.452 | 1.16 |
| 1.83 MB | 1.254 | 1.53 |
| 8.00 MB | 4.877 | 1.72 |
| 64.0 MB | 36.944 | 1.82 |

Pinned host memory changes nothing (1.211 ms vs 1.207 ms for a frame), and
staging through pinned memory is worse (1.376 ms). This is WSL2's GPU
paravirtualization boundary; the same transfer on native hardware would be
roughly an order of magnitude faster. The ~1.6 ms frame DMA that shows up as
~45% of a light frame is this, not a code defect.

Profiling note: nsys cannot get a GPU timeline through WSL2 either. It traces
the CUDA API fine but the capture contains no GPU kernel rows, and ncu fails
with `ERR_NVGPUCTRPERM` without a Windows-side driver permission change. The
kernels therefore time themselves with CUDA events, which
`tests/bin/profile_frame` reports.

## Bugs worth recording

**Fixed-capacity tile bins silently dropped geometry.** `TILE_CAP = 2048` per
tile. An earlier claim of "0 overflow on every model" was true only for the
framing that happened to be measured. Framed properly, `rumi/body.obj` at
400x400 packed 67,355 triangles into 625 tiles and overflowed 22,702 entries,
all dropped without a word: 3.40% coverage error, 10 solid missing-geometry
pixels. Replaced with count -> `cub::DeviceScan::ExclusiveSum` -> scatter into
each tile's exact slice, growing on demand, so there is no cap left to exceed.
Result 0.03% and 0 solid. The overflow counter is what caught it. Count the
thing you are dropping.

**The depth test was inverted.** This z-buffer keeps the largest depth as
nearest, so `atomicMax` is correct and `atomicMin` draws the far surface. A
coverage-only test cannot see the difference, because the set of lit pixels is
identical either way.

**`adjugate()/det()` is the inverse-transpose, not the inverse.** Used where the
real inverse was wanted, in the screen-space-to-shadow-map transform.

**`embed<4>()` fills the new component with 1.** Right for a position, wrong for
a direction: it adds the matrix's translation column. Both the light vector and
the normal are directions.

**A cache keyed on a pointer outlived the objects it keyed on.** Twice, in two
different places. `Engine::cudaMeshes` is keyed by `Model*` and `Scene::clear()`
frees every Model; the allocator then hands a newly loaded Model the address of
one just freed, and the stale entry draws the old geometry. Loading a completely
different model after a clear rendered the previous one. The background image
had the same shape of bug, since a load frees the old `TGAImage` before
allocating the new one. Both are fixed with a version counter on `Scene` rather
than a pointer comparison, because the pointer is exactly what cannot be
trusted. The general form: if freeing and reallocating can produce the same
address, identity is not a pointer.

**Normals were being reoriented toward the camera.** Forcing `nz >= 0` flips the
sign of `n.l` discontinuously wherever nz crosses zero, which shows up as
hard-edged bands across a curved surface. Near a silhouette an interpolated or
mapped normal legitimately points away.

**A differential test that agreed by construction.** `test_shaded`'s CPU
reference had the kernel's normal handling pasted into it, so the two agreed
automatically and a real shading bug survived. A reference has to be derived
independently or it is not a reference.

## What was removed, and why

**The ray tracer, 4,382 lines.** A separate ray tracer project already existed.
`cuda_triangle.cu`, the transform code and `model.cpp` had zero references to it
between them, so the rasterizer was never coupled to it; the cost was all in the
orchestration layer, which carried state that existed purely to decide whether
the frame could stay on the device with a second renderer competing for it.

**The blend shape system, 617 lines.** It was never character animation. No
importer existed, so shapes came only from a procedural expand/squash/twist or a
recorded manual sculpt, and weights stepped 0.2 per keypress with no
interpolation. GPU-accelerating it would have made a procedural squash run fast.

**The CPU rasterizer, from the engine.** It sat behind a `use_cuda_rendering`
flag with the GPU path as the default, so it was reachable only by pressing `K`
while still constraining `present()`, `render()` and the Engine's members around
it. `triangle()` and `IShader` now live in `tests/reference_raster.cpp`, which
nothing in `src/` links: their remaining job is to give the differential tests an
independent implementation to disagree with. `init()` fails outright when the
rasterizer will not start rather than falling back.

## Still open

- The frame DMA is on the critical path. It cannot be made faster here (see
  above), but it could be overlapped with the next frame's kernels, which would
  take a 3.35 ms frame to roughly 1.9 ms at the cost of one frame of latency.
  The loop is uncapped at ~275 FPS, so this buys throughput that is not short.
- `meshes()` is a function-local static that outlives the rasterizer. That is
  what lets uploaded geometry survive a supersampling change, but it is
  load-bearing by accident rather than by design.
- `cuda_triangle.cu` is ~1,840 lines and twelve kernels under a filename that
  says "triangle".
- Device mesh slots are never reused. The device memory is freed, but the
  descriptor vector only grows.
