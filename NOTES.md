# Engineering notes

Working notes from porting this renderer from a CPU rasterizer to a CUDA one.
Kept because the measurements are the interesting part: most of what follows is
a number that contradicted an assumption.

## Build environment

Two builds now: `CMakeLists.txt` at the root, which works natively on Windows
and Linux, and `src/Makefile`, which is the original WSL path and still works.

    cmake --preset windows      # or: --preset linux
    cmake --build build/windows
    ctest --test-dir build/windows

**Native Windows was never actually a dead end.** The note that used to sit
here said the Windows toolchain was one, because nvcc wants MSVC `cl.exe` and
MinGW g++ is not a host compiler nvcc supports. Both halves of that are true
and the conclusion did not follow: what it rules out is *MinGW*, not Windows.
With Visual Studio 2022 installed, nvcc has the `cl.exe` it was asking for.
The port needed no change to `src/` at all -- all six `.cu` files and every
`.cpp` compiled clean under MSVC 19.44 / CUDA 12.9 on the first attempt. The
only source change anywhere was `setenv` in the test programs, which is POSIX
and spelled `_putenv_s` in the MSVC CRT (`src/tests/compat.h`), plus
`_USE_MATH_DEFINES` for `M_PI`, which CMake supplies.

Worth knowing for the native build:

- **Use Ninja, not the Visual Studio generator.** The VS generator needs the
  CUDA MSBuild integration installed into VS itself, and here that copy was
  incomplete: the `.props`/`.targets`/`.xml` were in
  `MSBuild\Microsoft\VC\v170\BuildCustomizations` but
  `Nvda.Build.CudaTasks.v12.9.dll` was not, and CMake reports that as the
  unhelpful `No CUDA toolset found`. Ninja drives nvcc and `cl.exe` directly
  and needs none of it. It does need the MSVC environment, so configure and
  build from a Developer Command Prompt or after sourcing `vcvars64.bat`.
- **SDL2 has to be built for MSVC.** The `C:\SDL2` copy here is the MinGW
  devel package left over from that earlier attempt and will not link.
  `vcpkg install sdl2:x64-windows`, then point `VCPKG_ROOT` at the vcpkg tree
  so the preset finds the toolchain file.

In WSL: CUDA 12.0 at `/usr/local/cuda` (nvcc not on PATH), SDL2 headers in
`/usr/include/SDL2`. CUDA 12.0's nvcc rejects host gcc > 12 and the default here
is 14, so the Makefile pins `-ccbin g++-11`. Plain `make` from inside WSL.
The CMake build does not hardcode that pin -- it is a CUDA 12.0 constraint, and
the native toolchain is 12.9.

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

Sixteen kernels in `src/cuda/`, one file per pipeline stage. Geometry is
uploaded once per `Model` and lives on the device; only the matrices change per
frame. The finished frame is written top-down in R,G,B so it can go straight
into an SDL texture with no host per-pixel work.

| kernel | file | job |
|---|---|---|
| `mesh_setup_kernel` | `setup.cu` | transforms every queued mesh, backface + 5-plane frustum cull, appends survivors |
| `count_kernel` / `scatter_kernel` | `binning.cu` | bins triangles into 16x16 tiles via a cub exclusive scan |
| `tiled_raster_kernel` | `raster.cu` | forward path: one block per tile, one thread per pixel, shades inside the depth loop |
| `visibility_raster_kernel` | `raster.cu` | deferred path, front half: same walk, records the winning triangle instead of shading |
| `deferred_shade_kernel` | `shade.cu` | deferred path, back half: one shade per covered pixel |
| `shadow_raster_kernel` | `raster.cu` | depth-only, shares the same bins |
| `zbuffer_fill_kernel` | `raster.cu` | depth clear |
| `ssao_kernel` + blur + apply | `ssao.cu` | hemisphere occlusion over the finished frame |
| `background_kernel` | `resolve.cu` | composites the background image under the geometry |
| `downsample_kernel` | `resolve.cu` | supersample resolve |

`rasterizer.cu` owns every device allocation and sequences those stages. It
launches no kernel itself: each stage file exposes a host entry point that
carries its own grid and block geometry, declared in `stages.cuh`, so a launch
configuration lives next to the kernel it belongs to. `common.cuh` holds the
triangle and material layouts both sides read. cub is pulled in by `binning.cu`
alone.

`ssao_debug`, which renders the occlusion term, normals or depth on their own,
plus `visbuffer_fill_kernel` and the presentation copy in `present.cu`, make up
the rest. Deferred is the default; `cudaSetDeferredShading` switches paths at
run time so the two can be compared inside one process, and both call the same
`shade_fragment` in `shading.cuh`.

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

`tests/test_shaded` at its defaults (600x600, textured, with shadows):

    pixels_different      38499  (10.694%)
    significant_pixels        2  (0.001%, >8 in any channel)
    mean_abs_byte_diff   0.0614
    max_byte_diff            51

This section used to claim 0.17. That figure was stale -- it came from some
earlier framing that was never re-measured -- and the number above is what the
test actually prints.

### The two toolchains agree exactly

Running that same test from the MSVC build and the WSL gcc build gives
identical output: same 38,499 differing pixels, same 2 significant, same
0.0614, same max of 51. Not "close" -- the same numbers.

That is worth more than it looks. The differential suite's whole job is to
catch the GPU disagreeing with an independent CPU implementation, and if the
host toolchain perturbed the CPU reference at all, every threshold in the suite
would quietly mean something different on each platform. It does not, so a
failure on one platform is a real failure and not a floating-point dialect.
Checking this was cheap and the alternative was assuming it.

The same holds for the pipeline stats: `profile_frame` reports
`sub=136725 back=26778 off=80245 bin_entries=40589` on `rumi/body.obj` from
both builds. Identical, not close.

### Going native did not make the frame faster

The reason for porting off WSL was the readback. The section below records a
~1.8 GB/s device-to-host ceiling and predicts that the same transfer on native
hardware "would be roughly an order of magnitude faster". Measured warm, back
to back, same GPU, same source, `tests/bin/profile_frame` on `rumi/body.obj`
at 800x800, three runs each:

| stage | WSL2 | native Windows |
|---|---|---|
| raster kernel | 1.34-1.35 ms | 1.35-1.39 ms |
| frame DMA back | 0.41-0.48 ms | 0.59-0.89 ms |
| TOTAL | 2.19-2.33 ms | 3.05-3.12 ms |

The raster kernel is identical, which is the expected result: same device
code, same card. But the readback is *faster* under WSL, and the whole frame
with it. The prediction was wrong, and wrong in the direction that flattered
the plan, which is the direction nobody checks.

Two caveats, both of which weaken these numbers rather than the conclusion.
"frame DMA back" is derived, not measured -- `profile_frame` computes it as
wall time minus the CUDA event times, so it carries synchronization overhead
and is not a bandwidth figure. And this machine has large transient variance:
the same WSL binary reported 23.20 ms for this model on a cold run and 2.33 ms
warm, and one early native run reported 8186 ms with the GPU sitting at 0%
utilization, which was never reproduced and is still unexplained. Nothing here
should be published without a proper repeated measurement.

So the readback argument for going native does not survive contact with the
measurement. What does survive:

- the project builds from a clone on a machine with no WSL, which is the
  difference between a reviewer running it and not
- `ncu` and `nsys` can get a real GPU timeline natively, instead of the
  hand-rolled CUDA events being the only source of truth
- CUDA/OpenGL interop becomes reachable at all, and interop *removes* the
  copy rather than making it faster -- which is the only version of this
  argument the measurement still supports

### Profiling natively, which is the other thing the port was for

`src/profile.ps1` wraps Nsight Systems and, optionally, Nsight Compute.

Under WSL2, nsys traced the CUDA API fine and the capture contained no GPU
kernel rows at all, which is why every kernel in `src/cuda/` times itself with
CUDA events. Natively the timeline is there. `profile_frame`, one run:

| kernel | instances | avg | max |
|---|---|---|---|
| `tiled_raster_kernel` | 103 | 686 us | 1.41 ms |
| `mesh_setup_kernel` | 103 | 46 us | 133 us |
| `scatter_kernel` | 103 | 11 us | 15 us |
| `zbuffer_fill_kernel` | 103 | 9.2 us | 9.9 us |
| `count_kernel` | 103 | 7.0 us | 15 us |
| cub `DeviceScanKernel` | 103 | 4.0 us | 4.4 us |

The instance count is the first thing to check and it comes out exactly right:
`profile_frame` runs `reps+1` for three models, 51 + 31 + 21 = 103. The
`tiled_raster_kernel` max of 1.41 ms is `rumi/body.obj`, against the 1.35-1.39
ms the CUDA events report for the same model. The events were telling the
truth, which is worth knowing now that a second instrument can say so.

The capture also names the kernels `cub::CUB_200802_SM_860::...`, an
independent confirmation that the sm_86 architecture fix took -- the build was
silently producing sm_52 for a while and nothing but the configure summary
would have said otherwise.

**What the memory rows exposed.** The one thing the CUDA events could not see:

| operation | count | total | max |
|---|---|---|---|
| memcpy Device-to-Host | 415 | 16.08 ms | 156 us |
| memset | 827 | 4.64 ms | 25 us |
| memcpy Host-to-Device | 219 | 1.71 ms | 685 us |

103 of those 415 D2H copies are the frame, at ~156 us each, and they account
for 99.9% of the D2H time. That is 1.92 MB in 156 us, or 12.3 GB/s.

The other 312 are three per flush -- `num_tris`, `last_off`, `last_cnt` --
and 103 flushes times three is 309, which is the count. Each moves a handful
of bytes and each is a *blocking* readback that stalls the CPU on the GPU
mid-frame. They cost almost nothing in bandwidth, and they are the best
suspect for what the derived "frame DMA back" figure has been attributing to
the bus all along: 0.41-0.89 ms measured against 0.156 ms of actual transfer.
Removing them is now a concrete optimization rather than a hunch, and it is
exactly the sort of thing the hand-rolled event timing could not have found,
because it only ever wrapped the kernels.

**Nsight Compute still cannot read counters here.** `RmProfilingAdminOnly`
under `HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak` is already `0`, and has been across a reboot, and `ncu` still
returns `ERR_NVGPUCTRPERM` on driver 596.08. That is the only such key
anywhere under `nvlddmkm`. Running `ncu` from an elevated shell is the
remaining lever; `profile.ps1` detects the error and says so rather than
printing an empty report. nsys needs no elevation and is unaffected, and nsys
is the tool that matters for pipeline work -- ncu is for occupancy and
memory-throughput counters on one kernel, which is a Tier 1 concern.

### Interop, which is the version of that argument that held

`src/cuda/present.cu` registers an OpenGL pixel buffer with CUDA, and the
finished frame is copied into it device-to-device instead of being dragged
into host memory and uploaded again. `GLPresenter` owns the context, the
texture and the quad.

Both present paths were built to end identically -- same texture, same quad,
same context, same swap -- so the only difference is how pixels reach the
texture. That mattered: the obvious version of this comparison would have been
interop against the old `SDL_Renderer` path, which also differs in the driver
path, the texture format and the swap, and would have credited interop for all
three.

`unified_engine --bench-present` alternates the two in interleaved blocks of
120 frames. 800x800, 2x SSAA, 357 frames per mode per run:

| run | interop | host copy | difference |
|---|---|---|---|
| 1 | 1.114 ms | 2.245 ms | 1.131 ms |
| 2 | 1.188 ms | 2.355 ms | 1.167 ms |
| 3 | 1.949 ms | 2.983 ms | 1.034 ms |
| 4 | 2.209 ms | 3.184 ms | 0.975 ms |

Interop is faster in every run, by about 1.0-1.2 ms a frame.

The interleaving is why those numbers are usable at all. Look down the first
column: interop alone drifts from 1.11 ms to 2.21 ms across runs, so a
sequential "measure A, then measure B" would have produced anything from a
large win to a large loss depending on when each half happened to run. The
paired difference within a run is stable to about 0.2 ms while the absolute
numbers move by 100%. On this machine only the paired difference means
anything.

Two honest limits on the figure. The timer wraps the whole upload, and both
paths begin by flushing the pipeline and resolving supersampling, so each
absolute number includes waiting for the frame to finish rendering -- it is
not a pure transfer cost. Both pay that identically, which is what keeps the
difference clean. And this is one machine and one scene.

The fallback is real rather than theoretical, and WSL is what proves it. The
same binary logic there reports:

    GL renderer: llvmpipe (LLVM 20.1.2, 256 bits)
    CUDA/GL interop unavailable (OS call failed or operation not
    supported on this OS, 0 devices); presenting via host copy

which is the earlier `cudaGLGetDevices` finding reproduced from inside the
engine, and it drops to the host-copy path rather than failing. The headless
tests never get a GL context at all and fall back further, to the original
`SDL_Renderer`; all ten still pass.

### Deferred shading, and how much the answer depends on the scene

`tests/bench_overdraw` stacks one mesh N deep along the view axis so the
copies overlap on screen, and reports both paths at each depth. It counts
fragment-shader invocations as well as timing them, because the timing is not
self-validating: if both paths shade the same number of times then the scene
has no overdraw and the clock is answering a question nobody asked.

800x800, `obj/african_head.obj`, 2,492 fairly large triangles:

| layers | fwd shades | def shades | fwd colour | def colour | ratio |
|---|---|---|---|---|---|
| 1 | 180,241 | 176,553 | 0.163 ms | 0.134 ms | 1.21x |
| 2 | 273,001 | 178,785 | 0.277 ms | 0.208 ms | 1.33x |
| 4 | 400,961 | 183,589 | 0.514 ms | 0.356 ms | 1.44x |
| 8 | 603,381 | 194,156 | 0.896 ms | 0.577 ms | 1.55x |

The mechanism does what it was built to do. Deferred's invocation count is
essentially flat -- 176,553 to 194,156, a factor of 1.10 across an eightfold
increase in depth complexity -- while forward's triples. The gap in time widens
monotonically with depth, 1.21x to 1.55x, which is the shape the change was
supposed to produce. Zero differing pixels at every depth, and deferred's
invocation counts are identical run to run because one shade per covered pixel
is deterministic, where forward's move around with the depth-test race.

Neither timing curve is flat and neither can be. Both paths still walk every
triangle in the tile to work out coverage, and that is what grows with the
geometry; deferred removes only the repeated shading on top of it.

**And on the first scene I tried, it bought almost nothing.** `rumi/hair.obj`,
same sweep:

| layers | fwd shades | def shades | fwd colour | def colour | ratio |
|---|---|---|---|---|---|
| 1 | 1,830 | 1,297 | 1.147 ms | 0.990 ms | 1.16x |
| 8 | 9,547 | 1,593 | 7.594 ms | 7.220 ms | 1.05x |

The invocation ratio is even better there -- 6x rather than 3x -- and the time
barely moves. The counter is what explains it: 1,297 shaded pixels out of
640,000. 41,963 faces drawing roughly 1,300 pixels is sub-pixel geometry, so
the colour stage is almost entirely coverage testing and the shading it removes
was never the cost.

That is the useful part of the result. Deferred shading pays in proportion to
how much of the frame is *shading* rather than *coverage*: large triangles and
expensive fragment work, yes; dense micro-triangle meshes, hardly at all. Had I
measured only `hair.obj` -- the obvious choice, since it is the model with the
most geometry -- the conclusion would have been that the visibility buffer does
not work, and it would have been wrong. Had I measured only `african_head.obj`
I would have overclaimed. The invocation counter is what made the difference
legible rather than a mystery, and it cost about ten lines.

Worth keeping in mind for Tier 2: it also means the case for deferred gets
stronger as the shader gets more expensive, so PBR and multiple lights should
widen these ratios rather than leave them where they are.

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
roughly an order of magnitude faster.

**That sentence turned out to be right, and it still did not mean what it was
being used to mean.** A native nsys capture puts the frame readback at 156 us
for 1.92 MB: 12.3 GB/s, against 1.46 GB/s for the 1.83 MB row above. 8.4x, so
"roughly an order of magnitude faster" was a fair call.

What does not follow is the conclusion that was drawn from it. The frame did
not get faster (see "Going native did not make the frame faster" above),
because the figure labelled "frame DMA back" in `profile_frame` was never
mostly DMA. It is derived -- wall time minus the CUDA event times -- and at
0.41-0.89 ms it is three to five times the 0.156 ms the transfer actually
takes. The rest is synchronization, which does not care how fast the bus is.

Two claims were riding on one measurement and only one of them was ever
checked: the bandwidth claim was sound, and the frame-time claim rested on a
number that was quietly attributing synchronization to the bus. An earlier
draft of these notes said flatly that the prediction had been wrong, which was
an overcorrection written before there was a profiler to settle it. The ~1.6 ms frame DMA that shows up as
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

**The backface cull sign was inverted.** It kept back faces and discarded
front ones. From the front that leaves the inside of the far side of a closed
mesh, which keeps a plausible silhouette while lighting it by normals that
point away, so for a long time it read as bad shading rather than as a cull
bug; from behind, the face shows through the back of the head. Neither
differential test could see it, because both rasterizers shared the sign and
agreed exactly on a wrong image. What settles it is that culling is only ever
an optimization: for a closed mesh the render must be identical with it
disabled. Against a no-cull reference the old sign differs by a mean of 11.28
bytes per pixel and the corrected one by 0.00. The general form: when two
implementations are checked only against each other, a convention they share
is invisible; it needs a check against a version of the pipeline with the
stage removed.

**Five uninitialized members.** `cameraDistance`, `cameraRotationX/Y`,
`cameraTarget` and `orbitMode` were never initialized, and `resetCamera()` is
only reachable from a keypress, so the first orbit, pan or zoom of a session
read uninitialized floats and an uninitialized bool.

**Normals were being reoriented toward the camera.** Forcing `nz >= 0` flips the
sign of `n.l` discontinuously wherever nz crosses zero, which shows up as
hard-edged bands across a curved surface. Near a silhouette an interpolated or
mapped normal legitimately points away.

**A differential test that agreed by construction.** `test_shaded`'s CPU
reference had the kernel's normal handling pasted into it, so the two agreed
automatically and a real shading bug survived. A reference has to be derived
independently or it is not a reference.

**The CMake build silently targeted the wrong GPU.** `CMAKE_CUDA_ARCHITECTURES`
was being defaulted with `if(NOT DEFINED ...)` placed *after* `project()`, and
`project()` already defines it -- to a conservative 52. So the guard never
fired and the first native build produced sm_52 cubin for an sm_86 card. There
is no error and no visual difference: the driver JITs the embedded PTX and
every frame renders correctly, just not on code generated for the hardware it
is running on. The only symptom would have been performance numbers that were
quietly wrong, which is the worst possible symptom on a project whose point is
the measurements. It has to be set before `project()`. Same shape as the tile
overflow: the failure mode was silence, and it showed up only because the
configure summary prints the value.

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

- The frame DMA is off the critical path natively, where interop keeps the
  frame on the device. It is still there under WSL and in the headless
  fallback, and overlapping it with the next frame's kernels would still help
  those, at the cost of one frame of latency. Lower priority than it was: the
  path that motivated it is no longer the default one.
- `flush()` does three blocking few-byte device-to-host reads per frame
  (`num_tris`, `last_off`, `last_cnt`), each of which stalls the CPU on the GPU
  mid-frame. nsys counts 309 of them across a 103-frame run. They are nearly
  free in bandwidth and are the best current suspect for the gap between the
  0.156 ms the frame transfer actually takes and the 0.41-0.89 ms attributed to
  it. Keeping those counts on the device would remove all three.
- `--bench-present` measures the upload with a host timer that also covers the
  pipeline flush. Isolating the transfer itself would want CUDA events around
  the copy alone, which would also make the interop number comparable to the
  bandwidth table above rather than only to its own control.
- `meshes()` is a function-local static that outlives the rasterizer. That is
  what lets uploaded geometry survive a supersampling change, but it is
  load-bearing by accident rather than by design.
- Device mesh slots are never reused. The device memory is freed, but the
  descriptor vector only grows.
