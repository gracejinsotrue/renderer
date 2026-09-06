A 3D rendering engine written from scratch, where the whole graphics pipeline is
my own code running as CUDA kernels on the GPU. Nothing rasterizes but my own
kernels: no OpenGL, DirectX or Vulkan draws a triangle here, and there is no CPU
fallback. Without a GPU the engine refuses to start.

The one thing borrowed is a way to get the finished frame onto the screen. SDL2
opens the window, and OpenGL owns exactly one textured quad that the completed
frame is shown on -- so that the frame can be handed over on the device instead
of being copied out to the host and back. Every pixel on that quad was computed
by a CUDA kernel in this repo.

800x800, including getting the finished frame back to the screen, on a laptop
RTX A3000:

| scene | faces | ms/frame | FPS |
|---|---|---|---|
| `obj/african_head.obj` | 2,492 | ~1.9 | ~520 |
| `rumi/hair.obj` | 41,963 | ~3.3 | ~300 |
| `rumi/body.obj` | 136,725 | ~3.8 | ~260 |

Those are WSL numbers, where a large share of each frame is the readback. On a
native build the readback is gone entirely: the frame is handed to OpenGL on the
device. Both the port and the measurements that corrected my assumptions about it
-- including going native turning out *not* to make the frame faster, which is
not what I expected -- are written up in [NOTES.md](NOTES.md).

I started this in 2025 as a CPU rasterizer following ssloy's
[tinyrenderer](https://schmittl.github.io/tinyrenderer/), to teach myself
computer graphics and the heavy math under it from the ground up (so everything
here is algorithmic). In 2026 I picked it back up to move it onto the GPU, which
turned out to be less a port than a rebuild: the things that make a CPU
rasterizer reasonable, like walking one triangle at a time and rebuilding
geometry every frame, are exactly the things that make a GPU one slow.

GO TO:


[how to build it](#BUILD)


[technical features](#FEATURES)


[some interesting rendered images](#IMAGES)


# BUILD

Needs a CUDA-capable GPU, the CUDA toolkit, and SDL2. There is no CPU
fallback and no software path: without a GPU the engine exits rather than
pretending.

    cmake --preset windows          # or: --preset linux
    cmake --build build/windows
    ctest --test-dir build/windows  # 10 tests, see src/tests/README.md

**Linux.** `sudo apt install libsdl2-dev`, then the commands above with
`--preset linux`.

**Windows.** Build from a Developer Command Prompt, or any shell where
`vcvars64.bat` has been sourced, so nvcc can find MSVC's `cl.exe`. SDL2 has to
be one built for MSVC -- `vcpkg install sdl2:x64-windows`, with `VCPKG_ROOT`
pointing at the vcpkg tree. The presets use Ninja rather than the Visual Studio
generator, which needs the CUDA MSBuild integration installed into VS itself.

The build targets your own GPU's architecture by default
(`-DCMAKE_CUDA_ARCHITECTURES=86` to pin it instead).

`src/Makefile` still works and is still the WSL path; it predates the CMake
build and is kept because the measurements in [NOTES.md](NOTES.md) were taken
with it.






# FEATURES
interesting technical stuff:

1) **Tiled CUDA rasterization.** Each mesh is uploaded once and stays resident on
the device; only the matrices change per frame. One kernel transforms the whole
scene and does backface and frustum culling, and the survivors are binned into
16x16 screen tiles with a `cub` prefix sum, so there is no per-tile capacity that
could silently drop geometry (an earlier fixed-cap version was dropping 22,702
triangles a frame without saying so). The raster kernel runs one block per tile
and one thread per pixel, so a thread only ever walks the triangles touching its
own tile. The frame never comes back to the CPU to be composited: it is written
top-down in R,G,B, which is exactly the layout the present path (7) hands to
OpenGL, so no pixel is ever rearranged on the host.

2) **Scene graph hierarchy**, the generic
[kind](https://en.wikipedia.org/wiki/Scene_graph): parent-child transforms, so
moving a parent moves everything under it.

3) **Two-pass shadow mapping.** The first pass renders a depth buffer from the
light's point of view with an orthographic projection; the second samples it
during fragment shading, using an inverse transform to get from camera space to
light space. Shading is Phong, with diffuse, normal and specular maps sampled
bilinearly.

4) **Screen space ambient occlusion.** 24 samples over a hemisphere oriented by
an eye-space normal buffer, with a slope-scaled bias and a per-pixel rotation to
break up banding. Positions are reconstructed from depth rather than stored,
since `Viewport * Projection` is shared by every mesh in a frame.

5) **Supersampling.** 1x, 2x or 4x. The whole pipeline including shadows and
occlusion runs at the larger size and the frame is box-filtered back down on the
device, so edges, textures and specular highlights all get anti-aliased.

6) **A differential test suite.** The CPU rasterizer this started as now lives in
`src/tests/` and is not linked into the engine at all; its remaining job is to be
an independent implementation the kernels can be checked against. The two agree
to a mean byte difference of 0.0614, and agree to the byte across both the MSVC
and gcc builds. See
[src/tests/README.md](src/tests/README.md), which is mostly a list of the ways
each test can be written so that it passes without proving anything.

7) **The finished frame never leaves the GPU.** On a native build the frame is
copied device-to-device into an OpenGL pixel buffer registered with CUDA, and
the texture is fed from that buffer, so presenting costs no host round trip.
The older path -- device to host, then uploaded again -- is still there and
switchable at runtime with `K`, because that is the only way to measure the
two against each other honestly: both end at the same texture, quad and swap,
so nothing but the transfer differs. `unified_engine --bench-present`
alternates them in interleaved blocks and reports the difference, which is
about 1.0-1.2 ms a frame at 800x800. Under WSL, where OpenGL is llvmpipe and
CUDA can share no device with it, the engine says so and falls back.

8) **Profiling, with two instruments that check each other.** nsys cannot get a
GPU timeline through WSL2, so every kernel times itself with CUDA events and
`tests/bin/profile_frame` prints the per-stage breakdown. On a native build
[src/profile.ps1](src/profile.ps1) gets a real Nsight Systems timeline, and the
two agree: the raster kernel's self-reported 1.35-1.39 ms against a measured
1.41 ms max, over exactly the 103 launches the tool should have made. Having
the second instrument is what turned up the thing the first one structurally
could not see -- the frame readback takes 0.156 ms, not the 0.41-0.89 ms the
CUDA-event arithmetic was attributing to it, because `flush()` also does three
blocking few-byte readbacks a frame and those are stalls rather than transfers.
[nsys_easy](src/nsys_easy) is still there for host-side API timings under WSL.


# IMAGES

Rendered example of the food scene from kpop demon hunters
![alt text](src/rendered.png)

All the models are made by hand, here is how I did the modelling. 

https://github.com/user-attachments/assets/91802893-dd5c-45fb-9a43-e0ee252d52d3


Engine screenshot

<img width="680" height="798" alt="image" src="https://github.com/user-attachments/assets/a10605f2-7ac6-47f9-b6be-587a96ff11ae" />
<img width="1067" height="1006" alt="image" src="https://github.com/user-attachments/assets/50536bc3-e591-4c8a-8e58-9c664d20c707" />







https://github.com/user-attachments/assets/a302a7f6-5d5b-45d6-b2bc-8536bedeb28e





https://github.com/user-attachments/assets/7ff0dcd0-e98a-43ae-a273-07f3e8cb663c





"that's gore of my comfort character"
e.g. funny bugs

<img width="526" height="671" alt="image" src="https://github.com/user-attachments/assets/5b98dee0-e594-49f9-8092-00799caf3961" />


like any 3d engine, at certain angles, the geometry will look very weird.

<img width="1072" height="772" alt="image" src="https://github.com/user-attachments/assets/417d48da-3507-4827-9967-1f8074b4de8c" />

Bro the way I learned Blender from scratch for this too:

<img width="1356" height="1011" alt="image" src="https://github.com/user-attachments/assets/0f20a506-57b1-4adf-a0c1-bd9eb8528a91" />


<img width="1302" height="964" alt="image" src="https://github.com/user-attachments/assets/c44a0849-7d0d-4d02-8d66-5958cfba473a" />
