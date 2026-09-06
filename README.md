A 3D rendering engine written from scratch, where the whole graphics pipeline is
my own code running as CUDA kernels on the GPU. SDL2 opens a window and nothing
else is borrowed: no OpenGL, no DirectX, no Vulkan, and no CPU fallback. Without
a GPU the engine refuses to start.

800x800, including getting the finished frame back to the screen, on a laptop
RTX A3000:

| scene | faces | ms/frame | FPS |
|---|---|---|---|
| `obj/african_head.obj` | 2,492 | ~1.9 | ~520 |
| `rumi/hair.obj` | 41,963 | ~3.3 | ~300 |
| `rumi/body.obj` | 136,725 | ~3.8 | ~260 |

40% or more of each of those frames is the readback, which is capped near
1.8 GB/s by WSL2's GPU paravirtualization layer rather than by anything in the
code. That measurement and the rest of the port are written up in
[NOTES.md](NOTES.md).

I started this in 2025 as a CPU rasterizer following ssloy's
[tinyrenderer](https://schmittl.github.io/tinyrenderer/), to teach myself
computer graphics and the heavy math under it from the ground up (so everything
here is algorithmic). In 2026 I picked it back up to move it onto the GPU, which
turned out to be less a port than a rebuild: the things that make a CPU
rasterizer reasonable, like walking one triangle at a time and rebuilding
geometry every frame, are exactly the things that make a GPU one slow.

GO TO:



[technical features](#FEATURES)


[some interesting rendered images](#IMAGES)






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
top-down in R,G,B and DMA'd straight into the SDL texture.

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
to a mean byte difference of 0.17 with identical pixel coverage. See
[src/tests/README.md](src/tests/README.md), which is mostly a list of the ways
each test can be written so that it passes without proving anything.

7) **Profiling.** nsys cannot get a GPU timeline through WSL2, so the kernels
time themselves with CUDA events and `tests/bin/profile_frame` prints the
per-stage breakdown. There is also an easy Nsight wrapper script,
[nsys_easy](src/nsys_easy), for host-side API timings.


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
