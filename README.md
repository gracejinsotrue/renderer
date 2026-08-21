Real-time 3D graphics rendering engine from scratch except the part where i use sdl2. Has interactive vertex editing, so you can sculpt a mesh and watch it re-render live.

Supports multiple 3D objects via scene graph hierarchy. Cool camera too. 

Definitely a work-in-progress passion project meant to teach myself computer graphics  and the heavy math behind it from the ground up (therefore everything is algorithmic), rather than trying to make this production-ready. The CUDA path is the default renderer now and the CPU rasterizer is kept as a reference to diff against (press K to switch). That took the frame from about 100ms down to a couple of ms on my potato computer. Backface culling on its own was already worth 10FPS to 30FPS before any of the CUDA work.

GO TO: 

[some interesting rendered images](#IMAGES)


[technical features](#FEATURES)





# IMAGES

Ray Tracing example of the food scene from kpop demon hunters
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


# FEATURES
interesting technical stuff: 
1) generic computer graphics scene graph hierarchy: https://en.wikipedia.org/wiki/Scene_graph 
2) For the GPU acceleration aspect of my project, I used Nsight profiling. There exists Nsight profiling with a really easy script: [nsys_easy](src\nsys_easy) that I used sometimes. 
3) two pass shadow mapping: the first pass renders a depth buffer from the light's perspective with orthographic projection. then, the second pass samples shaodw buffer during fragment shading. then we use an inverse transform to convert between camera space and light space. 
4) CUDA-accelerated triangle rasterization, tiled. The CPU rasterizer does one triangle at a time on one core, so true. The CUDA path instead keeps each mesh resident on the device and re-uploads vertices only when the geometry actually changes. One kernel transforms a whole mesh and does backface and frustum culling, then the surviving triangles get binned into 16x16 screen tiles (counting kernel, prefix sum, scatter). The raster kernel runs one block per tile and one thread per pixel, so a thread only ever walks the triangles that touch its own tile instead of the whole batch. The frame never comes back to the CPU either: it gets DMA'd straight from device memory into the SDL texture.
5) The CPU and CUDA paths are kept numerically in sync on purpose, and the tests diff them against each other on a real textured model. That is how I caught the kernel reorienting normals toward the camera when the CPU shader did not, which was putting hard-edged bands across every curved surface. Worth knowing that a differential test only proves two implementations agree: the test had the same bug copied into its own reference, so it agreed with the wrong answer for ages.
6) I also wrote a CUDA path tracer at one point (Moller-Trumbore intersection, flat BVH, progressive accumulation, metal and glass) and then deleted it, since I have a separate ray tracer project and this one is a rasterizer. It is in the git history if you want it.
