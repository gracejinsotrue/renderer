Real-time 3D graphics rendering engine from scratch except the part where i use sdl2. Has vertex editing and animation caching (e.g., sculpt facial expressions and save them as reusable animations)

Supports multiple 3D objects via scene graph hierarchy. Cool camera too. 

Definitely a work-in-progress passion project. Currently adding CUDA acceleration, trying to optimize performance for larger models (e.g. with backface culling or some shadow map caching for my potato computer (the former actually sped it up from like 10FPS to 30FPS so reallll))

GO TO: 

[some interesting rendered images](#IMAGES)


[technical features](#FEATURES)


# IMAGES

Ray Tracing example of the food scene from kpop demon hunters
![alt text](src/rendered.png)

Engine screenshot

<img width="680" height="798" alt="image" src="https://github.com/user-attachments/assets/a10605f2-7ac6-47f9-b6be-587a96ff11ae" />
<img width="1067" height="1006" alt="image" src="https://github.com/user-attachments/assets/50536bc3-e591-4c8a-8e58-9c664d20c707" />







https://github.com/user-attachments/assets/a302a7f6-5d5b-45d6-b2bc-8536bedeb28e





https://github.com/user-attachments/assets/7ff0dcd0-e98a-43ae-a273-07f3e8cb663c





"oh, that's gore of my comfort character"
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
4)  Möller-Trumbore Ray-Triangle Intersection technique for ray tracing path, it's a classic fast algorithm for ray-mesh intersection testing. 
5) CUDA- accelerated triangle rasterization. The CPU rasterizing process processes one triangle at a time and loops through each pixel in a triangle's bounding box sequentially. This is slow. One core does all the work so true. But the CUDA implementation flips this arround by assigning one GPU thread to each pixel on the screen. The engine allocates GPU buffers for 1. vertices, 2. framebuffer and 3. depth and for each triangle it launches a kernel with 16x16 thread blocks. Each traingle INDEPENDENTLY computes barycentric coordinates to test whetehr the pixel lies inside the triangle, interpolates depth using those weights and writes to framebuffer if it does pass [z-buffer test](https://en.wikipedia.org/wiki/Z-buffering). Thousands of these pixels run in parallel across the GPU cores so true! Lowkey the tradeoff is memory transfer overhead because you need to transfer all vertices to GPU and then the finished frame back to the CPU so it wins on big fat models where paralleism outweighs copying cost. 