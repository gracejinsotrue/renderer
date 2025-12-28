Real-time 3D graphics rendering engine from scratch except the part where i use sdl2. Has vertex editing and animation caching (e.g., sculpt facial expressions and save them as reusable animations)

Supports multiple 3D objects via scene graph hierarchy. Cool camera too. 

Definitely a work-in-progress passion project. Currently adding CUDA acceleration, trying to optimize performance for larger models (e.g. with backface culling or some shadow map caching for my potato computer (the former actually sped it up from like 10FPS to 30FPS so reallll))

GOTO:
[some interesting rendered images](#images)


[technical features](#features)


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


# TECHNICAL FEATURES 
1) generic computer graphics scene graph hierarchy: https://en.wikipedia.org/wiki/Scene_graph 
2) Nsight profiling with a really easy script: [nsys_easy](src\nsys_easy)