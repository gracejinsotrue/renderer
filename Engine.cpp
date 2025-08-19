#include "Engine.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>

Engine::Engine(int winWidth, int winHeight, int renWidth, int renHeight)
    : window(nullptr), sdlRenderer(nullptr), frameTexture(nullptr),
      framebuffer(renWidth, renHeight, TGAImage::RGB), zbuffer(renWidth, renHeight, TGAImage::GRAYSCALE),
      running(false), wireframe(false), showStats(true),
      windowWidth(winWidth), windowHeight(winHeight), renderWidth(renWidth), renderHeight(renHeight),
      mouseX(0), mouseY(0), mouseDeltaX(0), mouseDeltaY(0), mousePressed(false),
      cameraDistance(5.0f), cameraRotationX(0.0f), cameraRotationY(0.0f),
      cameraTarget(0, 0, 0), orbitMode(true)
{
    // init imput state
    memset(keys, 0, sizeof(keys));

    // shadowbuffer
    shadowbuffer.resize(renderWidth * renderHeight);
    std::fill(shadowbuffer.begin(), shadowbuffer.end(), std::numeric_limits<float>::max());
}

Engine::~Engine()
{
    shutdown();
}

bool Engine::init()
{
    // ini SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0)
    {
        std::cerr << "SDL initialization failed: " << SDL_GetError() << std::endl;
        return false;
    }

    // window
    window = SDL_CreateWindow("MULTI OBJECT 3D ENGINE THIS BETTER WORK!!!",
                              SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                              windowWidth, windowHeight,
                              SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);

    if (!window)
    {
        std::cerr << "window creation failed: " << SDL_GetError() << std::endl;
        return false;
    }

    // create the renderer
    sdlRenderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!sdlRenderer)
    {
        std::cerr << "Renderer creation failed: " << SDL_GetError() << std::endl;
        return false;
    }

    // create texture for framebuffer
    frameTexture = SDL_CreateTexture(sdlRenderer, SDL_PIXELFORMAT_RGB24,
                                     SDL_TEXTUREACCESS_STREAMING,
                                     renderWidth, renderHeight);

    if (!frameTexture)
    {
        std::cerr << "Texture creation failed: " << SDL_GetError() << std::endl;
        return false;
    }

    // initialize timing
    lastTime = std::chrono::high_resolution_clock::now();

    running = true;

    // remove debugging later

    std::cout << "Multi-Object Engine initialized successfully!" << std::endl;
    std::cout << "\n=== MULTI-OBJECT CONTROLS ===" << std::endl;
    std::cout << "  TAB - Select next object" << std::endl;
    std::cout << "  SHIFT+TAB - Select previous object" << std::endl;
    std::cout << "  X - Delete selected object" << std::endl;
    std::cout << "  SHIFT+D - Duplicate selected object" << std::endl;
    std::cout << "  L - Load new model (test3.obj)" << std::endl;
    std::cout << "  N - Create empty node" << std::endl;
    std::cout << "  I - Print scene hierarchy" << std::endl;
    std::cout << "\n=== TRANSFORM SELECTED OBJECT ===" << std::endl;
    std::cout << "  CTRL + Numpad - Move object (4/6=X, 8/2=Z, +/-=Y)" << std::endl;
    std::cout << "  ALT + Numpad - Rotate object (4/6=Y, 8/2=X, 7/9=Z)" << std::endl;
    std::cout << "  SHIFT + Numpad +/- - Scale object uniformly" << std::endl;
    std::cout << "\n=== CAMERA CONTROLS ===" << std::endl;
    std::cout << "  Mouse + Left Click - Orbit/Look around" << std::endl;
    std::cout << "  Mouse Wheel - Zoom in/out" << std::endl;
    std::cout << "  WASD - Pan view (orbit mode) / Move camera (free mode)" << std::endl;
    std::cout << "  Q/E - Move up/down" << std::endl;
    std::cout << "  R/F - Zoom in/out (alternative to mouse wheel)" << std::endl;
    std::cout << "  G - Toggle camera mode (Orbit ↔ Free-look)" << std::endl;
    std::cout << "  H - Reset camera to default position" << std::endl;
    std::cout << "\n=== OTHER CONTROLS ===" << std::endl;
    std::cout << "  Arrow keys - Move light source" << std::endl;
    std::cout << "  F - Toggle wireframe mode" << std::endl;
    std::cout << "  T - Toggle stats display" << std::endl;
    std::cout << "  P - Capture frame (output.tga)" << std::endl;
    std::cout << "  B - Load background image" << std::endl;
    std::cout << "  C - Clear background" << std::endl;
    std::cout << "  ESC - Exit" << std::endl;
    std::cout << "\nDefault: Orbit Camera Mode - Mouse to orbit, WASD to pan, wheel to zoom" << std::endl;

    return true;
}

void Engine::zoomCamera(float amount)
{
    if (orbitMode)
    {
        // in orbit mode, change distance from target
        cameraDistance += amount;
        cameraDistance = std::max(0.5f, std::min(50.0f, cameraDistance)); // clamp distance

        // update camera position based on spherical coordinates
        updateCameraPosition();
    }
    else
    {
        // in free-look mode, move forward/backward
        Vec3f forward = (scene.camera.target - scene.camera.position).normalize();
        scene.camera.position = scene.camera.position + forward * amount;
        scene.camera.target = scene.camera.target + forward * amount;
    }
}

void Engine::panCamera(float deltaX, float deltaY)
{
    if (orbitMode)
    {
        // in orbit mode, move the target point
        Vec3f forward = (scene.camera.target - scene.camera.position).normalize();
        Vec3f right = cross(forward, scene.camera.up).normalize();
        Vec3f up = cross(right, forward).normalize();

        Vec3f panMovement = right * deltaX + up * deltaY;
        cameraTarget = cameraTarget + panMovement;
        scene.camera.target = cameraTarget;

        updateCameraPosition();
    }
    else
    {
        // in free-look mode, strafe and move up/down
        Vec3f forward = (scene.camera.target - scene.camera.position).normalize();
        Vec3f right = cross(forward, scene.camera.up).normalize();
        Vec3f up = cross(right, forward).normalize();

        Vec3f movement = right * deltaX + up * deltaY;
        scene.camera.position = scene.camera.position + movement;
        scene.camera.target = scene.camera.target + movement;
    }
}

void Engine::orbitCamera(float deltaYaw, float deltaPitch)
{
    if (orbitMode)
    {
        cameraRotationY += deltaYaw;
        cameraRotationX += deltaPitch;

        // clamp pitch to avoid gimbal lock
        cameraRotationX = std::max(-1.5f, std::min(1.5f, cameraRotationX));

        updateCameraPosition();
    }
}

void Engine::updateCameraPosition()
{
    if (orbitMode)
    {
        // convert spherical coordinates to cartesian!!
        float cosX = cos(cameraRotationX);
        float sinX = sin(cameraRotationX);
        float cosY = cos(cameraRotationY);
        float sinY = sin(cameraRotationY);

        Vec3f offset;
        offset.x = cameraDistance * cosX * sinY;
        offset.y = cameraDistance * sinX;
        offset.z = cameraDistance * cosX * cosY;

        scene.camera.position = cameraTarget + offset;
        scene.camera.target = cameraTarget;
    }
}

void Engine::resetCamera()
{
    // reset to default camera position
    cameraDistance = 5.0f;
    cameraRotationX = 0.0f;
    cameraRotationY = 0.0f;
    cameraTarget = Vec3f(0, 0, 0);

    scene.camera.position = Vec3f(0, 0, 5);
    scene.camera.target = Vec3f(0, 0, 0);
    scene.camera.up = Vec3f(0, 1, 0);

    if (orbitMode)
    {
        updateCameraPosition();
    }

    std::cout << "Camera reset to default position" << std::endl;
}

void Engine::toggleCameraMode()
{
    orbitMode = !orbitMode;
    if (orbitMode)
    {
        // switching to orbit mode, so must calculate current spherical coordinates
        Vec3f toCamera = scene.camera.position - scene.camera.target;
        cameraDistance = toCamera.norm();
        cameraTarget = scene.camera.target;
        cameraRotationY = atan2(toCamera.x, toCamera.z);
        cameraRotationX = asin(toCamera.y / cameraDistance);
        std::cout << "Switched to Orbit Camera Mode" << std::endl;
    }
    else
    {
        std::cout << "Switched to Free-Look Camera Mode" << std::endl;
    }
}

// multi-object scene management methods
SceneNode *Engine::loadModel(const std::string &filename, const std::string &nodeName)
{
    SceneNode *node = scene.loadModel(filename, nodeName);
    if (node)
    {
        std::cout << "Loaded model into scene: " << filename << std::endl;
        // auto-select newly loaded model
        scene.selectNode(node);
    }
    return node;
}

SceneNode *Engine::createEmptyNode(const std::string &nodeName)
{
    SceneNode *node = scene.createEmptyNode(nodeName);
    if (node)
    {
        scene.selectNode(node);
    }
    return node;
}

void Engine::loadBackground(const std::string &filename)
{
    scene.loadBackground(filename);
}

// OBJECT SELECTION MODELS
void Engine::selectNextObject()
{
    std::vector<SceneNode *> meshNodes;
    scene.getAllMeshNodes(meshNodes);

    if (meshNodes.empty())
        return;

    SceneNode *current = scene.getSelectedNode();
    auto it = std::find(meshNodes.begin(), meshNodes.end(), current);

    if (it == meshNodes.end())
    {
        // nothing selected or selected node not in mesh list
        scene.selectNode(meshNodes[0]);
    }
    else
    {
        // move to next, wrap around
        ++it;
        if (it == meshNodes.end())
            it = meshNodes.begin();
        scene.selectNode(*it);
    }
}

void Engine::selectPreviousObject()
{
    std::vector<SceneNode *> meshNodes;
    scene.getAllMeshNodes(meshNodes);

    if (meshNodes.empty())
        return;

    SceneNode *current = scene.getSelectedNode();
    auto it = std::find(meshNodes.begin(), meshNodes.end(), current);

    if (it == meshNodes.end())
    {
        // Nothing selected or selected node not in mesh list
        scene.selectNode(meshNodes.back());
    }
    else
    {
        // Move to previous, wrap around
        if (it == meshNodes.begin())
        {
            it = meshNodes.end();
        }
        --it;
        scene.selectNode(*it);
    }
}

void Engine::deleteSelectedObject()
{
    SceneNode *selected = scene.getSelectedNode();
    if (selected && selected->name != "Root")
    {
        std::string nodeName = selected->name;
        scene.deleteNode(nodeName);
        std::cout << "Deleted object: " << nodeName << std::endl;
    }
}
// TODO: limited duplicatign selection
void Engine::duplicateSelectedObject()
{
    SceneNode *selected = scene.getSelectedNode();
    if (selected && selected->hasModel())
    {
        // find the original model path (this is simplified for now. in practice we'd store this info)
        // for now, we'll create an empty node and mention this limitation
        SceneNode *duplicate = scene.createEmptyNode(selected->name + "_copy");
        if (duplicate)
        {
            // copy transform
            duplicate->localTransform = selected->localTransform;
            duplicate->localTransform.position.x += 1.0f;
            scene.selectNode(duplicate);
            std::cout << "Created duplicate (empty node): " << duplicate->name << std::endl;
            std::cout << "Note: Model duplication needs original file path - feature to be implemented" << std::endl;
        }
    }
}

// transform manipulation methods
void Engine::moveSelectedObject(const Vec3f &delta)
{
    SceneNode *selected = scene.getSelectedNode();
    if (selected)
    {
        selected->localTransform.position = selected->localTransform.position + delta;
        std::cout << "Moved " << selected->name << " by (" << delta.x << ", " << delta.y << ", " << delta.z << ")" << std::endl;
    }
}

void Engine::rotateSelectedObject(const Vec3f &delta)
{
    SceneNode *selected = scene.getSelectedNode();
    if (selected)
    {
        selected->localTransform.rotation = selected->localTransform.rotation + delta;
        std::cout << "Rotated " << selected->name << " by (" << delta.x << ", " << delta.y << ", " << delta.z << ")" << std::endl;
    }
}

void Engine::scaleSelectedObject(const Vec3f &delta)
{
    SceneNode *selected = scene.getSelectedNode();
    if (selected)
    {
        selected->localTransform.scale = selected->localTransform.scale + delta;
        // clamp scale to prevent negative values
        selected->localTransform.scale.x = std::max(0.1f, selected->localTransform.scale.x);
        selected->localTransform.scale.y = std::max(0.1f, selected->localTransform.scale.y);
        selected->localTransform.scale.z = std::max(0.1f, selected->localTransform.scale.z);
        std::cout << "Scaled " << selected->name << " by (" << delta.x << ", " << delta.y << ", " << delta.z << ")" << std::endl;
    }
}

void Engine::run()
{
    while (running)
    {
        auto currentTime = std::chrono::high_resolution_clock::now();

        // ensure minimum deltaTime to prevent FPS spikes
        float rawDeltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
        deltaTime = std::max(rawDeltaTime, 1.0f / 120.0f); // max 120 fps

        lastTime = currentTime;

        handleEvents();
        update();
        render();
        present();

        // SDL_Delay(33);
    }
}

void Engine::handleEvents()
{
    SDL_Event event;
    while (SDL_PollEvent(&event))
    {
        switch (event.type)
        {
        case SDL_QUIT:
            running = false;
            break;

        case SDL_KEYDOWN:
            keys[event.key.keysym.scancode] = true;

            // handle single-press keys
            switch (event.key.keysym.sym)
            {
            case SDLK_ESCAPE:
                running = false;
                break;
            case SDLK_f:
                wireframe = !wireframe;
                break;
            case SDLK_t:
                showStats = !showStats;
                break;
            case SDLK_p:
                captureFrame("output.tga");
                std::cout << "Frame captured!" << std::endl;
                break;
            case SDLK_b:
                scene.loadBackground("background.tga");
                break;
            case SDLK_c:
                scene.clearBackground();
                std::cout << "Background cleared!" << std::endl;
                break;
            case SDLK_g:
                toggleCameraMode();
                break;
            case SDLK_h:
                resetCamera();
                break;

            // object selection and manipulation
            case SDLK_TAB:
                if (keys[SDL_SCANCODE_LSHIFT])
                {
                    selectPreviousObject();
                }
                else
                {
                    selectNextObject();
                }
                break;
            case SDLK_x:
                deleteSelectedObject();
                break;
            case SDLK_d:
                if (keys[SDL_SCANCODE_LSHIFT])
                {
                    duplicateSelectedObject();
                }
                break;
            case SDLK_l:
                // Load a new model (example)
                loadModel("obj/test3.obj");
                break;
            case SDLK_n:
                // Create empty node
                createEmptyNode();
                break;
            case SDLK_i:
                // Print scene hierarchy
                scene.printSceneHierarchy();
                break;
            }
            break;

        case SDL_KEYUP:
            keys[event.key.keysym.scancode] = false;
            break;

        case SDL_MOUSEBUTTONDOWN:
            if (event.button.button == SDL_BUTTON_LEFT)
            {
                mousePressed = true;
                SDL_SetRelativeMouseMode(SDL_TRUE);
            }
            break;

        case SDL_MOUSEBUTTONUP:
            if (event.button.button == SDL_BUTTON_LEFT)
            {
                mousePressed = false;
                SDL_SetRelativeMouseMode(SDL_FALSE);
            }
            break;

        case SDL_MOUSEMOTION:
            if (mousePressed)
            {
                mouseDeltaX = event.motion.xrel;
                mouseDeltaY = event.motion.yrel;
            }
            break;

        case SDL_MOUSEWHEEL:
            if (event.wheel.y > 0)
            {
                zoomCamera(-0.5f); // Zoom in
            }
            else if (event.wheel.y < 0)
            {
                zoomCamera(0.5f); // Zoom out
            }
            break;
        }
    }
}

void Engine::update()
{

    static int frameCount = 0;
    frameCount++;
    if (frameCount % 30 == 0)
    { // Every 30 frames
        std::cout << "Frame " << frameCount << " - FPS: " << getFPS() << std::endl;
    }
    updateCamera();

    // update scene transforms
    scene.updateAllTransforms();

    // object manipulation with keyboard
    float moveSpeed = 2.0f * deltaTime;
    float rotSpeed = 1.0f * deltaTime;
    float scaleSpeed = 1.0f * deltaTime;

    SceneNode *selected = scene.getSelectedNode();
    if (selected)
    {
        Vec3f moveDelta(0, 0, 0);
        Vec3f rotDelta(0, 0, 0);
        Vec3f scaleDelta(0, 0, 0);

        // Movement w/ ctrl
        if (keys[SDL_SCANCODE_LCTRL])
        {
            if (keys[SDL_SCANCODE_KP_4])
                moveDelta.x -= moveSpeed; // Numpad 4
            if (keys[SDL_SCANCODE_KP_6])
                moveDelta.x += moveSpeed; // Numpad 6
            if (keys[SDL_SCANCODE_KP_8])
                moveDelta.z -= moveSpeed; // Numpad 8
            if (keys[SDL_SCANCODE_KP_2])
                moveDelta.z += moveSpeed; // Numpad 2
            if (keys[SDL_SCANCODE_KP_PLUS])
                moveDelta.y += moveSpeed; // Numpad +
            if (keys[SDL_SCANCODE_KP_MINUS])
                moveDelta.y -= moveSpeed; // Numpad -

            if (moveDelta.norm() > 0)
            {
                moveSelectedObject(moveDelta);
            }
        }
        // rotation (with Alt modifier)
        else if (keys[SDL_SCANCODE_LALT])
        {
            if (keys[SDL_SCANCODE_KP_4])
                rotDelta.y -= rotSpeed;
            if (keys[SDL_SCANCODE_KP_6])
                rotDelta.y += rotSpeed;
            if (keys[SDL_SCANCODE_KP_8])
                rotDelta.x -= rotSpeed;
            if (keys[SDL_SCANCODE_KP_2])
                rotDelta.x += rotSpeed;
            if (keys[SDL_SCANCODE_KP_7])
                rotDelta.z -= rotSpeed;
            if (keys[SDL_SCANCODE_KP_9])
                rotDelta.z += rotSpeed;

            if (rotDelta.norm() > 0)
            {
                rotateSelectedObject(rotDelta);
            }
        }
        // scale (with Shift modifier)
        else if (keys[SDL_SCANCODE_LSHIFT])
        {
            if (keys[SDL_SCANCODE_KP_PLUS])
                scaleDelta = Vec3f(scaleSpeed, scaleSpeed, scaleSpeed);
            if (keys[SDL_SCANCODE_KP_MINUS])
                scaleDelta = Vec3f(-scaleSpeed, -scaleSpeed, -scaleSpeed);

            if (scaleDelta.norm() > 0)
            {
                scaleSelectedObject(scaleDelta);
            }
        }
    }

    // light controls with arrow keys
    float lightSpeed = 2.0f * deltaTime;
    if (keys[SDL_SCANCODE_UP])
        scene.light.direction.z += lightSpeed;
    if (keys[SDL_SCANCODE_DOWN])
        scene.light.direction.z -= lightSpeed;
    if (keys[SDL_SCANCODE_LEFT])
        scene.light.direction.x -= lightSpeed;
    if (keys[SDL_SCANCODE_RIGHT])
        scene.light.direction.x += lightSpeed;

    scene.light.direction.normalize();

    updateWindowTitle();
}

void Engine::updateCamera()
{
    float moveSpeed = 3.0f * deltaTime;
    float panSpeed = 2.0f * deltaTime;
    float mouseSpeed = 0.003f;

    // mouse look / orbit
    if (mousePressed && (mouseDeltaX != 0 || mouseDeltaY != 0))
    {
        if (orbitMode)
        {
            orbitCamera(-mouseDeltaX * mouseSpeed, -mouseDeltaY * mouseSpeed);
        }
        else
        {
            //  free look mode
            Vec3f toCamera = scene.camera.position - scene.camera.target;
            float radius = toCamera.norm();

            float theta = atan2(toCamera.z, toCamera.x) - mouseDeltaX * mouseSpeed;
            float phi = acos(toCamera.y / radius) + mouseDeltaY * mouseSpeed;
            phi = std::max(0.1f, std::min(3.04f, phi));

            toCamera.x = radius * sin(phi) * cos(theta);
            toCamera.y = radius * cos(phi);
            toCamera.z = radius * sin(phi) * sin(theta);

            scene.camera.position = scene.camera.target + toCamera;
        }

        mouseDeltaX = mouseDeltaY = 0;
    }

    // for keyboard movement
    Vec3f movement(0, 0, 0);

    // WASD for movement/panning
    if (keys[SDL_SCANCODE_W])
        movement.z += 1.0f; // Forward/Pan up
    if (keys[SDL_SCANCODE_S])
        movement.z -= 1.0f; // Back/Pan down
    if (keys[SDL_SCANCODE_A])
        movement.x -= 1.0f; // Left/Pan left
    if (keys[SDL_SCANCODE_D])
        movement.x += 1.0f; // Right/Pan right
    if (keys[SDL_SCANCODE_Q])
        movement.y -= 1.0f; // Down
    if (keys[SDL_SCANCODE_E])
        movement.y += 1.0f; // Up

    // Zoom with R/F keys
    if (keys[SDL_SCANCODE_R])
        zoomCamera(-moveSpeed); // Zoom in
    if (keys[SDL_SCANCODE_F])
        zoomCamera(moveSpeed); // Zoom out

    if (movement.norm() > 0)
    {
        if (orbitMode)
        {
            // In orbit mode, WASD pans the view
            panCamera(movement.x * panSpeed, movement.y * panSpeed);

            // W/S also zoom in orbit mode
            if (movement.z != 0)
            {
                zoomCamera(movement.z * moveSpeed);
            }
        }
        else
        {
            // In free-look mode, use existing movement code
            scene.camera.move(movement.normalize(), moveSpeed);
        }
    }
}

void Engine::render()
{
    // Clear buffers
    framebuffer.clear();
    for (int i = 0; i < renderWidth * renderHeight; i++)
    {
        zbuffer.set(i % renderWidth, i / renderWidth, TGAColor(0));
    }

    // draw the background AND THEN RENDER 3D SCENE
    drawBackground();

    // then render 3D scene (but don't clear framebuffer in renderScene)
    renderScene();
}

void Engine::drawBackground()
{
    if (!scene.background)
        return;

    // scale and draw background to fill the framebuffer
    int bgWidth = scene.background->get_width();
    int bgHeight = scene.background->get_height();

    for (int y = 0; y < renderHeight; y++)
    {
        for (int x = 0; x < renderWidth; x++)
        {
            // Map framebuffer coordinates to background coordinates
            int bgX = (x * bgWidth) / renderWidth;
            int bgY = (y * bgHeight) / renderHeight;

            // Clamp to background bounds
            bgX = std::max(0, std::min(bgWidth - 1, bgX));
            bgY = std::max(0, std::min(bgHeight - 1, bgY));

            TGAColor bgColor = scene.background->get(bgX, bgY);
            framebuffer.set(x, y, bgColor);
        }
    }
}
// // single pass versin
// void Engine::renderScene()
// {
//     std::vector<SceneNode *> visibleMeshes;
//     scene.getVisibleMeshNodes(visibleMeshes);

//     if (visibleMeshes.empty())
//     {
//         return;
//     }

//     std::cout << "Rendering " << visibleMeshes.size() << " objects..." << std::endl;

//     // Set up camera
//     lookat(scene.camera.position, scene.camera.target, scene.camera.up);
//     viewport(renderWidth / 8, renderHeight / 8, renderWidth * 3 / 4, renderHeight * 3 / 4);
//     projection(scene.camera.fov);

//     // Store original ModelView
//     Matrix originalModelView = ModelView;

//     // SIMPLIFIED: Skip shadow mapping for now - just render objects directly
//     {
//         // Clear Z-buffer only (preserve background)
//         for (int i = 0; i < renderWidth * renderHeight; i++)
//         {
//             zbuffer.set(i % renderWidth, i / renderWidth, TGAColor(0));
//         }

//         // Simple shader without shadows for debugging
//         for (SceneNode *meshNode : visibleMeshes)
//         {
//             Model *model = meshNode->model;
//             Matrix nodeTransform = meshNode->getWorldMatrix();
//             Matrix currentModelView = originalModelView * nodeTransform;

//             // Set global shader variables
//             ::model = model;
//             light_dir = scene.light.direction;

//             // Use a simple shader instead of shadow mapping
//             struct SimpleShader : public IShader
//             {
//                 Matrix modelView;
//                 Matrix normalTransform;
//                 mat<2, 3, float> varying_uv;

//                 SimpleShader(Matrix mv) : modelView(mv), normalTransform(mv.invert_transpose()) {}

//                 virtual Vec4f vertex(int iface, int nthvert)
//                 {
//                     varying_uv.set_col(nthvert, ::model->uv(iface, nthvert));
//                     Vec4f gl_Vertex = Viewport * Projection * modelView * embed<4>(::model->vert(iface, nthvert));
//                     return gl_Vertex;
//                 }

//                 virtual bool fragment(Vec3f bar, TGAColor &color)
//                 {
//                     Vec2f uv = varying_uv * bar;

//                     // Simple diffuse lighting
//                     Vec3f n = proj<3>(normalTransform * embed<4>(::model->normal(uv))).normalize();
//                     Vec3f l = proj<3>(modelView * embed<4>(light_dir)).normalize();
//                     float diff = std::max(0.f, n * l);

//                     TGAColor c = ::model->diffuse(uv);
//                     for (int i = 0; i < 3; i++)
//                     {
//                         color[i] = std::min<float>(20 + c[i] * (0.3f + 0.7f * diff), 255);
//                     }
//                     return false;
//                 }
//             };

//             SimpleShader shader(currentModelView);
//             Matrix oldModelView = ModelView;
//             ModelView = currentModelView;

//             std::cout << "Rendering " << meshNode->name << " with " << model->nfaces() << " faces..." << std::endl;

//             for (int i = 0; i < model->nfaces(); i++)
//             {
//                 Vec4f screen_coords[3];
//                 for (int j = 0; j < 3; j++)
//                 {
//                     screen_coords[j] = shader.vertex(i, j);
//                 }
//                 triangle(screen_coords, shader, framebuffer, zbuffer);

//                 // Progress indicator for large models
//                 if (i % 1000 == 0 && i > 0)
//                 {
//                     std::cout << "  Face " << i << "/" << model->nfaces() << std::endl;
//                 }
//             }

//             ModelView = oldModelView;
//             std::cout << "Finished rendering " << meshNode->name << std::endl;
//         }
//     }

//     ModelView = originalModelView;
//     std::cout << "Frame rendering complete!" << std::endl;
// }
// TWO PASS RENDER PART!!
void Engine::renderScene()
{
    // get all visible mesh nodes instead of single animated model
    std::vector<SceneNode *> visibleMeshes;
    scene.getVisibleMeshNodes(visibleMeshes);

    if (visibleMeshes.empty())
    {
        return;
    }

    // set up camera
    lookat(scene.camera.position, scene.camera.target, scene.camera.up);
    viewport(renderWidth / 8, renderHeight / 8, renderWidth * 3 / 4, renderHeight * 3 / 4);
    projection(scene.camera.fov);

    // store original ModelView
    Matrix originalModelView = ModelView;

    // PASS 1: Shadow mapping
    std::fill(shadowbuffer.begin(), shadowbuffer.end(), std::numeric_limits<float>::max());

    Matrix M;
    {
        // render from light's perspective for shadow mapping
        lookat(scene.light.direction, Vec3f(0, 0, 0), scene.camera.up);
        viewport(renderWidth / 8, renderHeight / 8, renderWidth * 3 / 4, renderHeight * 3 / 4);
        projection(0); // Orthographic for directional light

        M = Viewport * Projection * ModelView;

        // create temporary buffers for shadow pass
        TGAImage tempFrame(renderWidth, renderHeight, TGAImage::RGB);
        TGAImage tempZ(renderWidth, renderHeight, TGAImage::GRAYSCALE);

        // render all visible meshes for shadows
        for (SceneNode *meshNode : visibleMeshes)
        {
            Model *model = meshNode->model;
            Matrix nodeTransform = meshNode->getWorldMatrix();
            Matrix shadowModelView = ModelView * nodeTransform;

            // Set global shader variables -- this is also a temporary solution
            ::model = model; // global variable for shaders

            DepthShader depthShader;
            Matrix oldModelView = ModelView;
            ModelView = shadowModelView;

            for (int i = 0; i < model->nfaces(); i++)
            {
                Vec4f screen_coords[3];
                for (int j = 0; j < 3; j++)
                {
                    screen_coords[j] = depthShader.vertex(i, j);
                }
                triangle(screen_coords, depthShader, tempFrame, tempZ);
            }

            ModelView = oldModelView;
        }
    }

    // PASS 2: Main rendering with shadows (preserve background)
    {
        // restore camera perspective
        lookat(scene.camera.position, scene.camera.target, scene.camera.up);
        viewport(renderWidth / 8, renderHeight / 8, renderWidth * 3 / 4, renderHeight * 3 / 4);
        projection(scene.camera.fov);
        ModelView = originalModelView;

        // only clear the Z-buffer, keep the background in framebuffer
        for (int i = 0; i < renderWidth * renderHeight; i++)
        {
            zbuffer.set(i % renderWidth, i / renderWidth, TGAColor(0));
        }

        // render ALL visible meshes
        for (SceneNode *meshNode : visibleMeshes)
        {
            Model *model = meshNode->model;
            Matrix nodeTransform = meshNode->getWorldMatrix();
            Matrix currentModelView = originalModelView * nodeTransform;

            Matrix current_transform = Viewport * Projection * currentModelView;
            Matrix shadow_transform = M * (current_transform.adjugate() / current_transform.det());

            // set global shader variables -- temporary solution until i figure something out better
            ::model = model;
            light_dir = scene.light.direction;

            ShadowMappingShader shader(currentModelView,
                                       (Projection * currentModelView).invert_transpose(),
                                       shadow_transform);

            Matrix oldModelView = ModelView;
            ModelView = currentModelView;

            for (int i = 0; i < model->nfaces(); i++)
            {
                Vec4f screen_coords[3];
                for (int j = 0; j < 3; j++)
                {
                    screen_coords[j] = shader.vertex(i, j);
                }
                triangle(screen_coords, shader, framebuffer, zbuffer);
            }

            ModelView = oldModelView;
        }
    }

    // restore original ModelView
    ModelView = originalModelView;
}

void Engine::present()
{
    // convert TGA framebuffer to SDL texture
    void *pixels;
    int pitch;

    if (SDL_LockTexture(frameTexture, NULL, &pixels, &pitch) == 0)
    {
        unsigned char *tgaData = framebuffer.buffer();
        unsigned char *sdlPixels = (unsigned char *)pixels;

        // convert BGR to RGB and flip vertically
        for (int y = 0; y < renderHeight; y++)
        {
            for (int x = 0; x < renderWidth; x++)
            {
                int tgaIndex = ((renderHeight - 1 - y) * renderWidth + x) * 3;
                int sdlIndex = (y * renderWidth + x) * 3;

                sdlPixels[sdlIndex + 0] = tgaData[tgaIndex + 2]; // R
                sdlPixels[sdlIndex + 1] = tgaData[tgaIndex + 1]; // G
                sdlPixels[sdlIndex + 2] = tgaData[tgaIndex + 0]; // B
            }
        }

        SDL_UnlockTexture(frameTexture);
    }

    // render to screen
    SDL_SetRenderDrawColor(sdlRenderer, 0, 0, 0, 255);
    SDL_RenderClear(sdlRenderer);

    // scale texture to fit window
    SDL_Rect dstRect;
    float scaleX = (float)windowWidth / renderWidth;
    float scaleY = (float)windowHeight / renderHeight;
    float scale = std::min(scaleX, scaleY);

    dstRect.w = (int)(renderWidth * scale);
    dstRect.h = (int)(renderHeight * scale);
    dstRect.x = (windowWidth - dstRect.w) / 2;
    dstRect.y = (windowHeight - dstRect.h) / 2;

    SDL_RenderCopy(sdlRenderer, frameTexture, NULL, &dstRect);

    SDL_RenderPresent(sdlRenderer);
}

void Engine::captureFrame(const std::string &filename)
{
    framebuffer.flip_vertically();
    framebuffer.write_tga_file(filename.c_str());
    framebuffer.flip_vertically(); // Flip back for next frame
}

void Engine::captureSequence(const std::string &baseName, int frameCount, float duration)
{
    std::cout << "Capturing " << frameCount << " frames over " << duration << " seconds..." << std::endl;

    Vec3f originalPos = scene.camera.position;
    float angleStep = 2.0f * M_PI / frameCount;
    float radius = (scene.camera.position - scene.camera.target).norm();

    for (int frame = 0; frame < frameCount; frame++)
    {
        // rotate camera around target
        float angle = frame * angleStep;
        scene.camera.position.x = scene.camera.target.x + radius * cos(angle);
        scene.camera.position.z = scene.camera.target.z + radius * sin(angle);

        render();

        std::ostringstream filename;
        filename << baseName << "_" << std::setfill('0') << std::setw(4) << frame << ".tga";
        captureFrame(filename.str());

        std::cout << "Frame " << (frame + 1) << "/" << frameCount << " captured" << std::endl;
    }

    scene.camera.position = originalPos;
    std::cout << "Sequence capture complete!" << std::endl;
}

void Engine::updateWindowTitle()
{
    if (showStats)
    {
        std::ostringstream title;
        title << "Multi-Object 3D Engine - FPS: " << std::fixed << std::setprecision(1) << getFPS()
              << " | Objects: " << scene.getMeshCount();

        SceneNode *selected = scene.getSelectedNode();
        if (selected)
        {
            title << " | Selected: " << selected->name;
            if (selected->hasModel())
            {
                title << " [MESH]";
            }
            else
            {
                title << " [EMPTY]";
            }

            // show transform info for selected object
            Vec3f pos = selected->localTransform.position;
            title << " | Pos:(" << std::setprecision(1)
                  << pos.x << "," << pos.y << "," << pos.z << ")";
        }

        title << " | Light: (" << std::setprecision(2)
              << scene.light.direction.x << ", "
              << scene.light.direction.y << ", "
              << scene.light.direction.z << ")";

        title << " | Camera: " << (orbitMode ? "Orbit" : "Free-look");

        SDL_SetWindowTitle(window, title.str().c_str());
    }
}
// die

void Engine::shutdown()
{
    if (frameTexture)
    {
        SDL_DestroyTexture(frameTexture);
        frameTexture = nullptr;
    }

    if (sdlRenderer)
    {
        SDL_DestroyRenderer(sdlRenderer);
        sdlRenderer = nullptr;
    }

    if (window)
    {
        SDL_DestroyWindow(window);
        window = nullptr;
    }

    SDL_Quit();
}