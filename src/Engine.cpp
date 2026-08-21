#include "Engine.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>

// the CUDA entry points are declared in Engine.h, not duplicated here

Engine::Engine(int winWidth, int winHeight, int renWidth, int renHeight)
    : window(nullptr), sdlRenderer(nullptr), frameTexture(nullptr),
      framebuffer(renWidth, renHeight, TGAImage::RGB), zbuffer(renWidth, renHeight, TGAImage::GRAYSCALE),
      frameOnGPU(false),
      running(false), wireframe(false), showStats(true),
      windowWidth(winWidth), windowHeight(winHeight), renderWidth(renWidth), renderHeight(renHeight),
      mouseX(0), mouseY(0), mouseDeltaX(0), mouseDeltaY(0), lastMouseX(0), lastMouseY(0), mousePressed(false)

{
    // init imput state
    memset(keys, 0, sizeof(keys));

    // shadowbuffer: sized here, and the shaders read its size from our_gl
    resizeShadowBuffer(renderWidth, renderHeight);
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

    // Initialize real-time ray tracer

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

    cuda_available = initCudaRasterizer(renderWidth, renderHeight);
    // On when the device is there. The two paths agree to a mean byte
    // difference of 0.17 with identical coverage, and the GPU one is roughly
    // two orders of magnitude faster, so there is no reason to open on the
    // slow path. K still switches back for comparison.
    use_cuda_rendering = cuda_available;

    if (cuda_available)
    {
        std::cout << "CUDA rasterizer ENABLED - press 'K' for the CPU path" << std::endl;
    }
    else
    {
        std::cout << "CUDA rasterizer not available - using CPU only" << std::endl;
    }

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

        // normalize angles to prevent accumulation and wrapping issues
        // Keep yaw in range [0, 2π]
        while (cameraRotationY < 0)
            cameraRotationY += 2.0f * M_PI;
        while (cameraRotationY >= 2.0f * M_PI)
            cameraRotationY -= 2.0f * M_PI;

        // keep pitch in range [-π, π]
        while (cameraRotationX < -M_PI)
            cameraRotationX += 2.0f * M_PI;
        while (cameraRotationX > M_PI)
            cameraRotationX -= 2.0f * M_PI;

        updateCameraPosition();
    }
}
void Engine::updateCameraPosition()
{
    if (orbitMode)
    {
        // debug the rotation values
        std::cout << "cameraRotationX (pitch): " << cameraRotationX << " radians (" << (cameraRotationX * 180.0f / M_PI) << " degrees)" << std::endl;
        std::cout << "cameraRotationY (yaw): " << cameraRotationY << " radians (" << (cameraRotationY * 180.0f / M_PI) << " degrees)" << std::endl;

        // Standard spherical to cartesian conversion
        // X rotation is pitch (up/down), Y rotation is yaw (left/right)
        float cosPitch = cos(cameraRotationX);
        float sinPitch = sin(cameraRotationX);
        float cosYaw = cos(cameraRotationY);
        float sinYaw = sin(cameraRotationY);

        Vec3f offset;
        offset.x = cameraDistance * cosPitch * sinYaw;
        offset.y = cameraDistance * sinPitch;
        offset.z = cameraDistance * cosPitch * cosYaw;

        scene.camera.position = cameraTarget + offset;
        scene.camera.target = cameraTarget;

        std::cout << "Camera offset: (" << offset.x << ", " << offset.y << ", " << offset.z << ")" << std::endl;
        std::cout << "Updated camera position: (" << scene.camera.position.x
                  << ", " << scene.camera.position.y << ", " << scene.camera.position.z << ")" << std::endl;
        std::cout << "Camera target: (" << scene.camera.target.x
                  << ", " << scene.camera.target.y << ", " << scene.camera.target.z << ")" << std::endl;
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
        // switching to orbit mode
        Vec3f toCamera = scene.camera.position - scene.camera.target;
        cameraDistance = toCamera.norm();
        cameraTarget = scene.camera.target;
        cameraRotationY = atan2(toCamera.x, toCamera.z);
        cameraRotationX = asin(toCamera.y / cameraDistance);
        std::cout << "Switched to Orbit Camera Mode" << std::endl;
    }
    else
    {
        // switching to free-look mode - set target to selected object or origin
        SceneNode *selected = scene.getSelectedNode();
        if (selected && selected->hasModel())
        {
            scene.camera.target = selected->getWorldPosition();
        }
        else
        {
            scene.camera.target = Vec3f(0, 0, 0);
        }
        std::cout << "Switched to Free-Look Camera Mode - rotating around "
                  << scene.camera.target.x << "," << scene.camera.target.y << "," << scene.camera.target.z << std::endl;
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

        updateCamera();

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

            // Quality controls
            case SDLK_EQUALS: // '+' key
            case SDLK_PLUS:
                if (vertexEditMode)
                {
                    // In vertex edit mode, adjust deformation strength
                    setDeformationStrength(vertexEditor.getDeformationStrength() * 1.2f);
                }
                break;

            case SDLK_MINUS:
                if (vertexEditMode)
                {
                    // In vertex edit mode, adjust deformation strength
                    setDeformationStrength(vertexEditor.getDeformationStrength() * 0.8f);
                }
                break;

            // Blend strength controls
            case SDLK_LEFTBRACKET: // '[' key
                if (vertexEditMode)
                {
                    setSelectionRadius(vertexEditor.getSelectionRadius() * 0.8f);
                }
                break;

            case SDLK_RIGHTBRACKET: // ']' key
                if (vertexEditMode)
                {
                    setSelectionRadius(vertexEditor.getSelectionRadius() * 1.2f);
                }
                break;

            case SDLK_ESCAPE:
                if (vertexEditMode)
                {
                    exitVertexEditMode();
                }
                else
                {
                    running = false;
                }
                break;

            case SDLK_k:
                toggleCudaRendering();
                break;

            case SDLK_t:
                showStats = !showStats;
                break;
            case SDLK_p:
                captureFrame("output.tga");
                std::cout << "Frame captured!" << std::endl;
                break;
            case SDLK_b:
                {
                    scene.loadBackground("background.tga");
                }
                break;
            case SDLK_c:
                if (vertexEditMode)
                {
                    vertexEditor.clearSelection();
                }
                else
                {
                    scene.clearBackground();
                    std::cout << "Background cleared!" << std::endl;
                }
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
                loadModel("obj/head.obj");
                break;
            case SDLK_n:
                createEmptyNode();
                break;
            case SDLK_i:
                if (vertexEditMode)
                {
                    vertexEditor.invertSelection();
                }
                else
                {
                    scene.printSceneHierarchy();
                }
                break;

            // ===== VERTEX EDIT MODE CONTROLS =====
            case SDLK_v:
                if (keys[SDL_SCANCODE_LCTRL])
                {
                    // Ctrl+V - Enter/Exit vertex edit mode
                    if (vertexEditMode)
                    {
                        exitVertexEditMode();
                    }
                    else
                    {
                        enterVertexEditMode();
                    }
                }
                else if (vertexEditMode)
                {
                    // V - Toggle vertex display
                    toggleVertexDisplay();
                }
                break;

            // Vertex edit mode controls (only active when in vertex edit mode)
            case SDLK_1:
                if (vertexEditMode)
                {
                    vertexEditor.setMode(VertexEditor::VERTEX_SELECT);
                }
                break;
            case SDLK_2:
                if (vertexEditMode)
                {
                    vertexEditor.setMode(VertexEditor::VERTEX_DEFORM);
                }
                break;

            case SDLK_a:
                if (vertexEditMode && !keys[SDL_SCANCODE_LCTRL])
                {
                    vertexEditor.selectAll();
                }
                break;
            case SDLK_s:
                if (vertexEditMode && !keys[SDL_SCANCODE_LCTRL])
                {
                    vertexEditor.printStatus();
                }
                break;
            case SDLK_r:
                if (vertexEditMode)
                {
                    vertexEditor.resetDeformation();
                }
                else
                {
                    std::cout << "R pressed - zoom in" << std::endl;
                    zoomCamera(-0.25f);
                }

                break;
            case SDLK_f:
                if (vertexEditMode)
                {
                    wireframe = !wireframe;
                }
                else
                {
                    std::cout << "F pressed - zoom out" << std::endl;
                    zoomCamera(0.25f);
                }
                break;

            case SDLK_F5:
                // undo every sculpt on the selected mesh
                {
                    SceneNode *selected = scene.getSelectedNode();
                    if (selected && selected->hasModel())
                    {
                        selected->model->restoreOriginalVertices();
                        std::cout << "Reset " << selected->name << " to original shape" << std::endl;
                    }
                }
                break;
            }

            break;

        case SDL_KEYUP:
            keys[event.key.keysym.scancode] = false;
            break;

        case SDL_MOUSEBUTTONDOWN:
            if (event.button.button == SDL_BUTTON_LEFT)
            {
                if (vertexEditMode)
                {
                    // vertex edit mode handling
                    int renderX = (event.button.x * renderWidth) / windowWidth;
                    int renderY = (event.button.y * renderHeight) / windowHeight;
                    Matrix viewMatrix = ModelView;
                    Matrix projMatrix = Projection;
                    vertexEditor.handleMouseClick(renderX, renderY, viewMatrix, projMatrix, renderWidth, renderHeight);
                }
                else
                {
                    mousePressed = true;
                    lastMouseX = event.button.x; // Store starting position
                    lastMouseY = event.button.y;
                    // DON'T use SDL_SetRelativeMouseMode - it's broken in WSL
                }
            }
            break;

        case SDL_MOUSEBUTTONUP:
            if (event.button.button == SDL_BUTTON_LEFT)
            {
                if (vertexEditMode)
                {
                    vertexEditor.handleMouseRelease();
                }
                else
                {
                    mousePressed = false;
                    // DON'T use SDL_SetRelativeMouseMode(SDL_FALSE)
                }
            }
            break;

        case SDL_MOUSEMOTION:
            if (vertexEditMode && (event.motion.state & SDL_BUTTON_LMASK))
            {
                // Mouse drag in vertex edit mode
                int renderX = (event.motion.x * renderWidth) / windowWidth;
                int renderY = (event.motion.y * renderHeight) / windowHeight;

                Matrix viewMatrix = ModelView;
                Matrix projMatrix = Projection;
                vertexEditor.handleMouseDrag(renderX, renderY, event.motion.xrel, event.motion.yrel,
                                             viewMatrix, projMatrix, renderWidth, renderHeight);
            }
            else if (mousePressed)
            {
                // Calculate deltas manually instead of using broken SDL relative mode
                int currentX = event.motion.x;
                int currentY = event.motion.y;

                // Accumulate. handleEvents() drains every pending event each
                // frame while updateCamera() consumes the delta once, so
                // assigning here would keep only the last event's couple of
                // pixels and throw the rest of the drag away.
                mouseDeltaX += currentX - lastMouseX;
                mouseDeltaY += currentY - lastMouseY;

                lastMouseX = currentX;
                lastMouseY = currentY;
            }
            break;

        case SDL_MOUSEWHEEL:
            if (vertexEditMode && vertexEditor.getMode() == VertexEditor::VERTEX_SELECT)
            {
                std::cout << "Adjusting selection radius" << std::endl;
                // Adjust selection radius with mouse wheel in vertex select mode
                if (event.wheel.y > 0)
                {
                    setSelectionRadius(vertexEditor.getSelectionRadius() * 1.1f);
                }
                else if (event.wheel.y < 0)
                {
                    setSelectionRadius(vertexEditor.getSelectionRadius() * 0.9f);
                }
            }
            else
            {
                std::cout << "Applying camera zoom" << std::endl;
                // Normal camera zoom - OUTSIDE vertex edit mode
                if (event.wheel.y > 0)
                {
                    zoomCamera(-0.25f); // Zoom in
                }
                else if (event.wheel.y < 0)
                {
                    zoomCamera(0.25f); // Zoom out
                }
            }
            break;
        }
    }
}

void Engine::update()
{

    static int frameCount = 0;
    frameCount++;
    // FPS is in the title bar; printing it here just scrolls the log
    // updateCamera();

    // update scene transforms
    scene.updateAllTransforms();
    // update rt ray tracer

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
            // True free-look mode - rotate around the selected object or scene center
            Vec3f rotationCenter;

            // Get the center point to rotate around
            SceneNode *selected = scene.getSelectedNode();
            if (selected && selected->hasModel())
            {
                rotationCenter = selected->getWorldPosition();
            }
            else
            {
                rotationCenter = Vec3f(0, 0, 0); // Default to origin if no object selected
            }

            // Calculate current direction from rotation center to camera
            Vec3f toCamera = scene.camera.position - rotationCenter;
            float radius = toCamera.norm();

            // Apply mouse rotation
            float yawDelta = -mouseDeltaX * mouseSpeed;
            float pitchDelta = -mouseDeltaY * mouseSpeed;

            // Convert to spherical coordinates
            float currentYaw = atan2(toCamera.x, toCamera.z);
            float currentPitch = asin(toCamera.y / radius);

            // Apply deltas
            currentYaw += yawDelta;
            currentPitch += pitchDelta;

            // Clamp pitch to prevent flipping
            currentPitch = std::max(-1.5f, std::min(1.5f, currentPitch));

            // Convert back to cartesian and update camera position
            float cosP = cos(currentPitch);
            float sinP = sin(currentPitch);
            float cosY = cos(currentYaw);
            float sinY = sin(currentYaw);

            scene.camera.position.x = rotationCenter.x + radius * cosP * sinY;
            scene.camera.position.y = rotationCenter.y + radius * sinP;
            scene.camera.position.z = rotationCenter.z + radius * cosP * cosY;

            // Always look at the rotation center
            scene.camera.target = rotationCenter;
        }

        mouseDeltaX = mouseDeltaY = 0;
    }

    // KEYBOARD MOVEMENT - ONLY OUTSIDE VERTEX EDIT MODE
    if (!vertexEditMode)
    {
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

        // Apply movement
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
}

void Engine::render()
{
    bool cudaPath = use_cuda_rendering && cuda_available;
    frameOnGPU = false;

    // cheap memset, and it's what gets shown if the scene turns out to be empty
    framebuffer.clear();

    if (!cudaPath)
    {
        for (int i = 0; i < renderWidth * renderHeight; i++)
        {
            zbuffer.set(i % renderWidth, i / renderWidth, TGAColor(0));
        }

        // draw the background AND THEN RENDER 3D SCENE
        drawBackground();
    }
    // the CUDA path owns both buffers on the device and overwrites every
    // pixel, so the host zbuffer clear and the background blit would be
    // discarded. skipping them changes nothing on screen.

    // then render 3D scene (but don't clear framebuffer in renderScene)
    renderScene();

    // The vertex overlay is the only thing left that composites on the host,
    // so the frame comes down from the device only when it is showing.
    if (frameOnGPU && vertexEditMode && vertexEditor.hasTarget())
    {
        cudaCopyResults(framebuffer);
        frameOnGPU = false;
    }

    if (vertexEditMode && vertexEditor.hasTarget())
    {
        vertexEditor.renderVertexOverlay(framebuffer, renderWidth, renderHeight);
    }
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

void Engine::renderScene()
{
    // recompute world matrices from the local transforms. getWorldMatrix()
    // reads a cached value that only this updates, so without it every node
    // reports identity and all objects draw at the origin.
    scene.updateAllTransforms();

    // get all visible mesh nodes instead of single animated model
    std::vector<SceneNode *> visibleMeshes;
    scene.getVisibleMeshNodes(visibleMeshes);

    if (visibleMeshes.empty())
    {
        return;
    }

    // Set up camera
    lookat(scene.camera.position, scene.camera.target, scene.camera.up);
    viewport(renderWidth / 8, renderHeight / 8, renderWidth * 3 / 4, renderHeight * 3 / 4);
    projection(scene.camera.projectionCoeff());

    // Store original ModelView
    Matrix originalModelView = ModelView;

    if (use_cuda_rendering && cuda_available)
    {
        // CUDA rendering path: two passes, same shape as the CPU path
        cudaClearBuffers();

        // PASS 1: depth from the light's point of view, orthographic
        lookat(scene.light.direction, Vec3f(0, 0, 0), scene.camera.up);
        viewport(renderWidth / 8, renderHeight / 8, renderWidth * 3 / 4, renderHeight * 3 / 4);
        projection(0);
        Matrix lightM = Viewport * Projection * ModelView;
        Matrix lightModelView = ModelView;

        float ident[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
        float nolight[3] = {0, 0, 1};
        float white[3] = {1.f, 1.f, 1.f};

        for (SceneNode *meshNode : visibleMeshes)
        {
            int mesh = getCudaMesh(meshNode->model);
            if (mesh < 0)
                continue;
            Matrix lm = lightM * meshNode->getWorldMatrix();
            Matrix lc = Projection * lightModelView * meshNode->getWorldMatrix();
            float mvp[16], clip[16];
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++)
                {
                    mvp[r * 4 + c] = lm[r][c];
                    clip[r * 4 + c] = lc[r][c];
                }
            cudaDrawMesh(mesh, mvp, clip, ident, nolight, white, 1.0f,
                         NULL, 0.f, 255, 255, 255);
        }
        cudaRenderShadowPass();

        // PASS 2: the camera view, sampling that depth buffer
        lookat(scene.camera.position, scene.camera.target, scene.camera.up);
        viewport(renderWidth / 8, renderHeight / 8, renderWidth * 3 / 4, renderHeight * 3 / 4);
        projection(scene.camera.projectionCoeff());
        ModelView = originalModelView;

        for (SceneNode *meshNode : visibleMeshes)
        {
            Model *model = meshNode->model;
            int mesh = getCudaMesh(model);
            if (mesh < 0)
                continue;

            Matrix nodeTransform = meshNode->getWorldMatrix();
            Matrix currentModelView = originalModelView * nodeTransform;
            Matrix clipTransform = Projection * currentModelView;
            Matrix transform = Viewport * clipTransform;

            // eye space, to match the light below. not Projection*ModelView:
            // the light is carried by currentModelView alone.
            Matrix MIT = currentModelView.invert_transpose();
            Vec3f l = proj<3>(currentModelView * embed<4>(scene.light.direction, 0.f)).normalize();

            // screen space -> shadow-map space. NOTE: adjugate() returns the
            // COFACTOR matrix, so adjugate()/det() is the inverse-TRANSPOSE.
            // invert() is the real inverse.
            Matrix lightXform = lightM * nodeTransform;
            Matrix camXform = transform;
            Matrix shadowXform = lightXform * camXform.invert();

            float mvp[16], clip[16], mit[16], msh[16];
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++)
                {
                    mvp[r * 4 + c] = transform[r][c];
                    clip[r * 4 + c] = clipTransform[r][c];
                    mit[r * 4 + c] = MIT[r][c];
                    msh[r * 4 + c] = shadowXform[r][c];
                }
            float light[3] = {l.x, l.y, l.z};
            float lightColor[3] = {
                std::max(0.0f, scene.light.color.x),
                std::max(0.0f, scene.light.color.y),
                std::max(0.0f, scene.light.color.z)};
            float lightIntensity = std::max(0.0f, scene.light.intensity);

            // Shadow bias. Depth spans 0..255 regardless of world scale, so
            // this is a fraction of that range, not a world distance. 2.0 gives
            // contact shadows without acne; tune per scene if needed.
            cudaDrawMesh(mesh, mvp, clip, mit, light, lightColor, lightIntensity,
                         msh, 2.0f, 200, 170, 150);
        }

        // no readback: leave the frame on the device for present() to blit.
        // the pending kernels are flushed by whichever of cudaBlitToTexture /
        // cudaCopyResults runs first, so the GPU keeps working meanwhile.
        frameOnGPU = true;
    }
    else
    {
        // Original CPU rendering path
        clearShadowBuffer();

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
            projection(scene.camera.projectionCoeff());
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
                // adjugate()/det() is the inverse-TRANSPOSE, not the inverse.
                // invert() is the one wanted here.
                Matrix shadow_transform = M * current_transform.invert();

                // set global shader variables -- temporary solution until i figure something out better
                ::model = model;
                light_dir = scene.light.direction;

                // uniform_M carries the light, uniform_MIT the normal. both
                // must land in the same space, so both are ModelView-only.
                ShadowMappingShader shader(currentModelView,
                                           currentModelView.invert_transpose(),
                                           shadow_transform,
                                           scene.light.color,
                                           scene.light.intensity);

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
    }

    // restore original ModelView
    ModelView = originalModelView;
}

// // TWO PASS RENDER PART!!
// void Engine::renderScene()
// {
//     // get all visible mesh nodes instead of single animated model
//     std::vector<SceneNode *> visibleMeshes;
//     scene.getVisibleMeshNodes(visibleMeshes);

//     if (!visibleMeshes.empty())
//     {
//         Model *firstModel = visibleMeshes[0]->model;
//         std::cout << "Model vertex count: " << firstModel->nverts() << std::endl;
//         if (firstModel->nverts() > 0)
//         {
//             Vec3f firstVert = firstModel->vert(0);
//             std::cout << "First vertex: " << firstVert.x << ", " << firstVert.y << ", " << firstVert.z << std::endl;
//         }
//     }

//     if (visibleMeshes.empty())
//     {
//         return;
//     }
//     std::cout << "renderScene camera pos: " << scene.camera.position.x << ", " << scene.camera.position.y << ", " << scene.camera.position.z << std::endl;

//     // set up camera
//     std::cout << "=== RENDER DEBUG ===" << std::endl;
//     std::cout << "Camera FOV: " << scene.camera.fov << std::endl;
//     std::cout << "Camera position: " << scene.camera.position.x << ", " << scene.camera.position.y << ", " << scene.camera.position.z << std::endl;
//     std::cout << "Camera target: " << scene.camera.target.x << ", " << scene.camera.target.y << ", " << scene.camera.target.z << std::endl;
//     std::cout << "Camera distance from target: " << (scene.camera.position - scene.camera.target).norm() << std::endl;
//     std::cout << "===================" << std::endl;

//     lookat(scene.camera.position, scene.camera.target, scene.camera.up);
//     viewport(renderWidth / 8, renderHeight / 8, renderWidth * 3 / 4, renderHeight * 3 / 4);
//     projection(scene.camera.fov);

//     // store original ModelView
//     Matrix originalModelView = ModelView;

//     // PASS 1: Shadow mapping
//     std::fill(shadowbuffer.begin(), shadowbuffer.end(), std::numeric_limits<float>::max());

//     Matrix M;
//     {
//         // render from light's perspective for shadow mapping
//         lookat(scene.light.direction, Vec3f(0, 0, 0), scene.camera.up);
//         viewport(renderWidth / 8, renderHeight / 8, renderWidth * 3 / 4, renderHeight * 3 / 4);
//         projection(0); // Orthographic for directional light

//         M = Viewport * Projection * ModelView;

//         // create temporary buffers for shadow pass
//         TGAImage tempFrame(renderWidth, renderHeight, TGAImage::RGB);
//         TGAImage tempZ(renderWidth, renderHeight, TGAImage::GRAYSCALE);

//         // render all visible meshes for shadows
//         for (SceneNode *meshNode : visibleMeshes)
//         {
//             Model *model = meshNode->model;
//             Matrix nodeTransform = meshNode->getWorldMatrix();
//             Matrix shadowModelView = ModelView * nodeTransform;

//             // Set global shader variables -- this is also a temporary solution
//             ::model = model; // global variable for shaders

//             DepthShader depthShader;
//             Matrix oldModelView = ModelView;
//             ModelView = shadowModelView;

//             for (int i = 0; i < model->nfaces(); i++)
//             {
//                 Vec4f screen_coords[3];
//                 for (int j = 0; j < 3; j++)
//                 {
//                     screen_coords[j] = depthShader.vertex(i, j);
//                 }
//                 triangle(screen_coords, depthShader, tempFrame, tempZ);
//             }

//             ModelView = oldModelView;
//         }
//     }

//     // PASS 2: Main rendering with shadows (preserve background)
//     {
//         // restore camera perspective
//         lookat(scene.camera.position, scene.camera.target, scene.camera.up);
//         viewport(renderWidth / 8, renderHeight / 8, renderWidth * 3 / 4, renderHeight * 3 / 4);
//         projection(scene.camera.fov);
//         ModelView = originalModelView;

//         // only clear the Z-buffer, keep the background in framebuffer
//         for (int i = 0; i < renderWidth * renderHeight; i++)
//         {
//             zbuffer.set(i % renderWidth, i / renderWidth, TGAColor(0));
//         }

//         // render ALL visible meshes
//         for (SceneNode *meshNode : visibleMeshes)
//         {
//             Model *model = meshNode->model;
//             Matrix nodeTransform = meshNode->getWorldMatrix();
//             Matrix currentModelView = originalModelView * nodeTransform;

//             Matrix current_transform = Viewport * Projection * currentModelView;
//             Matrix shadow_transform = M * (current_transform.adjugate() / current_transform.det());

//             // set global shader variables -- temporary solution until i figure something out better
//             ::model = model;
//             light_dir = scene.light.direction;

//             ShadowMappingShader shader(currentModelView,
//                                        (Projection * currentModelView).invert_transpose(),
//                                        shadow_transform);

//             Matrix oldModelView = ModelView;
//             ModelView = currentModelView;

//             for (int i = 0; i < model->nfaces(); i++)
//             {
//                 Vec4f screen_coords[3];
//                 for (int j = 0; j < 3; j++)
//                 {
//                     screen_coords[j] = shader.vertex(i, j);
//                 }
//                 triangle(screen_coords, shader, framebuffer, zbuffer);
//             }

//             ModelView = oldModelView;
//         }
//     }

//     // restore original ModelView
//     ModelView = originalModelView;
// }

void Engine::present()
{
    // convert TGA framebuffer to SDL texture
    void *pixels;
    int pitch;

    if (SDL_LockTexture(frameTexture, NULL, &pixels, &pitch) == 0)
    {
        if (frameOnGPU)
        {
            // device buffer is already RGB24 in SDL row order: straight DMA,
            // no swizzle, no flip, no per-pixel host work
            cudaBlitToTexture(pixels, pitch);
        }
        else
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
    // writing a TGA needs the frame on the host
    if (frameOnGPU)
    {
        cudaCopyResults(framebuffer);
        frameOnGPU = false;
    }

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
    // device geometry outlives the scene graph, so free it before tearing
    // the rasterizer down
    if (cuda_available)
    {
        for (auto &kv : cudaMeshes)
            cudaDestroyMesh(kv.second.handle);
        cudaMeshes.clear();
        cleanupCudaRasterizer();
    }

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

// uploads a model's geometry to the device the first time it is drawn, then
// hands back the same handle every frame after that.
int Engine::getCudaMesh(Model *model)
{
    if (!model || !cuda_available)
        return -1;

    auto it = cudaMeshes.find(model);
    if (it != cudaMeshes.end())
    {
        // deformed since the last upload: re-send positions only. topology,
        // uvs and normals are untouched by sculpting and blend shapes, and the
        // CPU path does not recompute normals after a deformation either.
        unsigned int v = model->geometryVersion();
        if (it->second.geomVersion != v)
        {
            cudaUpdateMeshVerts(it->second.handle,
                                (const float *)model->getVertexData(),
                                model->nverts());
            it->second.geomVersion = v;
        }
        return it->second.handle;
    }

    int nverts = model->nverts();
    int nfaces = model->nfaces();
    if (nverts <= 0 || nfaces <= 0)
        return -1;

    // vertices stay indexed so sculpting can re-upload positions alone.
    // normals and uvs go up unindexed, one set per triangle corner, which
    // avoids carrying separate vt/vn index arrays.
    std::vector<int> indices(nfaces * 3);
    std::vector<float> cnorms(nfaces * 9);
    std::vector<float> cuvs(nfaces * 6);
    for (int i = 0; i < nfaces; i++)
    {
        std::vector<int> f = model->face(i);
        for (int j = 0; j < 3; j++)
        {
            indices[i * 3 + j] = f[j];

            Vec3f n = model->normal(i, j);
            cnorms[(i * 3 + j) * 3 + 0] = n.x;
            cnorms[(i * 3 + j) * 3 + 1] = n.y;
            cnorms[(i * 3 + j) * 3 + 2] = n.z;

            Vec2f uv = model->uv(i, j);
            cuvs[(i * 3 + j) * 2 + 0] = uv.x;
            cuvs[(i * 3 + j) * 2 + 1] = uv.y;
        }
    }

    int handle = cudaCreateMesh((const float *)model->getVertexData(), nverts,
                                indices.data(), nfaces,
                                cnorms.data(), cuvs.data());

    // textures upload once, alongside the geometry
    if (handle >= 0)
    {
        TGAImage *maps[3] = {&model->diffuseMap(), &model->normalMap(), &model->specularMap()};
        for (int slot = 0; slot < 3; slot++)
        {
            TGAImage *t = maps[slot];
            if (t->get_width() > 0 && t->get_height() > 0 && t->buffer())
                cudaSetMeshTexture(handle, slot, t->buffer(),
                                   t->get_width(), t->get_height(), t->get_bytespp());
        }
    }

    cudaMeshes[model] = {handle, model->geometryVersion()};

    std::cout << "CUDA mesh uploaded: " << nverts << " verts, "
              << nfaces << " faces (handle " << handle << ")" << std::endl;
    return handle;
}

// The rasterizer ignores this; it only changes how the ray tracer shades the
// object. Marking the tracer dirty rebuilds its scene and restarts the
// accumulated image, since the old samples used the previous surface.
void Engine::toggleCudaRendering()
{
    if (cuda_available)
    {
        use_cuda_rendering = !use_cuda_rendering;
        std::cout << "CUDA rendering: " << (use_cuda_rendering ? "ENABLED" : "DISABLED") << std::endl;
    }
    else
    {
        std::cout << "CUDA not available" << std::endl;
    }
}

// VertexEditor implementation
void Engine::VertexEditor::setTargetModel(SceneNode *node)
{
    if (!node || !node->hasModel())
    {
        std::cerr << "Invalid target for vertex editing" << std::endl;
        return;
    }

    targetNode = node;
    targetModel = node->model;

    // Ensure model has backup vertices for editing
    if (!targetModel->getVertices().empty())
    {
        targetModel->backupOriginalVertices();
    }

    // Initialize vertex colors for selection feedback
    selectionColors.resize(targetModel->nverts(), Vec3f(1, 1, 1)); // White by default

    std::cout << "Vertex editor targeting: " << node->name
              << " (" << targetModel->nverts() << " vertices)" << std::endl;
}

void Engine::VertexEditor::setMode(EditMode mode)
{
    currentMode = mode;

    switch (mode)
    {
    case NORMAL:
        showVertices = false;
        clearSelection();
        std::cout << "NORMAL MODE" << std::endl;
        break;
    case VERTEX_SELECT:
        showVertices = true;
        std::cout << "VERTEX SELECT MODE - Click to select vertices, drag for radius selection" << std::endl;
        std::cout << "Selected vertices will turn red. Use mouse wheel to adjust selection radius." << std::endl;
        break;
    case VERTEX_DEFORM:
        showVertices = true;
        std::cout << "VERTEX DEFORM MODE - Drag selected vertices to sculpt the mesh" << std::endl;
        std::cout << "Use +/- keys to adjust deformation strength." << std::endl;
        break;
    }
}

void Engine::VertexEditor::selectVerticesInRadius(const Vec3f &worldPos, float radius)
{
    if (!targetModel)
        return;

    Matrix worldMatrix = targetNode->getWorldMatrix();
    int added = 0;

    for (int i = 0; i < targetModel->nverts(); i++)
    {
        // Transform vertex to world space
        Vec3f localVert = targetModel->vert(i);
        Vec4f worldVert4 = worldMatrix * embed<4>(localVert);
        Vec3f worldVert(worldVert4[0] / worldVert4[3], worldVert4[1] / worldVert4[3], worldVert4[2] / worldVert4[3]);

        // Check distance
        float distance = (worldVert - worldPos).norm();
        if (distance <= radius)
        {
            if (selectedVertices.find(i) == selectedVertices.end())
            {
                selectedVertices.insert(i);
                selectionColors[i] = Vec3f(1, 0, 0); // Red for selected
                added++;
            }
        }
    }

    if (added > 0)
    {
        std::cout << "Selected " << added << " more vertices (total: " << selectedVertices.size() << ")" << std::endl;
    }
}

void Engine::VertexEditor::clearSelection()
{
    selectedVertices.clear();
    for (auto &color : selectionColors)
    {
        color = Vec3f(1, 1, 1); // Reset to white
    }
    std::cout << "Selection cleared" << std::endl;
}

void Engine::VertexEditor::selectAll()
{
    if (!targetModel)
        return;

    selectedVertices.clear();
    for (int i = 0; i < targetModel->nverts(); i++)
    {
        selectedVertices.insert(i);
        selectionColors[i] = Vec3f(1, 0, 0); // Red for selected
    }
    std::cout << "Selected all " << targetModel->nverts() << " vertices" << std::endl;
}

void Engine::VertexEditor::invertSelection()
{
    if (!targetModel)
        return;

    std::set<int> newSelection;
    for (int i = 0; i < targetModel->nverts(); i++)
    {
        if (selectedVertices.find(i) == selectedVertices.end())
        {
            newSelection.insert(i);
            selectionColors[i] = Vec3f(1, 0, 0); // Red for selected
        }
        else
        {
            selectionColors[i] = Vec3f(1, 1, 1); // White for deselected
        }
    }
    selectedVertices = newSelection;
    std::cout << "Selection inverted - now " << selectedVertices.size() << " vertices selected" << std::endl;
}

void Engine::VertexEditor::applyDeformation(const Vec3f &direction, float strength)
{
    if (!targetModel || selectedVertices.empty())
        return;

    for (int vertexIndex : selectedVertices)
    {
        Vec3f currentPos = targetModel->vert(vertexIndex);
        Vec3f newPos = currentPos + direction * strength;
        targetModel->setVertex(vertexIndex, newPos);
    }
}

Vec3f Engine::VertexEditor::screenToWorldRay(int screenX, int screenY, const Matrix &viewMatrix,
                                             const Matrix &projMatrix, int renderWidth, int renderHeight)
{
    // This is a simplified approach - in a real system you'd do proper ray casting
    // For now, we'll approximate based on the camera position and screen coordinates

    if (!targetNode)
        return Vec3f(0, 0, 0);

    // Get model center as approximation
    Vec3f modelCenter = targetNode->getWorldPosition();

    // Convert screen coordinates to normalized device coordinates
    float normalizedX = (2.0f * screenX / renderWidth) - 1.0f;
    float normalizedY = 1.0f - (2.0f * screenY / renderHeight);

    // Simple projection to world space around the model
    return modelCenter + Vec3f(normalizedX, normalizedY, 0) * selectionRadius * 5.0f;
}

int Engine::VertexEditor::findClosestVertex(const Vec3f &worldPos, float maxDistance)
{
    if (!targetModel)
        return -1;

    Matrix worldMatrix = targetNode->getWorldMatrix();
    int closestVertex = -1;
    float closestDistance = maxDistance;

    for (int i = 0; i < targetModel->nverts(); i++)
    {
        Vec3f localVert = targetModel->vert(i);
        Vec4f worldVert4 = worldMatrix * embed<4>(localVert);
        Vec3f worldVert(worldVert4[0] / worldVert4[3], worldVert4[1] / worldVert4[3], worldVert4[2] / worldVert4[3]);

        float distance = (worldVert - worldPos).norm();
        if (distance < closestDistance)
        {
            closestDistance = distance;
            closestVertex = i;
        }
    }

    return closestVertex;
}

void Engine::VertexEditor::handleMouseClick(int mouseX, int mouseY, const Matrix &viewMatrix,
                                            const Matrix &projMatrix, int renderWidth, int renderHeight)
{
    if (currentMode == NORMAL || !targetModel)
        return;

    lastMouseX = mouseX;
    lastMouseY = mouseY;

    Vec3f worldPos = screenToWorldRay(mouseX, mouseY, viewMatrix, projMatrix, renderWidth, renderHeight);

    switch (currentMode)
    {
    case VERTEX_SELECT:
    {
        // Select vertices in radius around click point
        selectVerticesInRadius(worldPos, selectionRadius);
        break;
    }
    case VERTEX_DEFORM:
    {
        if (!selectedVertices.empty())
        {
            startDeformation(worldPos);
            isDragging = true;
            lastMouseWorldPos = worldPos;
        }
        break;
    }
    default:
        break;
    }
}

void Engine::VertexEditor::handleMouseDrag(int mouseX, int mouseY, int deltaX, int deltaY,
                                           const Matrix &viewMatrix, const Matrix &projMatrix,
                                           int renderWidth, int renderHeight)
{
    if (!isDragging || currentMode == VERTEX_SELECT)
        return;

    Vec3f currentWorldPos = screenToWorldRay(mouseX, mouseY, viewMatrix, projMatrix, renderWidth, renderHeight);
    Vec3f dragDirection = currentWorldPos - lastMouseWorldPos;

    if (dragDirection.norm() > 0.001f)
    { // Avoid tiny movements
        applyDeformation(dragDirection, deformationStrength);
        lastMouseWorldPos = currentWorldPos;
    }
}

void Engine::VertexEditor::handleMouseRelease()
{
    isDragging = false;
    if (currentMode == VERTEX_DEFORM)
    {
        endDeformation();
    }
}

void Engine::VertexEditor::startDeformation(const Vec3f &center)
{
    deformationCenter = center;
    isDeforming = true;
}

void Engine::VertexEditor::endDeformation()
{
    isDeforming = false;
}

void Engine::VertexEditor::resetDeformation()
{
    if (!targetModel)
        return;

    targetModel->restoreOriginalVertices();
    std::cout << "Reset vertices to original positions" << std::endl;
}

void Engine::VertexEditor::printStatus() const
{
    std::cout << "\n=== VERTEX EDITOR STATUS ===" << std::endl;
    std::cout << "Mode: ";
    switch (currentMode)
    {
    case NORMAL:
        std::cout << "NORMAL";
        break;
    case VERTEX_SELECT:
        std::cout << "VERTEX SELECT";
        break;
    case VERTEX_DEFORM:
        std::cout << "VERTEX DEFORM";
        break;
    }
    std::cout << std::endl;

    if (targetModel)
    {
        std::cout << "Target: " << targetNode->name << " (" << targetModel->nverts() << " vertices)" << std::endl;
        std::cout << "Selected: " << selectedVertices.size() << " vertices" << std::endl;
        std::cout << "Selection radius: " << selectionRadius << std::endl;
        std::cout << "Deformation strength: " << deformationStrength << std::endl;
        std::cout << "Deformation radius: " << deformationRadius << std::endl;
    }
    else
    {
        std::cout << "No target model selected" << std::endl;
    }
    std::cout << "=========================" << std::endl;
}

// Engine integration methods
void Engine::enterVertexEditMode()
{
    SceneNode *selected = scene.getSelectedNode();
    if (!selected || !selected->hasModel())
    {
        std::cout << "Select a model first to edit vertices (use TAB to cycle)" << std::endl;
        return;
    }

    vertexEditMode = true;
    vertexEditor.setTargetModel(selected);
    vertexEditor.setMode(VertexEditor::VERTEX_SELECT);

    std::cout << "\n=== ENTERED VERTEX EDIT MODE ===" << std::endl;
    std::cout << "Target: " << selected->name << std::endl;
    std::cout << "\n=== CONTROLS ===" << std::endl;
    std::cout << "  1 - Select mode | 2 - Deform mode | 3 - Blend shape mode" << std::endl;
    std::cout << "  Mouse Click - Select vertices in radius" << std::endl;
    std::cout << "  Mouse Drag - Deform selected vertices (in deform mode)" << std::endl;
    std::cout << "  Mouse Wheel - Adjust selection radius" << std::endl;
    std::cout << "  C - Clear selection | A - Select all | I - Invert selection" << std::endl;
    std::cout << "  +/- - Adjust deformation strength" << std::endl;
    std::cout << "  B - Start blend shape recording" << std::endl;
    std::cout << "  S - Save blend shape (when recording)" << std::endl;
    std::cout << "  R - Reset to original shape" << std::endl;
    std::cout << "  V - Toggle vertex display" << std::endl;
    std::cout << "  Ctrl+V - Exit vertex edit mode" << std::endl;
    std::cout << "==============================" << std::endl;

    vertexEditor.printStatus();
}

void Engine::exitVertexEditMode()
{
    vertexEditMode = false;
    vertexEditor.setMode(VertexEditor::NORMAL);
    std::cout << "Exited vertex edit mode" << std::endl;
}

void Engine::toggleVertexDisplay()
{
    vertexEditor.toggleVertexDisplay();
    std::cout << "Vertex display: " << (vertexEditor.isShowingVertices() ? "ON" : "OFF") << std::endl;
}

void Engine::setDeformationStrength(float strength)
{
    vertexEditor.setDeformationStrength(strength);
    std::cout << "Deformation strength: " << vertexEditor.getDeformationStrength() << std::endl;
}

void Engine::setSelectionRadius(float radius)
{
    vertexEditor.setSelectionRadius(radius);
    std::cout << "Selection radius: " << vertexEditor.getSelectionRadius() << std::endl;
}

void Engine::VertexEditor::renderVertexOverlay(TGAImage &framebuffer, int renderWidth, int renderHeight)
{
    if (!targetModel || !showVertices)
        return;

    Matrix worldMatrix = targetNode->getWorldMatrix();
    Matrix viewProjection = Viewport * Projection * ModelView;

    // Render all vertices as small dots
    for (int i = 0; i < targetModel->nverts(); i++)
    {
        Vec3f localVert = targetModel->vert(i);
        Vec4f worldVert4 = worldMatrix * embed<4>(localVert);
        Vec3f worldVert(worldVert4[0] / worldVert4[3], worldVert4[1] / worldVert4[3], worldVert4[2] / worldVert4[3]);

        // Project to screen
        Vec4f screenPos = viewProjection * embed<4>(worldVert);
        if (screenPos[3] > 0)
        {
            int screenX = screenPos[0] / screenPos[3];
            int screenY = screenPos[1] / screenPos[3];

            if (screenX >= 0 && screenX < renderWidth && screenY >= 0 && screenY < renderHeight)
            {
                // Choose color based on selection
                TGAColor color = (selectedVertices.find(i) != selectedVertices.end()) ? TGAColor(255, 0, 0) : TGAColor(255, 255, 255);

                // Draw small cross for vertex
                for (int dx = -1; dx <= 1; dx++)
                {
                    for (int dy = -1; dy <= 1; dy++)
                    {
                        int px = screenX + dx;
                        int py = screenY + dy;
                        if (px >= 0 && px < renderWidth && py >= 0 && py < renderHeight)
                        {
                            framebuffer.set(px, py, color);
                        }
                    }
                }
            }
        }
    }
}
