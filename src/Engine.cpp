#include "Engine.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>

// the CUDA entry points are declared in Engine.h, not duplicated here

Engine::Engine(int winWidth, int winHeight, int renWidth, int renHeight)
    : window(nullptr), sdlRenderer(nullptr), frameTexture(nullptr),
      wantGLPresent(false),
      captureStaging(renWidth, renHeight, TGAImage::RGB),
      uploadedBackgroundVersion(-1), cachedGeometryVersion(0),
      running(false), showStats(true), ssaaFactor(2),
      ssaoEnabled(true), ssaoRadius(0.18f), ssaoIntensity(0.85f), ssaoDebug(0),
      windowWidth(winWidth), windowHeight(winHeight), renderWidth(renWidth), renderHeight(renHeight),
      mouseX(0), mouseY(0), mouseDeltaX(0), mouseDeltaY(0), lastMouseX(0), lastMouseY(0), mousePressed(false),
      cameraRotationX(0.0f), cameraRotationY(0.0f), orbitMode(true)

{
    // init imput state
    memset(keys, 0, sizeof(keys));

    // Derived from the camera rather than hardcoded, so the orbit state agrees
    // with where the camera actually starts. At yaw 0 / pitch 0 the orbit
    // offset is (0, 0, cameraDistance), which is the Camera default, so the
    // first drag continues from the opening view instead of jumping.
    cameraTarget = scene.camera.target;
    cameraDistance = (scene.camera.position - scene.camera.target).norm();
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

    // Ask for a GL-capable window first, because the interop present path
    // needs one and the flag can only be set at creation. Headless (SDL's
    // dummy video driver) refuses it, which is not an error: the SDL_Renderer
    // fallback below handles that case and the tests run through it.
    const Uint32 baseFlags = SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE;
    const char *title = "MULTI OBJECT 3D ENGINE THIS BETTER WORK!!!";

    window = SDL_CreateWindow(title,
                              SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                              windowWidth, windowHeight,
                              baseFlags | SDL_WINDOW_OPENGL);
    bool glCapableWindow = (window != nullptr);

    if (!window)
    {
        window = SDL_CreateWindow(title,
                                  SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                  windowWidth, windowHeight, baseFlags);
    }

    if (!window)
    {
        std::cerr << "window creation failed: " << SDL_GetError() << std::endl;
        return false;
    }

    // The presenter is created after the rasterizer, further down: registering
    // a PBO with CUDA wants the CUDA context to exist already. Only if that
    // fails do the SDL_Renderer objects below get built.
    wantGLPresent = glCapableWindow;

    // initialize timing
    lastTime = std::chrono::high_resolution_clock::now();

    running = true;

    // remove debugging later

    std::cout << "Engine ready." << std::endl;

    std::cout << "\n=== SCENE ===" << std::endl;
    std::cout << "  TAB / SHIFT+TAB - Select next / previous object" << std::endl;
    std::cout << "  X               - Delete selected object" << std::endl;
    std::cout << "  SHIFT+D         - Duplicate selected object" << std::endl;
    std::cout << "  N               - Create empty node" << std::endl;
    std::cout << "  L               - Load a model" << std::endl;
    std::cout << "  I               - Print scene hierarchy" << std::endl;

    std::cout << "\n=== TRANSFORM SELECTED OBJECT ===" << std::endl;
    std::cout << "  CTRL + Numpad        - Move (4/6=X, 8/2=Z, +/-=Y)" << std::endl;
    std::cout << "  ALT + Numpad         - Rotate (4/6=Y, 8/2=X, 7/9=Z)" << std::endl;
    std::cout << "  SHIFT + Numpad +/-   - Scale uniformly" << std::endl;

    std::cout << "\n=== CAMERA ===" << std::endl;
    std::cout << "  Left drag    - Orbit" << std::endl;
    std::cout << "  Wheel, R/F   - Zoom" << std::endl;
    std::cout << "  WASD         - Pan (orbit mode) / move (free-look)" << std::endl;
    std::cout << "  Q/E          - Up / down" << std::endl;
    std::cout << "  G            - Orbit or free-look" << std::endl;
    std::cout << "  H            - Reset camera" << std::endl;

    std::cout << "\n=== RENDERING ===" << std::endl;
    std::cout << "  U            - Supersampling 1x / 2x / 4x" << std::endl;
    std::cout << "  O            - Ambient occlusion on or off" << std::endl;
    std::cout << "  , / .        - Occlusion strength" << std::endl;
    std::cout << "  ;            - Cycle occlusion debug views" << std::endl;
    std::cout << "  T            - Stats overlay" << std::endl;
    std::cout << "  K            - Present path: host copy or GL interop" << std::endl;
    std::cout << "  Arrow keys   - Move the light" << std::endl;
    std::cout << "  P            - Capture frame to output.tga" << std::endl;
    std::cout << "  B / C        - Load / clear background image" << std::endl;
    std::cout << "  ESC          - Exit" << std::endl;

    cuda_available = initCudaRasterizerSS(renderWidth, renderHeight, ssaaFactor);
    if (!cuda_available)
    {
        std::cerr << "CUDA rasterizer failed to initialize. Rendering happens "
                     "on the GPU; there is no CPU fallback." << std::endl;
        return false;
    }

    // The frame is always renderWidth x renderHeight by the time it reaches
    // the presenter: supersampling is resolved on the device, so changing the
    // SSAA factor never resizes anything here.
    if (wantGLPresent && glPresenter.create(window, renderWidth, renderHeight))
    {
        std::cout << "present path: " << glPresenter.modeName();
        if (!glPresenter.interopAvailable())
            std::cout << " (interop unavailable on this GL context)";
        else
            std::cout << " - K to switch";
        std::cout << std::endl;
    }
    else
    {
        sdlRenderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
        if (!sdlRenderer)
        {
            std::cerr << "Renderer creation failed: " << SDL_GetError() << std::endl;
            return false;
        }

        frameTexture = SDL_CreateTexture(sdlRenderer, SDL_PIXELFORMAT_RGB24,
                                         SDL_TEXTUREACCESS_STREAMING,
                                         renderWidth, renderHeight);
        if (!frameTexture)
        {
            std::cerr << "Texture creation failed: " << SDL_GetError() << std::endl;
            return false;
        }
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

            case SDLK_ESCAPE:
                running = false;
                break;

            case SDLK_o:
                toggleSSAO();
                break;

            // host copy vs CUDA/GL interop, switched in place so both can be
            // measured in one run under the same thermal and driver state
            case SDLK_k:
                togglePresentMode();
                break;

            // 1x, 2x, 4x supersampling
            case SDLK_u:
                setSSAA(ssaaFactor >= 4 ? 1 : ssaaFactor * 2);
                std::cout << "SSAA: " << ssaaFactor << "x" << std::endl;
                break;

            case SDLK_COMMA:
                setSSAOIntensity(ssaoIntensity - 0.1f);
                break;

            case SDLK_PERIOD:
                setSSAOIntensity(ssaoIntensity + 0.1f);
                break;

            // cycles the SSAO debug views: off, occlusion, normals, depth
            case SDLK_SEMICOLON:
                ssaoDebug = (ssaoDebug + 1) % 4;
                std::cout << "SSAO view: "
                          << (ssaoDebug == 0 ? "normal" :
                              ssaoDebug == 1 ? "occlusion term" :
                              ssaoDebug == 2 ? "normals" : "depth")
                          << std::endl;
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
                loadModel("obj/head.obj");
                break;
            case SDLK_n:
                createEmptyNode();
                break;
            case SDLK_i:
                scene.printSceneHierarchy();
                break;

            case SDLK_r:
                zoomCamera(-0.25f);
                break;
            case SDLK_f:
                zoomCamera(0.25f);
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
                lastMouseX = event.button.x; // Store starting position
                lastMouseY = event.button.y;
                // DON'T use SDL_SetRelativeMouseMode - it's broken in WSL
            }
            break;

        case SDL_MOUSEBUTTONUP:
            if (event.button.button == SDL_BUTTON_LEFT)
            {
                mousePressed = false;
                // DON'T use SDL_SetRelativeMouseMode(SDL_FALSE)
            }
            break;

        case SDL_MOUSEMOTION:
            if (mousePressed)
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
            {
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
    renderScene();
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

    syncBackground();
    syncGeometry();

    // before the early return: present() blits whatever is in device memory,
    // so an empty scene has to clear it or the last drawn frame persists.
    cudaClearBuffers();

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


    // the device buffers are ssaaFactor times larger in each axis, so the
    // viewport has to map to that, not to the display size. everything
    // downstream (shadow map, SSAO, resolve) follows from this.
    const int rw = renderWidth * ssaaFactor;
    const int rh = renderHeight * ssaaFactor;

    // PASS 1: depth from the light's point of view, orthographic
    lookat(scene.light.direction, Vec3f(0, 0, 0), scene.camera.up);
    viewport(rw / 8, rh / 8, rw * 3 / 4, rh * 3 / 4);
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
    viewport(rw / 8, rh / 8, rw * 3 / 4, rh * 3 / 4);
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

    // Ambient occlusion, straight over the finished device frame. The
    // depth buffer and the eye-space normals are already there, so this
    // costs one more pass and nothing comes back to the host.
    //
    // Viewport * Projection is shared by every mesh in the frame (only
    // ModelView differs), so inverting it lets the kernel rebuild eye-space
    // positions from depth alone instead of storing a position buffer.
    if (ssaoEnabled && ssaoIntensity > 0.f)
    {
        Matrix vp = Viewport * Projection;
        Matrix inv_vp = vp.invert();
        float vp16[16], inv16[16];
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++)
            {
                vp16[r * 4 + c] = vp[r][c];
                inv16[r * 4 + c] = inv_vp[r][c];
            }
        cudaApplySSAO(inv16, vp16, ssaoRadius, ssaoIntensity, 0.02f, ssaoDebug);
    }

    // no readback: the frame stays in device memory until present() blits it
    // or captureFrame() asks for it, so the GPU keeps working meanwhile.

    // restore original ModelView
    ModelView = originalModelView;
}

void Engine::present()
{
    // GL path: both of its modes end at the same texture and quad, so the
    // only thing that differs between them is how the pixels get there.
    if (glPresenter.isValid())
    {
        glPresenter.present();
        return;
    }

    // SDL_Renderer fallback, used when there is no GL context at all.
    void *pixels;
    int pitch;

    if (SDL_LockTexture(frameTexture, NULL, &pixels, &pitch) == 0)
    {
        // device buffer is already RGB24 in SDL row order: straight DMA,
        // no swizzle, no flip, no per-pixel host work
        cudaBlitToTexture(pixels, pitch);
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
    // writing a TGA is the one thing that still needs the frame on the host
    cudaCopyResults(captureStaging);

    captureStaging.flip_vertically();
    captureStaging.write_tga_file(filename.c_str());
    captureStaging.flip_vertically();
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
            cudaDestroyMesh(kv.second);
        cudaMeshes.clear();
        cleanupCudaRasterizer();
    }

    // before the window: the GL context is owned by the presenter and the
    // registered PBO has to be handed back to CUDA while it still exists
    glPresenter.destroy();

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
// cudaMeshes is keyed by Model*, and Scene::clear() frees every Model. Holding
// the cache across that leaks the device meshes, and worse: the allocator can
// hand a newly loaded Model the address of one just freed, and the stale entry
// then draws the old geometry.
void Engine::syncGeometry()
{
    if ((long)scene.geometryVersion == cachedGeometryVersion)
        return;

    for (auto &kv : cudaMeshes)
        cudaDestroyMesh(kv.second);
    cudaMeshes.clear();
    cachedGeometryVersion = (long)scene.geometryVersion;
}

// The background is a texture on the device, so it only needs re-uploading
// when it actually changes. clear() composites it; nothing here draws.
void Engine::syncBackground()
{
    if ((long)scene.backgroundVersion == uploadedBackgroundVersion)
        return;

    if (scene.background && scene.background->buffer())
        cudaSetBackground(scene.background->buffer(),
                          scene.background->get_width(),
                          scene.background->get_height(),
                          scene.background->get_bytespp());
    else
        cudaClearBackground();

    uploadedBackgroundVersion = (long)scene.backgroundVersion;
}

int Engine::getCudaMesh(Model *model)
{
    if (!model || !cuda_available)
        return -1;

    auto it = cudaMeshes.find(model);
    if (it != cudaMeshes.end())
        return it->second;

    int nverts = model->nverts();
    int nfaces = model->nfaces();
    if (nverts <= 0 || nfaces <= 0)
        return -1;

    // positions stay indexed; normals and uvs go up unindexed, one set per
    // triangle corner, which avoids carrying separate vt/vn index arrays.
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

    cudaMeshes[model] = handle;

    std::cout << "CUDA mesh uploaded: " << nverts << " verts, "
              << nfaces << " faces (handle " << handle << ")" << std::endl;
    return handle;
}

// Rebuilds the device buffers at the new size. Meshes live in storage that
// outlives the rasterizer instance, so the uploaded geometry and its textures
// survive and the handles stay valid.
bool Engine::benchmarkPresent(int blocks, int framesPerBlock)
{
    if (!glPresenter.isValid())
    {
        std::cerr << "no GL presenter: nothing to compare" << std::endl;
        return false;
    }
    if (!glPresenter.interopAvailable())
    {
        std::cerr << "interop unavailable on this GL context: nothing to compare"
                  << std::endl;
        return false;
    }

    double upload[2] = {0.0, 0.0};   // indexed by GLPresenter::Mode
    double frame[2] = {0.0, 0.0};
    long   count[2] = {0, 0};

    std::cout << "\npresent A/B: " << blocks << " blocks of " << framesPerBlock
              << " frames, alternating" << std::endl;

    for (int b = 0; b < blocks; b++)
    {
        GLPresenter::Mode m = (b % 2 == 0) ? GLPresenter::Interop
                                           : GLPresenter::HostCopy;
        glPresenter.setMode(m);

        for (int f = 0; f < framesPerBlock; f++)
        {
            scene.updateAllTransforms();
            render();

            auto t0 = std::chrono::high_resolution_clock::now();
            present();
            double frame_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::high_resolution_clock::now() - t0)
                                  .count();

            // The first block of each mode warms the driver's buffers; the
            // first frame after a mode switch also pays for the change.
            if (b < 2 || f == 0) continue;

            upload[m] += glPresenter.lastUploadMs();
            frame[m] += frame_ms;
            count[m] += 1;
        }
    }

    if (count[0] == 0 || count[1] == 0)
    {
        std::cerr << "not enough samples" << std::endl;
        return false;
    }

    std::cout << "\n  path        upload ms   present ms   frames" << std::endl;
    for (int m = 1; m >= 0; m--)
    {
        const char *name = (m == GLPresenter::Interop) ? "interop  " : "host copy";
        printf("  %s   %8.3f   %10.3f   %6ld\n", name,
               upload[m] / count[m], frame[m] / count[m], count[m]);
    }

    double d = (upload[GLPresenter::HostCopy] / count[GLPresenter::HostCopy]) -
               (upload[GLPresenter::Interop] / count[GLPresenter::Interop]);
    printf("\n  interop saves %.3f ms per frame on the upload\n", d);

    glPresenter.setMode(GLPresenter::Interop);
    return true;
}

void Engine::togglePresentMode()
{
    if (!glPresenter.isValid()) return;

    if (!glPresenter.interopAvailable())
    {
        std::cout << "interop is not available on this GL context" << std::endl;
        return;
    }

    glPresenter.setMode(glPresenter.getMode() == GLPresenter::Interop
                            ? GLPresenter::HostCopy
                            : GLPresenter::Interop);

    std::cout << "present path: " << glPresenter.modeName() << std::endl;
}

void Engine::setSSAA(int factor)
{
    if (factor < 1) factor = 1;
    if (factor > 4) factor = 4;
    if (factor == ssaaFactor || !cuda_available)
        return;

    ssaaFactor = factor;
    // the new rasterizer owns none of the old one's textures
    uploadedBackgroundVersion = -1;
    if (!initCudaRasterizerSS(renderWidth, renderHeight, ssaaFactor))
    {
        std::cout << "SSAA " << ssaaFactor << "x failed to allocate, falling back to 1x"
                  << std::endl;
        ssaaFactor = 1;
        cuda_available = initCudaRasterizerSS(renderWidth, renderHeight, 1);
    }
}

void Engine::toggleSSAO()
{
    ssaoEnabled = !ssaoEnabled;
    std::cout << "SSAO: " << (ssaoEnabled ? "ON" : "OFF") << std::endl;
}

void Engine::setSSAOIntensity(float v)
{
    ssaoIntensity = std::max(0.0f, std::min(1.0f, v));
    std::cout << "SSAO intensity: " << ssaoIntensity << std::endl;
}

void Engine::setSSAORadius(float v)
{
    ssaoRadius = std::max(0.01f, std::min(2.0f, v));
    std::cout << "SSAO radius: " << ssaoRadius << std::endl;
}

