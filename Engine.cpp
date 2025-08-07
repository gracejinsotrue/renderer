#include "Engine.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

// Transform implementation
Matrix Transform::getMatrix() const
{
    Matrix translation = Matrix::identity();
    translation[0][3] = position.x;
    translation[1][3] = position.y;
    translation[2][3] = position.z;

    // Simple rotation matrices (Y-axis rotation most common)
    Matrix rotY = Matrix::identity();
    float cosY = cos(rotation.y), sinY = sin(rotation.y);
    rotY[0][0] = cosY;
    rotY[0][2] = sinY;
    rotY[2][0] = -sinY;
    rotY[2][2] = cosY;

    Matrix rotX = Matrix::identity();
    float cosX = cos(rotation.x), sinX = sin(rotation.x);
    rotX[1][1] = cosX;
    rotX[1][2] = -sinX;
    rotX[2][1] = sinX;
    rotX[2][2] = cosX;

    Matrix scaling = Matrix::identity();
    scaling[0][0] = scale.x;
    scaling[1][1] = scale.y;
    scaling[2][2] = scale.z;

    return translation * rotY * rotX * scaling;
}

void Transform::interpolate(const Transform &other, float t, Transform &result) const
{
    result.position = position + (other.position - position) * t;
    result.rotation = rotation + (other.rotation - rotation) * t;
    result.scale = scale + (other.scale - scale) * t;
}

// AnimatedModel implementation
void AnimatedModel::startAnimation(AnimationType type, float duration, bool loop)
{
    currentAnimation = type;
    animation.duration = duration;
    animation.loop = loop;
    animation.play();
    baseTransform = transform;

    std::cout << "Started animation type " << type << " for " << duration << "s" << std::endl;
}

void AnimatedModel::update(float deltaTime)
{
    animation.update(deltaTime);

    if (!animation.playing && !animation.loop)
        return;

    float t = animation.getProgress();
    float time = animation.getTime();

    // Reset to base transform
    transform = baseTransform;

    switch (currentAnimation)
    {
    case ROTATE_Y:
        transform.rotation.y = baseTransform.rotation.y + t * 2.0f * M_PI;
        break;

    case BOUNCE:
        transform.position.y = baseTransform.position.y + abs(sin(t * M_PI * 4)) * 0.5f;
        break;

    case SCALE_PULSE:
    {
        float pulseScale = 1.0f + sin(t * M_PI * 4) * 0.3f;
        transform.scale = Vec3f(pulseScale, pulseScale, pulseScale);
    }
    break;

    case FIGURE_8:
        transform.position.x = baseTransform.position.x + sin(t * 2.0f * M_PI) * 0.8f;
        transform.position.y = baseTransform.position.y + sin(t * 4.0f * M_PI) * 0.4f;
        break;

    case HEAD_NOD:
        transform.rotation.x = baseTransform.rotation.x + sin(t * M_PI * 3) * 0.3f;
        break;

    case HEAD_SHAKE:
        transform.rotation.y = baseTransform.rotation.y + sin(t * M_PI * 4) * 0.4f;
        break;
    }
}

void Camera::rotate(float yaw, float pitch)
{
    // Convert to spherical coordinates for smooth rotation
    Vec3f direction = target - position;
    float radius = direction.norm();

    // Apply rotation
    float cosYaw = cos(yaw), sinYaw = sin(yaw);
    float cosPitch = cos(pitch), sinPitch = sin(pitch);

    direction.x = radius * sinPitch * cosYaw;
    direction.y = radius * cosPitch;
    direction.z = radius * sinPitch * sinYaw;

    target = position + direction;
}

void Camera::move(Vec3f direction, float speed)
{
    Vec3f forward = (target - position).normalize();
    Vec3f right = cross(forward, up).normalize();
    Vec3f realUp = cross(right, forward).normalize();

    Vec3f movement = right * direction.x + realUp * direction.y + forward * direction.z;
    position = position + movement * speed;
    target = target + movement * speed;
}

void Camera::lookAt(Vec3f eye, Vec3f center, Vec3f up_vec)
{
    position = eye;
    target = center;
    up = up_vec;
}

// Scene implementation
Scene::~Scene()
{
    clear();
    clearBackground();
}

void Scene::addModel(const std::string &filename)
{
    Model *model = new Model(filename.c_str());
    if (model->nverts() > 0)
    {
        AnimatedModel *animModel = new AnimatedModel(model);
        animatedModels.push_back(animModel);
        std::cout << "Loaded animated model: " << filename << std::endl;
    }
    else
    {
        delete model;
        std::cerr << "Failed to load model: " << filename << std::endl;
    }
}

void Scene::removeModel(int index)
{
    if (index >= 0 && index < animatedModels.size())
    {
        delete animatedModels[index]->model;
        delete animatedModels[index];
        animatedModels.erase(animatedModels.begin() + index);
    }
}

void Scene::clear()
{
    for (AnimatedModel *animModel : animatedModels)
    {
        delete animModel->model;
        delete animModel;
    }
    animatedModels.clear();
}

void Scene::update(float deltaTime)
{
    for (AnimatedModel *animModel : animatedModels)
    {
        animModel->update(deltaTime);
    }
}

void Scene::loadBackground(const std::string &filename)
{
    clearBackground();
    background = new TGAImage();
    if (background->read_tga_file(filename.c_str()))
    {
        std::cout << "Loaded background: " << filename << std::endl;
    }
    else
    {
        delete background;
        background = nullptr;
        std::cerr << "Failed to load background: " << filename << std::endl;
    }
}

void Scene::clearBackground()
{
    if (background)
    {
        delete background;
        background = nullptr;
    }
}

// Engine implementation
Engine::Engine(int winWidth, int winHeight, int renWidth, int renHeight)
    : window(nullptr), sdlRenderer(nullptr), frameTexture(nullptr), framebuffer(renWidth, renHeight, TGAImage::RGB), zbuffer(renWidth, renHeight, TGAImage::GRAYSCALE), running(false), wireframe(false), showStats(true), windowWidth(winWidth), windowHeight(winHeight), renderWidth(renWidth), renderHeight(renHeight), mouseX(0), mouseY(0), mouseDeltaX(0), mouseDeltaY(0), mousePressed(false), cameraDistance(5.0f), cameraRotationX(0.0f), cameraRotationY(0.0f), cameraTarget(0, 0, 0), orbitMode(true)
{
    // Initialize input state
    memset(keys, 0, sizeof(keys));

    // Initialize shadow buffer
    shadowbuffer.resize(renderWidth * renderHeight);
    std::fill(shadowbuffer.begin(), shadowbuffer.end(), std::numeric_limits<float>::max());
}

Engine::~Engine()
{
    shutdown();
}

bool Engine::init()
{
    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0)
    {
        std::cerr << "SDL initialization failed: " << SDL_GetError() << std::endl;
        return false;
    }

    // Create window
    window = SDL_CreateWindow("3D Renderer Engine",
                              SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                              windowWidth, windowHeight,
                              SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);

    if (!window)
    {
        std::cerr << "Window creation failed: " << SDL_GetError() << std::endl;
        return false;
    }

    // Create renderer
    sdlRenderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!sdlRenderer)
    {
        std::cerr << "Renderer creation failed: " << SDL_GetError() << std::endl;
        return false;
    }

    // Create texture for framebuffer
    frameTexture = SDL_CreateTexture(sdlRenderer, SDL_PIXELFORMAT_RGB24,
                                     SDL_TEXTUREACCESS_STREAMING,
                                     renderWidth, renderHeight);

    if (!frameTexture)
    {
        std::cerr << "Texture creation failed: " << SDL_GetError() << std::endl;
        return false;
    }

    // Initialize timing
    lastTime = std::chrono::high_resolution_clock::now();

    running = true;

    std::cout << "Engine initialized successfully!" << std::endl;
    std::cout << "\n=== CAMERA CONTROLS ===" << std::endl;
    std::cout << "  Mouse + Left Click - Orbit/Look around" << std::endl;
    std::cout << "  Mouse Wheel - Zoom in/out" << std::endl;
    std::cout << "  WASD - Pan view (orbit mode) / Move camera (free mode)" << std::endl;
    std::cout << "  Q/E - Move up/down" << std::endl;
    std::cout << "  R/F - Zoom in/out (alternative to mouse wheel)" << std::endl;
    std::cout << "  G - Toggle camera mode (Orbit ⟷ Free-look)" << std::endl;
    std::cout << "  H - Reset camera to default position" << std::endl;
    std::cout << "\n=== ANIMATION CONTROLS ===" << std::endl;
    std::cout << "  1-6 - Body animations (rotate, bounce, pulse, figure-8, nod, shake)" << std::endl;
    std::cout << "  Space - Pause/Resume animation" << std::endl;
    std::cout << "  R - Reset animation" << std::endl;
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
        // In orbit mode, change distance from target
        cameraDistance += amount;
        cameraDistance = std::max(0.5f, std::min(50.0f, cameraDistance)); // Clamp distance

        // Update camera position based on spherical coordinates
        updateCameraPosition();
    }
    else
    {
        // In free-look mode, move forward/backward
        Vec3f forward = (scene.camera.target - scene.camera.position).normalize();
        scene.camera.position = scene.camera.position + forward * amount;
        scene.camera.target = scene.camera.target + forward * amount;
    }
}

void Engine::panCamera(float deltaX, float deltaY)
{
    if (orbitMode)
    {
        // In orbit mode, move the target point
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
        // In free-look mode, strafe and move up/down
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

        // Clamp pitch to avoid gimbal lock
        cameraRotationX = std::max(-1.5f, std::min(1.5f, cameraRotationX));

        updateCameraPosition();
    }
}

void Engine::updateCameraPosition()
{
    if (orbitMode)
    {
        // Convert spherical coordinates to cartesian
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
    // Reset to default camera position
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
        // Switching to orbit mode - calculate current spherical coordinates
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

void Engine::run()
{
    while (running)
    {
        auto currentTime = std::chrono::high_resolution_clock::now();

        // Ensure minimum deltaTime to prevent FPS spikes
        float rawDeltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
        deltaTime = std::max(rawDeltaTime, 1.0f / 120.0f); // Minimum deltaTime = max 120 FPS

        lastTime = currentTime;

        handleEvents();
        update();
        render();
        present();

        // Force 30 FPS cap
        SDL_Delay(33);
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

            // Handle single-press keys
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
            case SDLK_SPACE:
                if (!scene.animatedModels.empty())
                {
                    AnimatedModel *animModel = scene.animatedModels[0];
                    if (animModel->animation.playing)
                    {
                        animModel->animation.pause();
                        std::cout << "Animation paused" << std::endl;
                    }
                    else
                    {
                        animModel->animation.play();
                        std::cout << "Animation resumed" << std::endl;
                    }
                }
                break;
            case SDLK_r:
                if (!scene.animatedModels.empty())
                {
                    scene.animatedModels[0]->animation.stop();
                    std::cout << "Animation reset" << std::endl;
                }
                break;
            case SDLK_1:
                if (!scene.animatedModels.empty())
                {
                    scene.animatedModels[0]->startAnimation(AnimatedModel::ROTATE_Y, 3.0f);
                }
                break;
            case SDLK_2:
                if (!scene.animatedModels.empty())
                {
                    scene.animatedModels[0]->startAnimation(AnimatedModel::BOUNCE, 2.0f);
                }
                break;
            case SDLK_3:
                if (!scene.animatedModels.empty())
                {
                    scene.animatedModels[0]->startAnimation(AnimatedModel::SCALE_PULSE, 2.5f);
                }
                break;
            case SDLK_4:
                if (!scene.animatedModels.empty())
                {
                    scene.animatedModels[0]->startAnimation(AnimatedModel::FIGURE_8, 4.0f);
                }
                break;
            case SDLK_5:
                if (!scene.animatedModels.empty())
                {
                    scene.animatedModels[0]->startAnimation(AnimatedModel::HEAD_NOD, 1.5f);
                }
                break;
            case SDLK_6:
                if (!scene.animatedModels.empty())
                {
                    scene.animatedModels[0]->startAnimation(AnimatedModel::HEAD_SHAKE, 1.8f);
                }
                break;
                // REMOVED: All facial animation key handlers (7-0, -, =)
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
            // Mouse wheel for zooming
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
    updateCamera();

    // Update scene animations
    scene.update(deltaTime);

    // Light controls with arrow keys
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

    // Mouse look / orbit
    if (mousePressed && (mouseDeltaX != 0 || mouseDeltaY != 0))
    {
        if (orbitMode)
        {
            orbitCamera(-mouseDeltaX * mouseSpeed, -mouseDeltaY * mouseSpeed);
        }
        else
        {
            // Free-look mode (existing code)
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

    // Keyboard movement
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

    // Draw background first
    drawBackground();

    // Then render 3D scene (but don't clear framebuffer in renderScene)
    renderScene();
}

void Engine::drawBackground()
{
    if (!scene.background)
        return;

    // Scale and draw background to fill the framebuffer
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
    if (scene.animatedModels.empty())
        return;

    // Use first animated model for now
    AnimatedModel *animModel = scene.animatedModels[0];
    Model *model = animModel->model;

    // Set up camera
    lookat(scene.camera.position, scene.camera.target, scene.camera.up);
    viewport(renderWidth / 8, renderHeight / 8, renderWidth * 3 / 4, renderHeight * 3 / 4);
    projection(scene.camera.fov);

    // Apply model transformation
    Matrix modelTransform = animModel->getTransformMatrix();
    Matrix originalModelView = ModelView;
    ModelView = ModelView * modelTransform;

    // PASS 1: Shadow mapping (use a separate temporary buffer to not mess with our background)
    std::fill(shadowbuffer.begin(), shadowbuffer.end(), std::numeric_limits<float>::max());

    Matrix M;
    {
        // Render from light's perspective for shadow mapping only
        lookat(scene.light.direction, Vec3f(0, 0, 0), scene.camera.up);
        viewport(renderWidth / 8, renderHeight / 8, renderWidth * 3 / 4, renderHeight * 3 / 4);
        projection(0); // Orthographic for directional light

        // Apply model transform to shadow pass too
        ModelView = ModelView * modelTransform;
        M = Viewport * Projection * ModelView;

        // Create temporary buffers for shadow pass
        TGAImage tempFrame(renderWidth, renderHeight, TGAImage::RGB);
        TGAImage tempZ(renderWidth, renderHeight, TGAImage::GRAYSCALE);

        DepthShader depthShader;
        for (int i = 0; i < model->nfaces(); i++)
        {
            Vec4f screen_coords[3];
            for (int j = 0; j < 3; j++)
            {
                screen_coords[j] = depthShader.vertex(i, j);
            }
            triangle(screen_coords, depthShader, tempFrame, tempZ); // Use temp buffers
        }
    }

    // PASS 2: Main rendering with shadows (preserve background)
    {
        // Restore camera perspective with model transform
        lookat(scene.camera.position, scene.camera.target, scene.camera.up);
        viewport(renderWidth / 8, renderHeight / 8, renderWidth * 3 / 4, renderHeight * 3 / 4);
        projection(scene.camera.fov);
        ModelView = originalModelView * modelTransform;

        Matrix current_transform = Viewport * Projection * ModelView;
        Matrix shadow_transform = M * (current_transform.adjugate() / current_transform.det());

        ShadowMappingShader shader(ModelView,
                                   (Projection * ModelView).invert_transpose(),
                                   shadow_transform);

        // Only clear the Z-buffer, keep the background in framebuffer
        for (int i = 0; i < renderWidth * renderHeight; i++)
        {
            zbuffer.set(i % renderWidth, i / renderWidth, TGAColor(0));
        }

        for (int i = 0; i < model->nfaces(); i++)
        {
            Vec4f screen_coords[3];
            for (int j = 0; j < 3; j++)
            {
                screen_coords[j] = shader.vertex(i, j);
            }
            triangle(screen_coords, shader, framebuffer, zbuffer); // Render onto background
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

    // Render to screen
    SDL_SetRenderDrawColor(sdlRenderer, 0, 0, 0, 255);
    SDL_RenderClear(sdlRenderer);

    // Scale texture to fit window
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
        // Rotate camera around target
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
        title << "Grace's shitty 3D Engine - FPS: " << std::fixed << std::setprecision(1) << getFPS()
              << " | Models: " << scene.animatedModels.size();

        if (!scene.animatedModels.empty())
        {
            AnimatedModel *animModel = scene.animatedModels[0];
            title << " | Anim: " << (animModel->animation.playing ? "Playing" : "Paused")
                  << " (" << std::setprecision(1) << animModel->animation.getProgress() * 100 << "%)";
        }

        title << " | Light: (" << std::setprecision(2)
              << scene.light.direction.x << ", "
              << scene.light.direction.y << ", "
              << scene.light.direction.z << ")";

        title << " | Camera: " << (orbitMode ? "Orbit" : "Free-look");

        SDL_SetWindowTitle(window, title.str().c_str());
    }
}

void Engine::loadModel(const std::string &filename)
{
    scene.addModel(filename);
}

void Engine::loadBackground(const std::string &filename)
{
    scene.loadBackground(filename);
}

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