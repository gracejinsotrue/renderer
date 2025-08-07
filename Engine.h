#ifndef __ENGINE_H__
#define __ENGINE_H__

#include <SDL2/SDL.h>
#include <vector>
#include <string>
#include <chrono>
#include "geometry.h"
#include "model.h"
#include "our_gl.h"
#include "tgaimage.h"
#include "shaders.h"

class Camera
{
public:
    Vec3f position;
    Vec3f target;
    Vec3f up;
    float fov;

    Camera(Vec3f pos = Vec3f(0, 0, 3), Vec3f tgt = Vec3f(0, 0, 0), Vec3f u = Vec3f(0, 1, 0))
        : position(pos), target(tgt), up(u), fov(-1.0f) {}

    void rotate(float yaw, float pitch);
    void move(Vec3f direction, float speed);
    void lookAt(Vec3f eye, Vec3f center, Vec3f up_vec);
};

class Light
{
public:
    Vec3f direction;
    Vec3f color;
    float intensity;

    Light(Vec3f dir = Vec3f(1, 1, 1), Vec3f col = Vec3f(1, 1, 1), float intens = 1.0f)
        : direction(dir), color(col), intensity(intens)
    {
        direction.normalize();
    }
};

class Animation
{
public:
    float duration;
    bool loop;
    bool playing;
    float currentTime;

    Animation(float dur = 1.0f, bool shouldLoop = true)
        : duration(dur), loop(shouldLoop), playing(false), currentTime(0.0f) {}

    void play()
    {
        playing = true;
        currentTime = 0.0f;
    }
    void pause() { playing = false; }
    void stop()
    {
        playing = false;
        currentTime = 0.0f;
    }

    void update(float deltaTime)
    {
        if (!playing)
            return;

        currentTime += deltaTime;
        if (currentTime >= duration)
        {
            if (loop)
            {
                currentTime = fmod(currentTime, duration);
            }
            else
            {
                currentTime = duration;
                playing = false;
            }
        }
    }

    float getProgress() const { return currentTime / duration; }
    float getTime() const { return currentTime; }
};

class Transform
{
public:
    Vec3f position;
    Vec3f rotation; // euler angles in radians
    Vec3f scale;

    Transform() : position(0, 0, 0), rotation(0, 0, 0), scale(1, 1, 1) {}

    Matrix getMatrix() const;
    void interpolate(const Transform &other, float t, Transform &result) const;
};

class AnimatedModel
{
public:
    Model *model;
    Transform transform;
    Animation animation;

    // Animation types
    enum AnimationType
    {
        ROTATE_Y,
        BOUNCE,
        SCALE_PULSE,
        FIGURE_8,
        HEAD_NOD,
        HEAD_SHAKE

    };

    AnimationType currentAnimation;
    Transform baseTransform; // Starting transform

    AnimatedModel(Model *m) : model(m), currentAnimation(ROTATE_Y)
    {
        baseTransform = transform;
        std::cout << "AnimatedModel created for model with " << model->nverts() << " vertices" << std::endl;
    }

    ~AnimatedModel()
    {
    }

    void startAnimation(AnimationType type, float duration = 2.0f, bool loop = true);
    void update(float deltaTime);
    Matrix getTransformMatrix() const { return transform.getMatrix(); }
};

class Scene
{
public:
    std::vector<AnimatedModel *> animatedModels;
    Camera camera;
    Light light;
    TGAImage *background;

    Scene() : background(nullptr) {}
    ~Scene();

    void addModel(const std::string &filename);
    void removeModel(int index);
    void clear();
    void loadBackground(const std::string &filename);
    void clearBackground();
    void update(float deltaTime); // update all animations
};

class Engine
{
private:
    SDL_Window *window;
    SDL_Renderer *sdlRenderer;
    SDL_Texture *frameTexture;

    Scene scene;
    TGAImage framebuffer;
    TGAImage zbuffer;

    // Timing
    std::chrono::high_resolution_clock::time_point lastTime;
    float deltaTime;

    // Input state
    bool keys[SDL_NUM_SCANCODES];
    int mouseX, mouseY;
    int mouseDeltaX, mouseDeltaY;
    bool mousePressed;

    // Engine state
    bool running;
    bool wireframe;
    bool showStats;

    // Rendering dimensions
    int renderWidth, renderHeight;
    int windowWidth, windowHeight;

    // Enhanced camera controls
    float cameraDistance;  // Distance from target
    float cameraRotationX; // Vertical rotation (pitch)
    float cameraRotationY; // Horizontal rotation (yaw)
    Vec3f cameraTarget;    // What we're looking at
    bool orbitMode;        // Toggle between orbit and free-look

public:
    Engine(int winWidth = 1024, int winHeight = 768, int renWidth = 800, int renHeight = 800);
    ~Engine();

    bool init();
    void run();
    void shutdown();

    // Main loop functions
    void handleEvents();
    void update();
    void render();
    void present();

    // Scene management
    Scene &getScene() { return scene; }
    void loadModel(const std::string &filename);
    void loadBackground(const std::string &filename);

    // Enhanced camera controls
    void updateCamera();
    void zoomCamera(float amount);
    void panCamera(float deltaX, float deltaY);
    void orbitCamera(float deltaYaw, float deltaPitch);
    void updateCameraPosition();
    void resetCamera();
    void toggleCameraMode();

    // Rendering
    void drawBackground();
    void renderScene();

    // Frame capture
    void captureFrame(const std::string &filename);
    void captureSequence(const std::string &baseName, int frameCount, float duration);

    // Utility
    void updateWindowTitle();
    float getFPS() const
    {
        // prevent division by zero and cap FPS display
        if (deltaTime <= 0.001f)
            return 60.0f; // If deltaTime too small, return reasonable FPS
        float fps = 1.0f / deltaTime;
        return std::min(fps, 60.0f); // Cap displayed FPS at 60 just because laptop skill issue
    }
};

#endif // __ENGINE_H__