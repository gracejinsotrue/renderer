#ifndef __ENGINE_H__
#define __ENGINE_H__

#include <SDL2/SDL.h>
#include <vector>
#include <string>
#include <chrono>
#include <unordered_map>
#include "geometry.h"
#include "Scene.h" // new scene system!
#include "our_gl.h"
#include "tgaimage.h"
#include "shaders.h"

extern "C"
{
    bool initCudaRasterizer(int width, int height);
    void cleanupCudaRasterizer();
    void cudaClearBuffers();
    void cudaRenderTriangle(const Vec4f &v0, const Vec4f &v1, const Vec4f &v2, const TGAColor &color);
    // fast path: device frame -> locked SDL texture, no host per-pixel work
    void cudaBlitToTexture(void *dst, int dst_pitch);
    // fallback: device frame -> host TGAImage, for overlays and TGA capture
    void cudaCopyResults(TGAImage &framebuffer);
    // per-frame triangle counts (offered / backface-culled / offscreen-culled)
    void cudaGetRasterStats(int *submitted, int *culled_back, int *culled_offscreen,
                            int *bin_overflow);
    // GPU milliseconds for the most recent flush (bin pass / raster pass)
    void cudaGetKernelTimings(float *upload_ms, float *bin_ms, float *raster_ms);
    // persistent device geometry: upload once, transform on the GPU each frame
    int cudaCreateMesh(const float *verts, int nverts, const int *faces, int nfaces,
                       const float *corner_normals, const float *corner_uvs);
    // slot: 0 diffuse, 1 normal map, 2 specular. px is a TGAImage buffer.
    void cudaSetMeshTexture(int handle, int slot, const unsigned char *px,
                            int w, int h, int bpp);
    void cudaSetLinearTextureFiltering(int enabled);
    // screen space ambient occlusion over the finished device frame
    void cudaApplySSAO(const float *inv_vp16, const float *vp16,
                       float radius, float intensity, float bias, int debug);
    void cudaDestroyMesh(int handle);
    // mshadow16 may be NULL for an unshadowed draw
    void cudaDrawMesh(int handle, const float *mvp16, const float *clip16,
                      const float *mit16,
                      const float *light3, const float *light_rgb3,
                      float light_intensity, const float *mshadow16, float shadow_bias,
                      unsigned char r, unsigned char g, unsigned char b);
    // rasterizes everything queued so far into the shadow depth buffer
    void cudaRenderShadowPass();
}

class Engine
{
private:
    SDL_Window *window;
    SDL_Renderer *sdlRenderer;
    SDL_Texture *frameTexture;

    Scene scene;
    TGAImage framebuffer;
    TGAImage zbuffer;

    // for timing
    std::chrono::high_resolution_clock::time_point lastTime;
    float deltaTime;

    // use cuda
    bool cuda_available;
    bool use_cuda_rendering;
    // true when the finished frame still lives only in device memory, so
    // present() can blit it directly instead of going through `framebuffer`
    bool frameOnGPU;
    // Model -> device mesh. meshes are immutable once loaded, so geometry is
    // uploaded once and only re-transformed on the GPU each frame.
    std::unordered_map<Model *, int> cudaMeshes;
    int getCudaMesh(Model *model);

    // Input state
    bool keys[SDL_NUM_SCANCODES];
    int mouseX, mouseY;
    int mouseDeltaX, mouseDeltaY;
    int lastMouseX, lastMouseY;
    bool mousePressed;

    // Engine state
    bool running;
    bool showStats;

    // SSAO. radius is in world units, so it scales with the scene rather
    // than the framebuffer; intensity 0 turns the pass off entirely.
    bool ssaoEnabled;
    float ssaoRadius;
    float ssaoIntensity;
    int ssaoDebug;    // 0 off, 1 ao term, 2 normals, 3 depth

    // Rendering dimensions
    int renderWidth, renderHeight;
    int windowWidth, windowHeight;

    // Enhanced camera controls
    float cameraDistance;
    float cameraRotationX;
    float cameraRotationY;
    Vec3f cameraTarget;
    bool orbitMode;





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

    // Scene management for multi-object
    Scene &getScene() { return scene; }
    SceneNode *loadModel(const std::string &filename, const std::string &nodeName = "");
    SceneNode *createEmptyNode(const std::string &nodeName = "");
    void loadBackground(const std::string &filename);

    // Object selection and manipulation
    void selectNextObject();
    void selectPreviousObject();
    void deleteSelectedObject();
    void duplicateSelectedObject();

    // Transform controls for selected object
    void moveSelectedObject(const Vec3f &delta);
    void rotateSelectedObject(const Vec3f &delta);
    void scaleSelectedObject(const Vec3f &delta);

    // Camera controls
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

    // utility
    void updateWindowTitle();
    float getFPS() const
    {
        if (deltaTime <= 0.001f)
            return 60.0f;
        float fps = 1.0f / deltaTime;
        return std::min(fps, 60.0f);
    }
    // for cuda
    void toggleCudaRendering();
    bool isCudaAvailable() const { return cuda_available; }
    bool isCudaRenderingEnabled() const { return use_cuda_rendering; }

    void toggleSSAO();
    bool isSSAOEnabled() const { return ssaoEnabled; }
    void setSSAODebug(int mode) { ssaoDebug = mode; }
    void setSSAOIntensity(float v);
    void setSSAORadius(float v);
    float getSSAOIntensity() const { return ssaoIntensity; }
    float getSSAORadius() const { return ssaoRadius; }
};

#endif // __ENGINE_H__