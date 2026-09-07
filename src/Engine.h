#ifndef __ENGINE_H__
#define __ENGINE_H__

#include <SDL2/SDL.h>
#include <vector>
#include <string>
#include <chrono>
#include <unordered_map>
#include "geometry.h"
#include "Scene.h" // new scene system!
#include "transform.h"
#include "tgaimage.h"
#include "GLPresenter.h"

extern "C"
{
    bool initCudaRasterizer(int width, int height);
    // renders at (width*ss) x (height*ss) and box-filters down on the device
    bool initCudaRasterizerSS(int width, int height, int ss);
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
    // composited under every frame by cudaClearBuffers, at render resolution
    void cudaSetBackground(const unsigned char *px, int w, int h, int bpp);
    void cudaClearBackground();
    // equirectangular HDR, linear RGBA float. drawn by cudaClearBuffers in
    // place of the background when both are set.
    void cudaSetEnvironment(const float *rgba, int w, int h);
    void cudaClearEnvironment();
    // row-major inverse of Viewport*Projection*ModelView, at render
    // resolution: what turns a pixel back into a world-space view ray. Both
    // are per-frame, so a rasterizer rebuild does not strand them.
    void cudaSetEnvironmentView(const float *inv16, float exposure);
    // screen space ambient occlusion over the finished device frame
    void cudaApplySSAO(const float *inv_vp16, const float *vp16,
                       float radius, float intensity, float bias, int debug);
    void cudaDestroyMesh(int handle);
    // meshes still holding device memory, for leak checks in the tests
    int cudaLiveMeshCount();
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
    // Only one of these two is live. GLPresenter is preferred; the
    // SDL_Renderer pair is the fallback for when there is no GL context at
    // all, which is how the headless tests run.
    SDL_Renderer *sdlRenderer;
    SDL_Texture *frameTexture;
    GLPresenter glPresenter;
    // window was created GL-capable, so a presenter is worth attempting
    bool wantGLPresent;

    Scene scene;
    // host copy, written only when captureFrame() asks for a TGA
    TGAImage captureStaging;

    // for timing
    std::chrono::high_resolution_clock::time_point lastTime;
    float deltaTime;

    bool cuda_available;
    // Model -> device mesh. meshes are immutable once loaded, so geometry is
    // uploaded once and only re-transformed on the GPU each frame.
    std::unordered_map<Model *, int> cudaMeshes;
    int getCudaMesh(Model *model);

    // Scene::backgroundVersion as of the last upload. -1 forces one, which is
    // how a rasterizer resize gets its textures back.
    long uploadedBackgroundVersion;
    void syncBackground();

    long uploadedEnvironmentVersion;
    void syncEnvironment();

    // Scene::geometryVersion as of the last time cudaMeshes was known good.
    long cachedGeometryVersion;
    void syncGeometry();

    // Input state
    bool keys[SDL_NUM_SCANCODES];
    int mouseX, mouseY;
    int mouseDeltaX, mouseDeltaY;
    int lastMouseX, lastMouseY;
    bool mousePressed;

    // Engine state
    bool running;
    bool showStats;

    // Supersampling factor. 1 disables it. The whole
    // pipeline (raster, shadows, SSAO) runs at the larger size and the frame
    // is averaged back down on the device, so edges, textures and specular
    // highlights are all anti-aliased.
    int ssaaFactor;

    // SSAO. radius is in world units, so it scales with the scene rather
    // than the render target; intensity 0 turns the pass off entirely.
    bool ssaoEnabled;
    float ssaoRadius;
    float ssaoIntensity;
    int ssaoDebug;    // 0 off, 1 ao term, 2 normals, 3 depth

    // shadow map comparison bias; see setShadowBias
    float shadowBias;

    // Linear multiplier on the environment's radiance before the tone curve.
    // Only the environment reads it: the shading path has no HDR values to
    // expose yet.
    float exposure;

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
    void loadEnvironment(const std::string &filename);

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
    bool isCudaAvailable() const { return cuda_available; }

    // Switches the present path between the host copy and CUDA/GL interop
    // in place, so the two can be compared inside one run rather than across
    // process launches. Does nothing when interop is unavailable.
    void togglePresentMode();

    // Alternates the two present paths in interleaved blocks and reports what
    // each costs. Blocks alternate rather than running all of one then all of
    // the other, because this machine's clocks drift enough over a run that a
    // sequential A-then-B would hand the difference to whichever went second.
    // Returns false if there is no GL presenter to measure.
    bool benchmarkPresent(int blocks = 8, int framesPerBlock = 120);

    void setSSAA(int factor);
    int getSSAA() const { return ssaaFactor; }

    void toggleSSAO();
    bool isSSAOEnabled() const { return ssaoEnabled; }
    void setSSAODebug(int mode) { ssaoDebug = mode; }
    void setSSAOIntensity(float v);
    void setSSAORadius(float v);

    // Depth spans 0..255 regardless of world scale, so this is a fraction of
    // that range rather than a world distance. A negative value switches the
    // shader into its shadow debug view, which emits (sz, stored, 0) instead
    // of shading -- that is how you see what the comparison is actually doing.
    void setShadowBias(float v) { shadowBias = v; }
    float getShadowBias() const { return shadowBias; }
    float getSSAOIntensity() const { return ssaoIntensity; }
    float getSSAORadius() const { return ssaoRadius; }

    void setExposure(float v);
    float getExposure() const { return exposure; }
};

#endif // __ENGINE_H__