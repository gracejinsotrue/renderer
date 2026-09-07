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

class UI;

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
    // Per-frame triangle counts: offered, backface-culled, offscreen-culled.
    // What actually rasterized is offered minus the two.
    //
    // bin_entries is not a cull count and not an error: bins are sized
    // exactly by the prefix sum, so nothing overflows. It is the number of
    // (triangle, tile) pairs, which is how far binning fans out -- a triangle
    // covering nine tiles is nine entries -- and so is normally larger than
    // the triangle count.
    void cudaGetRasterStats(int *submitted, int *culled_back, int *culled_offscreen,
                            int *bin_entries);
    // GPU milliseconds for the most recent flush (bin pass / raster pass)
    void cudaGetKernelTimings(float *upload_ms, float *bin_ms, float *raster_ms);
    // persistent device geometry: upload once, transform on the GPU each frame
    int cudaCreateMesh(const float *verts, int nverts, const int *faces, int nfaces,
                       const float *corner_normals, const float *corner_uvs);
    // slot: 0 diffuse, 1 normal map, 2 specular. px is a TGAImage buffer.
    void cudaSetMeshTexture(int handle, int slot, const unsigned char *px,
                            int w, int h, int bpp);
    void cudaSetLinearTextureFiltering(int enabled);
    // Which colour path runs. Deferred shades once per covered pixel; forward
    // shades inside the depth loop and pays per fragment that ever won.
    void cudaSetDeferredShading(int enabled);
    int cudaGetDeferredShading();
    // fragment shader invocations in the last flush, which is what separates
    // the two paths on a scene with overdraw
    int cudaGetShadeCount();
    void cudaSetToneMapping(int enabled);
    // equirectangular HDR, linear RGBA float. drawn by cudaClearBuffers as the
    // frame's backdrop, at render resolution.
    void cudaSetEnvironment(const float *rgba, int w, int h);
    void cudaClearEnvironment();
    // row-major inverse of Viewport*Projection*ModelView, at render
    // resolution: what turns a pixel back into a world-space view ray. Sent
    // every frame, so a rasterizer rebuild does not strand it.
    void cudaSetEnvironmentView(const float *inv16);
    // Row-major 3x3 eye -> world rotation, and how much of the environment's
    // light the shading takes. Diffuse IBL is off until an environment is
    // loaded; loading one builds the irradiance map.
    void cudaSetIBL(const float *e2w9, float intensity);
    void cudaSetMaterialParams(float metallic, float roughness);
    // linear multiplier applied by the tone map, ahead of the curve
    void cudaSetExposure(float exposure);
    // 0 emits the linear frame scaled to 0..255 with no exposure and no
    // curve. The differential tests want the shading path's own numbers.
    void cudaSetToneMapping(int enabled);
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

    // Optional control panel, created on the first present when enabled.
    // Held by pointer so imgui.h stays out of this header, which the headless
    // tests include. Null in every test: nothing constructs one unless
    // setUIEnabled(true) has been called.
    UI *ui;
    bool uiWanted;
    void ensureUI();

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

    // Scene::environmentVersion as of the last upload. -1 forces one, which is
    // how a rasterizer resize gets its textures back.
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

    // Linear multiplier applied to the whole frame before the tone curve.
    float exposure;

    // How much of the environment's irradiance reaches the shading. Has no
    // effect without an environment loaded.
    float iblIntensity;

    // Metallic-roughness, applied to every mesh. A per-model source would
    // come from the loader; there is none yet, so these are scene-wide.
    float pbrMetallic;
    float pbrRoughness;

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
    // Builds the panel for this frame, between render() and present(). A
    // separate step because the loop is driven by hand in several places --
    // the tests, the capture tools -- and those want the frame without it.
    // Does nothing unless setUIEnabled(true) was called.
    void buildUI();
    void render();
    void present();

    // Scene management for multi-object
    Scene &getScene() { return scene; }
    SceneNode *loadModel(const std::string &filename, const std::string &nodeName = "");
    SceneNode *createEmptyNode(const std::string &nodeName = "");
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

    // Opt-in, and off by default so the headless tests never build one.
    // main.cpp is the only caller.
    void setUIEnabled(bool on) { uiWanted = on; }

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
    int getSSAODebug() const { return ssaoDebug; }
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

    void setIBLIntensity(float v);
    float getIBLIntensity() const { return iblIntensity; }

    void setMetallic(float v);
    float getMetallic() const { return pbrMetallic; }
    void setRoughness(float v);
    float getRoughness() const { return pbrRoughness; }
};

#endif // __ENGINE_H__