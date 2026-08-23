#ifndef __ENGINE_H__
#define __ENGINE_H__

#include <SDL2/SDL.h>
#include <vector>
#include <string>
#include <chrono>
#include <set>
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
    void cudaUpdateMeshVerts(int handle, const float *verts, int nverts);
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
    // Model -> device mesh. geometry is uploaded once and re-transformed on
    // the GPU each frame. geomVersion is Model::geometryVersion() as of that
    // upload; when it diverges the positions are re-sent, so a sculpt or blend
    // shape never leaves stale geometry on the device.
    struct CudaMesh
    {
        int handle;
        unsigned int geomVersion;
    };
    std::unordered_map<Model *, CudaMesh> cudaMeshes;
    int getCudaMesh(Model *model);

    // Input state
    bool keys[SDL_NUM_SCANCODES];
    int mouseX, mouseY;
    int mouseDeltaX, mouseDeltaY;
    int lastMouseX, lastMouseY;
    bool mousePressed;

    // Engine state
    bool running;
    bool wireframe;
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



    class VertexEditor
    {
    public:
        enum EditMode
        {
            NORMAL,
            VERTEX_SELECT,
            VERTEX_DEFORM
        };

    private:
        EditMode currentMode;
        Model *targetModel;
        SceneNode *targetNode;

        // selection state
        std::set<int> selectedVertices;
        std::vector<Vec3f> selectionColors;
        bool showVertices;
        float vertexSize;
        float selectionRadius;

        // deformation state
        Vec3f deformationCenter;
        float deformationRadius;
        float deformationStrength;
        bool isDeforming;

        // blend shape creation

        // mouse interaction
        Vec3f lastMouseWorldPos;
        bool isDragging;
        int lastMouseX, lastMouseY;

    public:
        VertexEditor() : currentMode(NORMAL), targetModel(nullptr), targetNode(nullptr),
                         showVertices(false), vertexSize(3.0f), selectionRadius(0.05f),
                         deformationRadius(0.1f), deformationStrength(0.1f), isDeforming(false),
                         isDragging(false),
                         lastMouseX(0), lastMouseY(0) {}

        void setTargetModel(SceneNode *node);
        void setMode(EditMode mode);
        EditMode getMode() const { return currentMode; }

        // Vertex selection
        void selectVerticesInRadius(const Vec3f &worldPos, float radius);
        void addVertexToSelection(int vertexIndex);
        void removeVertexFromSelection(int vertexIndex);
        void clearSelection();
        void selectAll();
        void invertSelection();

        // Deformation
        void startDeformation(const Vec3f &center);
        void applyDeformation(const Vec3f &direction, float strength);
        void endDeformation();
        void resetDeformation();

        // Blend shape creation

        // Mouse handling
        void handleMouseClick(int mouseX, int mouseY, const Matrix &viewMatrix,
                              const Matrix &projMatrix, int renderWidth, int renderHeight);
        void handleMouseDrag(int mouseX, int mouseY, int deltaX, int deltaY,
                             const Matrix &viewMatrix, const Matrix &projMatrix,
                             int renderWidth, int renderHeight);
        void handleMouseRelease();

        // Utility
        Vec3f screenToWorldRay(int screenX, int screenY, const Matrix &viewMatrix,
                               const Matrix &projMatrix, int renderWidth, int renderHeight);
        int findClosestVertex(const Vec3f &worldPos, float maxDistance);

        // Settings
        void setSelectionRadius(float radius) { selectionRadius = std::max(0.01f, std::min(0.5f, radius)); }
        void setDeformationStrength(float strength) { deformationStrength = std::max(0.001f, std::min(1.0f, strength)); }
        void setDeformationRadius(float radius) { deformationRadius = std::max(0.01f, std::min(2.0f, radius)); }

        float getSelectionRadius() const { return selectionRadius; }
        float getDeformationStrength() const { return deformationStrength; }
        float getDeformationRadius() const { return deformationRadius; }

        // Info
        void printStatus() const;
        int getSelectedVertexCount() const { return selectedVertices.size(); }
        bool hasTarget() const { return targetModel != nullptr; }
        void toggleVertexDisplay() { showVertices = !showVertices; }
        bool isShowingVertices() const { return showVertices; }

        void renderSelectionInfo() const;

        void renderVertexOverlay(TGAImage &framebuffer, int renderWidth, int renderHeight);
    };

    VertexEditor vertexEditor;
    bool vertexEditMode;


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

    // for ray tracing


    // vertex editing interface
    void enterVertexEditMode();
    void exitVertexEditMode();
    void toggleVertexDisplay();
    void setVertexEditMode(VertexEditor::EditMode mode);

    // selection tools
    void selectVerticesInRadius(float radius);
    void clearVertexSelection();
    void selectAllVertices();
    void invertVertexSelection();

    // deformation tools
    void setDeformationStrength(float strength);
    void setDeformationRadius(float radius);
    void setSelectionRadius(float radius);
    void resetVertexDeformation();

    // blend shape tools

    // blend shape playback

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