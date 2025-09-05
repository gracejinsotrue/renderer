#ifndef __ENGINE_H__
#define __ENGINE_H__

#include <SDL2/SDL.h>
#include <vector>
#include <string>
#include <chrono>
#include <set>
#include "geometry.h"
#include "Scene.h" // new scene system!
#include "our_gl.h"
#include "tgaimage.h"
#include "shaders.h"
#include "ray_tracer_integration.h" // NEW: Ray tracer integration
#include "realtime_raytracer.h"

extern "C"
{
    bool initCudaRasterizer(int width, int height);
    void cleanupCudaRasterizer();
    void cudaClearBuffers();
    void cudaRenderTriangle(const Vec4f &v0, const Vec4f &v1, const Vec4f &v2, const TGAColor &color);
    void cudaCopyResults(TGAImage &framebuffer, TGAImage &zbuffer);
}

class Engine
{
private:
    SDL_Window *window;
    SDL_Renderer *sdlRenderer;
    SDL_Texture *frameTexture;

    Scene scene; // uses new scene system
    TGAImage framebuffer;
    TGAImage zbuffer;

    // for timing
    std::chrono::high_resolution_clock::time_point lastTime;
    float deltaTime;

    // use cuda
    bool cuda_available;
    bool use_cuda_rendering;

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
    float cameraDistance;
    float cameraRotationX;
    float cameraRotationY;
    Vec3f cameraTarget;
    bool orbitMode;

    // NEW: Ray tracing state
    bool rayTracingEnabled;

    std::unique_ptr<RealtimeRayTracer> realtimeRT;

    // Interactive vertex editing system
    class VertexEditor
    {
    public:
        enum EditMode
        {
            NORMAL,
            VERTEX_SELECT,
            VERTEX_DEFORM,
            BLEND_SHAPE_CREATE
        };

    private:
        EditMode currentMode;
        Model *targetModel;
        SceneNode *targetNode;

        // Selection state
        std::set<int> selectedVertices;
        std::vector<Vec3f> selectionColors;
        bool showVertices;
        float vertexSize;
        float selectionRadius;

        // Deformation state
        Vec3f deformationCenter;
        float deformationRadius;
        float deformationStrength;
        bool isDeforming;

        // Blend shape creation
        std::string currentBlendShapeName;
        bool recordingBlendShape;
        std::vector<Vec3f> blendShapeStart;

        // Mouse interaction
        Vec3f lastMouseWorldPos;
        bool isDragging;
        int lastMouseX, lastMouseY;

    public:
        VertexEditor() : currentMode(NORMAL), targetModel(nullptr), targetNode(nullptr),
                         showVertices(false), vertexSize(3.0f), selectionRadius(0.05f),
                         deformationRadius(0.1f), deformationStrength(0.1f), isDeforming(false),
                         recordingBlendShape(false), isDragging(false),
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
        void startBlendShape(const std::string &name);
        void saveBlendShape();
        void cancelBlendShape();

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

        // Rendering overlay (simplified - just status info)
        void renderSelectionInfo() const;

        void renderVertexOverlay(TGAImage &framebuffer, int renderWidth, int renderHeight);
    };

    VertexEditor vertexEditor;
    bool vertexEditMode;

    // Expression cycling state
    int currentExpressionIndex;
    std::vector<std::string> availableExpressions;
    void updateAvailableExpressions();

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

    // NEW: Ray tracing functionality
    void rayTraceCurrentScene();
    void handleRayTracingInput();

    void toggleRealtimeRayTracing();

    // Vertex editing interface
    void enterVertexEditMode();
    void exitVertexEditMode();
    void toggleVertexDisplay();
    void setVertexEditMode(VertexEditor::EditMode mode);

    // Selection tools
    void selectVerticesInRadius(float radius);
    void clearVertexSelection();
    void selectAllVertices();
    void invertVertexSelection();

    // Deformation tools
    void setDeformationStrength(float strength);
    void setDeformationRadius(float radius);
    void setSelectionRadius(float radius);
    void resetVertexDeformation();

    // Blend shape tools
    void startRecordingBlendShape(const std::string &name);
    void saveCurrentBlendShape();
    void cancelBlendShape();

    // Enhanced blend shape playback
    void listSavedBlendShapes();
    void triggerExpression(const std::string &name, float intensity = 1.0f);
    void clearAllExpressions();
    void cycleToNextExpression();
    void cycleToPreviousExpression();
    void blendExpressions(const std::string &expr1, const std::string &expr2, float blend);

    // Utility
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
};

#endif // __ENGINE_H__