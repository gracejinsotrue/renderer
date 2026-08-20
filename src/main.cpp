// main.cpp

#include <iostream>
#include "Engine.h"
#include "shaders.h"


// global variables for shaders
Model *model = NULL;
Vec3f light_dir(1, 1, 1);
const int width = 800;
const int height = 800;

int main(int argc, char **argv)
{
    std::cout << "Starting Multi-Object 3D Engine with Interactive Vertex Editor..." << std::endl;

    // create engine instance
    Engine engine(1024, 768, 800, 800);

    // initialize
    if (!engine.init())
    {
        std::cerr << "Failed to initialize engine!" << std::endl;
        return -1;
    }

    // Load models specified in command line
    if (argc > 1)
    {
        for (int i = 1; i < argc; i++)
        {
            SceneNode *node = engine.loadModel(argv[i]);
            if (node)
            {
                std::cout << "Loaded: " << argv[i] << " as node '" << node->name << "'" << std::endl;
            }
        }
    }
    else
    {
        // load default models
        engine.loadModel("obj/test3.obj", "MainModel");

        // create some example objects for testing
        SceneNode *empty1 = engine.createEmptyNode("Controller1");
        empty1->setPosition(Vec3f(2, 0, 0));

        SceneNode *empty2 = engine.createEmptyNode("Controller2");
        empty2->setPosition(Vec3f(-2, 0, 0));

        std::cout << "Loaded default scene with test objects" << std::endl;
    }

    // set up global pointers for shaders temporarily
    std::vector<SceneNode *> meshNodes;
    engine.getScene().getAllMeshNodes(meshNodes);
    if (!meshNodes.empty())
    {
        model = meshNodes[0]->model;
        light_dir = engine.getScene().light.direction;
    }

    // print scene hierarchy
    engine.getScene().printSceneHierarchy();

    std::cout << "\n=== INTERACTIVE VERTEX EDITOR ENGINE ===" << std::endl;
    std::cout << "Load a head model and sculpt facial expressions interactively!" << std::endl;

    std::cout << "\n=== VERTEX EDITING (MAIN FEATURE) ===" << std::endl;
    std::cout << "  Ctrl+V - Enter/Exit interactive vertex edit mode" << std::endl;
    std::cout << "  === IN VERTEX EDIT MODE: ===" << std::endl;
    std::cout << "    1 - SELECT mode (click to select vertices)" << std::endl;
    std::cout << "    2 - DEFORM mode (drag selected vertices)" << std::endl;
    std::cout << "    3 - BLEND SHAPE mode (create expressions)" << std::endl;
    std::cout << "    Mouse Click - Select vertices in radius" << std::endl;
    std::cout << "    Mouse Drag - Sculpt/deform selected vertices" << std::endl;
    std::cout << "    Mouse Wheel - Adjust selection radius" << std::endl;
    std::cout << "    C - Clear selection | A - Select all | I - Invert" << std::endl;
    std::cout << "    +/- - Adjust deformation strength" << std::endl;
    std::cout << "    [/] - Adjust selection radius" << std::endl;
    std::cout << "    B - Start recording blend shape expression" << std::endl;
    std::cout << "    S - Save recorded expression" << std::endl;
    std::cout << "    R - Reset to original shape" << std::endl;
    std::cout << "    V - Toggle vertex display" << std::endl;
    std::cout << "    Esc - Exit vertex edit mode (or cancel blend shape)" << std::endl;

    std::cout << "\n=== REAL-TIME RAY TRACING ===" << std::endl;
    std::cout << "  U - Toggle real-time ray tracing" << std::endl;
    std::cout << "  +/- - Increase/decrease ray tracing quality (outside edit mode)" << std::endl;
    std::cout << "  [/] - Decrease/increase blend strength (outside edit mode)" << std::endl;
    std::cout << "  O - Toggle progress overlay" << std::endl;
    std::cout << "  M - Toggle adaptive quality" << std::endl;
    std::cout << "  , - Toggle tile boundaries (debug)" << std::endl;
    std::cout << "  J - Show detailed ray tracing status" << std::endl;
    std::cout << "  Y - Offline ray trace to PPM file" << std::endl;

    std::cout << "\n=== TRANSFORM SELECTED OBJECT ===" << std::endl;
    std::cout << "  CTRL + Numpad - Move object (4/6=X, 8/2=Z, +/-=Y)" << std::endl;
    std::cout << "  ALT + Numpad - Rotate object (4/6=Y, 8/2=X, 7/9=Z)" << std::endl;
    std::cout << "  SHIFT + Numpad +/- - Scale object uniformly" << std::endl;

    std::cout << "\n=== CAMERA CONTROLS ===" << std::endl;
    std::cout << "  Mouse + Left Click - Orbit/Look around (outside edit mode)" << std::endl;
    std::cout << "  Mouse Wheel - Zoom in/out (outside edit mode)" << std::endl;
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
    std::cout << "  B - Load background image (outside edit mode)" << std::endl;
    std::cout << "  C - Clear background (outside edit mode)" << std::endl;
    std::cout << "  ESC - Exit" << std::endl;

    std::cout << "\n=== WORKFLOW FOR FACIAL ANIMATION ===" << std::endl;
    std::cout << "1. Load/select a head model using TAB" << std::endl;
    std::cout << "2. Press Ctrl+V to enter vertex edit mode" << std::endl;
    std::cout << "3. Use mode 1 (SELECT) to click and select facial regions" << std::endl;
    std::cout << "4. Use mode 2 (DEFORM) to drag and sculpt expressions" << std::endl;
    std::cout << "5. Use mode 3 (BLEND SHAPE) to save expressions as blend shapes" << std::endl;
    std::cout << "6. Press B to name and record expressions" << std::endl;
    std::cout << "7. Press S to save the expression for later use" << std::endl;
    std::cout << "=========================================" << std::endl;
    engine.run();

    std::cout << "Engine shutting down..." << std::endl;
    return 0;
}