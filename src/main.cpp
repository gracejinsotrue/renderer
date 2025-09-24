// main.cpp

#include <iostream>
#include "Engine.h"
#include "shaders.h"

// include our unified math system (this will include ray tracer headers)
#include "unified_math.h"

// global variables for shaders
Model *model = NULL;
Vec3f light_dir(1, 1, 1);
const int width = 800;
const int height = 800;

// tsest function for math system integration
void test_math_integration()
{
    std::cout << "\n=== TESTING UNIFIED MATH SYSTEM ===" << std::endl;

    // test 1: Basic conversion
    std::cout << "\n1. Testing basic vector conversion..." << std::endl;
    bool conversion_test = UnifiedMathTest::test_conversions();

    // test 2: Test unified math operations
    std::cout << "\n2. Testing unified math operations..." << std::endl;
    UnifiedVec3 v1(1.0f, 2.0f, 3.0f);
    UnifiedVec3 v2(4.0f, 5.0f, 6.0f);

    UnifiedVec3 cross_result = UnifiedMath::cross(v1, v2);
    float dot_result = UnifiedMath::dot(v1, v2);
    float length_result = UnifiedMath::length(v1);

    std::cout << "v1: (" << v1.x << ", " << v1.y << ", " << v1.z << ")" << std::endl;
    std::cout << "v2: (" << v2.x << ", " << v2.y << ", " << v2.z << ")" << std::endl;
    std::cout << "cross(v1, v2): (" << cross_result.x << ", " << cross_result.y << ", " << cross_result.z << ")" << std::endl;
    std::cout << "dot(v1, v2): " << dot_result << std::endl;
    std::cout << "length(v1): " << length_result << std::endl;

    // test 3: Ray conversion
    std::cout << "\n3. Testing ray conversion..." << std::endl;
    UnifiedRay unified_ray(UnifiedVec3(0, 0, 0), UnifiedVec3(1, 0, 0));
    rt_ray rt_ray_converted = unified_ray.to_rt_ray();
    UnifiedRay converted_back = UnifiedRay::from_rt_ray(rt_ray_converted);

    std::cout << "Original ray origin: (" << unified_ray.origin.x << ", " << unified_ray.origin.y << ", " << unified_ray.origin.z << ")" << std::endl;
    std::cout << "Converted back origin: (" << converted_back.origin.x << ", " << converted_back.origin.y << ", " << converted_back.origin.z << ")" << std::endl;

    // test 4: Color conversion
    std::cout << "\n4. Testing color conversion..." << std::endl;
    rt_color rt_col(0.5, 0.7, 0.3);
    TGAColor tga_color = ColorConversion::rt_color_to_tga(rt_col);
    rt_color converted_color = ColorConversion::tga_to_rt_color(tga_color);

    std::cout << "Original RT color: (" << rt_col.x() << ", " << rt_col.y() << ", " << rt_col.z() << ")" << std::endl;
    std::cout << "TGA color (RGB): (" << (int)tga_color[2] << ", " << (int)tga_color[1] << ", " << (int)tga_color[0] << ")" << std::endl;
    std::cout << "Converted back: (" << converted_color.x() << ", " << converted_color.y() << ", " << converted_color.z() << ")" << std::endl;

    std::cout << "\nMath integration test completed!" << std::endl;
    std::cout << "=====================================\n"
              << std::endl;
}

int main(int argc, char **argv)
{
    std::cout << "Starting Multi-Object 3D Engine with Interactive Vertex Editor..." << std::endl;

    // Test our unified math system first
    test_math_integration();

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