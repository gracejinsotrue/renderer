// main.cpp

#include <iostream>
#include "Engine.h"
#include "shaders.h"

// Global variables for shaders (temporary bridge solution)
Model *model = NULL;
Vec3f light_dir(1, 1, 1);
const int width = 800;
const int height = 800;

int main(int argc, char **argv)
{
    std::cout << "Starting Multi-Object 3D Engine..." << std::endl;

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

    std::cout << "\nEngine running! New Controls:" << std::endl;
    std::cout << "\n=== OBJECT MANIPULATION ===" << std::endl;
    std::cout << "  TAB - Select next object" << std::endl;
    std::cout << "  SHIFT+TAB - Select previous object" << std::endl;
    std::cout << "  X - Delete selected object" << std::endl;
    std::cout << "  SHIFT+D - Duplicate selected object" << std::endl;
    std::cout << "  L - Load new model (test3.obj)" << std::endl;
    std::cout << "  N - Create empty node" << std::endl;
    std::cout << "  I - Print scene hierarchy" << std::endl;
    std::cout << "\n=== TRANSFORM SELECTED OBJECT ===" << std::endl;
    std::cout << "  CTRL + Numpad - Move object (4/6=X, 8/2=Z, +/-=Y)" << std::endl;
    std::cout << "  ALT + Numpad - Rotate object (4/6=Y, 8/2=X, 7/9=Z)" << std::endl;
    std::cout << "  SHIFT + Numpad +/- - Scale object uniformly" << std::endl;
    std::cout << "\n=== CAMERA CONTROLS (unchanged) ===" << std::endl;
    std::cout << "  Mouse + Left Click - Orbit/Look around" << std::endl;
    std::cout << "  WASD - Pan view / Move camera" << std::endl;
    std::cout << "  Q/E - Move up/down" << std::endl;
    std::cout << "  R/F - Zoom in/out" << std::endl;
    std::cout << "  G - Toggle camera mode" << std::endl;
    std::cout << "  H - Reset camera" << std::endl;
    std::cout << "\n=== OTHER CONTROLS ===" << std::endl;
    std::cout << "  Arrow keys - Move light source" << std::endl;
    std::cout << "  F - Toggle wireframe" << std::endl;
    std::cout << "  T - Toggle stats" << std::endl;
    std::cout << "  P - Capture frame" << std::endl;
    std::cout << "  ESC - Exit" << std::endl;

    // Run the engine
    engine.run();

    std::cout << "Engine shutting down..." << std::endl;
    return 0;
}