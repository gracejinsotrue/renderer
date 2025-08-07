// main.cpp

#include <iostream>
#include "Engine.h"
#include "shaders.h"

// Global variables for shaders
Model *model = NULL;
Vec3f light_dir(1, 1, 1);
const int width = 800;
const int height = 800;

int main(int argc, char **argv)
{
    std::cout << "Starting 3D Engine..." << std::endl;

    // Create engine instance
    Engine engine(1024, 768, 800, 800);

    // Initialize
    if (!engine.init())
    {
        std::cerr << "Failed to initialize engine!" << std::endl;
        return -1;
    }

    // Load additional models if specified
    if (argc > 1)
    {
        for (int i = 1; i < argc; i++)
        {
            engine.loadModel(argv[i]);
        }
    }
    else
    {
        engine.loadModel("obj/test3.obj");
    }

    // Set up global pointers for shaders  (This is a temp bridge solution, gotta fix)
    if (!engine.getScene().animatedModels.empty())
    {
        model = engine.getScene().animatedModels[0]->model;
        light_dir = engine.getScene().light.direction;
    }

    std::cout << "Engine running! Check console for controls." << std::endl;

    // Run the engine
    engine.run();

    std::cout << "Engine shutting down..." << std::endl;
    return 0;
}