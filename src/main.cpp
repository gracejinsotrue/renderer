// main.cpp

#include <iostream>
#include <string>
#include "Engine.h"


int main(int argc, char **argv)
{

    // create engine instance
    Engine engine(1024, 768, 800, 800);

    engine.setUIEnabled(true);

    // initialize
    if (!engine.init())
    {
        std::cerr << "Failed to initialize engine!" << std::endl;
        return -1;
    }

    // --bench-present measures the two present paths against each other and
    // exits, rather than opening an interactive session.
    bool benchPresent = false;

    // Load models specified in command line
    if (argc > 1)
    {
        for (int i = 1; i < argc; i++)
        {
            if (std::string(argv[i]) == "--bench-present")
            {
                benchPresent = true;
                continue;
            }

            SceneNode *node = engine.loadModel(argv[i]);
            if (!node)
                std::cerr << "could not load " << argv[i] << std::endl;
        }
    }
    else
    {
        // No models named, so open the one scene that ships with the repository.
        // Paths are relative to the working directory; running from the repo
        // root is what the README documents.
        static const char *defaultScene[] = {
            "assets/toycar/ToyCar.obj", "assets/toycar/Fabric.obj",
            "assets/toycar/Glass.obj", "assets/scene/ground.obj",
        };
        int loaded = 0;
        for (const char *path : defaultScene)
            if (engine.loadModel(path))
                loaded++;

        if (loaded == 0)
            std::cerr << "No models loaded. Run from the repository root, or "
                         "pass a .obj on the command line." << std::endl;
        else
            std::cout << "Loaded the default scene: " << loaded << " meshes"
                      << std::endl;
    }

    // print scene hierarchy
    engine.getScene().printSceneHierarchy();

    if (benchPresent)
        return engine.benchmarkPresent() ? 0 : 1;

    engine.run();

    return 0;
}