// main.cpp

#include <iostream>
#include "Engine.h"


int main(int argc, char **argv)
{

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
            if (!node)
                std::cerr << "could not load " << argv[i] << std::endl;
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

    // print scene hierarchy
    engine.getScene().printSceneHierarchy();

    engine.run();

    return 0;
}