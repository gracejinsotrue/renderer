// headless capture of an arbitrary set of models, for eyeballing a scene
// without a window. usage: capture_scene out.tga model.obj [model.obj ...]
#include <cstdio>
#include <cstdlib>
#include "Engine.h"

// the pre-port Engine.h pulls in shaders.h, which declares these extern
Model *model = NULL;
Vec3f light_dir(1, 1, 1);

int main(int argc, char **argv)
{
    if (argc < 3) { printf("usage: %s out.tga model.obj [model.obj ...]\n", argv[0]); return 2; }
    setenv("SDL_VIDEODRIVER", "dummy", 1);
    setenv("SDL_RENDER_DRIVER", "software", 1);

    Engine engine(1024, 768, 800, 800);
    if (!engine.init()) { printf("init failed\n"); return 1; }

    for (int i = 2; i < argc; i++)
        if (!engine.loadModel(argv[i])) printf("could not load %s\n", argv[i]);

    if (getenv("NO_SSAO") && engine.isSSAOEnabled()) engine.toggleSSAO();

    if (getenv("SSAO_RADIUS")) engine.setSSAORadius((float)atof(getenv("SSAO_RADIUS")));

    if (getenv("SSAO_DEBUG")) engine.setSSAODebug(atoi(getenv("SSAO_DEBUG")));

    if (getenv("ORBIT"))
        engine.orbitCamera((float)atof(getenv("ORBIT")), 0.0f);

    engine.update();
    engine.render();
    engine.captureFrame(argv[1]);
    int sub=0, cb=0, co=0, ov=0;
    cudaGetRasterStats(&sub, &cb, &co, &ov);
    printf("STATS submitted=%d culled_back=%d culled_off=%d overflow=%d\n", sub, cb, co, ov);
    printf("wrote %s\n", argv[1]);
    return 0;
}
