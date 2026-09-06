// per-frame cost as mesh count grows, which is what the batched setup launch
// changes. renders through the real Engine so the two passes are included.
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include "Engine.h"
using Clock = std::chrono::high_resolution_clock;

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "../obj/african_head.obj";
    setenv("SDL_VIDEODRIVER", "dummy", 1);
    setenv("SDL_RENDER_DRIVER", "software", 1);

    Engine engine(1024, 768, 800, 800);
    if (!engine.init()) { printf("init failed\n"); return 1; }
    if (!engine.isCudaAvailable()) { printf("no CUDA\n"); return 0; }

    for (int n : {1, 8, 20, 40})
    {
        engine.getScene().clear();
        for (int i = 0; i < n; i++)
        {
            char nm[16]; snprintf(nm, sizeof(nm), "M%d", i);
            SceneNode *s = engine.loadModel(path, nm);
            if (s) s->localTransform.position = Vec3f((i % 8) * 0.35f - 1.2f,
                                                      (i / 8) * 0.35f - 0.7f, 0.f);
        }
        const int F = 40;
        engine.update(); engine.render();          // warm up
        auto t0 = Clock::now();
        for (int f = 0; f < F; f++) { engine.update(); engine.render(); }
        engine.captureFrame("/tmp/_bench_meshes.tga");   // force a sync
        auto t1 = Clock::now();
        double per = std::chrono::duration<double, std::milli>(t1 - t0).count() / F;
        printf("%3d meshes : %6.3f ms/frame\n", n, per);
    }
    return 0;
}
